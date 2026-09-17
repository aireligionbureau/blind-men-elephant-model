from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bme_model.cohort import build_fixture_cohort
from bme_model.front_pipeline import (
    StreamingFrontError,
    run_streaming_front_pipeline,
)
from bme_model.pipeline import (
    RunTimeBudgetExceeded,
    _cumulative_synthesis_usage,
    _execution_config,
    _require_run_budget,
)
from bme_model.providers import (
    DeepSeekTimeoutError,
    bounded_request_timeout_seconds,
)
from bme_model.runner import _run_one_retrieval_item
from bme_model.scheduler import AdaptiveConcurrencyGovernor
from bme_model.search import (
    FederatedHtmlSearchProvider,
    SearchResult,
)
from bme_model.synthesis import run_live_synthesis


def test_federated_search_races_sources_and_keeps_source_diversity() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return [_search_result(self.name, query)]

    provider = FederatedHtmlSearchProvider(
        [Provider("source-a"), Provider("source-b")]
    )
    batch = provider.search_with_trace("same question", limit=2)
    assert peak == 2
    assert batch.providers_succeeded == ["source-a", "source-b"]
    assert {item.source_name for item in batch.results} == {
        "source-a",
        "source-b",
    }


def test_federated_search_single_flight_fetches_identical_query_once() -> None:
    lock = threading.Lock()
    calls = 0

    class Provider:
        name = "single"

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.05)
            return [_search_result(self.name, query)]

    provider = FederatedHtmlSearchProvider([Provider()])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(provider.search_with_trace, "  Shared Query  ", limit=1),
            executor.submit(provider.search_with_trace, "shared query", limit=1),
        ]
        batches = [future.result() for future in futures]
    assert calls == 1
    assert sum(batch.cache_hit for batch in batches) == 1
    assert provider.search_with_trace("shared query", limit=1).cache_hit


def test_federated_search_opens_circuit_without_losing_other_sources() -> None:
    class FailingProvider:
        name = "blocked"

        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            self.calls += 1
            raise RuntimeError("verification challenge")

    class WorkingProvider:
        name = "working"

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            return [_search_result(self.name, query)]

    failing = FailingProvider()
    provider = FederatedHtmlSearchProvider(
        [failing, WorkingProvider()], circuit_failure_threshold=2
    )
    first = provider.search_with_trace("query one", limit=1)
    second = provider.search_with_trace("query two", limit=1)
    assert failing.calls == 1
    assert any("circuit_open" in item for item in second.providers_skipped)
    assert not second.provider_errors
    assert first.results and second.results


def test_unstable_provider_uses_one_cold_probe_across_parallel_queries() -> None:
    class FailingProvider:
        name = "blocked"

        def __init__(self) -> None:
            self.calls = 0
            self.lock = threading.Lock()

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            raise RuntimeError("verification challenge")

    class WorkingProvider:
        name = "working"

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            return [_search_result(self.name, query)]

    failing = FailingProvider()
    provider = FederatedHtmlSearchProvider(
        [failing, WorkingProvider()],
        single_flight_probe_names={"blocked"},
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(
            executor.map(
                lambda index: provider.search_with_trace(
                    f"query {index}", limit=1
                ),
                range(8),
            )
        )

    assert failing.calls == 1
    assert all(batch.results for batch in batches)
    assert sum(bool(batch.providers_skipped) for batch in batches) >= 7


def test_query_specific_empty_result_does_not_disable_provider_for_later_people() -> None:
    class SelectiveProvider:
        name = "selective"

        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            self.calls += 1
            if query == "niche query":
                raise RuntimeError("search returned no relevant results")
            return [_search_result(self.name, query)]

    class WorkingProvider:
        name = "working"

        def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
            return [_search_result(self.name, query)]

    selective = SelectiveProvider()
    provider = FederatedHtmlSearchProvider(
        [selective, WorkingProvider()],
        circuit_failure_threshold=1,
        single_flight_probe_names={"selective"},
    )

    first = provider.search_with_trace("niche query", limit=2)
    second = provider.search_with_trace("broad query", limit=2)

    assert first.results
    assert selective.calls == 2
    assert "selective" in second.providers_succeeded
    assert not any(
        item.startswith("selective: circuit_open")
        for item in second.providers_skipped
    )


def test_governor_caps_parallel_requests_without_changing_work() -> None:
    governor = AdaptiveConcurrencyGovernor(3)

    def work() -> int:
        with governor.slot():
            time.sleep(0.02)
            return 1

    with ThreadPoolExecutor(max_workers=12) as executor:
        values = list(executor.map(lambda _value: work(), range(18)))
    snapshot = governor.snapshot()
    assert sum(values) == 18
    assert snapshot.peak_active == 3
    assert snapshot.completed == 18


def test_optimized_modes_are_explicit_until_benchmark_activation() -> None:
    baseline = _execution_config()
    assert baseline["pipeline_mode"] == "sequential"
    assert baseline["synthesis_mode"] == "legacy_sequential"
    assert baseline["recovery_passes"] == 0
    assert baseline["time_budget_seconds"] == 570

    optimized = _execution_config(
        pipeline_mode="streaming_v2",
        synthesis_mode="optimized_v2",
    )
    assert optimized["concurrency"] == 24
    assert optimized["retrieval_concurrency"] == 12
    assert optimized["semantic_concurrency"] == 24
    assert optimized["model_concurrency"] == 24


def test_provider_timeout_fits_inside_remaining_run_budget() -> None:
    with patch("bme_model.providers.deepseek.time.time", return_value=100.0):
        assert bounded_request_timeout_seconds(200.0, 180) == 95
        try:
            bounded_request_timeout_seconds(108.0, 180)
            raise AssertionError("expected exhausted request budget")
        except DeepSeekTimeoutError:
            pass


def test_run_budget_stops_new_synthesis_work_before_deadline() -> None:
    journal = SimpleNamespace(
        payload={"created_at": "1970-01-01T00:01:40+00:00"}
    )
    with patch("bme_model.pipeline.time.time", return_value=155.0):
        try:
            _require_run_budget(
                journal,
                {"time_budget_seconds": 60},
                "relational_synthesis",
            )
            raise AssertionError("expected exhausted run budget")
        except RunTimeBudgetExceeded as exc:
            assert exc.retryable is False


def test_synthesis_usage_survives_targeted_recovery_passes() -> None:
    usage = _cumulative_synthesis_usage(
        [
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "calls": 2,
                "detective_usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 10,
                    "total_tokens": 90,
                },
            },
            {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
                "calls": 1,
                "contour_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 30,
                    "total_tokens": 80,
                },
            },
        ]
    )

    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 200
    assert usage["calls"] == 3
    assert usage["detective_usage"]["total_tokens"] == 90
    assert usage["contour_usage"]["total_tokens"] == 80


def test_speculative_contour_drafts_overlap_detective_adjudication() -> None:
    times: dict[str, float] = {}
    contour_kwargs: dict[str, object] = {}

    def fake_detective(*_args: object, **kwargs: object):
        callback = kwargs["proposal_ready_callback"]
        callback(
            {"question_id": "q1", "piece_index": []},
            [{"candidate_id": "candidate-1", "piece_ids": []}],
        )
        time.sleep(0.08)
        times["detective_end"] = time.perf_counter()
        return {"question_id": "q1"}, {
            "status": "succeeded",
            "usage": {},
            "audit": {"stages": []},
        }

    def fake_drafts(*_args: object, **_kwargs: object):
        times["draft_start"] = time.perf_counter()
        time.sleep(0.04)
        times["draft_end"] = time.perf_counter()
        bundle = {
            "mode": "speculative_parallel_v1",
            "main_draft": {"draft_id": "main"},
            "rival_drafts": [{"focus_id": "evidence_lineage"}],
        }
        return bundle, {
            "status": "succeeded",
            "usage": {},
            "audit": {"stages": []},
        }

    def fake_contour(*_args: object, **kwargs: object):
        contour_kwargs.update(kwargs)
        return {"direct_answer": "ok", "main_contour": "ok"}, {
            "status": "succeeded",
            "usage": {},
            "audit": {"stages": []},
        }

    with patch(
        "bme_model.synthesis.materialize_puzzle_pieces",
        return_value={"question_id": "q1", "pieces": []},
    ), patch(
        "bme_model.synthesis.run_live_detective",
        side_effect=fake_detective,
    ), patch(
        "bme_model.synthesis.run_provisional_contour_drafts",
        side_effect=fake_drafts,
    ), patch(
        "bme_model.synthesis.build_contour_constraints",
        return_value={"constraints": []},
    ), patch(
        "bme_model.synthesis.run_live_contour",
        side_effect=fake_contour,
    ):
        result = run_live_synthesis(
            "question",
            {},
            [],
            [],
            model="fixture-model",
            retries=0,
            synthesis_mode="optimized_v2",
        )

    assert times["draft_start"] < times["detective_end"]
    assert times["draft_end"] <= times["detective_end"]
    assert contour_kwargs["contour_mode"] == "speculative_parallel_v1"
    assert contour_kwargs["provisional_drafts"]
    assert result["contour_candidates"]["speculative_drafts"]["status"] == "succeeded"


def test_completed_speculative_drafts_checkpoint_before_detective_failure() -> None:
    draft_finished = threading.Event()
    checkpoints: list[dict] = []

    def fake_detective(*_args: object, **kwargs: object):
        callback = kwargs["proposal_ready_callback"]
        callback({"question": "question"}, [{"candidate_id": "candidate-1"}])
        assert draft_finished.wait(timeout=2)
        raise RuntimeError("adjudication interrupted")

    def fake_drafts(*_args: object, **_kwargs: object):
        record = {
            "status": "succeeded",
            "usage": {},
            "audit": {"stages": []},
        }
        draft_finished.set()
        return {"main_draft": {"draft_id": "main"}}, record

    with patch(
        "bme_model.synthesis.materialize_puzzle_pieces",
        return_value={"question_id": "q1", "pieces": []},
    ), patch(
        "bme_model.synthesis.run_live_detective",
        side_effect=fake_detective,
    ), patch(
        "bme_model.synthesis.run_provisional_contour_drafts",
        side_effect=fake_drafts,
    ):
        try:
            run_live_synthesis(
                "question",
                {},
                [],
                [],
                model="fixture-model",
                retries=0,
                synthesis_mode="optimized_v2",
                checkpoint_callback=checkpoints.append,
            )
        except RuntimeError as exc:
            assert "adjudication interrupted" in str(exc)
        else:
            raise AssertionError("detective interruption should propagate")

    assert checkpoints
    assert any(
        (item.get("speculative_drafts") or {}).get("status") == "succeeded"
        for item in checkpoints
    )


def test_governor_reduces_pressure_then_recovers_gradually() -> None:
    class PressureError(RuntimeError):
        status_code = 429
        retryable = True

    governor = AdaptiveConcurrencyGovernor(24, recovery_successes=3)
    try:
        with governor.slot():
            raise PressureError("rate limited")
    except PressureError:
        pass
    assert governor.snapshot().current_limit == 12
    for _ in range(3):
        with governor.slot():
            pass
    assert governor.snapshot().current_limit == 13


def test_streaming_front_starts_reasoning_before_slowest_search_finishes() -> None:
    question = "Should a city adopt a difficult long-term policy?"
    cohort = build_fixture_cohort(question, 4)
    lock = threading.Lock()
    retrieval_finished: dict[str, float] = {}
    person_started: dict[str, float] = {}

    def retrieval_runner(*args: object, **kwargs: object) -> dict:
        person = args[1]
        person_id = str(getattr(person, "id"))
        time.sleep(0.03 if "_p01_" in person_id else 0.12)
        result = _run_one_retrieval_item(*args, **kwargs)
        with lock:
            retrieval_finished[getattr(person, "id")] = time.perf_counter()
        return result

    def person_runner(
        value: str,
        person: object,
        _context: dict,
        _model: str,
        _max_tokens: int,
        _retries: int,
    ) -> dict:
        with lock:
            person_started[getattr(person, "id")] = time.perf_counter()
        time.sleep(0.02)
        return _person_result(value, person)

    with TemporaryDirectory() as root:
        result = run_streaming_front_pipeline(
            question,
            cohort,
            profile_name="standard",
            provider_name="mock",
            results_per_query=3,
            model="fixture-model",
            max_tokens=1024,
            semantic_verdicts=False,
            semantic_max_tokens=1024,
            retries=0,
            recovery_passes=0,
            retrieval_workers=4,
            people_workers=2,
            semantic_workers=2,
            model_concurrency=2,
            run_dir=root,
            retrieval_runner=retrieval_runner,
            person_runner=person_runner,
        )
    assert len(result.person_results) == 4
    assert min(person_started.values()) < max(retrieval_finished.values())
    assert result.metrics["mode"] == "streaming_v2"
    assert len(result.diagnosis["cognitive_chain_scans"]) == 4


def test_full_standard_front_pipeline_overlaps_all_three_stages() -> None:
    question = "Should a city make a difficult long-term investment?"
    cohort = build_fixture_cohort(question, 24)

    def retrieval_runner(*args: object, **kwargs: object) -> dict:
        time.sleep(0.06)
        return _run_one_retrieval_item(*args, **kwargs)

    def person_runner(
        value: str,
        person: object,
        _context: dict,
        _model: str,
        _max_tokens: int,
        _retries: int,
        **_kwargs: object,
    ) -> dict:
        time.sleep(0.04)
        return _person_result(value, person)

    def semantic_runner(
        case: dict,
        *,
        model: str,
        max_tokens: int,
        retries: int,
        **_kwargs: object,
    ) -> dict:
        del model, max_tokens, retries
        time.sleep(0.08)
        return {
            "person_id": case["person_id"],
            "person_name": case["person_name"],
            "status": "no_finding",
            "attempts": 1,
            "verdict": {
                "person_id": case["person_id"],
                "headline": "暂未检出需要语义裁决的问题。",
                "verdicts": [],
                "unresolved": ["测试运行只验证调度，不评价内容。"],
                "validation": {
                    "valid": True,
                    "accepted_verdict_count": 0,
                    "dropped_verdicts": [],
                },
            },
            "usage": {},
            "raw": {},
            "errors": [],
        }

    with TemporaryDirectory() as root:
        result = run_streaming_front_pipeline(
            question,
            cohort,
            profile_name="standard",
            provider_name="mock",
            results_per_query=3,
            model="fixture-model",
            max_tokens=1024,
            semantic_verdicts=True,
            semantic_max_tokens=1024,
            retries=0,
            recovery_passes=0,
            retrieval_workers=6,
            people_workers=6,
            semantic_workers=6,
            model_concurrency=6,
            run_dir=root,
            retrieval_runner=retrieval_runner,
            person_runner=person_runner,
            semantic_runner=semantic_runner,
        )
    stages = result.metrics["stages"]
    serial_envelope = sum(item["wall_seconds"] for item in stages.values())
    assert len(result.person_results) == 24
    assert len(result.semantic_records) == 24
    assert result.metrics["total_wall_seconds"] < serial_envelope * 0.75
    assert (
        stages["digital_people"]["started_offset_seconds"]
        < stages["retrieval"]["completed_offset_seconds"]
    )
    assert (
        stages["semantic_verdicts"]["started_offset_seconds"]
        < stages["digital_people"]["completed_offset_seconds"]
    )


def test_streaming_front_repairs_one_semantic_gap_without_stage_restart() -> None:
    question = "Should a city make a difficult long-term investment?"
    cohort = build_fixture_cohort(question, 4)
    target_id = cohort.personas[0].id
    calls: dict[str, int] = {}

    def person_runner(
        value: str,
        person: object,
        _context: dict,
        _model: str,
        _max_tokens: int,
        _retries: int,
        **_kwargs: object,
    ) -> dict:
        return _person_result(value, person)

    def semantic_runner(
        case: dict,
        *,
        model: str,
        max_tokens: int,
        retries: int,
        **_kwargs: object,
    ) -> dict:
        del model, max_tokens, retries
        person_id = case["person_id"]
        calls[person_id] = calls.get(person_id, 0) + 1
        if person_id == target_id and calls[person_id] == 1:
            return {
                "person_id": person_id,
                "person_name": case["person_name"],
                "status": "fallback",
                "attempts": 1,
                "retryable": True,
                "verdict": {},
                "usage": {},
                "raw": {},
                "errors": ["one evidence-lock gap"],
            }
        return {
            "person_id": person_id,
            "person_name": case["person_name"],
            "status": "no_finding",
            "attempts": 1,
            "verdict": {
                "person_id": person_id,
                "headline": "No semantic issue was established.",
                "verdicts": [],
                "unresolved": ["The fixture only checks targeted repair."],
                "validation": {
                    "valid": True,
                    "accepted_verdict_count": 0,
                    "dropped_verdicts": [],
                },
            },
            "usage": {},
            "raw": {},
            "errors": [],
        }

    with TemporaryDirectory() as root:
        result = run_streaming_front_pipeline(
            question,
            cohort,
            profile_name="standard",
            provider_name="mock",
            results_per_query=3,
            model="fixture-model",
            max_tokens=1024,
            semantic_verdicts=True,
            semantic_max_tokens=1024,
            retries=0,
            recovery_passes=0,
            retrieval_workers=4,
            people_workers=4,
            semantic_workers=4,
            model_concurrency=4,
            run_dir=root,
            person_runner=person_runner,
            semantic_runner=semantic_runner,
        )

    assert len(result.semantic_records) == 4
    assert calls[target_id] == 2
    assert all(count == 1 for person_id, count in calls.items() if person_id != target_id)


def test_streaming_front_harvests_successes_before_reporting_one_failure() -> None:
    question = "Should a city make a difficult long-term investment?"
    cohort = build_fixture_cohort(question, 4)
    failed_person_id = cohort.personas[0].id

    def person_runner(
        value: str,
        person: object,
        _context: dict,
        _model: str,
        _max_tokens: int,
        _retries: int,
        **_kwargs: object,
    ) -> dict:
        if getattr(person, "id") == failed_person_id:
            raise RuntimeError("one bounded person failure")
        time.sleep(0.04)
        return _person_result(value, person)

    with TemporaryDirectory() as root:
        try:
            run_streaming_front_pipeline(
                question,
                cohort,
                profile_name="standard",
                provider_name="mock",
                results_per_query=3,
                model="fixture-model",
                max_tokens=1024,
                semantic_verdicts=False,
                semantic_max_tokens=1024,
                retries=0,
                recovery_passes=0,
                retrieval_workers=4,
                people_workers=4,
                semantic_workers=2,
                model_concurrency=4,
                run_dir=root,
                person_runner=person_runner,
            )
            raise AssertionError("expected one person failure")
        except StreamingFrontError as exc:
            assert exc.stage == "digital_people"
            assert failed_person_id in exc.person_id

        saved_people = sorted((Path(root) / "person_outputs").glob("*.json"))
        assert len(saved_people) == 3
        assert all(failed_person_id not in path.name for path in saved_people)


def _search_result(source: str, query: str) -> SearchResult:
    return SearchResult(
        id=f"{source}-{query}",
        query=query,
        title=f"{source} result",
        url=f"https://{source}.example/item",
        snippet="independent evidence",
        source_name=source,
        source_type="primary",
        evidence_type="measurement",
    )


def _person_result(question: str, person: object) -> dict:
    person_id = getattr(person, "id")
    return {
        "person_id": person_id,
        "person_name": getattr(person, "name"),
        "person": asdict(person),
        "status": "succeeded",
        "attempts": 1,
        "duration_seconds": 0.02,
        "output": {
            "person_id": person_id,
            "question": question,
            "search_strategy": {},
            "source_layer_usage": {},
            "answer_position": {"exact_answer": "uncertain"},
            "model_prior_before_search": [],
            "evidence_ledger": [],
            "value_judgements": [],
            "what_i_underweighted": [],
            "core_assumptions": ["available evidence is incomplete"],
            "reasoning_path": ["check what the evidence can support"],
            "what_would_change_my_mind": ["new verifiable evidence"],
            "conclusion": "uncertain",
            "confidence": 0.4,
        },
        "usage": {},
        "raw": {},
    }
