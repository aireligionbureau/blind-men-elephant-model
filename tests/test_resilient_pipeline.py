from __future__ import annotations

import json
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bme_model.cohort import build_fixture_cohort
from bme_model.batch import _run_person_with_retries
from bme_model.model import select_personas
from bme_model.pipeline import (
    PipelineIncompleteError,
    _run_streaming_front_stages,
    _retrieval_quality,
    audit_resilient_run,
    resume_resilient_batch,
    run_resilient_batch,
)
from bme_model.front_pipeline import _explicit_external_gap_recorded
from bme_model.runner import (
    RetrievalStack,
    _run_one_retrieval_item,
    run_filtered_retrieval,
)
from bme_model.runtime import (
    RunAlreadyActiveError,
    RunJournal,
    RunLock,
    atomic_write_json,
    atomic_write_text,
    read_json,
)
from bme_model.providers import (
    DeepSeekAPIError,
    DeepSeekClient,
    OpenAICompatibleClient,
)


class RuntimeStateTests(unittest.TestCase):
    def test_manual_resume_starts_a_new_bounded_analysis_window(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = Path(root) / "resume-window"
            journal = RunJournal.create(
                run_dir,
                question="中断后的任务能否在新的受限窗口内恢复？",
                config={
                    "provider_name": "mock",
                    "model": "fixture-model",
                    "profile_name": "standard",
                },
                stage_names=["preflight"],
            )
            journal.payload["execution"] = {}
            journal.save()

            def capture_window(
                resumed: RunJournal,
                *,
                config: dict,
                execution: dict,
            ) -> dict:
                del config, execution
                return {
                    "analysis_window_started_at": resumed.payload.get(
                        "analysis_window_started_at"
                    )
                }

            with patch(
                "bme_model.pipeline._execute_pipeline",
                side_effect=capture_window,
            ):
                result = resume_resilient_batch(
                    run_dir,
                    renew_time_budget=True,
                )

            self.assertTrue(result["analysis_window_started_at"])

    def test_atomic_write_retries_a_transient_windows_replace_lock(self) -> None:
        with TemporaryDirectory() as root:
            target = Path(root) / "result.json"
            from bme_model import runtime

            real_replace = runtime.os.replace
            attempts = 0

            def flaky_replace(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("temporary scanner lock")
                real_replace(source, destination)

            with patch("bme_model.runtime.os.replace", side_effect=flaky_replace), patch(
                "bme_model.runtime.time.sleep"
            ):
                atomic_write_text(target, "ok")

            self.assertEqual(target.read_text(encoding="utf-8"), "ok")
            self.assertEqual(attempts, 2)

    def test_manifest_records_failure_and_resume_without_losing_history(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = Path(root) / "run"
            journal = RunJournal.create(
                run_dir,
                question="一个复杂问题",
                config={"model": "fixture"},
                stage_names=["cohort"],
            )
            journal.start_stage("cohort")
            journal.fail_stage("cohort", "temporary failure", retryable=True)

            resumed = RunJournal.load(run_dir)
            resumed.begin_resume()
            resumed.start_stage("cohort")
            artifact = run_dir / "cohort.json"
            atomic_write_json(artifact, {"ok": True})
            resumed.complete_stage("cohort", artifacts=[artifact])
            resumed.finish()

            saved = read_json(run_dir / "run_manifest.json")
            self.assertEqual(saved["status"], "succeeded")
            self.assertEqual(saved["resume_count"], 1)
            self.assertEqual(saved["stages"]["cohort"]["attempts"], 2)
            self.assertEqual(len(saved["stages"]["cohort"]["artifacts"]), 1)
            self.assertTrue(any(item["event"] == "stage_failed" for item in saved["events"]))

    def test_run_lock_prevents_duplicate_writer_and_releases_cleanly(self) -> None:
        with TemporaryDirectory() as root:
            with RunLock(root):
                with self.assertRaises(RunAlreadyActiveError):
                    with RunLock(root):
                        pass
            with RunLock(root):
                self.assertTrue((Path(root) / ".run.lock").exists())

    def test_terminal_provider_error_is_not_retried(self) -> None:
        question = "鉴权错误是否会被盲目重试？"
        person = select_personas(question, count=1)[0]

        class TerminalClient:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def chat(self, *_args: object, **_kwargs: object):
                raise DeepSeekAPIError(
                    "unauthorized",
                    status_code=401,
                    retryable=False,
                )

        with patch("bme_model.batch.DeepSeekClient", TerminalClient):
            result = _run_person_with_retries(
                question,
                person,
                {},
                "fixture-model",
                1024,
                3,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["error_kind"], "provider_configuration")

    def test_missing_api_key_is_a_terminal_configuration_error(self) -> None:
        with patch.dict(
            "os.environ",
            {"BME_LLM_API_KEY": "", "DEEPSEEK_API_KEY": ""},
        ):
            client = DeepSeekClient(api_key=None, model="fixture-model")
            with self.assertRaises(DeepSeekAPIError) as raised:
                client.chat([])
        self.assertFalse(raised.exception.retryable)

    def test_generic_provider_settings_take_priority_and_omit_unsupported_thinking(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BME_LLM_API_KEY": "generic-key",
                "DEEPSEEK_API_KEY": "legacy-key",
            },
        ):
            client = OpenAICompatibleClient(
                base_url="https://provider.example/v1",
                model="generic-model",
                supports_thinking=False,
            )
        captured: list[dict] = []

        def capture(payload: dict):
            captured.append(payload)
            return payload

        with patch.object(client, "_send_chat_request", side_effect=capture):
            result = client.chat(
                [{"role": "user", "content": "test"}],
                thinking={"type": "enabled"},
                reasoning_effort="high",
            )

        self.assertEqual(client.api_key, "generic-key")
        self.assertEqual(result["model"], "generic-model")
        self.assertNotIn("thinking", captured[0])
        self.assertNotIn("reasoning_effort", captured[0])


class RetrievalCheckpointTests(unittest.TestCase):
    def test_corrupt_retrieval_item_reruns_only_that_person(self) -> None:
        question = "城市是否应该延长图书馆开放时间？"
        cohort = build_fixture_cohort(question, 4)
        with TemporaryDirectory() as root:
            checkpoint_dir = Path(root)
            first = run_filtered_retrieval(
                question,
                person_count=4,
                provider_name="hybrid-mock",
                cohort=cohort,
                model="fixture-model",
                concurrency=3,
                checkpoint_dir=checkpoint_dir,
            )
            victim = sorted(checkpoint_dir.glob("*.json"))[0]
            victim.write_text("{broken", encoding="utf-8")

            with patch(
                "bme_model.runner._run_one_retrieval_item",
                wraps=_run_one_retrieval_item,
            ) as rerun:
                second = run_filtered_retrieval(
                    question,
                    person_count=4,
                    provider_name="hybrid-mock",
                    cohort=cohort,
                    model="fixture-model",
                    concurrency=3,
                    checkpoint_dir=checkpoint_dir,
                )

            self.assertEqual(rerun.call_count, 1)
            self.assertEqual(len(first["retrieval_records"]), 4)
            self.assertEqual(len(second["retrieval_records"]), 4)

    def test_empty_external_search_becomes_an_explicit_diagnosable_gap(self) -> None:
        question = "韩国股市会不会崩盘"
        cohort = build_fixture_cohort(question, 1)

        class EmptyProvider:
            name = "empty-live"

            def search(self, _query: str, *, limit: int = 5):
                del limit
                raise RuntimeError("all providers returned zero results")

        stack = RetrievalStack(
            name="model-prior+empty-live",
            external_provider=EmptyProvider(),
            include_model_prior=True,
            tolerate_external_errors=False,
            evidence_mode="three_layer_live",
        )
        item = _run_one_retrieval_item(
            question,
            cohort.personas[0],
            cohort,
            "three-layer",
            3,
            "fixture-model",
            0,
            stack=stack,
        )

        record = item["retrieval_record"]
        self.assertEqual(item["status"], "succeeded")
        self.assertEqual(
            record["external_search_status"],
            "unavailable_after_question_centered_recovery",
        )
        self.assertTrue(record["retrieval_stack"]["external_errors"])
        self.assertTrue(
            all(
                source["retrieval_layer"] == "model_prior"
                for source in record["candidate_sources"]
            )
        )
        self.assertTrue(_explicit_external_gap_recorded(item, stack))

    def test_three_layer_quality_allows_only_a_small_visible_gap(self) -> None:
        def record(index: int, *, external: bool) -> dict:
            return {
                "person_id": f"p{index:02d}",
                "candidate_sources": (
                    [{"retrieval_layer": "external_search"}] if external else []
                ),
            }

        one_gap = {
            "evidence_mode": "three_layer_live",
            "retrieval_records": [
                record(index, external=index != 24) for index in range(1, 25)
            ],
        }
        quality = _retrieval_quality(one_gap)
        self.assertEqual(quality["status"], "passed")
        self.assertEqual(quality["external_coverage"], 0.9583)
        self.assertEqual(quality["warnings"], [
            "1 digital people have no external search result"
        ])

        three_gaps = {
            "evidence_mode": "three_layer_live",
            "retrieval_records": [
                record(index, external=index <= 21) for index in range(1, 25)
            ],
        }
        blocked = _retrieval_quality(three_gaps)
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(len(blocked["blocking_person_ids"]), 3)


class ResilientPipelineTests(unittest.TestCase):
    def test_streaming_recovery_discards_only_failed_retrieval_checkpoints(self) -> None:
        question = "流式恢复是否只重试证据不合格的数字人？"
        cohort = build_fixture_cohort(question, 24)
        retrieval = {
            "evidence_mode": "three_layer_live",
            "retrieval_records": [
                {
                    "person_id": person.id,
                    "candidate_sources": (
                        [{"retrieval_layer": "external_search"}]
                        if index <= 21
                        else [{"retrieval_layer": "model_prior"}]
                    ),
                }
                for index, person in enumerate(cohort.personas, start=1)
            ],
        }
        with TemporaryDirectory() as root:
            run_dir = Path(root) / "run"
            journal = RunJournal.create(
                run_dir,
                question=question,
                config={"model": "fixture"},
                stage_names=["retrieval", "digital_people", "semantic_verdicts"],
            )
            item_dir = run_dir / "retrieval_items"
            item_dir.mkdir(parents=True)
            for person in cohort.personas:
                atomic_write_json(item_dir / f"{person.id}.json", {"ok": True})

            front = SimpleNamespace(
                retrieval=retrieval,
                diagnosis={},
                metrics={},
                person_results=[],
                changed_people=set(),
                semantic_records=[],
            )
            config = {
                "profile_name": "standard",
                "provider_name": "three-layer",
                "results_per_query": 5,
                "model": "fixture",
                "max_tokens": 100,
                "semantic_verdicts": True,
                "semantic_max_tokens": 100,
            }
            execution = {
                "retries": 0,
                "recovery_passes": 0,
                "retrieval_concurrency": 2,
                "concurrency": 2,
                "semantic_concurrency": 2,
                "model_concurrency": 2,
            }
            with patch(
                "bme_model.pipeline.run_streaming_front_pipeline",
                return_value=front,
            ), self.assertRaises(PipelineIncompleteError):
                _run_streaming_front_stages(
                    journal,
                    question,
                    cohort,
                    config,
                    execution,
                )

            for person in cohort.personas[:21]:
                self.assertTrue((item_dir / f"{person.id}.json").exists())
            for person in cohort.personas[21:]:
                self.assertFalse((item_dir / f"{person.id}.json").exists())

    def test_terminal_person_failures_do_not_enter_stage_recovery_loops(self) -> None:
        question = "鉴权失败时是否会停止整阶段盲目补跑？"
        call_count = 0

        def fixture_cohort(value: str, count: int, **_kwargs: object):
            return build_fixture_cohort(value, count)

        def terminal_person(
            _value: str,
            person: object,
            _context: dict,
            _model: str,
            _max_tokens: int,
            _retries: int,
        ) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "person_id": getattr(person, "id"),
                "person_name": getattr(person, "name"),
                "person": asdict(person),
                "status": "failed",
                "attempts": 1,
                "duration_seconds": 0.0,
                "error": "unauthorized",
                "error_kind": "provider_configuration",
                "retryable": False,
                "output": {},
                "usage": {},
            }

        with TemporaryDirectory() as root, patch(
            "bme_model.pipeline.generate_live_cohort", fixture_cohort
        ), patch(
            "bme_model.pipeline._run_person_with_retries", terminal_person
        ):
            with self.assertRaises(PipelineIncompleteError) as raised:
                run_resilient_batch(
                    question,
                    profile_name="standard",
                    provider_name="mock",
                    output_dir=root,
                    semantic_verdicts=False,
                    relational_synthesis=False,
                    concurrency=4,
                    retries=0,
                    recovery_passes=2,
                )
            self.assertEqual(raised.exception.stage, "digital_people")
            self.assertEqual(call_count, 24)
            manifest = read_json(raised.exception.run_dir / "run_manifest.json")
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(
                manifest["stages"]["digital_people"]["error"]["retryable"]
            )

    def test_transient_person_failure_is_recovered_in_same_run(self) -> None:
        question = "一次临时失败会不会拖垮整轮？"
        gate = threading.Lock()
        failed_once = False
        call_count = 0

        def fixture_cohort(value: str, count: int, **_kwargs: object):
            return build_fixture_cohort(value, count)

        def flaky_person(
            value: str,
            person: object,
            _context: dict,
            _model: str,
            _max_tokens: int,
            _retries: int,
        ) -> dict:
            nonlocal failed_once, call_count
            with gate:
                call_count += 1
                should_fail = not failed_once
                if should_fail:
                    failed_once = True
            if should_fail:
                return {
                    "person_id": getattr(person, "id"),
                    "person_name": getattr(person, "name"),
                    "person": asdict(person),
                    "status": "failed",
                    "attempts": 1,
                    "duration_seconds": 0.0,
                    "error": "temporary provider failure",
                    "error_kind": "provider_or_parse",
                    "output": {},
                    "usage": {},
                }
            return _person_result(value, person)

        with TemporaryDirectory() as root:
            with patch(
                "bme_model.pipeline.generate_live_cohort", fixture_cohort
            ), patch(
                "bme_model.pipeline._run_person_with_retries", flaky_person
            ):
                result = run_resilient_batch(
                    question,
                    profile_name="standard",
                    provider_name="mock",
                    output_dir=root,
                    semantic_verdicts=False,
                    relational_synthesis=False,
                    concurrency=4,
                    retries=0,
                    recovery_passes=1,
                )

            recovered = [
                item
                for item in result["person_results"]
                if item.get("recovery_history")
            ]
            self.assertEqual(result["run_quality"]["status"], "passed")
            self.assertEqual(call_count, 25)
            self.assertEqual(len(recovered), 1)
            manifest = read_json(Path(result["run_dir"]) / "run_manifest.json")
            progress = manifest["stages"]["digital_people"]["progress"]
            self.assertEqual(progress["completed"], 24)
            self.assertEqual(progress["total"], 24)

    def test_full_resume_repairs_only_corrupt_person_output(self) -> None:
        question = "这个系统能否稳定恢复？"

        def fixture_cohort(value: str, count: int, **_kwargs: object):
            return build_fixture_cohort(value, count)

        def fixture_person(
            value: str,
            person: object,
            _context: dict,
            _model: str,
            _max_tokens: int,
            _retries: int,
        ) -> dict:
            return _person_result(value, person)

        with TemporaryDirectory() as root:
            with patch(
                "bme_model.pipeline.generate_live_cohort", fixture_cohort
            ), patch(
                "bme_model.pipeline._run_person_with_retries", fixture_person
            ):
                first = run_resilient_batch(
                    question,
                    profile_name="standard",
                    provider_name="mock",
                    output_dir=root,
                    semantic_verdicts=False,
                    relational_synthesis=False,
                    concurrency=4,
                    retries=0,
                    recovery_passes=1,
                )

            run_dir = Path(first["run_dir"])
            victim = sorted((run_dir / "person_outputs").glob("*.json"))[0]
            victim.write_text("{broken", encoding="utf-8")
            calls: list[str] = []

            def counted_person(
                value: str,
                person: object,
                context: dict,
                model: str,
                max_tokens: int,
                retries: int,
            ) -> dict:
                calls.append(getattr(person, "id"))
                return fixture_person(value, person, context, model, max_tokens, retries)

            with patch(
                "bme_model.pipeline._run_person_with_retries", counted_person
            ):
                resumed = resume_resilient_batch(
                    run_dir,
                    retries=0,
                    recovery_passes=1,
                )

            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["run_quality"]["status"], "passed")
            self.assertEqual(resumed["run_quality"]["status"], "passed")
            self.assertEqual(len(calls), 1)
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["resume_count"], 1)
            audit = audit_resilient_run(run_dir)
            self.assertTrue(audit["passed"], audit)


def _person_result(question: str, person: object) -> dict:
    person_id = getattr(person, "id")
    person_name = getattr(person, "name")
    return {
        "person_id": person_id,
        "person_name": person_name,
        "person": asdict(person),
        "status": "succeeded",
        "attempts": 1,
        "duration_seconds": 0.0,
        "output": {
            "person_id": person_id,
            "question": question,
            "search_strategy": {},
            "source_layer_usage": {},
            "answer_position": {"exact_answer": "目前无法确定"},
            "model_prior_before_search": [],
            "evidence_ledger": [],
            "value_judgements": [],
            "what_i_underweighted": [],
            "core_assumptions": ["信息仍然有限"],
            "reasoning_path": ["检查当前材料能够支持到哪里"],
            "what_would_change_my_mind": ["出现可核验的新证据"],
            "conclusion": "目前无法确定",
            "confidence": 0.4,
        },
        "usage": {},
        "raw": {},
    }


if __name__ == "__main__":
    unittest.main()
