from __future__ import annotations

import hashlib
import inspect
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .batch import (
    _build_retrieval_context,
    _merge_usage_payloads,
    _read_person_results,
    _run_person_with_retries,
    validate_digital_person_output,
)
from .cohort import CohortBundle
from .config import RUN_PROFILES
from .context import validate_run_isolation
from .meta_model import (
    attach_semantic_verdicts,
    build_semantic_verdict_case,
    diagnose_cognitive_shadows,
    diagnose_one_cognitive_chain,
    evaluate_meta_model_overall,
    evaluate_semantic_verdict_strength,
    load_lenses,
    run_semantic_verdict_case,
    semantic_case_fingerprint,
)
from .meta_model.semantic_verdicts import load_semantic_verdict_records
from .model import DigitalPerson
from .providers import DeepSeekClient, bounded_request_timeout_seconds
from .runner import (
    RetrievalStack,
    _build_retrieval_stack,
    _load_retrieval_checkpoint,
    _run_one_retrieval_item,
    assemble_retrieval_result,
)
from .runtime import atomic_write_json
from .scheduler import AdaptiveConcurrencyGovernor


@dataclass(frozen=True)
class StreamingFrontResult:
    retrieval: dict[str, Any]
    person_results: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    semantic_records: list[dict[str, Any]]
    changed_people: set[str]
    metrics: dict[str, Any]


class StreamingFrontError(RuntimeError):
    def __init__(
        self,
        stage: str,
        person_id: str,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        self.stage = stage
        self.person_id = person_id
        self.retryable = retryable
        super().__init__(f"{stage} failed for {person_id}: {message}")


def run_streaming_front_pipeline(
    question: str,
    cohort: CohortBundle,
    *,
    profile_name: str,
    provider_name: str,
    results_per_query: int,
    model: str,
    max_tokens: int,
    semantic_verdicts: bool,
    semantic_max_tokens: int,
    retries: int,
    recovery_passes: int,
    retrieval_workers: int,
    people_workers: int,
    semantic_workers: int,
    model_concurrency: int,
    run_dir: str | Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    retrieval_runner: Callable[..., dict[str, Any]] = _run_one_retrieval_item,
    person_runner: Callable[..., dict[str, Any]] = _run_person_with_retries,
    semantic_runner: Callable[..., dict[str, Any]] = run_semantic_verdict_case,
    deadline_epoch: float | None = None,
) -> StreamingFrontResult:
    """Flow each person from retrieval to reasoning to semantic diagnosis.

    The work units and validators are identical to the sequential pipeline.
    Only readiness and scheduling determine when each unit starts.
    """

    run_path = Path(run_dir)
    retrieval_dir = run_path / "retrieval_items"
    people_dir = run_path / "person_outputs"
    verdict_dir = run_path / "semantic_verdicts"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    people_dir.mkdir(parents=True, exist_ok=True)
    verdict_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    stack = _build_retrieval_stack(provider_name)
    governor = AdaptiveConcurrencyGovernor(max(1, model_concurrency))
    user_id = "bme_" + hashlib.sha256(
        str(run_path.resolve()).encode("utf-8")
    ).hexdigest()[:24]
    client_factory = lambda: DeepSeekClient(  # noqa: E731 - injected factory.
        model=model,
        timeout_seconds=bounded_request_timeout_seconds(
            deadline_epoch, 180
        ),
        request_governor=governor,
        user_id=user_id,
    )
    personas = list(cohort.personas)
    persona_dicts = {person.id: asdict(person) for person in personas}
    lenses = load_lenses()

    retrieval_items: dict[str, dict[str, Any]] = {}
    person_results: dict[str, dict[str, Any]] = {}
    semantic_records: dict[str, dict[str, Any]] = {}
    semantic_inference_mode = "validated_fast_v1"
    chain_scans: dict[str, dict[str, Any]] = {}
    changed_people: set[str] = set()
    saved_people = {
        item.get("person_id"): item
        for item in _read_person_results(people_dir)
        if item.get("person_id")
    }
    saved_semantic = {
        item.get("person_id"): item
        for item in load_semantic_verdict_records(verdict_dir)
        if item.get("person_id")
    }
    attempts: dict[tuple[str, str], int] = {}
    futures: dict[Future[Any], tuple[str, DigitalPerson]] = {}
    terminal_failures: list[dict[str, Any]] = []
    # One in-stage semantic repair prevents a few invalid verdicts from making
    # the durable coordinator restart an otherwise complete 24-person front.
    semantic_repair_passes = max(1, recovery_passes)
    stage_marks = {
        name: {"started_offset_seconds": None, "completed_offset_seconds": None}
        for name in ("retrieval", "digital_people", "semantic_verdicts")
    }

    def mark_started(stage: str) -> None:
        if stage_marks[stage]["started_offset_seconds"] is None:
            stage_marks[stage]["started_offset_seconds"] = round(
                time.perf_counter() - started, 3
            )

    def mark_completed(stage: str) -> None:
        if stage_marks[stage]["completed_offset_seconds"] is None:
            stage_marks[stage]["completed_offset_seconds"] = round(
                time.perf_counter() - started, 3
            )

    def emit(activity: str) -> None:
        if not progress_callback:
            return
        progress_callback(
            {
                "activity": activity,
                "retrieval": {
                    "completed": len(retrieval_items),
                    "total": len(personas),
                },
                "digital_people": {
                    "completed": len(person_results),
                    "total": len(personas),
                },
                "semantic_verdicts": {
                    "completed": len(semantic_records),
                    "total": len(personas) if semantic_verdicts else 0,
                },
                "model_governor": governor.snapshot().to_dict(),
            }
        )

    def remember_failure(
        stage: str,
        person: DigitalPerson,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        terminal_failures.append(
            {
                "stage": stage,
                "person_id": person.id,
                "message": message,
                "retryable": retryable,
            }
        )
        emit(f"{stage}_failure_checkpointed")

    def submit_retrieval(
        executor: ThreadPoolExecutor,
        person: DigitalPerson,
        *,
        delay: float = 0.0,
    ) -> None:
        stage = "retrieval"
        attempts[(stage, person.id)] = attempts.get((stage, person.id), 0) + 1
        mark_started(stage)
        future = executor.submit(
            _delayed_retrieval,
            delay,
            retrieval_runner,
            question,
            person,
            cohort,
            provider_name,
            results_per_query,
            model,
            retries,
            stack,
            client_factory,
        )
        futures[future] = (stage, person)

    def submit_person(
        executor: ThreadPoolExecutor,
        person: DigitalPerson,
        item: dict[str, Any],
        *,
        delay: float = 0.0,
    ) -> None:
        stage = "digital_people"
        attempts[(stage, person.id)] = attempts.get((stage, person.id), 0) + 1
        mark_started(stage)
        context = _retrieval_context(person.id, item)
        future = executor.submit(
            _delayed_person,
            delay,
            person_runner,
            question,
            person,
            context,
            model,
            max_tokens,
            retries,
            client_factory,
        )
        futures[future] = (stage, person)

    def submit_semantic(
        executor: ThreadPoolExecutor,
        person: DigitalPerson,
        item: dict[str, Any],
        person_record: dict[str, Any],
        *,
        token_limit: int | None = None,
        delay: float = 0.0,
    ) -> None:
        if not semantic_verdicts:
            return
        stage = "semantic_verdicts"
        ledger = item["evidence_ledger"]
        output = person_record.get("output") or {}
        scan = diagnose_one_cognitive_chain(
            question,
            persona_dicts[person.id],
            ledger,
            output,
            lenses=lenses,
        )
        chain_scans[person.id] = scan
        case = build_semantic_verdict_case(
            question,
            persona_dicts[person.id],
            ledger,
            output,
            scan,
        )
        saved = saved_semantic.get(person.id) or {}
        if _valid_semantic_record(
            saved,
            case,
            inference_mode=semantic_inference_mode,
        ):
            semantic_records[person.id] = saved
            emit(stage)
            return
        attempts[(stage, person.id)] = attempts.get((stage, person.id), 0) + 1
        mark_started(stage)
        future = executor.submit(
            _delayed_semantic,
            delay,
            semantic_runner,
            case,
            model,
            token_limit or semantic_max_tokens,
            retries,
            client_factory,
            semantic_inference_mode,
        )
        setattr(future, "bme_case", case)
        futures[future] = (stage, person)

    retrieval_executor = ThreadPoolExecutor(max_workers=max(1, retrieval_workers))
    people_executor = ThreadPoolExecutor(max_workers=max(1, people_workers))
    semantic_executor = ThreadPoolExecutor(max_workers=max(1, semantic_workers))
    try:
        for person in personas:
            cached = _load_retrieval_checkpoint(
                retrieval_dir / f"{person.id}.json",
                question=question,
                cohort_id=cohort.cohort_id,
                person_id=person.id,
                provider_name=provider_name,
                results_per_query=results_per_query,
                model=model,
            )
            if cached and (
                _required_external_present(cached, stack)
                or _explicit_external_gap_recorded(cached, stack)
            ):
                retrieval_items[person.id] = cached
                _continue_from_retrieval(
                    person,
                    cached,
                    saved_people,
                    person_results,
                    question,
                    people_executor,
                    semantic_executor,
                    submit_person,
                    submit_semantic,
                )
            else:
                (retrieval_dir / f"{person.id}.json").unlink(missing_ok=True)
                submit_retrieval(retrieval_executor, person)
        emit("streaming_front")

        while futures:
            done, _pending = wait(
                list(futures), return_when=FIRST_COMPLETED
            )
            for future in done:
                stage, person = futures.pop(future)
                try:
                    value = future.result()
                except Exception as exc:  # noqa: BLE001 - retry only the failed unit.
                    retryable = bool(getattr(exc, "retryable", True))
                    if retryable and attempts[(stage, person.id)] <= recovery_passes:
                        delay = min(2 ** (attempts[(stage, person.id)] - 1), 4)
                        if stage == "retrieval":
                            submit_retrieval(
                                retrieval_executor, person, delay=delay
                            )
                            continue
                    remember_failure(
                        stage,
                        person,
                        str(exc),
                        retryable=retryable,
                    )
                    continue

                if stage == "retrieval":
                    if not _required_external_present(value, stack):
                        if attempts[(stage, person.id)] <= recovery_passes:
                            submit_retrieval(
                                retrieval_executor,
                                person,
                                delay=min(
                                    2 ** (attempts[(stage, person.id)] - 1), 4
                                ),
                            )
                            continue
                        if not _explicit_external_gap_recorded(value, stack):
                            remember_failure(
                                stage,
                                person,
                                "no external result survived retrieval",
                                retryable=True,
                            )
                            continue
                    atomic_write_json(
                        retrieval_dir / f"{person.id}.json", value
                    )
                    retrieval_items[person.id] = value
                    _continue_from_retrieval(
                        person,
                        value,
                        saved_people,
                        person_results,
                        question,
                        people_executor,
                        semantic_executor,
                        submit_person,
                        submit_semantic,
                    )
                    if len(retrieval_items) == len(personas):
                        mark_completed(stage)

                elif stage == "digital_people":
                    previous = saved_people.get(person.id) or {}
                    context = _retrieval_context(
                        person.id, retrieval_items[person.id]
                    )
                    fingerprint = _person_fingerprint(
                        question, person, context
                    )
                    value["pipeline_input_fingerprint"] = fingerprint
                    value = _merge_person_recovery(previous, value)
                    atomic_write_json(people_dir / f"{person.id}.json", value)
                    saved_people[person.id] = value
                    if value.get("status") != "succeeded" or validate_digital_person_output(
                        question, person.id, value.get("output")
                    ):
                        retryable = bool(value.get("retryable", True))
                        if retryable and attempts[(stage, person.id)] <= recovery_passes:
                            submit_person(
                                people_executor,
                                person,
                                retrieval_items[person.id],
                                delay=min(
                                    2 ** (attempts[(stage, person.id)] - 1), 4
                                ),
                            )
                            continue
                        remember_failure(
                            stage,
                            person,
                            str(value.get("error") or "invalid person output"),
                            retryable=retryable,
                        )
                        continue
                    person_results[person.id] = value
                    changed_people.add(person.id)
                    submit_semantic(
                        semantic_executor,
                        person,
                        retrieval_items[person.id],
                        value,
                    )
                    if len(person_results) == len(personas):
                        mark_completed(stage)

                else:
                    case = getattr(future, "bme_case")
                    value["input_fingerprint"] = semantic_case_fingerprint(
                        case,
                        execution_mode=semantic_inference_mode,
                    )
                    previous = saved_semantic.get(person.id) or {}
                    value = _merge_semantic_recovery(previous, value)
                    atomic_write_json(verdict_dir / f"{person.id}.json", value)
                    saved_semantic[person.id] = value
                    if value.get("status") == "fallback":
                        retryable = bool(value.get("retryable", True))
                        if (
                            retryable
                            and attempts[(stage, person.id)]
                            <= semantic_repair_passes
                        ):
                            submit_semantic(
                                semantic_executor,
                                person,
                                retrieval_items[person.id],
                                person_results[person.id],
                                token_limit=min(
                                    semantic_max_tokens
                                    * (2 ** attempts[(stage, person.id)]),
                                    32768,
                                ),
                                delay=min(
                                    2 ** (attempts[(stage, person.id)] - 1), 4
                                ),
                            )
                            continue
                        remember_failure(
                            stage,
                            person,
                            "; ".join(value.get("errors") or ["semantic fallback"]),
                            retryable=retryable,
                        )
                        continue
                    if not (value.get("verdict") or {}).get(
                        "validation", {}
                    ).get("valid"):
                        remember_failure(
                            stage,
                            person,
                            "semantic verdict failed evidence-lock validation",
                            retryable=False,
                        )
                        continue
                    semantic_records[person.id] = value
                    if len(semantic_records) == len(personas):
                        mark_completed(stage)
                emit(stage)
    finally:
        retrieval_executor.shutdown(wait=True, cancel_futures=False)
        people_executor.shutdown(wait=True, cancel_futures=False)
        semantic_executor.shutdown(wait=True, cancel_futures=False)

    if terminal_failures:
        stage_order = {
            "retrieval": 0,
            "digital_people": 1,
            "semantic_verdicts": 2,
        }
        ordered_failures = sorted(
            terminal_failures,
            key=lambda item: (
                stage_order.get(str(item.get("stage")), 99),
                str(item.get("person_id")),
            ),
        )
        primary_stage = str(ordered_failures[0]["stage"])
        failed_ids = ",".join(
            str(item["person_id"])
            for item in ordered_failures
            if item["stage"] == primary_stage
        )
        details = "; ".join(
            f"{item['stage']}/{item['person_id']}: {item['message']}"
            for item in ordered_failures[:8]
        )
        raise StreamingFrontError(
            primary_stage,
            failed_ids or "batch",
            details,
            retryable=all(
                bool(item.get("retryable", True))
                for item in ordered_failures
            ),
        )

    if len(retrieval_items) != len(personas):
        raise StreamingFrontError(
            "retrieval", "batch", "retrieval coverage is incomplete", retryable=True
        )
    if len(person_results) != len(personas):
        raise StreamingFrontError(
            "digital_people", "batch", "person coverage is incomplete", retryable=True
        )
    if semantic_verdicts and len(semantic_records) != len(personas):
        raise StreamingFrontError(
            "semantic_verdicts",
            "batch",
            "semantic coverage is incomplete",
            retryable=True,
        )
    if not semantic_verdicts:
        mark_completed("semantic_verdicts")

    ordered_retrieval = [retrieval_items[person.id] for person in personas]
    retrieval = assemble_retrieval_result(
        question,
        cohort,
        provider_name=provider_name,
        profile_payload=asdict(RUN_PROFILES[profile_name]),
        results_per_query=results_per_query,
        tool_retries=retries,
        model=model,
        stack=stack,
        ordered_items=ordered_retrieval,
    )
    ordered_people = [person_results[person.id] for person in personas]
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in ordered_people
    ]
    validate_run_isolation(
        question,
        cohort.cohort_id,
        retrieval["personas"],
        question_frame=retrieval.get("question_frame"),
        cohort_generation=retrieval.get("cohort_generation"),
        retrieval_records=retrieval.get("retrieval_records"),
        evidence_ledgers=retrieval.get("evidence_ledgers"),
        person_results=ordered_people,
    )
    diagnosis = diagnose_cognitive_shadows(
        question,
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        person_outputs=successful_outputs,
    )
    ordered_semantic = (
        [semantic_records[person.id] for person in personas]
        if semantic_verdicts
        else []
    )
    if semantic_verdicts:
        attach_semantic_verdicts(diagnosis, ordered_semantic)
        diagnosis["semantic_verdict_evaluation"] = (
            evaluate_semantic_verdict_strength(diagnosis)
        )
        diagnosis["meta_model_overall_evaluation"] = (
            evaluate_meta_model_overall(diagnosis)
        )

    total = round(time.perf_counter() - started, 3)
    for stage, marks in stage_marks.items():
        if marks["started_offset_seconds"] is None:
            marks["started_offset_seconds"] = 0.0
        if marks["completed_offset_seconds"] is None:
            marks["completed_offset_seconds"] = total
        marks["wall_seconds"] = round(
            float(marks["completed_offset_seconds"])
            - float(marks["started_offset_seconds"]),
            3,
        )
    metrics = {
        "schema": "bme.streaming-front-metrics.v1",
        "mode": "streaming_v2",
        "semantic_inference_mode": semantic_inference_mode,
        "digital_person_prompt_context_mode": "lossless_packet_v2",
        "total_wall_seconds": total,
        "stages": stage_marks,
        "workers": {
            "retrieval": max(1, retrieval_workers),
            "digital_people": max(1, people_workers),
            "semantic_verdicts": max(1, semantic_workers),
        },
        "model_governor": governor.snapshot().to_dict(),
        "reused": {
            "retrieval": len(personas)
            - sum(attempts.get(("retrieval", person.id), 0) > 0 for person in personas),
            "digital_people": len(personas)
            - sum(attempts.get(("digital_people", person.id), 0) > 0 for person in personas),
            "semantic_verdicts": (
                len(personas)
                - sum(
                    attempts.get(("semantic_verdicts", person.id), 0) > 0
                    for person in personas
                )
                if semantic_verdicts
                else 0
            ),
        },
    }
    return StreamingFrontResult(
        retrieval=retrieval,
        person_results=ordered_people,
        diagnosis=diagnosis,
        semantic_records=ordered_semantic,
        changed_people=changed_people,
        metrics=metrics,
    )


def _continue_from_retrieval(
    person: DigitalPerson,
    item: dict[str, Any],
    saved_people: dict[str, dict[str, Any]],
    person_results: dict[str, dict[str, Any]],
    question: str,
    people_executor: ThreadPoolExecutor,
    semantic_executor: ThreadPoolExecutor,
    submit_person: Callable[..., None],
    submit_semantic: Callable[..., None],
) -> None:
    context = _retrieval_context(person.id, item)
    fingerprint = _person_fingerprint(question, person, context)
    saved = saved_people.get(person.id) or {}
    if (
        saved.get("status") == "succeeded"
        and saved.get("pipeline_input_fingerprint") == fingerprint
        and not validate_digital_person_output(
            question, person.id, saved.get("output")
        )
    ):
        person_results[person.id] = saved
        submit_semantic(semantic_executor, person, item, saved)
        return
    submit_person(people_executor, person, item)


def _retrieval_context(person_id: str, item: dict[str, Any]) -> dict[str, Any]:
    context = _build_retrieval_context(
        person_id,
        {person_id: item["retrieval_record"]},
        {person_id: item["evidence_ledger"]},
    )
    context["_prompt_context_mode"] = "lossless_packet_v2"
    return context


def _person_fingerprint(
    question: str,
    person: DigitalPerson,
    context: dict[str, Any],
) -> str:
    encoded = json.dumps(
        {"question": question, "person": asdict(person), "retrieval": context},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_external_present(
    item: dict[str, Any], stack: RetrievalStack
) -> bool:
    if stack.evidence_mode not in {
        "three_layer_live",
        "hybrid_live",
        "live_external",
    }:
        return True
    count = sum(
        source.get("retrieval_layer") == "external_search"
        for source in (item.get("retrieval_record") or {}).get(
            "candidate_sources", []
        )
        or []
    )
    if stack.evidence_mode == "three_layer_live":
        return count > 0
    return True


def _explicit_external_gap_recorded(
    item: dict[str, Any], stack: RetrievalStack
) -> bool:
    """Permit a transparent, bounded evidence gap after targeted retries.

    This does not turn a model prior into external evidence. It only lets the
    remaining pipeline preserve and diagnose the failed search as a blind spot.
    The batch quality gate still limits how many such gaps a live run may carry.
    """

    if stack.evidence_mode != "three_layer_live":
        return False
    record = item.get("retrieval_record") or {}
    if record.get("external_search_status") != (
        "unavailable_after_question_centered_recovery"
    ):
        return False
    sources = record.get("candidate_sources") or []
    has_prior = any(
        source.get("retrieval_layer") == "model_prior" for source in sources
    )
    errors = (record.get("retrieval_stack") or {}).get("external_errors") or []
    trace = item.get("tool_trace") or {}
    return bool(has_prior and errors and trace.get("external_evidence_gap"))


def _valid_semantic_record(
    record: dict[str, Any],
    case: dict[str, Any],
    *,
    inference_mode: str = "provider_default",
) -> bool:
    if record.get("status") not in {"succeeded", "no_finding"}:
        return False
    if record.get("input_fingerprint") != semantic_case_fingerprint(
        case,
        execution_mode=inference_mode,
    ):
        return False
    return bool((record.get("verdict") or {}).get("validation", {}).get("valid"))


def _merge_person_recovery(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    if not previous or previous.get("status") == "succeeded":
        return current
    current["recovery_history"] = [
        *(previous.get("recovery_history") or []),
        {
            "status": previous.get("status"),
            "attempts": previous.get("attempts", 0),
            "error": previous.get("error"),
            "error_kind": previous.get("error_kind"),
            "usage": previous.get("usage", {}),
        },
    ][-10:]
    current["attempts"] = int(previous.get("attempts") or 0) + int(
        current.get("attempts") or 0
    )
    current["usage"] = _merge_usage_payloads(
        [previous.get("usage") or {}, current.get("usage") or {}]
    )
    return current


def _merge_semantic_recovery(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    if not previous or previous.get("status") in {"succeeded", "no_finding"}:
        return current
    current["recovery_history"] = [
        *(previous.get("recovery_history") or []),
        {
            "status": previous.get("status"),
            "attempts": previous.get("attempts", 0),
            "errors": previous.get("errors", []),
            "usage": previous.get("usage", {}),
        },
    ][-10:]
    current["attempts"] = int(previous.get("attempts") or 0) + int(
        current.get("attempts") or 0
    )
    current["usage"] = _merge_usage_payloads(
        [previous.get("usage") or {}, current.get("usage") or {}]
    )
    return current


def _supports_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _delayed_retrieval(
    delay: float,
    runner: Callable[..., dict[str, Any]],
    question: str,
    person: DigitalPerson,
    cohort: CohortBundle,
    provider_name: str,
    results_per_query: int,
    model: str,
    retries: int,
    stack: RetrievalStack,
    client_factory: Callable[[], DeepSeekClient],
) -> dict[str, Any]:
    if delay:
        time.sleep(delay)
    kwargs: dict[str, Any] = {}
    if _supports_keyword(runner, "stack"):
        kwargs["stack"] = stack
    if _supports_keyword(runner, "client_factory"):
        kwargs["client_factory"] = client_factory
    return runner(
        question,
        person,
        cohort,
        provider_name,
        results_per_query,
        model,
        retries,
        **kwargs,
    )


def _delayed_person(
    delay: float,
    runner: Callable[..., dict[str, Any]],
    question: str,
    person: DigitalPerson,
    context: dict[str, Any],
    model: str,
    max_tokens: int,
    retries: int,
    client_factory: Callable[[], DeepSeekClient],
) -> dict[str, Any]:
    if delay:
        time.sleep(delay)
    kwargs = (
        {"client_factory": client_factory}
        if _supports_keyword(runner, "client_factory")
        else {}
    )
    return runner(
        question,
        person,
        context,
        model,
        max_tokens,
        retries,
        **kwargs,
    )


def _delayed_semantic(
    delay: float,
    runner: Callable[..., dict[str, Any]],
    case: dict[str, Any],
    model: str,
    max_tokens: int,
    retries: int,
    client_factory: Callable[[], DeepSeekClient],
    inference_mode: str,
) -> dict[str, Any]:
    if delay:
        time.sleep(delay)
    return runner(
        case,
        model=model,
        max_tokens=max_tokens,
        retries=retries,
        client_factory=client_factory,
        inference_mode=inference_mode,
    )
