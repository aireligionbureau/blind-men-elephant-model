from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .agentic_search import SourceSelectionTrace, select_sources_with_filter
from .cohort import generate_live_cohort
from .config import DEFAULT_COHORT_MODEL, DEFAULT_MODEL, RUN_PROFILES
from .context import validate_artifact_questions, validate_run_isolation
from .evidence import build_evidence_ledger
from .json_utils import parse_json_content
from .meta_model import (
    attach_semantic_verdicts,
    diagnose_cognitive_shadows,
    evaluate_meta_model_overall,
    evaluate_semantic_verdict_strength,
    run_semantic_verdict_batch,
)
from .meta_model.semantic_verdicts import (
    load_semantic_verdict_records,
    summarize_semantic_verdict_usage,
)
from .model import DigitalPerson, digital_person_from_dict
from .prompts import render_digital_person_messages
from .providers import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekTimeoutError,
    retry_delay_seconds,
)
from .puzzle import build_shadow_puzzle
from .report import build_run_report
from .runtime import atomic_write_json
from .runner import run_filtered_retrieval
from .search import SearchResult
from .synthesis import (
    build_deterministic_synthesis,
    restore_or_build_synthesis,
    run_live_synthesis,
    write_synthesis_artifacts,
)


def run_live_batch(
    question: str,
    *,
    profile_name: str = "standard",
    provider_name: str = "three-layer",
    results_per_query: int = 5,
    model: str = DEFAULT_MODEL,
    cohort_model: str = DEFAULT_COHORT_MODEL,
    max_tokens: int = 8192,
    concurrency: int = 24,
    retries: int = 1,
    output_dir: str | Path | None = None,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
    semantic_verdicts: bool = True,
    semantic_max_tokens: int = 8192,
    relational_synthesis: bool = True,
    relational_max_tokens: int = 8192,
    retrieval_concurrency: int | None = None,
    semantic_concurrency: int | None = None,
    model_concurrency: int | None = None,
    pipeline_mode: str = "sequential",
    synthesis_mode: str = "legacy_sequential",
    recovery_passes: int = 0,
    time_budget_seconds: int = 570,
) -> dict[str, Any]:
    """Run the checkpointed production pipeline.

    The implementation is imported lazily so saved-run repair helpers in this
    module remain available without creating an import cycle.
    """

    from .pipeline import run_resilient_batch

    return run_resilient_batch(
        question,
        profile_name=profile_name,
        provider_name=provider_name,
        results_per_query=results_per_query,
        model=model,
        cohort_model=cohort_model,
        max_tokens=max_tokens,
        concurrency=concurrency,
        retrieval_concurrency=retrieval_concurrency,
        semantic_concurrency=semantic_concurrency,
        model_concurrency=model_concurrency,
        pipeline_mode=pipeline_mode,
        synthesis_mode=synthesis_mode,
        retries=retries,
        recovery_passes=recovery_passes,
        time_budget_seconds=time_budget_seconds,
        output_dir=output_dir,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
        semantic_verdicts=semantic_verdicts,
        semantic_max_tokens=semantic_max_tokens,
        relational_synthesis=relational_synthesis,
        relational_max_tokens=relational_max_tokens,
    )


def _run_live_batch_one_shot(
    question: str,
    *,
    profile_name: str = "standard",
    provider_name: str = "mock",
    results_per_query: int = 5,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    concurrency: int = 6,
    retries: int = 2,
    output_dir: str | Path | None = None,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
    semantic_verdicts: bool = True,
    semantic_max_tokens: int = 8192,
    relational_synthesis: bool = True,
    relational_max_tokens: int = 8192,
) -> dict[str, Any]:
    profile = RUN_PROFILES[profile_name]
    run_dir = _make_run_dir(output_dir, profile_name)
    people_dir = run_dir / "person_outputs"
    people_dir.mkdir(parents=True, exist_ok=True)

    cohort = generate_live_cohort(
        question,
        profile.person_count,
        model=model,
        retries=retries,
    )
    _write_json(run_dir / "cohort.json", cohort.to_dict())
    retrieval = run_filtered_retrieval(
        question,
        profile_name=profile_name,
        provider_name=provider_name,
        results_per_query=results_per_query,
        cohort=cohort,
        model=model,
        tool_retries=retries,
    )
    _write_json(run_dir / "retrieval.json", retrieval)

    personas = cohort.personas
    records_by_person = {record["person_id"]: record for record in retrieval["retrieval_records"]}
    ledgers_by_person = {ledger["person_id"]: ledger for ledger in retrieval["evidence_ledgers"]}

    started_at = datetime.now().isoformat(timespec="seconds")
    person_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_person_with_retries,
                question,
                person,
                _build_retrieval_context(person.id, records_by_person, ledgers_by_person),
                model,
                max_tokens,
                retries,
            ): person
            for person in personas
        }
        for future in as_completed(futures):
            person = futures[future]
            result = future.result()
            person_results.append(result)
            _write_json(people_dir / f"{person.id}.json", result)

    person_results = sorted(person_results, key=lambda item: item["person_id"])
    validate_run_isolation(
        question,
        retrieval["cohort_id"],
        retrieval["personas"],
        question_frame=retrieval.get("question_frame"),
        cohort_generation=retrieval.get("cohort_generation"),
        retrieval_records=retrieval.get("retrieval_records"),
        evidence_ledgers=retrieval.get("evidence_ledgers"),
        person_results=person_results,
    )
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in person_results
        if item["status"] == "succeeded"
    ]

    diagnosis = diagnose_cognitive_shadows(
        question,
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        person_outputs=successful_outputs,
    )
    semantic_records: list[dict[str, Any]] = []
    if semantic_verdicts:
        semantic_records = run_semantic_verdict_batch(
            question,
            retrieval["personas"],
            retrieval["evidence_ledgers"],
            successful_outputs,
            diagnosis,
            output_dir=run_dir / "semantic_verdicts",
            model=model,
            max_tokens=semantic_max_tokens,
            concurrency=concurrency,
            retries=retries,
        )
        attach_semantic_verdicts(diagnosis, semantic_records)
        diagnosis["semantic_verdict_evaluation"] = evaluate_semantic_verdict_strength(diagnosis)
        diagnosis["meta_model_overall_evaluation"] = evaluate_meta_model_overall(diagnosis)
    puzzle = build_shadow_puzzle(question, diagnosis, person_outputs=successful_outputs)
    if relational_synthesis:
        synthesis = run_live_synthesis(
            question,
            diagnosis,
            successful_outputs,
            retrieval["evidence_ledgers"],
            question_frame=retrieval.get("question_frame"),
            model=model,
            max_tokens=relational_max_tokens,
            retries=retries,
        )
    else:
        synthesis = build_deterministic_synthesis(
            question,
            diagnosis,
            successful_outputs,
            retrieval["evidence_ledgers"],
            question_frame=retrieval.get("question_frame"),
        )
    usage = summarize_usage(
        person_results,
        model=model,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    semantic_usage = summarize_semantic_verdict_usage(semantic_records, model=model)
    semantic_usage["pricing"] = _usage_pricing(
        semantic_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    cohort_usage = dict(cohort.generation.get("usage") or {})
    cohort_usage["model"] = cohort.generation.get("model") or model
    cohort_usage["pricing"] = _usage_pricing(
        cohort_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    search_tool_usage = dict(retrieval.get("search_tool_usage") or {})
    search_tool_usage["pricing"] = _usage_pricing(
        search_tool_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    relational_usage = dict(synthesis["relational_synthesis_usage"])
    relational_usage["pricing"] = _usage_pricing(
        relational_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    combined_usage = _combine_usage(
        usage,
        semantic_usage,
        cohort_usage,
        search_tool_usage,
        relational_usage,
    )

    completed_at = datetime.now().isoformat(timespec="seconds")
    result = {
        "question": question,
        "profile": asdict(profile),
        "cohort_id": retrieval["cohort_id"],
        "run_dir": str(run_dir),
        "started_at": started_at,
        "completed_at": completed_at,
        "model": model,
        "provider": provider_name,
        "results_per_query": results_per_query,
        "concurrency": concurrency,
        "retries": retries,
        "person_results": person_results,
        "cohort_generation": cohort.to_dict(),
        "cohort_generation_usage": cohort_usage,
        "search_tool_usage": search_tool_usage,
        "usage": usage,
        "semantic_verdict_usage": semantic_usage,
        "relational_synthesis_usage": relational_usage,
        "combined_usage": combined_usage,
        "semantic_verdict_results": semantic_records,
        "retrieval": retrieval,
        "meta_model": diagnosis,
        "shadow_puzzle": puzzle,
        "puzzle_materials": synthesis["puzzle_materials"],
        "detective": synthesis["detective"],
        "contour_constraints": synthesis["contour_constraints"],
        "contour_candidates": synthesis["contour_candidates"],
        "truth_contour": synthesis["truth_contour"],
    }
    _write_json(run_dir / "diagnosis.json", diagnosis)
    _write_json(run_dir / "shadow_puzzle.json", puzzle)
    write_synthesis_artifacts(run_dir, synthesis)
    _write_json(run_dir / "summary.json", _summary_payload(result))
    _write_json(run_dir / "run.json", result)
    build_run_report(run_dir)
    return result


def resume_live_batch(
    run_dir: str | Path,
    *,
    concurrency: int | None = None,
    retrieval_concurrency: int | None = None,
    semantic_concurrency: int | None = None,
    model_concurrency: int | None = None,
    pipeline_mode: str | None = None,
    synthesis_mode: str | None = None,
    retries: int | None = None,
    recovery_passes: int | None = None,
    time_budget_seconds: int | None = None,
) -> dict[str, Any]:
    """Resume only missing or invalid work in a checkpointed run."""

    from .pipeline import resume_resilient_batch

    return resume_resilient_batch(
        run_dir,
        concurrency=concurrency,
        retrieval_concurrency=retrieval_concurrency,
        semantic_concurrency=semantic_concurrency,
        model_concurrency=model_concurrency,
        pipeline_mode=pipeline_mode,
        synthesis_mode=synthesis_mode,
        retries=retries,
        recovery_passes=recovery_passes,
        time_budget_seconds=time_budget_seconds,
    )


def reselect_saved_run_sources(
    run_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 6,
    retries: int = 2,
    max_tokens: int = 8192,
    semantic_max_tokens: int = 8192,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    retrieval_path = run_path / "retrieval.json"
    retrieval = _read_json(retrieval_path)
    question = str(retrieval["question"])
    personas = [digital_person_from_dict(item) for item in retrieval["personas"]]
    records_by_person = {
        record["person_id"]: record
        for record in retrieval["retrieval_records"]
    }
    traces_by_person = {
        trace["person_id"]: trace
        for trace in retrieval.get("search_tool_traces", [])
    }

    cache_dir = run_path / "source_selection_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, dict[str, Any]] = {}
    pending_personas: list[DigitalPerson] = []
    for person in personas:
        cache_path = cache_dir / f"{person.id}.json"
        if not cache_path.exists():
            pending_personas.append(person)
            continue
        cached = _read_json(cache_path)
        trace_payload = cached.get("trace") or {}
        if cached.get("question") != question or not trace_payload.get("source_decisions"):
            pending_personas.append(person)
            continue
        selections[person.id] = {
            "trace": SourceSelectionTrace(
                source_decisions=trace_payload["source_decisions"],
                selection_summary=str(trace_payload.get("selection_summary") or ""),
                usage=trace_payload.get("usage") or {},
            ),
            "attempts": int(cached.get("attempts") or 1),
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _select_sources_with_retries,
                question,
                person,
                _external_results(records_by_person[person.id]),
                model,
                retries,
            ): person
            for person in pending_personas
        }
        for future in as_completed(futures):
            person = futures[future]
            selection = future.result()
            selections[person.id] = selection
            _write_json(
                cache_dir / f"{person.id}.json",
                {
                    "question": question,
                    "person_id": person.id,
                    "attempts": selection["attempts"],
                    "trace": asdict(selection["trace"]),
                },
            )

    ledgers: list[dict[str, Any]] = []
    selection_usage_records: list[dict[str, Any]] = []
    for person in personas:
        record = records_by_person[person.id]
        candidates = [SearchResult(**item) for item in record.get("candidate_sources", [])]
        selection = selections[person.id]
        trace: SourceSelectionTrace = selection["trace"]
        selection_usage_records.append(trace.usage)
        search_strategy = record.get("search_strategy") or {}
        search_strategy["selection_summary"] = trace.selection_summary
        search_strategy["source_selection_mode"] = "deepseek_semantic_filter"
        ledger = build_evidence_ledger(
            question,
            person,
            search_strategy,
            candidates,
            semantic_source_decisions=trace.source_decisions,
        )
        ledger_payload = asdict(ledger)
        ledgers.append(ledger_payload)
        record["search_strategy"] = search_strategy
        record["ledger_summary"] = {
            "accepted": len(ledger.accepted_sources),
            "rejected": len(ledger.rejected_sources),
            "ignored": len(ledger.ignored_sources),
            "information_shadow_hints": ledger.information_shadow_hints,
        }
        tool_trace = traces_by_person.get(person.id)
        if tool_trace is not None:
            tool_trace["source_decisions"] = trace.source_decisions
            tool_trace["selection_summary"] = trace.selection_summary
            tool_trace["selection_reprocessed"] = True
            tool_trace["selection_attempts"] = selection["attempts"]
            tool_trace["usage"] = _merge_usage_payloads([tool_trace.get("usage") or {}, trace.usage])

    selection_usage = _merge_usage_payloads(selection_usage_records)
    search_tool_usage = dict(retrieval.get("search_tool_usage") or {})
    search_tool_usage["calls"] = int(search_tool_usage.get("calls") or 0) + int(
        selection_usage.get("api_calls") or 0
    )
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        search_tool_usage[key] = int(search_tool_usage.get(key) or 0) + int(selection_usage.get(key) or 0)
    search_tool_usage["source_selection_calls"] = int(selection_usage.get("api_calls") or 0)
    retrieval["evidence_ledgers"] = ledgers
    retrieval["search_tool_usage"] = search_tool_usage
    retrieval["source_selection_reprocessed_at"] = datetime.now().isoformat(timespec="seconds")

    archive_dir = _archive_saved_run_stage(run_path)
    _write_json(retrieval_path, retrieval)

    people_dir = run_path / "person_outputs"
    people_dir.mkdir(parents=True, exist_ok=True)
    ledgers_by_person = {ledger["person_id"]: ledger for ledger in ledgers}
    person_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_person_with_retries,
                question,
                person,
                _build_retrieval_context(person.id, records_by_person, ledgers_by_person),
                model,
                max_tokens,
                retries,
            ): person
            for person in personas
        }
        for future in as_completed(futures):
            person = futures[future]
            person_result = future.result()
            person_results.append(person_result)
            _write_json(people_dir / f"{person.id}.json", person_result)
    person_results = sorted(person_results, key=lambda item: item["person_id"])

    validate_run_isolation(
        question,
        retrieval["cohort_id"],
        retrieval["personas"],
        question_frame=retrieval.get("question_frame"),
        cohort_generation=retrieval.get("cohort_generation"),
        retrieval_records=retrieval.get("retrieval_records"),
        evidence_ledgers=ledgers,
        person_results=person_results,
    )
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in person_results
        if item.get("status") == "succeeded"
    ]
    diagnosis = diagnose_cognitive_shadows(
        question,
        retrieval["personas"],
        ledgers,
        person_outputs=successful_outputs,
    )
    semantic_records = run_semantic_verdict_batch(
        question,
        retrieval["personas"],
        ledgers,
        successful_outputs,
        diagnosis,
        output_dir=run_path / "semantic_verdicts",
        model=model,
        max_tokens=semantic_max_tokens,
        concurrency=concurrency,
        retries=retries,
        resume=False,
    )
    result = rebuild_saved_run(run_path)
    reprocess_summary = {
        "archive_dir": str(archive_dir),
        "selection_cache_dir": str(cache_dir),
        "person_count": len(personas),
        "selection_usage": selection_usage,
        "external_decisions": _external_decision_counts(ledgers),
        "person_succeeded": sum(item.get("status") == "succeeded" for item in person_results),
        "person_failed": sum(item.get("status") != "succeeded" for item in person_results),
        "semantic_record_count": len(semantic_records),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["source_selection_reprocess"] = reprocess_summary
    _write_json(run_path / "source_selection_reprocess.json", reprocess_summary)
    _write_json(run_path / "run.json", result)
    return result


def repair_saved_run_person_outputs(
    run_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 6,
    retries: int = 2,
    max_tokens: int = 8192,
    semantic_max_tokens: int = 8192,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    retrieval = _read_json(run_path / "retrieval.json")
    question = str(retrieval["question"])
    personas = [digital_person_from_dict(item) for item in retrieval["personas"]]
    people_by_id = {item.get("person_id"): item for item in _read_person_results(run_path / "person_outputs")}
    invalid: list[tuple[DigitalPerson, list[str]]] = []
    for person in personas:
        saved = people_by_id.get(person.id) or {}
        errors = validate_digital_person_output(question, person.id, saved.get("output"))
        if saved.get("status") != "succeeded":
            errors.insert(0, f"saved status is {saved.get('status') or 'missing'}")
        if errors:
            invalid.append((person, errors))
    if not invalid:
        return rebuild_saved_run(run_path)

    records_by_person = {
        record["person_id"]: record
        for record in retrieval["retrieval_records"]
    }
    ledgers_by_person = {
        ledger["person_id"]: ledger
        for ledger in retrieval["evidence_ledgers"]
    }
    repaired: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_person_with_retries,
                question,
                person,
                _build_retrieval_context(person.id, records_by_person, ledgers_by_person),
                model,
                max_tokens,
                retries,
            ): person
            for person, _errors in invalid
        }
        for future in as_completed(futures):
            person = futures[future]
            repaired[person.id] = future.result()

    failed = [item for item in repaired.values() if item.get("status") != "succeeded"]
    if failed:
        _write_json(
            run_path / "person_output_repair_failures.json",
            {
                "question": question,
                "failed": failed,
                "attempted_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        failed_ids = ", ".join(item["person_id"] for item in failed)
        raise RuntimeError(f"Digital-person chain repair failed for: {failed_ids}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive_dir = run_path / "archive" / f"before-person-chain-repair-{stamp}"
    archived_people = archive_dir / "person_outputs"
    archived_verdicts = archive_dir / "semantic_verdicts"
    archived_people.mkdir(parents=True, exist_ok=False)
    archived_verdicts.mkdir(parents=True, exist_ok=False)
    semantic_dir = run_path / "semantic_verdicts"
    for person, _errors in invalid:
        person_path = run_path / "person_outputs" / f"{person.id}.json"
        verdict_path = semantic_dir / f"{person.id}.json"
        if person_path.exists():
            person_path.rename(archived_people / person_path.name)
        if verdict_path.exists():
            verdict_path.rename(archived_verdicts / verdict_path.name)
        _write_json(person_path, repaired[person.id])

    person_results = _read_person_results(run_path / "person_outputs")
    validate_run_isolation(
        question,
        retrieval["cohort_id"],
        retrieval["personas"],
        question_frame=retrieval.get("question_frame"),
        cohort_generation=retrieval.get("cohort_generation"),
        retrieval_records=retrieval.get("retrieval_records"),
        evidence_ledgers=retrieval.get("evidence_ledgers"),
        person_results=person_results,
    )
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in person_results
        if item.get("status") == "succeeded"
    ]
    diagnosis = diagnose_cognitive_shadows(
        question,
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        person_outputs=successful_outputs,
    )
    semantic_records = run_semantic_verdict_batch(
        question,
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        successful_outputs,
        diagnosis,
        output_dir=semantic_dir,
        model=model,
        max_tokens=semantic_max_tokens,
        concurrency=concurrency,
        retries=retries,
        resume=True,
    )
    result = rebuild_saved_run(run_path)
    repair_summary = {
        "archive_dir": str(archive_dir),
        "repaired_person_ids": [person.id for person, _errors in invalid],
        "repaired_count": len(invalid),
        "semantic_record_count": len(semantic_records),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["person_chain_repair"] = repair_summary
    _write_json(run_path / "person_chain_repair.json", repair_summary)
    _write_json(run_path / "run.json", result)
    return result


def _select_sources_with_retries(
    question: str,
    person: DigitalPerson,
    results: list[SearchResult],
    model: str,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            trace = select_sources_with_filter(
                question,
                person,
                results,
                model=model,
            )
            return {"trace": trace, "attempts": attempt + 1}
        except Exception as exc:  # noqa: BLE001 - saved-run recovery should retry transient model failures.
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"Source selection failed for {person.id}: {last_error}") from last_error


def _external_results(record: dict[str, Any]) -> list[SearchResult]:
    results = [
        SearchResult(**item)
        for item in record.get("candidate_sources", [])
        if item.get("retrieval_layer") == "external_search"
    ]
    if not results:
        raise RuntimeError(f"Saved retrieval has no external results for {record.get('person_id')}")
    return results


def _merge_usage_payloads(records: list[dict[str, Any]]) -> dict[str, Any]:
    active = [record for record in records if isinstance(record, dict) and record]
    prompt_tokens = sum(int(record.get("prompt_tokens") or record.get("input_tokens") or 0) for record in active)
    completion_tokens = sum(
        int(record.get("completion_tokens") or record.get("output_tokens") or 0)
        for record in active
    )
    total_tokens = sum(int(record.get("total_tokens") or 0) for record in active)
    api_calls = sum(int(record.get("api_calls") or 1) for record in active)
    return {
        "api_calls": api_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
    }


def _archive_saved_run_stage(run_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive_dir = run_path / "archive" / f"before-source-selection-{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for name in ["retrieval.json", "diagnosis.json", "shadow_puzzle.json", "summary.json", "run.json", "report.html"]:
        source = run_path / name
        if source.exists():
            shutil.copy2(source, archive_dir / name)
    for name in ["person_outputs", "semantic_verdicts"]:
        source = run_path / name
        if source.exists():
            source.rename(archive_dir / name)
    return archive_dir


def _external_decision_counts(ledgers: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"accept": 0, "reject": 0, "ignore": 0}
    for ledger in ledgers:
        for item in ledger.get("source_decisions", []):
            result = item.get("result") or {}
            decision = item.get("decision")
            if result.get("retrieval_layer") == "external_search" and decision in counts:
                counts[decision] += 1
    return counts


def summarize_usage(
    person_results: list[dict[str, Any]],
    *,
    model: str,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    succeeded = 0
    failed = 0
    for item in person_results:
        if item.get("status") == "succeeded":
            succeeded += 1
        else:
            failed += 1
        usage = item.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    input_price = _price_from_arg_or_env(
        input_price_per_1m_cny,
        "BME_INPUT_PRICE_PER_1M_CNY",
        "DEEPSEEK_INPUT_PRICE_PER_1M_CNY",
    )
    output_price = _price_from_arg_or_env(
        output_price_per_1m_cny,
        "BME_OUTPUT_PRICE_PER_1M_CNY",
        "DEEPSEEK_OUTPUT_PRICE_PER_1M_CNY",
    )
    cost = None
    if input_price is not None and output_price is not None:
        cost = round(prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price, 6)

    return {
        "model": model,
        "succeeded": succeeded,
        "failed": failed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "pricing": {
            "input_price_per_1m_cny": input_price,
            "output_price_per_1m_cny": output_price,
            "estimated_cost_cny": cost,
            "note": "若价格为空，请设置 BME_INPUT_PRICE_PER_1M_CNY 和 BME_OUTPUT_PRICE_PER_1M_CNY 后重新统计。",
        },
    }


def _run_person_with_retries(
    question: str,
    person: DigitalPerson,
    retrieval_context: dict[str, Any],
    model: str,
    max_tokens: int,
    retries: int,
    *,
    client_factory: Callable[[], DeepSeekClient] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt_context_mode = str(
        retrieval_context.get("_prompt_context_mode", "legacy_full")
    )
    attempts = 0
    last_error = None
    last_error_kind = "unknown"
    last_raw: dict[str, Any] = {}
    usage_records: list[dict[str, Any]] = []
    attempt_log: list[dict[str, Any]] = []
    base_messages = render_digital_person_messages(question, person, retrieval_context)
    messages = list(base_messages)
    token_limit = max_tokens
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            client = (
                client_factory()
                if client_factory is not None
                else DeepSeekClient(model=model)
            )
            response = client.chat(
                messages,
                max_tokens=token_limit,
                response_format={"type": "json_object"},
            )
            usage_records.append(response.usage)
            last_raw = response.raw
            output = parse_json_content(response.content)
            validation_errors = validate_digital_person_output(question, person.id, output)
            if validation_errors:
                last_error_kind = "validation"
                last_error = "Incomplete digital-person chain: " + "; ".join(validation_errors)
                finish_reason = str(
                    (((response.raw.get("choices") or [{}])[0]) or {}).get("finish_reason")
                    or ""
                )
                attempt_log.append(
                    {
                        "attempt": attempts,
                        "status": "invalid",
                        "finish_reason": finish_reason,
                        "errors": validation_errors,
                        "max_tokens": token_limit,
                    }
                )
                if finish_reason == "length":
                    token_limit = min(max(token_limit + 2048, int(token_limit * 1.5)), 32768)
                messages = list(base_messages) + [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": (
                            "上一版没有通过完整认知链校验。请保留本数字人的认知身份证，"
                            "完全重写一个完整 JSON，并修复："
                            + "；".join(validation_errors)
                        ),
                    },
                ]
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                break
            return {
                "person_id": person.id,
                "person_name": person.name,
                "person": asdict(person),
                "status": "succeeded",
                "attempts": attempts,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "output": output,
                "usage": _merge_usage_payloads(usage_records),
                "raw": response.raw,
                "attempt_log": attempt_log,
                "prompt_context_mode": prompt_context_mode,
            }
        except DeepSeekTimeoutError as exc:
            last_error = str(exc)
            last_error_kind = "uncertain_timeout"
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "uncertain_timeout",
                    "error": last_error,
                    "max_tokens": token_limit,
                }
            )
            break
        except DeepSeekAPIError as exc:
            last_error = str(exc)
            last_error_kind = (
                "transient_provider" if exc.retryable else "provider_configuration"
            )
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "retryable_provider_error" if exc.retryable else "terminal_provider_error",
                    "error": last_error,
                    "status_code": exc.status_code,
                    "max_tokens": token_limit,
                }
            )
            if not exc.retryable:
                break
            if attempt < retries:
                time.sleep(retry_delay_seconds(exc, attempt))
        except Exception as exc:  # noqa: BLE001 - batch runner must preserve failures.
            last_error = str(exc)
            last_error_kind = "provider_or_parse"
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "failed",
                    "error": last_error,
                    "max_tokens": token_limit,
                }
            )
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {
        "person_id": person.id,
        "person_name": person.name,
        "person": asdict(person),
        "status": "failed",
        "attempts": attempts,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "error": last_error,
        "error_kind": last_error_kind,
        "retryable": last_error_kind not in {
            "provider_configuration",
            "uncertain_timeout",
        },
        "output": {},
        "usage": _merge_usage_payloads(usage_records),
        "raw": last_raw,
        "attempt_log": attempt_log,
        "prompt_context_mode": prompt_context_mode,
    }


def validate_digital_person_output(
    question: str,
    person_id: str,
    output: Any,
) -> list[str]:
    if not isinstance(output, dict):
        return ["output is not a JSON object"]
    errors: list[str] = []
    if output.get("person_id") != person_id:
        errors.append("person_id is missing or mismatched")
    if output.get("question") != question:
        errors.append("question is missing or mismatched")
    for key in ["search_strategy", "source_layer_usage", "answer_position"]:
        if not isinstance(output.get(key), dict):
            errors.append(f"{key} is missing or not an object")
    for key in ["model_prior_before_search", "evidence_ledger", "value_judgements", "what_i_underweighted"]:
        if not isinstance(output.get(key), list):
            errors.append(f"{key} is missing or not a list")
    for key in ["core_assumptions", "reasoning_path", "what_would_change_my_mind"]:
        value = output.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{key} is missing or empty")
    if not str(output.get("conclusion") or "").strip():
        errors.append("conclusion is missing or empty")
    answer_position = output.get("answer_position")
    if isinstance(answer_position, dict) and not str(answer_position.get("exact_answer") or "").strip():
        errors.append("answer_position.exact_answer is missing or empty")
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        errors.append("confidence is missing or not numeric")
    return errors


def _build_retrieval_context(
    person_id: str,
    records_by_person: dict[str, dict[str, Any]],
    ledgers_by_person: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = records_by_person.get(person_id, {})
    ledger = ledgers_by_person.get(person_id, {})
    return {
        "question_context_id": record.get("question_context_id"),
        "question_frame": record.get("question_frame", {}),
        "search_strategy": record.get("search_strategy", {}),
        "retrieval_stack": record.get("retrieval_stack", {}),
        "candidate_sources": record.get("candidate_sources", []),
        "evidence_ledger": ledger,
        "source_layer_rules": {
            "model_prior": "搜索前先验，用来暴露你的默认记忆、历史类比、价值排序和注意力偏向；不能当作外部事实。",
            "external_search": "外部搜索材料，可以作为事实线索，但仍要按你的信息过滤器采信、忽略或排斥。",
            "mock_search": "模拟材料，只能用于流程测试，不能当作真实证据。",
        },
        "instruction": "只能按你的认知身份证处理当前问题；必须区分模型先验和外部证据，并说明你采信、排斥、低估了什么。任何上下文标识不一致的材料都必须拒绝。",
    }


def _make_run_dir(output_dir: str | Path | None, profile_name: str) -> Path:
    base = Path(output_dir) if output_dir is not None else Path("runs")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = base / f"{stamp}-{profile_name}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _price_from_arg_or_env(value: float | None, *env_names: str) -> float | None:
    if value is not None:
        return value
    for env_name in env_names:
        if (
            env_name.startswith("DEEPSEEK_")
            and os.getenv("BME_LLM_PROVIDER_PRESET", "deepseek") != "deepseek"
        ):
            continue
        raw = os.getenv(env_name)
        if raw is not None and raw.strip() != "":
            return float(raw)
    return None


def _summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": result["question"],
        "profile": result["profile"],
        "cohort_id": result["cohort_id"],
        "run_dir": result["run_dir"],
        "model": result["model"],
        "usage": result["usage"],
        "cohort_generation_usage": result.get("cohort_generation_usage", {}),
        "search_tool_usage": result.get("search_tool_usage", {}),
        "semantic_verdict_usage": result.get("semantic_verdict_usage", {}),
        "relational_synthesis_usage": result.get("relational_synthesis_usage", {}),
        "combined_usage": result.get("combined_usage", result["usage"]),
        "readiness_evaluation": result["meta_model"].get("readiness_evaluation", {}),
        "cognitive_chain_evaluation": result["meta_model"].get("cognitive_chain_evaluation", {}),
        "semantic_verdict_evaluation": result["meta_model"].get("semantic_verdict_evaluation", {}),
        "meta_model_overall_evaluation": result["meta_model"].get("meta_model_overall_evaluation", {}),
        "puzzle_quality": result["shadow_puzzle"].get("puzzle_quality", {}),
        "detective_evaluation": result.get("detective", {}).get("detective_evaluation", {}),
        "contour_evaluation": result.get("truth_contour", {}).get("contour_evaluation", {}),
        "failed_person_ids": [
            item["person_id"]
            for item in result["person_results"]
            if item["status"] != "succeeded"
        ],
    }


def rebuild_saved_run(
    run_dir: str | Path,
    *,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    retrieval = _read_json(run_path / "retrieval.json")
    run_payload = _read_json(run_path / "run.json") if (run_path / "run.json").exists() else {}
    person_results = _read_person_results(run_path / "person_outputs")
    validate_artifact_questions(retrieval["question"], {"run": run_payload})
    validate_run_isolation(
        retrieval["question"],
        retrieval["cohort_id"],
        retrieval["personas"],
        question_frame=retrieval.get("question_frame"),
        cohort_generation=retrieval.get("cohort_generation"),
        retrieval_records=retrieval.get("retrieval_records"),
        evidence_ledgers=retrieval.get("evidence_ledgers"),
        person_results=person_results,
    )

    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in person_results
        if item.get("status") == "succeeded"
    ]
    diagnosis = diagnose_cognitive_shadows(
        retrieval["question"],
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        person_outputs=successful_outputs,
    )
    semantic_records = load_semantic_verdict_records(run_path / "semantic_verdicts")
    if semantic_records:
        attach_semantic_verdicts(diagnosis, semantic_records)
        diagnosis["semantic_verdict_evaluation"] = evaluate_semantic_verdict_strength(diagnosis)
        diagnosis["meta_model_overall_evaluation"] = evaluate_meta_model_overall(diagnosis)
    puzzle = build_shadow_puzzle(retrieval["question"], diagnosis, person_outputs=successful_outputs)
    synthesis = restore_or_build_synthesis(
        run_path,
        retrieval["question"],
        diagnosis,
        successful_outputs,
        retrieval["evidence_ledgers"],
        question_frame=retrieval.get("question_frame"),
    )
    model = run_payload.get("model") or _read_model_from_summary(run_path) or DEFAULT_MODEL
    usage = summarize_usage(
        person_results,
        model=model,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    semantic_usage = summarize_semantic_verdict_usage(semantic_records, model=model)
    semantic_usage["pricing"] = _usage_pricing(
        semantic_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    cohort_usage = run_payload.get("cohort_generation_usage") or retrieval.get("cohort_generation", {}).get("usage", {})
    if cohort_usage and "pricing" not in cohort_usage:
        cohort_usage = {**cohort_usage, "model": model}
        cohort_usage["pricing"] = _usage_pricing(
            cohort_usage,
            input_price_per_1m_cny=input_price_per_1m_cny,
            output_price_per_1m_cny=output_price_per_1m_cny,
        )
    search_tool_usage = dict(retrieval.get("search_tool_usage") or run_payload.get("search_tool_usage") or {})
    search_tool_usage["pricing"] = _usage_pricing(
        search_tool_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    relational_usage = dict(synthesis["relational_synthesis_usage"])
    relational_usage["pricing"] = _usage_pricing(
        relational_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    result = {
        **run_payload,
        "question": retrieval["question"],
        "profile": run_payload.get("profile") or retrieval.get("profile"),
        "cohort_id": retrieval["cohort_id"],
        "run_dir": str(run_path),
        "model": model,
        "provider": run_payload.get("provider") or retrieval.get("search_provider"),
        "results_per_query": retrieval.get("results_per_query"),
        "person_results": person_results,
        "usage": usage,
        "cohort_generation_usage": cohort_usage,
        "search_tool_usage": search_tool_usage,
        "semantic_verdict_usage": semantic_usage,
        "relational_synthesis_usage": relational_usage,
        "combined_usage": _combine_usage(
            usage,
            semantic_usage,
            cohort_usage,
            search_tool_usage,
            relational_usage,
        ),
        "semantic_verdict_results": semantic_records,
        "retrieval": retrieval,
        "meta_model": diagnosis,
        "shadow_puzzle": puzzle,
        "puzzle_materials": synthesis["puzzle_materials"],
        "detective": synthesis["detective"],
        "contour_constraints": synthesis["contour_constraints"],
        "contour_candidates": synthesis["contour_candidates"],
        "truth_contour": synthesis["truth_contour"],
        "rebuilt_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_path / "diagnosis.json", diagnosis)
    _write_json(run_path / "shadow_puzzle.json", puzzle)
    write_synthesis_artifacts(run_path, synthesis)
    _write_json(run_path / "summary.json", _summary_payload(result))
    _write_json(run_path / "run.json", result)
    build_run_report(run_path)
    return result


def enrich_saved_run_with_semantic_verdicts(
    run_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    concurrency: int = 6,
    retries: int = 2,
    force: bool = False,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    base = rebuild_saved_run(
        run_path,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    retrieval = base["retrieval"]
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in base["person_results"]
        if item.get("status") == "succeeded"
    ]
    diagnosis = base["meta_model"]
    records = run_semantic_verdict_batch(
        retrieval["question"],
        retrieval["personas"],
        retrieval["evidence_ledgers"],
        successful_outputs,
        diagnosis,
        output_dir=run_path / "semantic_verdicts",
        model=model,
        max_tokens=max_tokens,
        concurrency=concurrency,
        retries=retries,
        resume=not force,
    )
    attach_semantic_verdicts(diagnosis, records)
    diagnosis["semantic_verdict_evaluation"] = evaluate_semantic_verdict_strength(diagnosis)
    diagnosis["meta_model_overall_evaluation"] = evaluate_meta_model_overall(diagnosis)
    puzzle = build_shadow_puzzle(retrieval["question"], diagnosis, person_outputs=successful_outputs)
    synthesis = build_deterministic_synthesis(
        retrieval["question"],
        diagnosis,
        successful_outputs,
        retrieval["evidence_ledgers"],
        question_frame=retrieval.get("question_frame"),
    )
    semantic_usage = summarize_semantic_verdict_usage(records, model=model)
    semantic_usage["pricing"] = _usage_pricing(
        semantic_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    relational_usage = dict(synthesis["relational_synthesis_usage"])
    relational_usage["pricing"] = _usage_pricing(
        relational_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    result = {
        **base,
        "model": model,
        "meta_model": diagnosis,
        "shadow_puzzle": puzzle,
        "semantic_verdict_results": records,
        "semantic_verdict_usage": semantic_usage,
        "relational_synthesis_usage": relational_usage,
        "puzzle_materials": synthesis["puzzle_materials"],
        "detective": synthesis["detective"],
        "contour_constraints": synthesis["contour_constraints"],
        "contour_candidates": synthesis["contour_candidates"],
        "truth_contour": synthesis["truth_contour"],
        "combined_usage": _combine_usage(
            base["usage"],
            semantic_usage,
            base.get("cohort_generation_usage"),
            base.get("search_tool_usage"),
            relational_usage,
        ),
        "semantic_verdicts_enriched_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_path / "diagnosis.json", diagnosis)
    _write_json(run_path / "shadow_puzzle.json", puzzle)
    write_synthesis_artifacts(run_path, synthesis)
    _write_json(run_path / "summary.json", _summary_payload(result))
    _write_json(run_path / "run.json", result)
    build_run_report(run_path)
    return result


def enrich_saved_run_with_truth_contour(
    run_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    retries: int = 2,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
) -> dict[str, Any]:
    """Run only the live detective and contour layers for a saved case."""

    run_path = Path(run_dir)
    previous_candidates = (
        _read_json(run_path / "contour_candidates.json")
        if (run_path / "contour_candidates.json").exists()
        else {}
    )
    base = rebuild_saved_run(
        run_path,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    retrieval = base["retrieval"]
    successful_outputs = [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in base["person_results"]
        if item.get("status") == "succeeded"
    ]
    synthesis = run_live_synthesis(
        retrieval["question"],
        base["meta_model"],
        successful_outputs,
        retrieval["evidence_ledgers"],
        question_frame=retrieval.get("question_frame"),
        model=model,
        max_tokens=max_tokens,
        retries=retries,
        resume_record=previous_candidates,
    )
    relational_usage = dict(synthesis["relational_synthesis_usage"])
    relational_usage["pricing"] = _usage_pricing(
        relational_usage,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    result = {
        **base,
        "model": model,
        "puzzle_materials": synthesis["puzzle_materials"],
        "detective": synthesis["detective"],
        "contour_constraints": synthesis["contour_constraints"],
        "contour_candidates": synthesis["contour_candidates"],
        "truth_contour": synthesis["truth_contour"],
        "relational_synthesis_usage": relational_usage,
        "combined_usage": _combine_usage(
            base["usage"],
            base.get("semantic_verdict_usage", {}),
            base.get("cohort_generation_usage"),
            base.get("search_tool_usage"),
            relational_usage,
        ),
        "truth_contour_enriched_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_synthesis_artifacts(run_path, synthesis)
    _write_json(run_path / "summary.json", _summary_payload(result))
    _write_json(run_path / "run.json", result)
    build_run_report(run_path)
    return result


def _read_person_results(people_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(people_dir.glob("*.json")):
        try:
            item = _read_json(path)
        except (OSError, json.JSONDecodeError):
            # A process may have died while an older non-atomic writer owned the
            # file. Treat it as a missing checkpoint so the resilient runner can
            # regenerate only this person.
            continue
        if not isinstance(item, dict):
            continue
        output = item.get("output")
        if isinstance(output, dict) and isinstance(output.get("raw_text"), str):
            recovered = parse_json_content(output["raw_text"])
            if isinstance(recovered, dict) and "raw_text" not in recovered:
                item["output"] = recovered
                item["recovered_from_raw_text"] = True
                _write_json(path, item)
        results.append(item)
    return sorted(results, key=lambda item: item.get("person_id", ""))


def _read_model_from_summary(run_path: Path) -> str | None:
    summary_path = run_path / "summary.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    return summary.get("model")


def _usage_pricing(
    usage: dict[str, Any],
    *,
    input_price_per_1m_cny: float | None,
    output_price_per_1m_cny: float | None,
) -> dict[str, Any]:
    input_price = _price_from_arg_or_env(
        input_price_per_1m_cny,
        "BME_INPUT_PRICE_PER_1M_CNY",
        "DEEPSEEK_INPUT_PRICE_PER_1M_CNY",
    )
    output_price = _price_from_arg_or_env(
        output_price_per_1m_cny,
        "BME_OUTPUT_PRICE_PER_1M_CNY",
        "DEEPSEEK_OUTPUT_PRICE_PER_1M_CNY",
    )
    cost = None
    if input_price is not None and output_price is not None:
        cost = round(
            int(usage.get("prompt_tokens") or 0) / 1_000_000 * input_price
            + int(usage.get("completion_tokens") or 0) / 1_000_000 * output_price,
            6,
        )
    return {
        "input_price_per_1m_cny": input_price,
        "output_price_per_1m_cny": output_price,
        "estimated_cost_cny": cost,
    }


def _combine_usage(
    person_usage: dict[str, Any],
    semantic_usage: dict[str, Any],
    cohort_usage: dict[str, Any] | None = None,
    search_tool_usage: dict[str, Any] | None = None,
    relational_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cohort_usage = cohort_usage or {}
    search_tool_usage = search_tool_usage or {}
    relational_usage = relational_usage or {}
    prompt_tokens = (
        int(person_usage.get("prompt_tokens") or 0)
        + int(semantic_usage.get("prompt_tokens") or 0)
        + int(cohort_usage.get("prompt_tokens") or cohort_usage.get("input_tokens") or 0)
        + int(search_tool_usage.get("prompt_tokens") or search_tool_usage.get("input_tokens") or 0)
        + int(relational_usage.get("prompt_tokens") or relational_usage.get("input_tokens") or 0)
    )
    completion_tokens = (
        int(person_usage.get("completion_tokens") or 0)
        + int(semantic_usage.get("completion_tokens") or 0)
        + int(cohort_usage.get("completion_tokens") or cohort_usage.get("output_tokens") or 0)
        + int(search_tool_usage.get("completion_tokens") or search_tool_usage.get("output_tokens") or 0)
        + int(relational_usage.get("completion_tokens") or relational_usage.get("output_tokens") or 0)
    )
    total_tokens = (
        int(person_usage.get("total_tokens") or 0)
        + int(semantic_usage.get("total_tokens") or 0)
        + int(cohort_usage.get("total_tokens") or 0)
        + int(search_tool_usage.get("total_tokens") or 0)
        + int(relational_usage.get("total_tokens") or 0)
    )
    person_cost = person_usage.get("pricing", {}).get("estimated_cost_cny")
    semantic_cost = semantic_usage.get("pricing", {}).get("estimated_cost_cny")
    cohort_cost = cohort_usage.get("pricing", {}).get("estimated_cost_cny")
    search_tool_cost = search_tool_usage.get("pricing", {}).get("estimated_cost_cny")
    relational_cost = relational_usage.get("pricing", {}).get("estimated_cost_cny")
    combined_cost = None
    if all(
        value is not None
        for value in [
            person_cost,
            semantic_cost,
            cohort_cost,
            search_tool_cost,
            relational_cost,
        ]
    ):
        combined_cost = round(
            float(person_cost)
            + float(semantic_cost)
            + float(cohort_cost)
            + float(search_tool_cost)
            + float(relational_cost),
            6,
        )
    return {
        "model": person_usage.get("model") or semantic_usage.get("model"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "digital_person_succeeded": person_usage.get("succeeded", 0),
        "digital_person_failed": person_usage.get("failed", 0),
        "cohort_generation_calls": 1 if cohort_usage else 0,
        "search_tool_calls": search_tool_usage.get("calls", 0),
        "executed_search_requests": search_tool_usage.get("executed_search_requests", 0),
        "external_result_count": search_tool_usage.get("external_result_count", 0),
        "semantic_verdict_succeeded": semantic_usage.get("succeeded", 0),
        "semantic_verdict_no_finding": semantic_usage.get("no_finding", 0),
        "semantic_verdict_fallback": semantic_usage.get("fallback", 0),
        "relational_synthesis_calls": relational_usage.get("calls", 0),
        "relational_synthesis_mode": relational_usage.get("mode", "not_run"),
        "estimated_cost_cny": combined_cost,
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
