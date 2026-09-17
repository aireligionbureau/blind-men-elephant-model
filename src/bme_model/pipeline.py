from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch import (
    _build_retrieval_context,
    _combine_usage,
    _make_run_dir,
    _merge_usage_payloads,
    _read_person_results,
    _run_person_with_retries,
    _summary_payload,
    _usage_pricing,
    summarize_usage,
    validate_digital_person_output,
)
from .cohort import CohortBundle, generate_live_cohort
from .config import DEFAULT_COHORT_MODEL, DEFAULT_MODEL, RUN_PROFILES
from .context import validate_run_isolation
from .contour.schemas import validate_truth_contour
from .contour.live import (
    CONTOUR_PIPELINE_VERSION,
    FINAL_PROMPT_REVISION,
    SPECULATIVE_FINAL_PROMPT_REVISION,
)
from .detective.live import (
    ADJUDICATION_PROMPT_REVISION,
    DETECTIVE_PIPELINE_VERSION,
    PROPOSAL_PROMPT_REVISION,
)
from .detective.schemas import validate_detective_output, validate_puzzle_materials
from .front_pipeline import StreamingFrontError, run_streaming_front_pipeline
from .meta_model import (
    attach_semantic_verdicts,
    diagnose_cognitive_shadows,
    evaluate_meta_model_overall,
    evaluate_semantic_verdict_strength,
    run_semantic_verdict_batch,
)
from .meta_model.semantic_verdicts import summarize_semantic_verdict_usage
from .model import DigitalPerson, digital_person_from_dict
from .puzzle import build_shadow_puzzle
from .providers import (
    DeepSeekClient,
    bounded_request_timeout_seconds,
    synthesis_request_timeout_seconds,
)
from .report import build_run_report
from .runner import (
    RETRIEVAL_IMPLEMENTATION_REVISION,
    RetrievalBatchError,
    _build_retrieval_stack,
    run_filtered_retrieval,
)
from .runtime import (
    RunAlreadyActiveError,
    RunJournal,
    RunLock,
    RunStateError,
    artifact_record,
    atomic_write_json,
    read_json,
    utc_now,
)
from .scheduler import AdaptiveConcurrencyGovernor
from .synthesis import (
    build_deterministic_synthesis,
    run_live_synthesis,
    write_synthesis_artifacts,
)


PIPELINE_STAGES = [
    "preflight",
    "cohort",
    "retrieval",
    "digital_people",
    "diagnosis",
    "semantic_verdicts",
    "shadow_puzzle",
    "relational_synthesis",
    "finalize",
]
DEFAULT_RUN_TIME_BUDGET_SECONDS = 570


class RunTimeBudgetExceeded(RuntimeError):
    """Stop bounded recovery before a slow provider can turn into an hour loop."""

    retryable = False


class PipelineIncompleteError(RuntimeError):
    """A run stopped safely and can continue from its saved checkpoint."""

    def __init__(self, run_dir: Path, stage: str, error: Exception | str) -> None:
        self.run_dir = run_dir
        self.stage = stage
        self.original_error = error
        super().__init__(
            f"Run stopped at recoverable stage {stage!r}: {error}. "
            f"Resume from {run_dir}"
        )


def run_resilient_batch(
    question: str,
    *,
    profile_name: str = "standard",
    provider_name: str = "three-layer",
    results_per_query: int = 5,
    model: str = DEFAULT_MODEL,
    cohort_model: str = DEFAULT_COHORT_MODEL,
    max_tokens: int = 8192,
    concurrency: int = 24,
    retrieval_concurrency: int | None = None,
    semantic_concurrency: int | None = None,
    model_concurrency: int | None = None,
    pipeline_mode: str = "sequential",
    synthesis_mode: str = "legacy_sequential",
    retries: int = 1,
    recovery_passes: int = 0,
    time_budget_seconds: int = DEFAULT_RUN_TIME_BUDGET_SECONDS,
    output_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
    semantic_verdicts: bool = True,
    semantic_max_tokens: int = 8192,
    relational_synthesis: bool = True,
    relational_max_tokens: int = 8192,
) -> dict[str, Any]:
    if profile_name not in RUN_PROFILES:
        raise ValueError(f"Unknown run profile: {profile_name}")
    target_run_dir = (
        Path(run_dir)
        if run_dir is not None
        else _make_run_dir(output_dir, profile_name)
    )
    config = _immutable_config(
        profile_name=profile_name,
        provider_name=provider_name,
        results_per_query=results_per_query,
        model=model,
        cohort_model=cohort_model,
        max_tokens=max_tokens,
        semantic_verdicts=semantic_verdicts,
        semantic_max_tokens=semantic_max_tokens,
        relational_synthesis=relational_synthesis,
        relational_max_tokens=relational_max_tokens,
    )
    execution = _execution_config(
        concurrency=concurrency,
        retrieval_concurrency=retrieval_concurrency,
        semantic_concurrency=semantic_concurrency,
        model_concurrency=model_concurrency,
        pipeline_mode=pipeline_mode,
        synthesis_mode=synthesis_mode,
        retries=retries,
        recovery_passes=recovery_passes,
        time_budget_seconds=time_budget_seconds,
        input_price_per_1m_cny=input_price_per_1m_cny,
        output_price_per_1m_cny=output_price_per_1m_cny,
    )
    with RunLock(target_run_dir):
        if run_dir is not None:
            unexpected = [
                path.name
                for path in target_run_dir.iterdir()
                if path.name not in {"queued_job.json", ".run.lock"}
            ]
            if unexpected:
                raise RunStateError(
                    f"Explicit run directory is not empty: {target_run_dir} ({unexpected})"
                )
        journal = RunJournal.create(
            target_run_dir,
            question=question,
            config=config,
            stage_names=PIPELINE_STAGES,
        )
        journal.payload["execution"] = execution
        journal.save()
        return _execute_pipeline(journal, config=config, execution=execution)


def resume_resilient_batch(
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
    renew_time_budget: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    with RunLock(run_path):
        journal = (
            RunJournal.load(run_path)
            if (run_path / "run_manifest.json").exists()
            else _bootstrap_legacy_journal(run_path)
        )
        config = dict(journal.payload.get("config") or {})
        execution = dict(journal.payload.get("execution") or {})
        overrides = {
            "concurrency": concurrency,
            "retrieval_concurrency": retrieval_concurrency,
            "semantic_concurrency": semantic_concurrency,
            "model_concurrency": model_concurrency,
            "pipeline_mode": pipeline_mode,
            "synthesis_mode": synthesis_mode,
            "retries": retries,
            "recovery_passes": recovery_passes,
            "time_budget_seconds": time_budget_seconds,
        }
        for key, value in overrides.items():
            if value is not None:
                execution[key] = value
        execution = _execution_config(**execution)
        journal.assert_compatible(
            question=str(journal.payload.get("question") or ""),
            config=config,
        )
        journal.payload["execution"] = execution
        if renew_time_budget:
            journal.payload["analysis_window_started_at"] = utc_now()
        journal.begin_resume()
        return _execute_pipeline(journal, config=config, execution=execution)


def read_run_status(run_dir: str | Path) -> dict[str, Any]:
    journal = RunJournal.load(run_dir)
    stages = journal.payload.get("stages", {})
    failed_stage = next(
        (
            name
            for name in reversed(PIPELINE_STAGES)
            if ((stages.get(name) or {}).get("status"))
            in {"recoverable_failed", "failed"}
        ),
        None,
    )
    failed_record = (stages.get(failed_stage) or {}) if failed_stage else {}
    failed_error = failed_record.get("error") or {}
    return {
        "run_dir": str(Path(run_dir)),
        "run_id": journal.payload.get("run_id"),
        "question": journal.payload.get("question"),
        "status": journal.payload.get("status"),
        "current_stage": journal.payload.get("current_stage"),
        "failed_stage": failed_stage,
        "error": failed_error.get("message"),
        "resume_count": journal.payload.get("resume_count", 0),
        "created_at": journal.payload.get("created_at"),
        "updated_at": journal.payload.get("updated_at"),
        "completed_at": journal.payload.get("completed_at"),
        "stages": stages,
        "quality": journal.payload.get("quality", {}),
    }


def recover_incomplete_runs(
    output_dir: str | Path = "runs",
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Resume abandoned recoverable runs, suitable for service startup."""

    base = Path(output_dir)
    results: list[dict[str, Any]] = []
    if not base.exists():
        return results
    candidates = sorted(
        base.rglob("run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    for manifest_path in candidates:
        if len(results) >= max(0, limit):
            break
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        status = str(manifest.get("status") or "")
        if status not in {"initialized", "running", "recoverable_failed"}:
            continue
        run_dir = manifest_path.parent
        try:
            resumed = resume_resilient_batch(run_dir)
            results.append(
                {
                    "run_dir": str(run_dir),
                    "status": "succeeded",
                    "quality": (resumed.get("run_quality") or {}).get("status"),
                }
            )
        except RunAlreadyActiveError:
            results.append({"run_dir": str(run_dir), "status": "already_active"})
        except PipelineIncompleteError as exc:
            results.append(
                {
                    "run_dir": str(run_dir),
                    "status": "recoverable_failed",
                    "stage": exc.stage,
                    "error": str(exc.original_error),
                }
            )
        except Exception as exc:  # noqa: BLE001 - one damaged run must not block the rest.
            results.append(
                {
                    "run_dir": str(run_dir),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return results


def audit_resilient_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify persisted recovery state without calling a model or search source."""

    run_path = Path(run_dir)
    violations: list[str] = []
    warnings: list[str] = []
    try:
        journal = RunJournal.load(run_path)
    except Exception as exc:
        return {
            "run_dir": str(run_path),
            "passed": False,
            "violations": [str(exc)],
            "warnings": [],
        }
    manifest = journal.payload
    if manifest.get("status") != "succeeded":
        violations.append(f"run status is {manifest.get('status')}")
    for name in PIPELINE_STAGES:
        status = ((manifest.get("stages") or {}).get(name) or {}).get("status")
        if status not in {"succeeded", "reused"}:
            violations.append(f"stage {name} is {status or 'missing'}")
    if (manifest.get("quality") or {}).get("status") != "passed":
        violations.append("final quality gate did not pass")
    if (run_path / ".run.lock").exists():
        warnings.append("run lock exists; the run may currently be active")
    temporary_files = [
        str(path.relative_to(run_path))
        for path in run_path.rglob("*.tmp")
    ]
    if temporary_files:
        violations.append("orphaned atomic-write temp files are present")

    latest_artifacts: dict[str, dict[str, Any]] = {}
    for name in PIPELINE_STAGES:
        stage = ((manifest.get("stages") or {}).get(name) or {})
        for item in stage.get("artifacts", []) or []:
            if item.get("path"):
                latest_artifacts[str(item["path"])] = item
    digest_mismatches: list[str] = []
    for relative, expected in latest_artifacts.items():
        path = Path(relative)
        target = path if path.is_absolute() else run_path / path
        if not target.exists():
            digest_mismatches.append(f"missing:{relative}")
            continue
        actual = artifact_record(target)
        if (
            actual.get("sha256") != expected.get("sha256")
            or actual.get("size_bytes") != expected.get("size_bytes")
        ):
            digest_mismatches.append(f"changed:{relative}")
    if digest_mismatches:
        violations.append("artifact integrity mismatch")

    run_payload = _read_optional_json(run_path / "run.json")
    expected_people = int(
        ((run_payload.get("profile") or {}).get("person_count")) or 0
    )
    person_results = _read_person_results(run_path / "person_outputs")
    valid_people = sum(
        item.get("status") == "succeeded"
        and not validate_digital_person_output(
            str(run_payload.get("question") or ""),
            str(item.get("person_id") or ""),
            item.get("output"),
        )
        for item in person_results
    )
    if expected_people and valid_people != expected_people:
        violations.append(
            f"valid digital-person outputs are {valid_people}/{expected_people}"
        )
    return {
        "run_dir": str(run_path),
        "passed": not violations,
        "status": manifest.get("status"),
        "resume_count": manifest.get("resume_count", 0),
        "valid_person_count": valid_people,
        "expected_person_count": expected_people,
        "verified_artifact_count": len(latest_artifacts),
        "artifact_digest_mismatches": digest_mismatches,
        "orphaned_temp_files": temporary_files,
        "violations": violations,
        "warnings": warnings,
    }


def _execute_pipeline(
    journal: RunJournal,
    *,
    config: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    run_path = journal.run_dir
    question = str(journal.payload.get("question") or "").strip()
    profile = RUN_PROFILES[str(config["profile_name"])]
    legacy = bool(journal.payload.get("legacy_bootstrap"))

    _run_preflight(journal, question, config, execution)
    cohort, cohort_reused = _run_cohort_stage(
        journal,
        question,
        profile.person_count,
        config,
        execution,
    )
    if execution.get("pipeline_mode") == "streaming_v2":
        (
            retrieval,
            person_results,
            diagnosis,
            semantic_records,
        ) = _run_streaming_front_stages(
            journal,
            question,
            cohort,
            config,
            execution,
        )
    else:
        retrieval, retrieval_reused = _run_retrieval_stage(
            journal,
            question,
            cohort,
            config,
            execution,
        )
        person_results, changed_people = _run_people_stage(
            journal,
            question,
            cohort.personas,
            retrieval,
            config,
            execution,
            allow_legacy=(legacy and cohort_reused and retrieval_reused),
        )
        successful_outputs = _successful_person_outputs(person_results)
        diagnosis = _run_diagnosis_stage(
            journal,
            question,
            retrieval,
            successful_outputs,
        )
        diagnosis, semantic_records = _run_semantic_stage(
            journal,
            question,
            retrieval,
            successful_outputs,
            diagnosis,
            config,
            execution,
            changed_people=changed_people,
        )
    successful_outputs = _successful_person_outputs(person_results)
    puzzle = _run_puzzle_stage(
        journal,
        question,
        diagnosis,
        successful_outputs,
    )
    synthesis = _run_synthesis_stage(
        journal,
        question,
        retrieval,
        successful_outputs,
        diagnosis,
        config,
        execution,
        allow_legacy=legacy,
    )
    return _finalize_run(
        journal,
        question,
        profile,
        cohort,
        retrieval,
        person_results,
        semantic_records,
        diagnosis,
        puzzle,
        synthesis,
        config,
        execution,
    )


def _run_preflight(
    journal: RunJournal,
    question: str,
    config: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    stage = "preflight"
    journal.start_stage(stage)
    try:
        if not question:
            raise ValueError("question is empty")
        if int(execution["concurrency"]) < 1:
            raise ValueError("concurrency must be positive")
        if int(execution["retrieval_concurrency"]) < 1:
            raise ValueError("retrieval_concurrency must be positive")
        if int(execution["semantic_concurrency"]) < 1:
            raise ValueError("semantic_concurrency must be positive")
        if int(execution["model_concurrency"]) < 1:
            raise ValueError("model_concurrency must be positive")
        if int(execution["retries"]) < 0 or int(execution["recovery_passes"]) < 0:
            raise ValueError("retry counts cannot be negative")
        _build_retrieval_stack(str(config["provider_name"]))
        probe = journal.run_dir / ".write-probe.json"
        atomic_write_json(probe, {"ok": True, "at": utc_now()})
        probe.unlink(missing_ok=True)
        journal.complete_stage(
            stage,
            quality={
                "status": "passed",
                "checks": [
                    "question_present",
                    "configuration_valid",
                    "retrieval_provider_constructible",
                    "output_directory_writable",
                ],
                "api_key_persisted": False,
            },
        )
    except Exception as exc:
        _stop_stage(journal, stage, exc, retryable=False)


def _run_cohort_stage(
    journal: RunJournal,
    question: str,
    expected_count: int,
    config: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[CohortBundle, bool]:
    stage = "cohort"
    path = journal.run_dir / "cohort.json"
    journal.start_stage(stage)
    try:
        cohort = _load_valid_cohort(path, question, expected_count)
        if cohort:
            journal.complete_stage(stage, artifacts=[path], reused=True)
            return cohort, True
        cohort = generate_live_cohort(
            question,
            expected_count,
            model=str(config.get("cohort_model") or config["model"]),
            retries=int(execution["retries"]),
            deadline_epoch=_run_deadline_epoch(journal, execution),
        )
        atomic_write_json(path, cohort.to_dict())
        validated = _load_valid_cohort(path, question, expected_count)
        if not validated:
            raise RuntimeError("cohort checkpoint failed validation after writing")
        journal.complete_stage(
            stage,
            artifacts=[path],
            quality={
                "person_count": len(validated.personas),
                "status": "passed",
                "cohort_geometry": dict(
                    validated.generation.get("cohort_geometry") or {}
                ),
                "epistemic_passes": int(
                    validated.generation.get("epistemic_passes") or 1
                ),
            },
        )
        return validated, False
    except Exception as exc:
        _stop_stage(
            journal,
            stage,
            exc,
            retryable=bool(getattr(exc, "retryable", True)),
        )


def _run_streaming_front_stages(
    journal: RunJournal,
    question: str,
    cohort: CohortBundle,
    config: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    active_stages = ["retrieval", "digital_people", "semantic_verdicts"]
    for stage in active_stages:
        journal.start_stage(stage)
    journal.payload["current_stage"] = "streaming_front"
    journal.save()

    def save_progress(progress: dict[str, Any]) -> None:
        retrieval_progress = dict(progress.get("retrieval") or {})
        retrieval_progress.update(
            {
                "failed": 0,
                "reused": 0,
                "streaming": True,
                "activity": progress.get("activity"),
            }
        )
        people_progress = dict(progress.get("digital_people") or {})
        people_progress.update(
            {
                "failed_in_pass": 0,
                "streaming": True,
                "activity": progress.get("activity"),
            }
        )
        semantic_progress = dict(progress.get("semantic_verdicts") or {})
        semantic_progress.update(
            {
                "fallback": 0,
                "streaming": True,
                "activity": progress.get("activity"),
            }
        )
        journal.payload["current_stage"] = "streaming_front"
        journal.set_progress("retrieval", retrieval_progress)
        journal.payload["current_stage"] = "streaming_front"
        journal.set_progress("digital_people", people_progress)
        journal.payload["current_stage"] = "streaming_front"
        journal.set_progress("semantic_verdicts", semantic_progress)

    try:
        front = run_streaming_front_pipeline(
            question,
            cohort,
            profile_name=str(config["profile_name"]),
            provider_name=str(config["provider_name"]),
            results_per_query=int(config["results_per_query"]),
            model=str(config["model"]),
            max_tokens=int(config["max_tokens"]),
            semantic_verdicts=bool(config["semantic_verdicts"]),
            semantic_max_tokens=int(config["semantic_max_tokens"]),
            retries=int(execution["retries"]),
            recovery_passes=int(execution["recovery_passes"]),
            retrieval_workers=int(execution["retrieval_concurrency"]),
            people_workers=int(execution["concurrency"]),
            semantic_workers=int(execution["semantic_concurrency"]),
            model_concurrency=int(execution["model_concurrency"]),
            run_dir=journal.run_dir,
            progress_callback=save_progress,
            person_runner=_run_person_with_retries,
            deadline_epoch=_run_deadline_epoch(journal, execution),
        )
        retrieval_path = journal.run_dir / "retrieval.json"
        diagnosis_path = journal.run_dir / "diagnosis.json"
        metrics_path = journal.run_dir / "streaming_front_metrics.json"
        atomic_write_json(retrieval_path, front.retrieval)
        atomic_write_json(diagnosis_path, front.diagnosis)
        atomic_write_json(metrics_path, front.metrics)

        retrieval_quality = _retrieval_quality(front.retrieval)
        if retrieval_quality["blocking_person_ids"]:
            _discard_retrieval_checkpoints(
                journal.run_dir / "retrieval_items",
                retrieval_quality["blocking_person_ids"],
            )
            raise StreamingFrontError(
                "retrieval",
                ",".join(retrieval_quality["blocking_person_ids"]),
                "external evidence coverage is below the quality gate",
                retryable=True,
            )
        retrieval_quality["streaming_timing"] = (
            front.metrics["stages"]["retrieval"]
        )
        journal.complete_stage(
            "retrieval",
            artifacts=[retrieval_path, metrics_path],
            quality=retrieval_quality,
            reused=(front.metrics["reused"]["retrieval"] == len(cohort.personas)),
        )

        journal.complete_stage(
            "digital_people",
            artifacts=[
                journal.run_dir / "person_outputs" / f"{person.id}.json"
                for person in cohort.personas
            ],
            quality={
                "status": "passed",
                "expected": len(cohort.personas),
                "succeeded": len(front.person_results),
                "repaired_person_ids": sorted(front.changed_people),
                "streaming_timing": front.metrics["stages"]["digital_people"],
            },
            reused=(
                front.metrics["reused"]["digital_people"]
                == len(cohort.personas)
            ),
        )

        journal.start_stage("diagnosis")
        journal.complete_stage(
            "diagnosis",
            artifacts=[diagnosis_path],
            quality={
                "status": "passed",
                "cognitive_chain_score": (
                    front.diagnosis.get("cognitive_chain_evaluation") or {}
                ).get("score"),
                "streamed_person_scans_used_same_detectors": True,
            },
        )

        if bool(config["semantic_verdicts"]):
            semantic_quality = {
                "status": "passed",
                "record_count": len(front.semantic_records),
                "succeeded": sum(
                    item.get("status") == "succeeded"
                    for item in front.semantic_records
                ),
                "no_finding": sum(
                    item.get("status") == "no_finding"
                    for item in front.semantic_records
                ),
                "fallback": 0,
                "streaming_timing": front.metrics["stages"][
                    "semantic_verdicts"
                ],
            }
            semantic_artifacts = [
                journal.run_dir / "semantic_verdicts" / f"{person.id}.json"
                for person in cohort.personas
            ] + [diagnosis_path]
        else:
            semantic_quality = {
                "status": "skipped_by_configuration",
                "record_count": 0,
                "fallback": 0,
            }
            semantic_artifacts = [diagnosis_path]
        journal.complete_stage(
            "semantic_verdicts",
            artifacts=semantic_artifacts,
            quality=semantic_quality,
            reused=(
                bool(config["semantic_verdicts"])
                and front.metrics["reused"]["semantic_verdicts"]
                == len(cohort.personas)
            ),
        )
        return (
            front.retrieval,
            front.person_results,
            front.diagnosis,
            front.semantic_records,
        )
    except StreamingFrontError as exc:
        _stop_stage(
            journal,
            exc.stage,
            exc,
            retryable=exc.retryable,
        )
    except Exception as exc:
        _stop_stage(journal, "digital_people", exc)


def _run_retrieval_stage(
    journal: RunJournal,
    question: str,
    cohort: CohortBundle,
    config: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    stage = "retrieval"
    path = journal.run_dir / "retrieval.json"
    item_dir = journal.run_dir / "retrieval_items"
    journal.start_stage(stage)
    try:
        existing = _load_valid_retrieval(path, question, cohort, config)
        if existing:
            quality = _retrieval_quality(existing)
            if not quality["blocking_person_ids"]:
                _seed_retrieval_checkpoints(item_dir, existing, config)
                journal.complete_stage(
                    stage,
                    artifacts=[path],
                    quality=quality,
                    reused=True,
                )
                return existing, True
            _seed_retrieval_checkpoints(item_dir, existing, config)
            _discard_retrieval_checkpoints(item_dir, quality["blocking_person_ids"])

        last_error: Exception | None = None
        for recovery_pass in range(int(execution["recovery_passes"]) + 1):
            try:
                retrieval = run_filtered_retrieval(
                    question,
                    profile_name=str(config["profile_name"]),
                    provider_name=str(config["provider_name"]),
                    results_per_query=int(config["results_per_query"]),
                    cohort=cohort,
                    model=str(config["model"]),
                    tool_retries=int(execution["retries"]),
                    concurrency=int(execution["retrieval_concurrency"]),
                    checkpoint_dir=item_dir,
                    progress_callback=lambda progress: journal.set_progress(
                        stage,
                        {**progress, "recovery_pass": recovery_pass},
                    ),
                )
                retrieval["requested_provider"] = str(config["provider_name"])
                atomic_write_json(path, retrieval)
                validated = _load_valid_retrieval(path, question, cohort, config)
                if not validated:
                    _discard_retrieval_checkpoints(
                        item_dir, [person.id for person in cohort.personas]
                    )
                    last_error = RuntimeError(
                        "Retrieval batch failed structural and run-isolation validation"
                    )
                    if recovery_pass < int(execution["recovery_passes"]):
                        time.sleep(min(2**recovery_pass, 8))
                        continue
                    raise last_error
                retrieval = validated
                quality = _retrieval_quality(retrieval)
                if not quality["blocking_person_ids"]:
                    journal.complete_stage(
                        stage,
                        artifacts=[path],
                        quality=quality,
                    )
                    return retrieval, False
                last_error = RuntimeError(
                    "Live external evidence coverage is below the quality gate for: "
                    + ", ".join(quality["blocking_person_ids"])
                )
                _discard_retrieval_checkpoints(
                    item_dir, quality["blocking_person_ids"]
                )
            except RetrievalBatchError as exc:
                last_error = exc
                if not exc.retryable:
                    break
            if recovery_pass < int(execution["recovery_passes"]):
                time.sleep(min(2**recovery_pass, 8))
        raise last_error or RuntimeError("retrieval did not produce a valid result")
    except Exception as exc:
        _stop_stage(
            journal,
            stage,
            exc,
            retryable=bool(getattr(exc, "retryable", True)),
        )


def _run_people_stage(
    journal: RunJournal,
    question: str,
    personas: list[DigitalPerson],
    retrieval: dict[str, Any],
    config: dict[str, Any],
    execution: dict[str, Any],
    *,
    allow_legacy: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    stage = "digital_people"
    people_dir = journal.run_dir / "person_outputs"
    people_dir.mkdir(parents=True, exist_ok=True)
    records_by_person = {
        item["person_id"]: item for item in retrieval["retrieval_records"]
    }
    ledgers_by_person = {
        item["person_id"]: item for item in retrieval["evidence_ledgers"]
    }
    journal.start_stage(stage)
    changed_people: set[str] = set()
    stage_retryable = True
    try:
        last_failures: list[dict[str, Any]] = []
        for recovery_pass in range(int(execution["recovery_passes"]) + 1):
            saved = {
                item.get("person_id"): item
                for item in _read_person_results(people_dir)
                if item.get("person_id")
            }
            pending: list[tuple[DigitalPerson, dict[str, Any], str]] = []
            valid: dict[str, dict[str, Any]] = {}
            for person in personas:
                context = _build_retrieval_context(
                    person.id, records_by_person, ledgers_by_person
                )
                fingerprint = _payload_fingerprint(
                    {"question": question, "person": asdict(person), "retrieval": context}
                )
                item = saved.get(person.id) or {}
                errors = validate_digital_person_output(
                    question, person.id, item.get("output")
                )
                fingerprint_matches = item.get("pipeline_input_fingerprint") == fingerprint
                legacy_matches = allow_legacy and not item.get("pipeline_input_fingerprint")
                if (
                    item.get("status") == "succeeded"
                    and not errors
                    and (fingerprint_matches or legacy_matches)
                ):
                    valid[person.id] = item
                    continue
                pending.append((person, context, fingerprint))

            if not pending:
                results = [valid[person.id] for person in personas]
                validate_run_isolation(
                    question,
                    retrieval["cohort_id"],
                    retrieval["personas"],
                    question_frame=retrieval.get("question_frame"),
                    cohort_generation=retrieval.get("cohort_generation"),
                    retrieval_records=retrieval.get("retrieval_records"),
                    evidence_ledgers=retrieval.get("evidence_ledgers"),
                    person_results=results,
                )
                journal.complete_stage(
                    stage,
                    artifacts=[people_dir / f"{person.id}.json" for person in personas],
                    quality={
                        "status": "passed",
                        "expected": len(personas),
                        "succeeded": len(results),
                        "repaired_person_ids": sorted(changed_people),
                    },
                    reused=not changed_people,
                )
                return results, changed_people

            last_failures = []
            completed_this_pass = 0
            with ThreadPoolExecutor(
                max_workers=max(1, int(execution["concurrency"]))
            ) as executor:
                futures = {
                    executor.submit(
                        _run_person_with_retries,
                        question,
                        person,
                        context,
                        str(config["model"]),
                        int(config["max_tokens"]),
                        int(execution["retries"]),
                    ): (person, fingerprint)
                    for person, context, fingerprint in pending
                }
                for future in as_completed(futures):
                    person, fingerprint = futures[future]
                    previous = saved.get(person.id) or {}
                    result = future.result()
                    result["pipeline_input_fingerprint"] = fingerprint
                    result = _merge_recovery_record(previous, result)
                    atomic_write_json(people_dir / f"{person.id}.json", result)
                    changed_people.add(person.id)
                    if result.get("status") != "succeeded":
                        last_failures.append(result)
                    completed_this_pass += 1
                    journal.set_progress(
                        stage,
                        {
                            "completed": len(valid) + completed_this_pass,
                            "total": len(personas),
                            "failed_in_pass": len(last_failures),
                            "recovery_pass": recovery_pass,
                        },
                    )
            if not last_failures:
                reloaded = {
                    item.get("person_id"): item
                    for item in _read_person_results(people_dir)
                    if item.get("person_id")
                }
                if all(
                    reloaded.get(person.id, {}).get("status") == "succeeded"
                    and not validate_digital_person_output(
                        question,
                        person.id,
                        reloaded.get(person.id, {}).get("output"),
                    )
                    for person in personas
                ):
                    results = [reloaded[person.id] for person in personas]
                    validate_run_isolation(
                        question,
                        retrieval["cohort_id"],
                        retrieval["personas"],
                        question_frame=retrieval.get("question_frame"),
                        cohort_generation=retrieval.get("cohort_generation"),
                        retrieval_records=retrieval.get("retrieval_records"),
                        evidence_ledgers=retrieval.get("evidence_ledgers"),
                        person_results=results,
                    )
                    journal.complete_stage(
                        stage,
                        artifacts=[
                            people_dir / f"{person.id}.json" for person in personas
                        ],
                        quality={
                            "status": "passed",
                            "expected": len(personas),
                            "succeeded": len(results),
                            "repaired_person_ids": sorted(changed_people),
                        },
                    )
                    return results, changed_people
            terminal_failures = [
                item for item in last_failures if item.get("retryable") is False
            ]
            if terminal_failures:
                stage_retryable = False
                break
            if recovery_pass < int(execution["recovery_passes"]):
                time.sleep(min(2**recovery_pass, 8))

        failed_ids = [str(item.get("person_id")) for item in last_failures]
        raise RuntimeError(
            "Complete digital-person chains are still missing after recovery: "
            + ", ".join(failed_ids)
        )
    except Exception as exc:
        _stop_stage(
            journal,
            stage,
            exc,
            retryable=(
                stage_retryable
                and bool(getattr(exc, "retryable", True))
            ),
        )


def _run_diagnosis_stage(
    journal: RunJournal,
    question: str,
    retrieval: dict[str, Any],
    successful_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    stage = "diagnosis"
    path = journal.run_dir / "diagnosis.json"
    journal.start_stage(stage)
    try:
        diagnosis = diagnose_cognitive_shadows(
            question,
            retrieval["personas"],
            retrieval["evidence_ledgers"],
            person_outputs=successful_outputs,
        )
        atomic_write_json(path, diagnosis)
        journal.complete_stage(
            stage,
            artifacts=[path],
            quality={
                "status": "passed",
                "cognitive_chain_score": (
                    diagnosis.get("cognitive_chain_evaluation") or {}
                ).get("score"),
            },
        )
        return diagnosis
    except Exception as exc:
        _stop_stage(journal, stage, exc, retryable=False)


def _run_semantic_stage(
    journal: RunJournal,
    question: str,
    retrieval: dict[str, Any],
    successful_outputs: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    config: dict[str, Any],
    execution: dict[str, Any],
    *,
    changed_people: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage = "semantic_verdicts"
    verdict_dir = journal.run_dir / "semantic_verdicts"
    verdict_dir.mkdir(parents=True, exist_ok=True)
    journal.start_stage(stage)
    stage_retryable = True
    try:
        if not bool(config["semantic_verdicts"]):
            journal.complete_stage(
                stage,
                quality={"status": "skipped_by_configuration"},
            )
            return diagnosis, []
        _archive_changed_semantic_records(verdict_dir, changed_people)
        records: list[dict[str, Any]] = []
        for recovery_pass in range(int(execution["recovery_passes"]) + 1):
            records = run_semantic_verdict_batch(
                question,
                retrieval["personas"],
                retrieval["evidence_ledgers"],
                successful_outputs,
                diagnosis,
                output_dir=verdict_dir,
                model=str(config["model"]),
                max_tokens=(
                    int(config["semantic_max_tokens"])
                    if recovery_pass == 0
                    else min(int(config["semantic_max_tokens"]) * 2, 32768)
                ),
                concurrency=int(execution["concurrency"]),
                retries=int(execution["retries"]),
                resume=True,
            )
            fallbacks = [item for item in records if item.get("status") == "fallback"]
            journal.set_progress(
                stage,
                {
                    "records": len(records),
                    "fallback": len(fallbacks),
                    "recovery_pass": recovery_pass,
                },
            )
            terminal_fallbacks = [
                item
                for item in fallbacks
                if item.get("retryable") is False
            ]
            if terminal_fallbacks:
                stage_retryable = False
                break
            if not fallbacks:
                break
            if recovery_pass < int(execution["recovery_passes"]):
                time.sleep(min(2**recovery_pass, 8))
        fallbacks = [item for item in records if item.get("status") == "fallback"]
        if fallbacks:
            raise RuntimeError(
                "Semantic verdicts still use operational fallback for: "
                + ", ".join(str(item.get("person_id")) for item in fallbacks)
            )
        expected_ids = {item["person_id"] for item in successful_outputs}
        record_ids = {str(item.get("person_id") or "") for item in records}
        if record_ids != expected_ids:
            missing = sorted(expected_ids - record_ids)
            extra = sorted(record_ids - expected_ids)
            raise RuntimeError(
                f"Semantic verdict coverage mismatch; missing={missing}, extra={extra}"
            )
        invalid_records = [
            str(item.get("person_id"))
            for item in records
            if not (item.get("verdict") or {}).get("validation", {}).get("valid")
        ]
        if invalid_records:
            raise RuntimeError(
                "Semantic verdict evidence lock is invalid for: "
                + ", ".join(invalid_records)
            )
        attach_semantic_verdicts(diagnosis, records)
        diagnosis["semantic_verdict_evaluation"] = evaluate_semantic_verdict_strength(
            diagnosis
        )
        diagnosis["meta_model_overall_evaluation"] = evaluate_meta_model_overall(
            diagnosis
        )
        diagnosis_path = journal.run_dir / "diagnosis.json"
        atomic_write_json(diagnosis_path, diagnosis)
        journal.complete_stage(
            stage,
            artifacts=[
                verdict_dir / f"{item['person_id']}.json" for item in records
            ]
            + [diagnosis_path],
            quality={
                "status": "passed",
                "record_count": len(records),
                "succeeded": sum(item.get("status") == "succeeded" for item in records),
                "no_finding": sum(item.get("status") == "no_finding" for item in records),
                "fallback": 0,
            },
        )
        return diagnosis, records
    except Exception as exc:
        _stop_stage(journal, stage, exc, retryable=stage_retryable)


def _run_puzzle_stage(
    journal: RunJournal,
    question: str,
    diagnosis: dict[str, Any],
    successful_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    stage = "shadow_puzzle"
    path = journal.run_dir / "shadow_puzzle.json"
    journal.start_stage(stage)
    try:
        puzzle = build_shadow_puzzle(
            question, diagnosis, person_outputs=successful_outputs
        )
        atomic_write_json(path, puzzle)
        journal.complete_stage(
            stage,
            artifacts=[path],
            quality={"status": "passed", **(puzzle.get("puzzle_quality") or {})},
        )
        return puzzle
    except Exception as exc:
        _stop_stage(journal, stage, exc, retryable=False)


def _run_synthesis_stage(
    journal: RunJournal,
    question: str,
    retrieval: dict[str, Any],
    successful_outputs: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    config: dict[str, Any],
    execution: dict[str, Any],
    *,
    allow_legacy: bool,
) -> dict[str, Any]:
    stage = "relational_synthesis"
    checkpoint_path = journal.run_dir / "contour_candidates.json"
    input_path = journal.run_dir / "synthesis_checkpoint.json"
    input_fingerprint = _payload_fingerprint(
        {
            "question": question,
            "diagnosis": diagnosis,
            "person_outputs": successful_outputs,
            "evidence_ledgers": retrieval["evidence_ledgers"],
            "question_frame": retrieval.get("question_frame"),
            "synthesis_mode": execution.get("synthesis_mode"),
        }
    )
    journal.start_stage(stage)
    stage_retryable = True
    try:
        if not bool(config["relational_synthesis"]):
            synthesis = build_deterministic_synthesis(
                question,
                diagnosis,
                successful_outputs,
                retrieval["evidence_ledgers"],
                question_frame=retrieval.get("question_frame"),
            )
            write_synthesis_artifacts(journal.run_dir, synthesis)
            journal.complete_stage(
                stage,
                artifacts=_synthesis_paths(journal.run_dir),
                quality={"status": "deterministic_by_configuration"},
            )
            return synthesis

        resume_record: dict[str, Any] = {}
        saved_input = _read_optional_json(input_path)
        if saved_input.get("input_fingerprint") == input_fingerprint:
            resume_record = _read_optional_json(checkpoint_path)
        elif (
            allow_legacy
            and execution.get("synthesis_mode") == "legacy_sequential"
            and not saved_input
            and checkpoint_path.exists()
        ):
            resume_record = _read_optional_json(checkpoint_path)

        def save_checkpoint(payload: dict[str, Any]) -> None:
            atomic_write_json(checkpoint_path, payload)
            atomic_write_json(
                input_path,
                {
                    "schema": "bme.synthesis-checkpoint.v1",
                    "input_fingerprint": input_fingerprint,
                    "updated_at": utc_now(),
                },
            )
            journal.set_progress(
                stage,
                {
                    "detective_status": (payload.get("detective") or {}).get("status"),
                    "speculative_draft_status": (
                        payload.get("speculative_drafts") or {}
                    ).get("status"),
                    "contour_status": (payload.get("contour") or {}).get("status"),
                    "checkpointed": True,
                },
            )

        synthesis_governor = AdaptiveConcurrencyGovernor(
            max(1, int(execution["model_concurrency"]))
        )
        synthesis_user_id = "bme_synthesis_" + hashlib.sha256(
            str(journal.run_dir.resolve()).encode("utf-8")
        ).hexdigest()[:24]
        synthesis_client_factory = lambda: DeepSeekClient(  # noqa: E731
            model=str(config["model"]),
            timeout_seconds=bounded_request_timeout_seconds(
                _run_deadline_epoch(journal, execution),
                synthesis_request_timeout_seconds(),
            ),
            request_governor=synthesis_governor,
            user_id=synthesis_user_id,
        )
        synthesis: dict[str, Any] = {}
        quality: dict[str, Any] = {}
        recovery_history: list[dict[str, Any]] = []
        cumulative_usage: list[dict[str, Any]] = []
        synthesis_started = time.perf_counter()
        for recovery_pass in range(int(execution["recovery_passes"]) + 1):
            _require_run_budget(
                journal,
                execution,
                stage,
                minimum_seconds=60,
            )
            pass_started = time.perf_counter()
            synthesis = run_live_synthesis(
                question,
                diagnosis,
                successful_outputs,
                retrieval["evidence_ledgers"],
                question_frame=retrieval.get("question_frame"),
                model=str(config["model"]),
                max_tokens=(
                    int(config["relational_max_tokens"])
                    if recovery_pass == 0
                    else min(int(config["relational_max_tokens"]) * 2, 32768)
                ),
                retries=int(execution["retries"]),
                synthesis_mode=str(execution["synthesis_mode"]),
                client_factory=synthesis_client_factory,
                resume_record=resume_record,
                checkpoint_callback=save_checkpoint,
            )
            synthesis.setdefault("relational_synthesis_usage", {})[
                "model_governor"
            ] = synthesis_governor.snapshot().to_dict()
            pass_usage = dict(
                synthesis.get("relational_synthesis_usage") or {}
            )
            cumulative_usage.append(pass_usage)
            write_synthesis_artifacts(journal.run_dir, synthesis)
            resume_record = _read_optional_json(checkpoint_path)
            quality = _synthesis_quality(resume_record)
            artifact_errors = _synthesis_artifact_errors(synthesis, question)
            quality["artifact_validation_errors"] = artifact_errors
            quality["blocking_issues"].extend(artifact_errors)
            if artifact_errors:
                quality["status"] = "blocked"
            quality["recovery_pass"] = recovery_pass
            recovery_history.append(
                {
                    "pass": recovery_pass,
                    "duration_seconds": round(
                        time.perf_counter() - pass_started, 3
                    ),
                    "detective_status": quality.get("detective_status"),
                    "contour_status": quality.get("contour_status"),
                    "blocking_issues": list(
                        dict.fromkeys(quality.get("blocking_issues") or [])
                    ),
                    "usage": pass_usage,
                }
            )
            if _has_terminal_synthesis_failure(resume_record):
                quality["terminal_provider_failure"] = True
                stage_retryable = False
                break
            if not quality["blocking_issues"]:
                break
            if recovery_pass < int(execution["recovery_passes"]):
                time.sleep(min(2**recovery_pass, 8))
        if quality.get("blocking_issues"):
            raise RuntimeError("; ".join(quality["blocking_issues"]))
        cumulative = _cumulative_synthesis_usage(cumulative_usage)
        cumulative.update(
            {
                "mode": "live",
                "model": str(config["model"]),
                "synthesis_mode": str(execution["synthesis_mode"]),
                "wall_seconds": round(
                    time.perf_counter() - synthesis_started, 3
                ),
                "recovery_pass_count": max(0, len(recovery_history) - 1),
                "recovery_history": recovery_history,
                "model_governor": synthesis_governor.snapshot().to_dict(),
            }
        )
        synthesis["relational_synthesis_usage"] = cumulative
        candidates = synthesis.setdefault("contour_candidates", {})
        candidates["recovery_history"] = recovery_history
        candidates["cumulative_usage"] = cumulative
        write_synthesis_artifacts(journal.run_dir, synthesis)
        journal.complete_stage(
            stage,
            artifacts=_synthesis_paths(journal.run_dir) + [input_path],
            quality=quality,
        )
        return synthesis
    except Exception as exc:
        _stop_stage(
            journal,
            stage,
            exc,
            retryable=(
                stage_retryable
                and bool(getattr(exc, "retryable", True))
            ),
        )


def _cumulative_synthesis_usage(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep every paid synthesis call visible across targeted recovery passes."""

    active = [item for item in records if isinstance(item, dict)]
    prompt = sum(int(item.get("prompt_tokens") or 0) for item in active)
    completion = sum(
        int(item.get("completion_tokens") or 0) for item in active
    )
    total = sum(int(item.get("total_tokens") or 0) for item in active)
    calls = sum(int(item.get("calls") or 0) for item in active)

    def component(name: str) -> dict[str, int]:
        values = [
            item.get(name) or {}
            for item in active
            if isinstance(item.get(name), dict)
        ]
        component_prompt = sum(
            int(item.get("prompt_tokens") or 0) for item in values
        )
        component_completion = sum(
            int(item.get("completion_tokens") or 0) for item in values
        )
        component_total = sum(
            int(item.get("total_tokens") or 0) for item in values
        )
        return {
            "prompt_tokens": component_prompt,
            "completion_tokens": component_completion,
            "total_tokens": component_total
            or component_prompt + component_completion,
        }

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
        "calls": calls,
        "detective_usage": component("detective_usage"),
        "speculative_draft_usage": component(
            "speculative_draft_usage"
        ),
        "contour_usage": component("contour_usage"),
    }


def _finalize_run(
    journal: RunJournal,
    question: str,
    profile: Any,
    cohort: CohortBundle,
    retrieval: dict[str, Any],
    person_results: list[dict[str, Any]],
    semantic_records: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    puzzle: dict[str, Any],
    synthesis: dict[str, Any],
    config: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    stage = "finalize"
    journal.start_stage(stage)
    try:
        usage = summarize_usage(
            person_results,
            model=str(config["model"]),
            input_price_per_1m_cny=execution.get("input_price_per_1m_cny"),
            output_price_per_1m_cny=execution.get("output_price_per_1m_cny"),
        )
        semantic_usage = summarize_semantic_verdict_usage(
            semantic_records, model=str(config["model"])
        )
        semantic_usage["pricing"] = _usage_pricing(
            semantic_usage,
            input_price_per_1m_cny=execution.get("input_price_per_1m_cny"),
            output_price_per_1m_cny=execution.get("output_price_per_1m_cny"),
        )
        cohort_usage = dict(cohort.generation.get("usage") or {})
        cohort_usage["model"] = (
            cohort.generation.get("model") or str(config["model"])
        )
        cohort_usage["pricing"] = _usage_pricing(
            cohort_usage,
            input_price_per_1m_cny=execution.get("input_price_per_1m_cny"),
            output_price_per_1m_cny=execution.get("output_price_per_1m_cny"),
        )
        search_tool_usage = dict(retrieval.get("search_tool_usage") or {})
        search_tool_usage["pricing"] = _usage_pricing(
            search_tool_usage,
            input_price_per_1m_cny=execution.get("input_price_per_1m_cny"),
            output_price_per_1m_cny=execution.get("output_price_per_1m_cny"),
        )
        relational_usage = dict(synthesis["relational_synthesis_usage"])
        relational_usage["pricing"] = _usage_pricing(
            relational_usage,
            input_price_per_1m_cny=execution.get("input_price_per_1m_cny"),
            output_price_per_1m_cny=execution.get("output_price_per_1m_cny"),
        )
        combined_usage = _combine_usage(
            usage,
            semantic_usage,
            cohort_usage,
            search_tool_usage,
            relational_usage,
        )
        quality = _overall_quality(
            profile.person_count,
            retrieval,
            person_results,
            semantic_records,
            synthesis,
        )
        elapsed_seconds = _run_elapsed_seconds(journal)
        time_budget_seconds = int(execution["time_budget_seconds"])
        quality["metrics"].update(
            {
                "run_wall_seconds": elapsed_seconds,
                "analysis_time_budget_seconds": time_budget_seconds,
                "within_analysis_time_budget": (
                    elapsed_seconds <= time_budget_seconds
                ),
            }
        )
        atomic_write_json(journal.run_dir / "quality.json", quality)
        journal.set_quality(quality)
        if quality["blocking_issues"]:
            raise RuntimeError("; ".join(quality["blocking_issues"]))

        result = {
            "question": question,
            "profile": asdict(profile),
            "cohort_id": retrieval["cohort_id"],
            "run_dir": str(journal.run_dir),
            "started_at": journal.payload.get("created_at"),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "model": str(config["model"]),
            "provider": str(config["provider_name"]),
            "results_per_query": int(config["results_per_query"]),
            "pipeline_mode": str(execution["pipeline_mode"]),
            "concurrency": int(execution["concurrency"]),
            "retrieval_concurrency": int(execution["retrieval_concurrency"]),
            "semantic_concurrency": int(execution["semantic_concurrency"]),
            "model_concurrency": int(execution["model_concurrency"]),
            "retries": int(execution["retries"]),
            "recovery_passes": int(execution["recovery_passes"]),
            "time_budget_seconds": time_budget_seconds,
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
            "run_quality": quality,
            "resilient_pipeline": {
                "manifest": "run_manifest.json",
                "pipeline_version": journal.payload.get("pipeline_version"),
                "resume_count": journal.payload.get("resume_count", 0),
            },
        }
        diagnosis_path = journal.run_dir / "diagnosis.json"
        puzzle_path = journal.run_dir / "shadow_puzzle.json"
        summary_path = journal.run_dir / "summary.json"
        run_path = journal.run_dir / "run.json"
        atomic_write_json(diagnosis_path, diagnosis)
        atomic_write_json(puzzle_path, puzzle)
        write_synthesis_artifacts(journal.run_dir, synthesis)
        atomic_write_json(summary_path, _summary_payload(result))
        atomic_write_json(run_path, result)
        report_path = build_run_report(journal.run_dir)
        journal.complete_stage(
            stage,
            artifacts=[
                diagnosis_path,
                puzzle_path,
                summary_path,
                run_path,
                report_path,
                journal.run_dir / "quality.json",
            ],
            quality=quality,
        )
        journal.finish()
        return result
    except PipelineIncompleteError:
        raise
    except Exception as exc:
        _stop_stage(journal, stage, exc)


def _load_valid_cohort(
    path: Path, question: str, expected_count: int
) -> CohortBundle | None:
    payload = _read_optional_json(path)
    if not payload:
        return None
    try:
        personas = [digital_person_from_dict(item) for item in payload["personas"]]
        cohort = CohortBundle(
            cohort_id=str(payload["cohort_id"]),
            question=str(payload["question"]),
            question_frame=dict(payload["question_frame"]),
            personas=personas,
            generation=dict(payload["generation"]),
        )
        if cohort.question != question or len(personas) != expected_count:
            return None
        if len({person.id for person in personas}) != expected_count:
            return None
        if cohort.question_frame.get("exact_question") != question:
            return None
        if cohort.question_frame.get("context_id") != cohort.cohort_id:
            return None
        if cohort.generation.get("context_isolation") != "only_current_question":
            return None
        return cohort
    except Exception:
        return None


def _load_valid_retrieval(
    path: Path,
    question: str,
    cohort: CohortBundle,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    payload = _read_optional_json(path)
    if not payload:
        return None
    try:
        if payload.get("question") != question:
            return None
        if payload.get("cohort_id") != cohort.cohort_id:
            return None
        if int(payload.get("person_count") or 0) != len(cohort.personas):
            return None
        if int(payload.get("results_per_query") or 0) != int(
            config["results_per_query"]
        ):
            return None
        if (
            payload.get("retrieval_implementation_revision")
            != RETRIEVAL_IMPLEMENTATION_REVISION
        ):
            return None
        requested = payload.get("requested_provider")
        if requested and requested != config["provider_name"]:
            return None
        validate_run_isolation(
            question,
            cohort.cohort_id,
            payload["personas"],
            question_frame=payload.get("question_frame"),
            cohort_generation=payload.get("cohort_generation"),
            retrieval_records=payload.get("retrieval_records"),
            evidence_ledgers=payload.get("evidence_ledgers"),
        )
        expected_ids = {person.id for person in cohort.personas}
        record_ids = {
            item.get("person_id") for item in payload.get("retrieval_records", [])
        }
        ledger_ids = {
            item.get("person_id") for item in payload.get("evidence_ledgers", [])
        }
        if record_ids != expected_ids or ledger_ids != expected_ids:
            return None
        return payload
    except Exception:
        return None


def _retrieval_quality(retrieval: dict[str, Any]) -> dict[str, Any]:
    mode = str(retrieval.get("evidence_mode") or "")
    live_mode = mode in {"three_layer_live", "hybrid_live", "live_external"}
    records = retrieval.get("retrieval_records", []) or []
    external_by_person: dict[str, int] = {}
    for record in records:
        external_by_person[str(record.get("person_id"))] = sum(
            (item.get("retrieval_layer") == "external_search")
            for item in record.get("candidate_sources", []) or []
        )
    deficient = [
        person_id
        for person_id, count in external_by_person.items()
        if live_mode and count == 0
    ]
    coverage = (
        sum(count > 0 for count in external_by_person.values()) / len(records)
        if records
        else 0.0
    )
    if mode == "three_layer_live" and coverage < 0.9:
        blocking = deficient
    elif live_mode and coverage < 0.8:
        blocking = deficient
    else:
        blocking = []
    warnings = []
    if mode == "test_fixture":
        warnings.append("synthetic retrieval fixture; not suitable as live evidence")
    if mode == "prior_only":
        warnings.append("model-prior-only run has no external evidence")
    if live_mode and deficient and not blocking:
        warnings.append(
            f"{len(deficient)} digital people have no external search result"
        )
    return {
        "status": "passed" if not blocking else "blocked",
        "evidence_mode": mode,
        "person_count": len(records),
        "external_coverage": round(coverage, 4),
        "blocking_person_ids": blocking,
        "warnings": warnings,
    }


def _seed_retrieval_checkpoints(
    item_dir: Path, retrieval: dict[str, Any], config: dict[str, Any]
) -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    ledgers = {
        item["person_id"]: item for item in retrieval.get("evidence_ledgers", [])
    }
    traces = {
        item["person_id"]: item for item in retrieval.get("search_tool_traces", [])
    }
    for record in retrieval.get("retrieval_records", []) or []:
        person_id = record.get("person_id")
        if not person_id or person_id not in ledgers:
            continue
        path = item_dir / f"{person_id}.json"
        if path.exists():
            continue
        atomic_write_json(
            path,
            {
                "schema": "bme.retrieval-item.v2",
                "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
                "status": "succeeded",
                "question": retrieval.get("question"),
                "cohort_id": retrieval.get("cohort_id"),
                "person_id": person_id,
                "provider_name": config.get("provider_name"),
                "results_per_query": config.get("results_per_query"),
                "model": config.get("model"),
                "retrieval_record": record,
                "evidence_ledger": ledgers[person_id],
                "tool_trace": traces.get(person_id, {}),
            },
        )


def _discard_retrieval_checkpoints(item_dir: Path, person_ids: list[str]) -> None:
    for person_id in person_ids:
        (item_dir / f"{person_id}.json").unlink(missing_ok=True)


def _archive_changed_semantic_records(
    verdict_dir: Path, changed_people: set[str]
) -> None:
    if not changed_people:
        return
    archive = verdict_dir / "archive" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    moved = False
    for person_id in sorted(changed_people):
        path = verdict_dir / f"{person_id}.json"
        if not path.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        path.replace(archive / path.name)
        moved = True
    if not moved and archive.exists():
        archive.rmdir()


def _merge_recovery_record(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    if not previous:
        return current
    history = list(previous.get("recovery_history") or [])
    history.append(
        {
            "status": previous.get("status"),
            "attempts": previous.get("attempts", 0),
            "error": previous.get("error"),
            "error_kind": previous.get("error_kind"),
            "usage": previous.get("usage", {}),
        }
    )
    current["recovery_history"] = history[-10:]
    current["attempts"] = int(previous.get("attempts") or 0) + int(
        current.get("attempts") or 0
    )
    current["usage"] = _merge_usage_payloads(
        [previous.get("usage") or {}, current.get("usage") or {}]
    )
    return current


def _successful_person_outputs(
    person_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "person_id": item["person_id"],
            "person": item.get("person", {}),
            "output": item.get("output", {}),
            "usage": item.get("usage", {}),
        }
        for item in person_results
        if item.get("status") == "succeeded"
    ]


def _synthesis_quality(record: dict[str, Any]) -> dict[str, Any]:
    detective = record.get("detective") or {}
    contour = record.get("contour") or {}
    blocking: list[str] = []
    detective_status = str(detective.get("status") or "")
    contour_status = str(contour.get("status") or "")
    if detective_status != "succeeded":
        blocking.append(f"detective stage is {detective_status or 'missing'}")
    contour_stages = (contour.get("audit") or {}).get("stages", []) or []
    failed_contour_stages = [
        item.get("stage")
        for item in contour_stages
        if item.get("status") == "failed"
    ]
    justified_fallback = (
        contour_status == "fallback"
        and any(item.get("stage") == "constructive_relation_gate" for item in contour_stages)
    )
    if contour_status != "succeeded" and not justified_fallback:
        blocking.append(f"contour stage is {contour_status or 'missing'}")
    if failed_contour_stages:
        blocking.append(
            "failed contour sub-stages: " + ", ".join(failed_contour_stages)
        )
    return {
        "status": "passed" if not blocking else "blocked",
        "detective_status": detective_status,
        "contour_status": contour_status,
        "justified_conservative_fallback": justified_fallback,
        "blocking_issues": blocking,
    }


def _has_terminal_synthesis_failure(record: dict[str, Any]) -> bool:
    for section_name in ("detective", "contour"):
        section = record.get(section_name) or {}
        stages = (section.get("audit") or {}).get("stages", []) or []
        if any(
            item.get("status") == "failed" and item.get("retryable") is False
            for item in stages
            if isinstance(item, dict)
        ):
            return True
    return False


def _synthesis_artifact_errors(
    synthesis: dict[str, Any], question: str
) -> list[str]:
    materials = synthesis.get("puzzle_materials") or {}
    detective = synthesis.get("detective") or {}
    constraints = synthesis.get("contour_constraints") or {}
    contour = synthesis.get("truth_contour") or {}
    errors = [
        *validate_puzzle_materials(materials, question),
        *validate_detective_output(detective, materials),
        *validate_truth_contour(
            contour,
            question=question,
            detective=detective,
            constraints=constraints,
        ),
    ]
    return [str(item) for item in errors[:20]]


def _overall_quality(
    expected_people: int,
    retrieval: dict[str, Any],
    person_results: list[dict[str, Any]],
    semantic_records: list[dict[str, Any]],
    synthesis: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    retrieval_quality = _retrieval_quality(retrieval)
    if retrieval_quality["blocking_person_ids"]:
        blocking.append("retrieval external-evidence coverage is below its gate")
    warnings.extend(retrieval_quality["warnings"])
    succeeded_people = sum(item.get("status") == "succeeded" for item in person_results)
    if succeeded_people != expected_people:
        blocking.append(
            f"digital-person completeness is {succeeded_people}/{expected_people}"
        )
    fallback_count = sum(item.get("status") == "fallback" for item in semantic_records)
    if fallback_count:
        blocking.append(f"{fallback_count} semantic verdicts use fallback")
    no_finding = sum(item.get("status") == "no_finding" for item in semantic_records)
    if no_finding:
        warnings.append(
            f"{no_finding} people legitimately had no evidence-locked semantic finding"
        )
    candidate_record = synthesis.get("contour_candidates") or {}
    deterministic = (
        (synthesis.get("relational_synthesis_usage") or {}).get("mode")
        == "deterministic"
    )
    synthesis_quality = (
        {
            "detective_status": "deterministic",
            "contour_status": "deterministic",
            "blocking_issues": [],
        }
        if deterministic
        else _synthesis_quality(candidate_record)
    )
    blocking.extend(synthesis_quality["blocking_issues"])
    truth = synthesis.get("truth_contour") or {}
    direct_answer = truth.get("direct_answer")
    direct_answer_text = (
        direct_answer.get("text")
        if isinstance(direct_answer, dict)
        else direct_answer
    )
    if not str(direct_answer_text or "").strip():
        blocking.append("truth contour has no direct answer")
    detective = synthesis.get("detective") or {}
    relations = detective.get("relation_certificates", []) or []
    transport_unresolved_relation_ids = [
        str(item.get("relation_id") or "")
        for item in relations
        if (item.get("provenance") or {}).get(
            "adjudication_transport_fallback"
        )
    ]
    transport_fallback_accepted_ids = [
        str(item.get("relation_id") or "")
        for item in relations
        if item.get("status") == "accepted"
        and (item.get("provenance") or {}).get(
            "adjudication_transport_fallback"
        )
    ]
    if transport_unresolved_relation_ids:
        warnings.append(
            f"{len(transport_unresolved_relation_ids)} relation candidates remained "
            "unresolved after bounded transport recovery and were excluded from the contour"
        )
    if transport_fallback_accepted_ids:
        blocking.append(
            "transport-unresolved relations were incorrectly accepted"
        )
    directional_relation_count = sum(
        item.get("status") == "accepted"
        and item.get("relation_type") == "independent_convergence"
        and (item.get("independence_profile") or {}).get(
            "directional_support_authorized", False
        )
        for item in relations
    )
    accepted_verification_counts = _accepted_verification_counts(retrieval)
    verified_accepted_evidence_count = sum(
        accepted_verification_counts.get(status, 0)
        for status in ("page_verified", "primary_verified", "cross_verified")
    )
    contour_evaluation = truth.get("contour_evaluation") or {}
    sentence_binding_count = int(
        contour_evaluation.get("sentence_binding_count") or 0
    )
    sentence_binding_coverage = float(
        contour_evaluation.get("sentence_binding_coverage") or 0.0
    )
    verified_observation_count = int(
        contour_evaluation.get("verified_observation_count") or 0
    )
    directional_judgment_count = int(
        contour_evaluation.get("directional_judgment_count") or 0
    )
    evidence_binding_contract = str(
        truth.get("evidence_binding_contract") or "missing"
    )
    validation_drop_count = len(truth.get("validation_drops") or [])

    if evidence_binding_contract == "sentence_level.v1":
        if sentence_binding_coverage < 1.0 or sentence_binding_count < 1:
            blocking.append("sentence-level evidence binding is incomplete")
        if verified_observation_count and not verified_accepted_evidence_count:
            blocking.append("verified observations lack verified accepted evidence")
        if directional_judgment_count and not directional_relation_count:
            blocking.append("directional judgment lacks an authorized independent relation")
    elif truth.get("generation_mode") == "live_adversarial_inference":
        warnings.append(
            "live contour used legacy statement-level provenance; sentence binding was not claimed"
        )
    if not verified_accepted_evidence_count:
        warnings.append(
            "no accepted evidence was page, primary, or cross verified; the contour must remain structural or conditional"
        )
    if not directional_relation_count:
        warnings.append(
            "no independent relation is authorized for directional judgment"
        )
    if validation_drop_count:
        warnings.append(
            f"{validation_drop_count} inadmissible contour statements were removed locally"
        )
    return {
        "schema": "bme.run-quality.v2",
        "status": "passed" if not blocking else "blocked",
        "blocking_issues": blocking,
        "warnings": warnings,
        "metrics": {
            "expected_people": expected_people,
            "succeeded_people": succeeded_people,
            "retrieval_external_coverage": retrieval_quality["external_coverage"],
            "semantic_fallback_count": fallback_count,
            "semantic_no_finding_count": no_finding,
            "detective_status": synthesis_quality["detective_status"],
            "contour_status": synthesis_quality["contour_status"],
            "accepted_evidence_verification_counts": accepted_verification_counts,
            "verified_accepted_evidence_count": verified_accepted_evidence_count,
            "accepted_directional_support_relation_count": directional_relation_count,
            "transport_unresolved_relation_count": len(
                transport_unresolved_relation_ids
            ),
            "transport_fallback_accepted_relation_count": len(
                transport_fallback_accepted_ids
            ),
            "evidence_binding_contract": evidence_binding_contract,
            "sentence_binding_count": sentence_binding_count,
            "sentence_binding_coverage": sentence_binding_coverage,
            "verified_observation_count": verified_observation_count,
            "directional_judgment_count": directional_judgment_count,
            "contour_validation_drop_count": validation_drop_count,
        },
    }


def _accepted_verification_counts(retrieval: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ledger in retrieval.get("evidence_ledgers", []) or []:
        for decision in ledger.get("source_decisions", []) or []:
            if str(decision.get("decision") or "").lower() not in {
                "accept",
                "accepted",
            }:
                continue
            status = str(
                (decision.get("result") or {}).get("verification_status")
                or "unknown"
            )
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _synthesis_paths(run_dir: Path) -> list[Path]:
    return [
        run_dir / "puzzle_pieces.json",
        run_dir / "detective.json",
        run_dir / "contour_constraints.json",
        run_dir / "contour_candidates.json",
        run_dir / "truth_contour.json",
    ]


def _immutable_config(
    *,
    profile_name: str,
    provider_name: str,
    results_per_query: int,
    model: str,
    cohort_model: str | None,
    max_tokens: int,
    semantic_verdicts: bool,
    semantic_max_tokens: int,
    relational_synthesis: bool,
    relational_max_tokens: int,
) -> dict[str, Any]:
    return {
        "profile_name": profile_name,
        "provider_name": provider_name,
        "results_per_query": int(results_per_query),
        "model": model,
        "cohort_model": cohort_model or model,
        "max_tokens": int(max_tokens),
        "semantic_verdicts": bool(semantic_verdicts),
        "semantic_max_tokens": int(semantic_max_tokens),
        "relational_synthesis": bool(relational_synthesis),
        "relational_max_tokens": int(relational_max_tokens),
        "protocol_revisions": {
            "retrieval": RETRIEVAL_IMPLEMENTATION_REVISION,
            "detective_pipeline": DETECTIVE_PIPELINE_VERSION,
            "detective_proposal": PROPOSAL_PROMPT_REVISION,
            "detective_adjudication": ADJUDICATION_PROMPT_REVISION,
            "contour_pipeline": CONTOUR_PIPELINE_VERSION,
            "contour_final": FINAL_PROMPT_REVISION,
            "contour_speculative_final": SPECULATIVE_FINAL_PROMPT_REVISION,
            "question_grammar": "bme.question-grammar.v1",
            "sentence_evidence_binding": "sentence_level.v1",
        },
    }


def _execution_config(
    *,
    concurrency: int = 24,
    retrieval_concurrency: int | None = None,
    semantic_concurrency: int | None = None,
    model_concurrency: int | None = None,
    pipeline_mode: str = "sequential",
    synthesis_mode: str = "legacy_sequential",
    retries: int = 1,
    recovery_passes: int = 0,
    time_budget_seconds: int = DEFAULT_RUN_TIME_BUDGET_SECONDS,
    input_price_per_1m_cny: float | None = None,
    output_price_per_1m_cny: float | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    if pipeline_mode not in {"sequential", "streaming_v2"}:
        raise ValueError(
            "pipeline_mode must be 'sequential' or 'streaming_v2'"
        )
    if synthesis_mode not in {"legacy_sequential", "optimized_v2"}:
        raise ValueError(
            "synthesis_mode must be 'legacy_sequential' or 'optimized_v2'"
        )
    if int(time_budget_seconds) < 60:
        raise ValueError("time_budget_seconds must be at least 60")
    people_limit = max(1, int(concurrency))
    retrieval_limit = int(
        retrieval_concurrency
        if retrieval_concurrency is not None
        else (
            min(people_limit, 12)
            if pipeline_mode == "streaming_v2"
            else min(people_limit, 4)
        )
    )
    semantic_limit = int(
        semantic_concurrency
        if semantic_concurrency is not None
        else people_limit
    )
    return {
        "pipeline_mode": pipeline_mode,
        "synthesis_mode": synthesis_mode,
        "concurrency": people_limit,
        "retrieval_concurrency": max(1, retrieval_limit),
        "semantic_concurrency": max(1, semantic_limit),
        "model_concurrency": max(
            1,
            int(
                model_concurrency
                if model_concurrency is not None
                else max(people_limit, retrieval_limit, semantic_limit)
            ),
        ),
        "retries": int(retries),
        "recovery_passes": int(recovery_passes),
        "time_budget_seconds": int(time_budget_seconds),
        "input_price_per_1m_cny": input_price_per_1m_cny,
        "output_price_per_1m_cny": output_price_per_1m_cny,
    }


def _bootstrap_legacy_journal(run_path: Path) -> RunJournal:
    retrieval = _read_optional_json(run_path / "retrieval.json")
    run_payload = _read_optional_json(run_path / "run.json")
    summary = _read_optional_json(run_path / "summary.json")
    if not retrieval:
        raise RunStateError("Cannot bootstrap a legacy run without retrieval.json")
    question = str(retrieval.get("question") or "")
    profile_payload = run_payload.get("profile") or retrieval.get("profile") or {}
    profile_name = str(profile_payload.get("name") or "standard")
    provider_name = str(
        run_payload.get("provider")
        or retrieval.get("requested_provider")
        or "three-layer"
    )
    config = _immutable_config(
        profile_name=profile_name,
        provider_name=provider_name,
        results_per_query=int(retrieval.get("results_per_query") or 5),
        model=str(run_payload.get("model") or summary.get("model") or DEFAULT_MODEL),
        cohort_model=str(
            (
                ((run_payload.get("cohort_generation") or {}).get("generation") or {}).get("model")
            )
            or run_payload.get("model")
            or summary.get("model")
            or DEFAULT_MODEL
        ),
        max_tokens=8192,
        semantic_verdicts=(run_path / "semantic_verdicts").exists(),
        semantic_max_tokens=8192,
        relational_synthesis=(run_path / "contour_candidates.json").exists(),
        relational_max_tokens=8192,
    )
    execution = _execution_config(
        concurrency=int(run_payload.get("concurrency") or 6),
        retrieval_concurrency=int(run_payload.get("retrieval_concurrency") or 4),
        retries=int(run_payload.get("retries") or 1),
        recovery_passes=int(run_payload.get("recovery_passes") or 0),
        time_budget_seconds=DEFAULT_RUN_TIME_BUDGET_SECONDS,
    )
    journal = RunJournal.create(
        run_path,
        question=question,
        config=config,
        stage_names=PIPELINE_STAGES,
    )
    journal.payload["legacy_bootstrap"] = True
    journal.payload["execution"] = execution
    journal.payload["legacy_artifacts_detected_at"] = utc_now()
    journal.save()
    return journal


def _payload_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_deadline_epoch(
    journal: RunJournal,
    execution: dict[str, Any],
) -> float:
    created_at = str(
        journal.payload.get("analysis_window_started_at")
        or journal.payload.get("created_at")
        or ""
    )
    try:
        started_epoch = datetime.fromisoformat(created_at).timestamp()
    except ValueError:
        started_epoch = time.time()
    return started_epoch + int(
        execution.get("time_budget_seconds")
        or DEFAULT_RUN_TIME_BUDGET_SECONDS
    )


def _run_elapsed_seconds(journal: RunJournal) -> float:
    created_at = str(journal.payload.get("created_at") or "")
    try:
        started_epoch = datetime.fromisoformat(created_at).timestamp()
    except ValueError:
        return 0.0
    return round(max(0.0, time.time() - started_epoch), 3)


def _require_run_budget(
    journal: RunJournal,
    execution: dict[str, Any],
    stage: str,
    *,
    minimum_seconds: int = 10,
) -> None:
    if (
        _run_deadline_epoch(journal, execution) - time.time()
        < max(1, int(minimum_seconds))
    ):
        budget_seconds = int(
            execution.get("time_budget_seconds")
            or DEFAULT_RUN_TIME_BUDGET_SECONDS
        )
        raise RunTimeBudgetExceeded(
            f"{stage} stopped before starting another model call because the "
            f"{budget_seconds}-second analysis budget was exhausted; completed checkpoints "
            "remain saved"
        )


def _stop_stage(
    journal: RunJournal,
    stage: str,
    error: Exception,
    *,
    retryable: bool = True,
) -> None:
    journal.fail_stage(stage, error, retryable=retryable)
    raise PipelineIncompleteError(journal.run_dir, stage, error) from error
