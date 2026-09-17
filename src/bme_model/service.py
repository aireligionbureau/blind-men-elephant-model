from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .config import DEFAULT_COHORT_MODEL, DEFAULT_MODEL
from .pipeline import (
    PipelineIncompleteError,
    read_run_status,
    resume_resilient_batch,
    run_resilient_batch,
)
from .runtime import RUN_PIPELINE_VERSION, atomic_write_json, read_json, utc_now
from .user_settings import (
    UserModelSettings,
    UserSettingsError,
    apply_settings_to_environment,
    default_settings_path,
    load_effective_settings,
    probe_provider_credentials,
    public_settings,
    save_settings,
    settings_from_payload,
)


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_RUNTIME_REVISION = "2026-09-17-model-choice-v20"


class ProviderConfigurationError(ValueError):
    """A run cannot start until its model provider is configured."""


def _provider_readiness(provider: str, *, timeout_seconds: float = 4.0) -> dict[str, Any]:
    """Check process-level provider access without making a billable model call."""
    if provider == "mock":
        return {"ready": True, "reason": None}

    api_key = os.getenv("BME_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"ready": False, "reason": "api_key_missing"}

    base_url = (
        os.getenv("BME_LLM_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    endpoint = urlparse(base_url)
    host = endpoint.hostname
    if not host:
        return {"ready": False, "reason": "provider_url_invalid"}
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    try:
        connection = socket.create_connection((host, port), timeout=timeout_seconds)
        connection.close()
    except OSError as exc:
        return {
            "ready": False,
            "reason": "provider_network_unreachable",
            "detail": str(exc),
        }
    return {"ready": True, "reason": None}


class RunCoordinator:
    """Small persistent job coordinator for the local web experience."""

    def __init__(
        self,
        *,
        runs_dir: str | Path,
        provider: str = "three-layer",
        model: str = DEFAULT_MODEL,
        cohort_model: str = DEFAULT_COHORT_MODEL,
        run_workers: int = 1,
        concurrency: int = 24,
        retrieval_concurrency: int = 12,
        semantic_concurrency: int = 24,
        model_concurrency: int = 24,
        pipeline_mode: str = "sequential",
        synthesis_mode: str = "legacy_sequential",
        retries: int = 1,
        recovery_passes: int = 0,
        time_budget_seconds: int = 570,
        auto_resume_attempts: int = 1,
        auto_resume_delay_seconds: float = 2.0,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model = model
        self.cohort_model = cohort_model
        self.concurrency = concurrency
        self.retrieval_concurrency = retrieval_concurrency
        self.semantic_concurrency = semantic_concurrency
        self.model_concurrency = model_concurrency
        self.pipeline_mode = pipeline_mode
        self.synthesis_mode = synthesis_mode
        self.retries = retries
        self.recovery_passes = recovery_passes
        self.time_budget_seconds = max(60, int(time_budget_seconds))
        self.auto_resume_attempts = max(0, auto_resume_attempts)
        self.auto_resume_delay_seconds = max(0.0, auto_resume_delay_seconds)
        self.executor = ThreadPoolExecutor(max_workers=max(1, run_workers))
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_keys: dict[str, str] = {}

    def submit(self, question: str) -> tuple[dict[str, Any], bool]:
        normalized = " ".join(question.split()).strip()
        if len(normalized) < 3:
            raise ValueError("问题太短，无法形成复杂问题分析。")
        if len(normalized) > 2000:
            raise ValueError("问题超过 2000 个字符。")
        self._require_provider_configuration()
        key = self._job_key(normalized)
        existing_active_id: str | None = None
        with self._lock:
            existing_id = self._active_keys.get(key)
            if existing_id:
                existing = self._jobs.get(existing_id) or {}
                if existing.get("status") in {"queued", "running"}:
                    existing_active_id = existing_id
            if existing_active_id is not None:
                run_id = existing_active_id
            else:
                run_id = self._new_run_id()
                run_dir = self.runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                queued = {
                    "schema": "bme.queued-job.v1",
                    "run_id": run_id,
                    "question": normalized,
                    "created_at": utc_now(),
                    "provider": self.provider,
                    "model": self.model,
                    "cohort_model": self.cohort_model,
                    "profile": "standard",
                    "pipeline_mode": self.pipeline_mode,
                    "synthesis_mode": self.synthesis_mode,
                }
                atomic_write_json(run_dir / "queued_job.json", queued)
                self._jobs[run_id] = {
                    **queued,
                    "status": "queued",
                    "run_dir": str(run_dir),
                    "error": None,
                }
                self._active_keys[key] = run_id
        if existing_active_id is not None:
            return self.status(existing_active_id), True
        # The durable queue record is written before the worker is exposed.
        self.executor.submit(self._run_job, run_id, normalized, key)
        return self.status(run_id), False

    def status(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        with self._lock:
            memory = dict(self._jobs.get(run_id) or {})
        manifest_path = run_dir / "run_manifest.json"
        persisted: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                persisted = read_run_status(run_dir)
            except Exception as exc:
                persisted = {
                    "status": "recoverable_failed",
                    "error": f"运行清单暂时无法读取：{exc}",
                }
        persisted_status = str(persisted.get("status") or "")
        memory_status = str(memory.get("status") or "")
        memory_updated = str(memory.get("updated_at") or memory.get("created_at") or "")
        persisted_updated = str(persisted.get("updated_at") or "")
        use_memory = bool(memory_status) and (
            not persisted_status or memory_updated >= persisted_updated
        )
        status = memory_status if use_memory else persisted_status or "unknown"
        stages = persisted.get("stages") or {}
        completed_stages = sum(
            (item.get("status") in {"succeeded", "reused"})
            for item in stages.values()
            if isinstance(item, dict)
        )
        people_progress = (stages.get("digital_people") or {}).get("progress") or {}
        retrieval_progress = (stages.get("retrieval") or {}).get("progress") or {}
        semantic_progress = (stages.get("semantic_verdicts") or {}).get("progress") or {}
        synthesis_progress = (
            (stages.get("relational_synthesis") or {}).get("progress") or {}
        )
        successful_stage_states = {"succeeded", "reused"}

        def stage_state(name: str) -> str:
            return str((stages.get(name) or {}).get("status") or "pending")

        forensic_stage_states = [
            stage_state("diagnosis"),
            stage_state("semantic_verdicts"),
        ]
        if all(state in successful_stage_states for state in forensic_stage_states):
            forensic_status = "succeeded"
        elif any(state == "running" for state in forensic_stage_states) or (
            persisted.get("current_stage")
            in {
                "streaming_front",
                "retrieval",
                "digital_people",
                "diagnosis",
                "semantic_verdicts",
            }
        ):
            forensic_status = "running"
        else:
            forensic_status = "pending"

        relational_state = stage_state("relational_synthesis")
        shadow_puzzle_state = stage_state("shadow_puzzle")
        checkpoint_detective = str(
            synthesis_progress.get("detective_status") or ""
        )
        checkpoint_contour = str(synthesis_progress.get("contour_status") or "")
        if relational_state in successful_stage_states or (
            checkpoint_detective == "succeeded"
        ):
            detective_status = "succeeded"
        elif (
            relational_state == "running"
            or shadow_puzzle_state == "running"
            or shadow_puzzle_state in successful_stage_states
        ):
            detective_status = "running"
        else:
            detective_status = "pending"

        if relational_state in successful_stage_states or (
            checkpoint_contour == "succeeded"
        ):
            contour_status = "succeeded"
        elif relational_state == "running" and checkpoint_detective == "succeeded":
            contour_status = "running"
        elif persisted.get("current_stage") == "finalize":
            contour_status = "succeeded"
        else:
            contour_status = "pending"
        cohort_stage = stages.get("cohort") or {}
        cohort_count = int((cohort_stage.get("quality") or {}).get("person_count") or 0)
        if cohort_count == 0 and (run_dir / "cohort.json").exists():
            try:
                cohort_count = len(read_json(run_dir / "cohort.json").get("personas") or [])
            except (OSError, json.JSONDecodeError, AttributeError):
                cohort_count = 0
        report_exists = (run_dir / "report.html").exists()
        pipeline_mode, synthesis_mode = self._saved_execution_modes(run_dir)
        return {
            "run_id": run_id,
            "question": persisted.get("question") or memory.get("question"),
            "status": status,
            "current_stage": (
                memory.get("failed_stage")
                if use_memory and memory.get("failed_stage")
                else persisted.get("current_stage")
                or persisted.get("failed_stage")
            ),
            "completed_stage_count": completed_stages,
            "total_stage_count": 9,
            "cohort": {
                "completed": cohort_count,
                "total": max(cohort_count, 24),
            },
            "retrieval": {
                "completed": int(retrieval_progress.get("completed") or 0),
                "failed": int(retrieval_progress.get("failed") or 0),
                "reused": int(retrieval_progress.get("reused") or 0),
                "total": int(retrieval_progress.get("total") or 24),
            },
            "digital_people": {
                "completed": int(people_progress.get("completed") or 0),
                "total": int(people_progress.get("total") or 24),
                "failed_in_pass": int(people_progress.get("failed_in_pass") or 0),
            },
            "semantic_verdicts": {
                "completed": int(semantic_progress.get("completed") or 0),
                "total": int(semantic_progress.get("total") or 24),
                "fallback": int(semantic_progress.get("fallback") or 0),
            },
            "analysis_layers": {
                "forensic": {
                    "status": forensic_status,
                    "completed": int(semantic_progress.get("completed") or 0),
                    "total": int(semantic_progress.get("total") or 24),
                },
                "detective": {"status": detective_status},
                "contour": {"status": contour_status},
            },
            "synthesis": {
                "detective_status": checkpoint_detective or None,
                "contour_status": checkpoint_contour or None,
                "checkpointed": bool(synthesis_progress.get("checkpointed")),
            },
            "quality": persisted.get("quality") or {},
            "error": (
                None
                if status == "succeeded"
                else (
                    memory.get("error")
                    if use_memory
                    else persisted.get("error") or memory.get("error")
                )
            ),
            "report_ready": report_exists and status == "succeeded",
            "report_url": f"/runs/{run_id}/report.html" if report_exists else None,
            "pipeline_mode": pipeline_mode,
            "synthesis_mode": synthesis_mode,
            "created_at": persisted.get("created_at") or memory.get("created_at"),
            "updated_at": persisted.get("updated_at") or memory.get("updated_at"),
            "deduplicated": False,
        }

    def resume(self, run_id: str) -> tuple[dict[str, Any], bool]:
        self._require_provider_configuration()
        current = self.status(run_id)
        question = str(current.get("question") or "").strip()
        run_dir = self._run_dir(run_id)
        queued_path = run_dir / "queued_job.json"
        if not question and queued_path.exists():
            queued = read_json(queued_path)
            question = str(queued.get("question") or "").strip()
        if not question:
            raise ValueError("saved run has no question")
        key = self._saved_job_key(run_dir, question)
        active_id: str | None = None
        with self._lock:
            candidate_id = self._active_keys.get(key)
            candidate = self._jobs.get(candidate_id or "") or {}
            if candidate.get("status") in {"queued", "running"}:
                active_id = candidate_id
            else:
                self._jobs.setdefault(run_id, {}).update(
                    {
                        "run_id": run_id,
                        "question": question,
                        "run_dir": str(run_dir),
                        "status": "queued",
                        "error": None,
                        "failed_stage": None,
                        "updated_at": utc_now(),
                    }
                )
                self._active_keys[key] = run_id
        if active_id:
            return self.status(active_id), True
        if (run_dir / "run_manifest.json").exists():
            self.executor.submit(self._resume_job, run_id, key, True)
        else:
            self.executor.submit(self._run_job, run_id, question, key)
        return self.status(run_id), False

    def recover_startup_jobs(self, *, limit: int = 10) -> None:
        scheduled: set[str] = set()
        for queued_path in sorted(self.runs_dir.glob("*/queued_job.json")):
            if len(scheduled) >= max(0, limit):
                break
            if (queued_path.parent / "run_manifest.json").exists():
                continue
            try:
                queued = read_json(queued_path)
                run_id = str(queued["run_id"])
                question = str(queued["question"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            key = self._job_key(
                question,
                provider=str(queued.get("provider") or self.provider),
                model=str(queued.get("model") or self.model),
                cohort_model=str(
                    queued.get("cohort_model") or self.cohort_model
                ),
                profile=str(queued.get("profile") or "standard"),
                pipeline_mode=str(
                    queued.get("pipeline_mode") or self.pipeline_mode
                ),
                synthesis_mode=str(
                    queued.get("synthesis_mode") or self.synthesis_mode
                ),
            )
            with self._lock:
                active_id = self._active_keys.get(key)
                active = self._jobs.get(active_id or "") or {}
                if active.get("status") in {"queued", "running"}:
                    continue
                self._jobs[run_id] = {
                    **queued,
                    "status": "queued",
                    "run_dir": str(queued_path.parent),
                    "error": None,
                }
                self._active_keys[key] = run_id
            self.executor.submit(self._run_job, run_id, question, key)
            scheduled.add(run_id)

        manifests = sorted(
            self.runs_dir.glob("*/run_manifest.json"),
            key=lambda path: path.stat().st_mtime,
        )
        for manifest_path in manifests:
            if len(scheduled) >= max(0, limit):
                break
            run_id = manifest_path.parent.name
            if run_id in scheduled:
                continue
            try:
                manifest = read_json(manifest_path)
                status = str(manifest.get("status") or "")
                question = str(manifest.get("question") or "").strip()
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if status not in {"initialized", "running", "recoverable_failed"}:
                continue
            if not question or not RUN_ID_PATTERN.fullmatch(run_id):
                continue
            key = self._saved_job_key(manifest_path.parent, question)
            with self._lock:
                active_id = self._active_keys.get(key)
                active = self._jobs.get(active_id or "") or {}
                if active.get("status") in {"queued", "running"}:
                    continue
                self._jobs[run_id] = {
                    "run_id": run_id,
                    "question": question,
                    "status": "queued",
                    "run_dir": str(manifest_path.parent),
                    "error": None,
                    "updated_at": utc_now(),
                }
                self._active_keys[key] = run_id
            self.executor.submit(self._resume_job, run_id, key, True)
            scheduled.add(run_id)

    def close(self, *, wait: bool = False) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=False)

    def has_active_jobs(self) -> bool:
        with self._lock:
            return any(
                job.get("status") in {"queued", "running"}
                for job in self._jobs.values()
            )

    def apply_provider_settings(
        self,
        settings: UserModelSettings,
        *,
        settings_path: str | Path | None = None,
    ) -> None:
        with self._lock:
            if any(
                job.get("status") in {"queued", "running"}
                for job in self._jobs.values()
            ):
                raise UserSettingsError(
                    "当前问题仍在运行，完成后再修改模型设置。",
                    code="run_active",
                )
            if settings_path is not None:
                save_settings(settings, settings_path)
            apply_settings_to_environment(settings, overwrite=True)
            self.model = settings.model
            self.cohort_model = settings.cohort_model

    def _require_provider_configuration(self) -> None:
        if self.provider == "mock":
            return
        if not (
            os.getenv("BME_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        ):
            raise ProviderConfigurationError(
                "请先在模型设置中填写并验证 API Key。"
            )

    def _run_job(self, run_id: str, question: str, key: str) -> None:
        self._execute_job(
            run_id,
            key,
            question=question,
            start_fresh=True,
            renew_time_budget_on_first_resume=False,
        )

    def _resume_job(
        self,
        run_id: str,
        key: str,
        renew_time_budget: bool = False,
    ) -> None:
        self._execute_job(
            run_id,
            key,
            question=None,
            start_fresh=False,
            renew_time_budget_on_first_resume=renew_time_budget,
        )

    def _execute_job(
        self,
        run_id: str,
        key: str,
        *,
        question: str | None,
        start_fresh: bool,
        renew_time_budget_on_first_resume: bool,
    ) -> None:
        run_dir = self.runs_dir / run_id
        pipeline_mode, synthesis_mode = self._saved_execution_modes(run_dir)
        self._set_memory(
            run_id,
            status="running",
            updated_at=utc_now(),
            error=None,
            failed_stage=None,
        )
        fresh_call = start_fresh
        renew_resume_window = renew_time_budget_on_first_resume
        auto_resume_attempt = 0
        auto_resume_by_stage: dict[str, int] = {}
        auto_resume_total_limit = max(
            self.auto_resume_attempts,
            self.auto_resume_attempts * 2,
        )
        try:
            while True:
                try:
                    if fresh_call:
                        if question is None:
                            raise ValueError("fresh run is missing its question")
                        result = run_resilient_batch(
                            question,
                            profile_name="standard",
                            provider_name=self.provider,
                            model=self.model,
                            cohort_model=self.cohort_model,
                            concurrency=self.concurrency,
                            retrieval_concurrency=self.retrieval_concurrency,
                            semantic_concurrency=self.semantic_concurrency,
                            model_concurrency=self.model_concurrency,
                            pipeline_mode=pipeline_mode,
                            synthesis_mode=synthesis_mode,
                            retries=self.retries,
                            recovery_passes=self.recovery_passes,
                            time_budget_seconds=self.time_budget_seconds,
                            run_dir=run_dir,
                        )
                    else:
                        renew_this_resume = renew_resume_window
                        renew_resume_window = False
                        result = resume_resilient_batch(
                            run_dir,
                            concurrency=self.concurrency,
                            retrieval_concurrency=self.retrieval_concurrency,
                            semantic_concurrency=self.semantic_concurrency,
                            model_concurrency=self.model_concurrency,
                            pipeline_mode=pipeline_mode,
                            synthesis_mode=synthesis_mode,
                            retries=self.retries,
                            recovery_passes=self.recovery_passes,
                            time_budget_seconds=self.time_budget_seconds,
                            renew_time_budget=renew_this_resume,
                        )
                    self._set_memory(
                        run_id,
                        status="succeeded",
                        updated_at=utc_now(),
                        quality=(result.get("run_quality") or {}).get("status"),
                        error=None,
                        failed_stage=None,
                        auto_recovery_attempt=auto_resume_attempt,
                    )
                    return
                except PipelineIncompleteError as exc:
                    retryable = self._pipeline_failure_is_retryable(run_dir, exc)
                    stage_auto_resume_attempt = auto_resume_by_stage.get(exc.stage, 0)
                    if not retryable:
                        self._set_memory(
                            run_id,
                            status="failed",
                            updated_at=utc_now(),
                            error=str(exc.original_error),
                            failed_stage=exc.stage,
                            auto_recovery_attempt=auto_resume_attempt,
                        )
                        return
                    if (
                        stage_auto_resume_attempt >= self.auto_resume_attempts
                        or auto_resume_attempt >= auto_resume_total_limit
                    ):
                        self._set_memory(
                            run_id,
                            status="recoverable_failed",
                            updated_at=utc_now(),
                            error=str(exc.original_error),
                            failed_stage=exc.stage,
                            auto_recovery_attempt=auto_resume_attempt,
                        )
                        return
                    auto_resume_attempt += 1
                    stage_auto_resume_attempt += 1
                    auto_resume_by_stage[exc.stage] = stage_auto_resume_attempt
                    self._set_memory(
                        run_id,
                        status="running",
                        updated_at=utc_now(),
                        error=None,
                        failed_stage=None,
                        auto_recovery_attempt=auto_resume_attempt,
                        auto_recovery_limit=auto_resume_total_limit,
                        auto_recovery_stage=exc.stage,
                        auto_recovery_stage_attempt=stage_auto_resume_attempt,
                        auto_recovery_stage_limit=self.auto_resume_attempts,
                    )
                    delay = (
                        self.auto_resume_delay_seconds * stage_auto_resume_attempt
                    )
                    if delay:
                        time.sleep(delay)
                    fresh_call = False
                except Exception as exc:  # noqa: BLE001 - persist worker state.
                    self._set_memory(
                        run_id,
                        status="failed",
                        updated_at=utc_now(),
                        error=str(exc),
                    )
                    return
        finally:
            with self._lock:
                if self._active_keys.get(key) == run_id:
                    self._active_keys.pop(key, None)

    @staticmethod
    def _pipeline_failure_is_retryable(
        run_dir: Path,
        error: PipelineIncompleteError,
    ) -> bool:
        try:
            manifest = read_json(run_dir / "run_manifest.json")
            stage_error = (
                ((manifest.get("stages") or {}).get(error.stage) or {}).get("error")
                or {}
            )
            retryable = stage_error.get("retryable")
            if isinstance(retryable, bool):
                return retryable
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
        return bool(getattr(error.original_error, "retryable", True))

    def _set_memory(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs.setdefault(run_id, {}).update(changes)

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("invalid run id")
        path = (self.runs_dir / run_id).resolve()
        if path.parent != self.runs_dir:
            raise ValueError("invalid run path")
        return path

    def _saved_job_key(self, run_dir: Path, question: str) -> str:
        config: dict[str, Any] = {}
        manifest_path = run_dir / "run_manifest.json"
        queued_path = run_dir / "queued_job.json"
        try:
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                config = {
                    **dict(manifest.get("config") or {}),
                    **dict(manifest.get("execution") or {}),
                }
            elif queued_path.exists():
                config = dict(read_json(queued_path))
        except (OSError, json.JSONDecodeError, TypeError):
            config = {}
        return self._job_key(
            question,
            provider=str(
                config.get("provider_name") or config.get("provider") or self.provider
            ),
            model=str(config.get("model") or self.model),
            cohort_model=str(
                config.get("cohort_model") or self.cohort_model
            ),
            profile=str(config.get("profile_name") or config.get("profile") or "standard"),
            pipeline_mode=str(
                config.get("pipeline_mode") or self.pipeline_mode
            ),
            synthesis_mode=str(
                config.get("synthesis_mode") or self.synthesis_mode
            ),
        )

    def _saved_execution_modes(self, run_dir: Path) -> tuple[str, str]:
        manifest_path = run_dir / "run_manifest.json"
        queued_path = run_dir / "queued_job.json"
        payload: dict[str, Any] = {}
        try:
            if manifest_path.exists():
                payload = dict(
                    (read_json(manifest_path).get("execution") or {})
                )
            elif queued_path.exists():
                payload = dict(read_json(queued_path))
        except (OSError, json.JSONDecodeError, TypeError):
            payload = {}
        return (
            str(payload.get("pipeline_mode") or self.pipeline_mode),
            str(payload.get("synthesis_mode") or self.synthesis_mode),
        )

    def _job_key(
        self,
        question: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        cohort_model: str | None = None,
        profile: str = "standard",
        pipeline_mode: str | None = None,
        synthesis_mode: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "question": question,
                "provider": provider or self.provider,
                "model": model or self.model,
                "cohort_model": cohort_model or self.cohort_model,
                "profile": profile,
                "pipeline_mode": pipeline_mode or self.pipeline_mode,
                "synthesis_mode": synthesis_mode or self.synthesis_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_run_id() -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return f"{stamp}-standard-{uuid.uuid4().hex[:6]}"


class BMEHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        coordinator: RunCoordinator,
        homepage_dir: Path,
        settings_path: str | Path | None = None,
    ) -> None:
        super().__init__(address, BMERequestHandler)
        self.coordinator = coordinator
        self.homepage_dir = homepage_dir.resolve()
        self.settings_path = (
            Path(settings_path).resolve()
            if settings_path is not None
            else default_settings_path()
        )


class BMERequestHandler(BaseHTTPRequestHandler):
    server: BMEHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API.
        if not self._allow_local_request():
            return
        parsed_request = urlparse(self.path)
        path = unquote(parsed_request.path)
        if path == "/api/health":
            coordinator = self.server.coordinator
            health = {
                "status": "ok",
                "service": "blind-men-elephant-model",
                "runtime_revision": SERVICE_RUNTIME_REVISION,
                "project_root": str(PROJECT_ROOT),
                "pipeline_version": RUN_PIPELINE_VERSION,
                "execution_profile": (
                    "candidate"
                    if coordinator.pipeline_mode == "streaming_v2"
                    and coordinator.synthesis_mode == "optimized_v2"
                    else "baseline"
                ),
                "pipeline_mode": coordinator.pipeline_mode,
                "synthesis_mode": coordinator.synthesis_mode,
                "runs_namespace": coordinator.runs_dir.name,
            }
            probe = parse_qs(parsed_request.query).get("probe") or []
            if "provider" in probe:
                provider = _provider_readiness(coordinator.provider)
                health["provider_ready"] = provider["ready"]
                health["provider_probe"] = provider
            self._json(health)
            return
        if path == "/api/settings":
            settings = self._current_settings()
            payload = public_settings(
                settings,
                can_edit=not self.server.coordinator.has_active_jobs(),
            )
            payload["api_key_required"] = self.server.coordinator.provider != "mock"
            if self.server.coordinator.provider == "mock":
                payload["configured"] = True
            self._json(payload)
            return
        match = re.fullmatch(r"/api/runs/([A-Za-z0-9._-]+)", path)
        if match:
            try:
                self._json(self.server.coordinator.status(match.group(1)))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        report = re.fullmatch(
            r"/runs/([A-Za-z0-9._-]+)/report\.html", path
        )
        if report:
            try:
                run_dir = self.server.coordinator._run_dir(report.group(1))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self._file(run_dir / "report.html", content_type="text/html; charset=utf-8")
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API.
        if not self._allow_local_request():
            return
        path = unquote(urlparse(self.path).path)
        if path == "/api/settings":
            self._configure_provider()
            return
        resume_match = re.fullmatch(
            r"/api/runs/([A-Za-z0-9._-]+)/resume", path
        )
        if resume_match:
            try:
                result, deduplicated = self.server.coordinator.resume(
                    resume_match.group(1)
                )
                result["deduplicated"] = deduplicated
                self._json(result, status=HTTPStatus.ACCEPTED)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if path != "/api/runs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json_body()
            question = str(payload.get("question") or "")
            result, deduplicated = self.server.coordinator.submit(question)
            result["deduplicated"] = deduplicated
            self._json(result, status=HTTPStatus.ACCEPTED)
        except ProviderConfigurationError as exc:
            self._json(
                {"error": str(exc), "code": "provider_not_configured"},
                status=HTTPStatus.PRECONDITION_REQUIRED,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _configure_provider(self) -> None:
        try:
            payload = self._read_json_body()
            current = self._current_settings()
            candidate = settings_from_payload(payload, current=current)
            if self.server.coordinator.has_active_jobs():
                raise UserSettingsError(
                    "当前问题仍在运行，完成后再修改模型设置。",
                    code="run_active",
                )
            probe = probe_provider_credentials(candidate)
            self.server.coordinator.apply_provider_settings(
                candidate,
                settings_path=self.server.settings_path,
            )
            response = public_settings(
                candidate,
                can_edit=True,
                verified=bool(probe.get("verified")),
            )
            response["api_key_required"] = True
            response["probe"] = {
                "ready": bool(probe.get("ready")),
                "verified": bool(probe.get("verified")),
                "reason": probe.get("reason"),
                "model_available": probe.get("model_available"),
            }
            self._json(response)
        except UserSettingsError as exc:
            status = (
                HTTPStatus.CONFLICT
                if exc.code == "run_active"
                else HTTPStatus.BAD_GATEWAY
                if exc.code in {"provider_unavailable", "provider_unreachable", "model_list_invalid"}
                else HTTPStatus.BAD_REQUEST
            )
            self._json(
                {"error": str(exc), "code": exc.code},
                status=status,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except OSError:
            self._json(
                {
                    "error": "设置已经验证，但无法安全保存到这台电脑。",
                    "code": "settings_save_failed",
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _current_settings(self) -> UserModelSettings:
        settings = load_effective_settings(
            self.server.settings_path, prefer_saved=True
        )
        return replace(
            settings,
            model=self.server.coordinator.model,
            cohort_model=self.server.coordinator.cohort_model,
        )

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > 65536:
            raise ValueError("request body must be between 1 and 65536 bytes")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _allow_local_request(self) -> bool:
        host = urlparse(f"//{self.headers.get('Host', '')}")
        bound_host = str(self.server.server_address[0]).casefold()
        allowed_hosts = (
            {"127.0.0.1", "localhost"}
            if bound_host in {"127.0.0.1", "localhost"}
            else {bound_host}
        )
        try:
            valid_host = (
                host.hostname is not None
                and host.hostname.casefold() in allowed_hosts
                and host.port == self.server.server_port
                and not host.username
                and not host.password
            )
            origin = self.headers.get("Origin")
            valid_origin = not origin or origin == f"http://{self.headers['Host']}"
        except ValueError:
            valid_host = valid_origin = False
        if valid_host and valid_origin:
            return True
        self.send_error(HTTPStatus.FORBIDDEN)
        return False

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (self.server.homepage_dir / relative).resolve()
        try:
            target.relative_to(self.server.homepage_dir)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        self._file(target)

    def _file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers(include_content_policy=mime.startswith("text/html"))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers(include_content_policy=False)
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self, *, include_content_policy: bool) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if include_content_policy:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    runs_dir: str | Path = "runs",
    homepage_dir: str | Path | None = None,
    provider: str = "three-layer",
    model: str | None = None,
    cohort_model: str | None = None,
    run_workers: int = 1,
    pipeline_mode: str = "sequential",
    synthesis_mode: str = "legacy_sequential",
    auto_recover: bool = True,
    settings_path: str | Path | None = None,
) -> None:
    resolved_settings_path = (
        Path(settings_path).resolve()
        if settings_path is not None
        else default_settings_path()
    )
    effective_settings = load_effective_settings(
        resolved_settings_path, prefer_saved=True
    )
    if effective_settings.configured:
        apply_settings_to_environment(effective_settings, overwrite=True)
    coordinator = RunCoordinator(
        runs_dir=runs_dir,
        provider=provider,
        model=model or effective_settings.model,
        cohort_model=cohort_model or effective_settings.cohort_model,
        run_workers=run_workers,
        pipeline_mode=pipeline_mode,
        synthesis_mode=synthesis_mode,
    )
    if auto_recover and (
        provider == "mock" or effective_settings.configured
    ):
        coordinator.recover_startup_jobs()
    server = BMEHTTPServer(
        (host, port),
        coordinator,
        Path(homepage_dir) if homepage_dir else PROJECT_ROOT / "homepage",
        resolved_settings_path,
    )
    try:
        print(f"Blind Men Elephant Model: http://{host}:{port}")
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        coordinator.close()
