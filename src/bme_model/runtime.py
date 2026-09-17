from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import AbstractContextManager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


RUN_MANIFEST_SCHEMA = "bme.run-manifest.v1"
RUN_PIPELINE_VERSION = "bme.resilient-run.v2"


class RunStateError(RuntimeError):
    """Raised when a saved run cannot be resumed without mixing run state."""


class RunAlreadyActiveError(RunStateError):
    """Raised when another live process owns the same run directory."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: str | Path, value: str) -> Path:
    """Replace a text artifact atomically within its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_contention_retry(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def _replace_with_contention_retry(source: Path, target: Path) -> None:
    """Tolerate brief Windows scanner/indexer locks without slowing normal writes."""

    delays = (0.03, 0.06, 0.12)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    body = target.read_bytes()
    return {
        "path": target.name,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RunLock(AbstractContextManager["RunLock"]):
    """A crash-recoverable single-writer lock for one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / ".run.lock"
        self.owner = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                existing = _read_lock_owner(self.path)
                if _lock_owner_is_alive(existing):
                    raise RunAlreadyActiveError(
                        "This run is already active"
                        f" (pid={existing.get('pid')}, host={existing.get('host')})."
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.owner, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            self.acquired = True
            return self
        raise RunAlreadyActiveError("Could not acquire the run lock.")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired:
            return
        existing = _read_lock_owner(self.path)
        if int(existing.get("pid") or -1) == os.getpid():
            self.path.unlink(missing_ok=True)
        self.acquired = False


class RunJournal:
    """Persisted stage state used by both CLI runs and future web workers."""

    def __init__(self, run_dir: str | Path, payload: dict[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "run_manifest.json"
        self.payload = payload

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        *,
        question: str,
        config: dict[str, Any],
        stage_names: list[str],
    ) -> "RunJournal":
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        payload = {
            "schema": RUN_MANIFEST_SCHEMA,
            "pipeline_version": RUN_PIPELINE_VERSION,
            "run_id": run_path.name,
            "question": question,
            "config": deepcopy(config),
            "config_fingerprint": config_fingerprint(config),
            "status": "initialized",
            "current_stage": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "resume_count": 0,
            "stages": {
                name: {
                    "status": "pending",
                    "attempts": 0,
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "quality": {},
                    "artifacts": [],
                }
                for name in stage_names
            },
            "quality": {
                "status": "not_evaluated",
                "blocking_issues": [],
                "warnings": [],
                "metrics": {},
            },
            "events": [],
        }
        journal = cls(run_path, payload)
        journal.save()
        return journal

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunJournal":
        run_path = Path(run_dir)
        path = run_path / "run_manifest.json"
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get("schema") != RUN_MANIFEST_SCHEMA:
            raise RunStateError("run_manifest.json is missing or has an unsupported schema")
        return cls(run_path, payload)

    def assert_compatible(self, *, question: str, config: dict[str, Any]) -> None:
        if str(self.payload.get("question") or "") != question:
            raise RunStateError("Resume question does not match the saved run")
        expected = str(self.payload.get("config_fingerprint") or "")
        actual = config_fingerprint(config)
        if expected != actual:
            raise RunStateError(
                "Resume configuration differs from the saved immutable run configuration"
            )

    def begin_resume(self) -> None:
        self.payload["resume_count"] = int(self.payload.get("resume_count") or 0) + 1
        self.payload["status"] = "running"
        self.payload["completed_at"] = None
        self._event("run_resumed", None, {})
        self.save()

    def start_stage(self, name: str) -> None:
        stage = self._stage(name)
        stage["status"] = "running"
        stage["attempts"] = int(stage.get("attempts") or 0) + 1
        stage["started_at"] = utc_now()
        stage["completed_at"] = None
        stage["error"] = None
        self.payload["status"] = "running"
        self.payload["current_stage"] = name
        self._event("stage_started", name, {"attempt": stage["attempts"]})
        self.save()

    def complete_stage(
        self,
        name: str,
        *,
        artifacts: list[str | Path] | None = None,
        quality: dict[str, Any] | None = None,
        reused: bool = False,
    ) -> None:
        stage = self._stage(name)
        stage["status"] = "reused" if reused else "succeeded"
        stage["completed_at"] = utc_now()
        stage["error"] = None
        stage["quality"] = deepcopy(quality or {})
        stage["artifacts"] = []
        for path in artifacts or []:
            target = Path(path)
            if not target.exists():
                continue
            record = artifact_record(target)
            try:
                record["path"] = str(
                    target.resolve().relative_to(self.run_dir.resolve())
                ).replace("\\", "/")
            except ValueError:
                record["path"] = str(target.resolve())
            stage["artifacts"].append(record)
        self.payload["current_stage"] = None
        self._event("stage_completed", name, {"reused": reused})
        self.save()

    def fail_stage(
        self,
        name: str,
        error: Exception | str,
        *,
        retryable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        stage = self._stage(name)
        stage["status"] = "recoverable_failed" if retryable else "failed"
        stage["completed_at"] = utc_now()
        stage["error"] = {
            "message": str(error),
            "retryable": retryable,
            "details": deepcopy(details or {}),
        }
        self.payload["status"] = stage["status"]
        self.payload["current_stage"] = None
        self._event("stage_failed", name, stage["error"])
        self.save()

    def set_progress(self, name: str, progress: dict[str, Any]) -> None:
        stage = self._stage(name)
        stage["progress"] = deepcopy(progress)
        self.save()

    def set_quality(self, quality: dict[str, Any]) -> None:
        self.payload["quality"] = deepcopy(quality)
        self.save()

    def finish(self, *, degraded: bool = False) -> None:
        self.payload["status"] = "degraded" if degraded else "succeeded"
        self.payload["current_stage"] = None
        self.payload["completed_at"] = utc_now()
        self._event("run_completed", None, {"degraded": degraded})
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def _stage(self, name: str) -> dict[str, Any]:
        stages = self.payload.setdefault("stages", {})
        if name not in stages:
            stages[name] = {
                "status": "pending",
                "attempts": 0,
                "started_at": None,
                "completed_at": None,
                "error": None,
                "quality": {},
                "artifacts": [],
            }
        return stages[name]

    def _event(
        self,
        event: str,
        stage: str | None,
        details: dict[str, Any],
    ) -> None:
        events = self.payload.setdefault("events", [])
        events.append(
            {
                "at": utc_now(),
                "event": event,
                "stage": stage,
                "details": deepcopy(details),
            }
        )
        if len(events) > 200:
            del events[:-200]


def _read_lock_owner(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lock_owner_is_alive(owner: dict[str, Any]) -> bool:
    pid = int(owner.get("pid") or 0)
    host = str(owner.get("host") or "")
    if _lock_is_expired(owner):
        return False
    if pid <= 0 or (host and host != socket.gethostname()):
        return bool(host and host != socket.gethostname())
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_is_expired(owner: dict[str, Any]) -> bool:
    raw = str(owner.get("acquired_at") or "")
    if not raw:
        return True
    try:
        acquired = datetime.fromisoformat(raw)
    except ValueError:
        return True
    if acquired.tzinfo is None:
        acquired = acquired.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - acquired > timedelta(hours=48)
