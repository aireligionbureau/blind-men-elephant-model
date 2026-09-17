from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GovernorSnapshot:
    configured_limit: int
    current_limit: int
    active: int
    peak_active: int
    completed: int
    throttled: int
    failed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "configured_limit": self.configured_limit,
            "current_limit": self.current_limit,
            "active": self.active,
            "peak_active": self.peak_active,
            "completed": self.completed,
            "throttled": self.throttled,
            "failed": self.failed,
        }


class AdaptiveConcurrencyGovernor:
    """Provider-neutral request gate with conservative pressure recovery.

    The governor only controls when a request may start. It never changes the
    model, prompt, token budget, response, retry policy, or quality gate.
    """

    def __init__(
        self,
        max_concurrency: int,
        *,
        min_concurrency: int = 2,
        recovery_successes: int = 12,
        pressure_cooldown_seconds: float = 1.0,
    ) -> None:
        configured = max(1, int(max_concurrency))
        self.configured_limit = configured
        self.current_limit = configured
        self.min_concurrency = max(1, min(int(min_concurrency), configured))
        self.recovery_successes = max(1, int(recovery_successes))
        self.pressure_cooldown_seconds = max(
            0.0, float(pressure_cooldown_seconds)
        )
        self._condition = threading.Condition()
        self._active = 0
        self._peak_active = 0
        self._completed = 0
        self._throttled = 0
        self._failed = 0
        self._successes_since_pressure = 0
        self._cooldown_until = 0.0

    def slot(self) -> "_GovernorSlot":
        return _GovernorSlot(self)

    def snapshot(self) -> GovernorSnapshot:
        with self._condition:
            return GovernorSnapshot(
                configured_limit=self.configured_limit,
                current_limit=self.current_limit,
                active=self._active,
                peak_active=self._peak_active,
                completed=self._completed,
                throttled=self._throttled,
                failed=self._failed,
            )

    def _acquire(self) -> None:
        with self._condition:
            while True:
                cooldown = self._cooldown_until - time.monotonic()
                if self._active < self.current_limit and cooldown <= 0:
                    self._active += 1
                    self._peak_active = max(self._peak_active, self._active)
                    return
                self._condition.wait(
                    timeout=max(0.05, min(cooldown, 0.5))
                    if cooldown > 0
                    else 0.5
                )

    def _release(self, error: BaseException | None) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            if error is None:
                self._completed += 1
                self._successes_since_pressure += 1
                if (
                    self.current_limit < self.configured_limit
                    and self._successes_since_pressure >= self.recovery_successes
                ):
                    self.current_limit += 1
                    self._successes_since_pressure = 0
            else:
                self._failed += 1
                status_code = getattr(error, "status_code", None)
                retryable = bool(getattr(error, "retryable", False))
                pressure = status_code in {408, 425, 429, 500, 502, 503, 504}
                if pressure or retryable:
                    self._throttled += 1
                    self.current_limit = max(
                        self.min_concurrency,
                        max(1, self.current_limit // 2),
                    )
                    self._successes_since_pressure = 0
                    self._cooldown_until = max(
                        self._cooldown_until,
                        time.monotonic() + self.pressure_cooldown_seconds,
                    )
            self._condition.notify_all()


class _GovernorSlot(AbstractContextManager[None]):
    def __init__(self, governor: AdaptiveConcurrencyGovernor) -> None:
        self.governor = governor

    def __enter__(self) -> None:
        self.governor._acquire()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        del exc_type, traceback
        self.governor._release(exc)
