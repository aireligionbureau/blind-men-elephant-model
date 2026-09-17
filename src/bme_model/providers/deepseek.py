from __future__ import annotations

import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL
from ..scheduler import AdaptiveConcurrencyGovernor


class DeepSeekAPIError(RuntimeError):
    """Structured provider failure used by retry and recovery policy."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        uncertain_remote_completion: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.uncertain_remote_completion = uncertain_remote_completion


class DeepSeekTimeoutError(DeepSeekAPIError):
    """A request may still be running remotely, so callers should not retry blindly."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            retryable=False,
            uncertain_remote_completion=True,
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]
    usage: dict[str, Any]
    message: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client with DeepSeek defaults.

    Generic BME_LLM_* settings take precedence; DEEPSEEK_* remains a backwards-
    compatible fallback. Do not hardcode keys in repository files.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 120,
        request_governor: AdaptiveConcurrencyGovernor | None = None,
        user_id: str | None = None,
        supports_thinking: bool | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("BME_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("BME_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("BME_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or DEFAULT_MODEL
        )
        self.timeout_seconds = timeout_seconds
        self.request_governor = request_governor
        self.user_id = user_id
        self.supports_thinking = (
            supports_thinking
            if supports_thinking is not None
            else _supports_thinking(self.base_url)
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        thinking: bool | str | dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise DeepSeekAPIError(
                "BME_LLM_API_KEY or DEEPSEEK_API_KEY is not set.",
                retryable=False,
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if thinking is not None and self.supports_thinking:
            if isinstance(thinking, dict):
                payload["thinking"] = thinking
            else:
                enabled = thinking is True or compact_thinking(thinking) == "enabled"
                payload["thinking"] = {
                    "type": "enabled" if enabled else "disabled"
                }
        if reasoning_effort is not None and self.supports_thinking:
            payload["reasoning_effort"] = reasoning_effort
        if self.user_id and urlparse(self.base_url).hostname == "api.deepseek.com":
            payload["user_id"] = self.user_id

        if self.request_governor is None:
            return self._send_chat_request(payload)
        with self.request_governor.slot():
            return self._send_chat_request(payload)

    def _send_chat_request(self, payload: dict[str, Any]) -> LLMResponse:

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                try:
                    body = response.read()
                except http.client.IncompleteRead as exc:
                    body = exc.partial
                    try:
                        raw = json.loads(body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as parse_exc:
                        raise DeepSeekAPIError(
                            "LLM provider response ended before the JSON body was complete.",
                            retryable=True,
                        ) from parse_exc
                else:
                    raw = json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            raise DeepSeekAPIError(
                f"LLM provider HTTP {exc.code}: {body}",
                status_code=exc.code,
                retryable=retryable,
                retry_after_seconds=_retry_after_seconds(
                    (exc.headers or {}).get("Retry-After")
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise DeepSeekTimeoutError(
                    f"LLM provider request exceeded {self.timeout_seconds} seconds; automatic retry disabled to avoid duplicate billing."
                ) from exc
            raise DeepSeekAPIError(
                f"LLM provider request failed: {exc}",
                retryable=True,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise DeepSeekTimeoutError(
                f"LLM provider request exceeded {self.timeout_seconds} seconds; automatic retry disabled to avoid duplicate billing."
            ) from exc

        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM provider response has no choices: {raw}")

        message = choices[0].get("message") or {}
        return LLMResponse(
            content=message.get("content") or "",
            raw=raw,
            usage=raw.get("usage", {}),
            message=message,
            tool_calls=list(message.get("tool_calls") or []),
            reasoning_content=message.get("reasoning_content"),
        )


def compact_thinking(value: Any) -> str:
    return str(value or "").strip().lower()


def synthesis_request_timeout_seconds(default: int = 150) -> int:
    raw = os.getenv("BME_LLM_SYNTHESIS_TIMEOUT_SECONDS")
    if raw is None:
        return default
    try:
        configured = int(raw)
    except ValueError:
        return default
    return max(60, min(configured, 360))


def bounded_request_timeout_seconds(
    deadline_epoch: float | None,
    default: int,
    *,
    reserve_seconds: int = 5,
    minimum_seconds: int = 5,
) -> int:
    """Fit one provider request inside the remaining end-to-end run budget."""

    configured = max(minimum_seconds, int(default))
    if deadline_epoch is None:
        return configured
    remaining = int(deadline_epoch - time.time() - reserve_seconds)
    if remaining < minimum_seconds:
        raise DeepSeekTimeoutError(
            "The run time budget is exhausted; no new model request was started."
        )
    return min(configured, remaining)


def _supports_thinking(base_url: str) -> bool:
    configured = os.getenv("BME_LLM_SUPPORTS_THINKING")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return "deepseek.com" in base_url.casefold()


def retry_delay_seconds(error: Exception, attempt: int, *, cap: float = 30.0) -> float:
    requested = getattr(error, "retry_after_seconds", None)
    if requested is not None:
        return max(0.0, min(float(requested), cap))
    return min(float(2**max(attempt, 0)), cap)


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


# Provider-neutral names are the public integration surface. The historical
# DeepSeek names remain aliases so existing runs and downstream imports keep working.
OpenAICompatibleClient = DeepSeekClient
ProviderAPIError = DeepSeekAPIError
ProviderTimeoutError = DeepSeekTimeoutError
