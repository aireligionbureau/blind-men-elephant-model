from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from ctypes import wintypes
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import DEFAULT_BASE_URL, DEFAULT_COHORT_MODEL, DEFAULT_MODEL


SETTINGS_SCHEMA = "bme.user-settings.v1"
DEFAULT_PROVIDER_PRESET = "deepseek"
SUPPORTED_PROVIDER_PRESETS = {"deepseek", "openai", "openai-compatible"}
PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "cohort_model": DEFAULT_COHORT_MODEL,
        "supports_thinking": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
        "cohort_model": "gpt-4.1-mini",
        "supports_thinking": False,
    },
}
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")
ENVIRONMENT_KEYS = (
    "BME_LLM_API_KEY",
    "DEEPSEEK_API_KEY",
)


class UserSettingsError(ValueError):
    """A user-facing configuration or credential validation error."""

    def __init__(self, message: str, *, code: str = "settings_invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UserModelSettings:
    provider_preset: str = DEFAULT_PROVIDER_PRESET
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    cohort_model: str = DEFAULT_COHORT_MODEL
    supports_thinking: bool = True
    api_key: str = field(default="", repr=False)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def default_settings_path() -> Path:
    override = os.getenv("BME_SETTINGS_FILE")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        root = Path(os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming")
        return root / "BlindMenElephantModel" / "settings.json"
    root = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "blind-men-elephant-model" / "settings.json"


def load_saved_settings(path: str | Path | None = None) -> UserModelSettings:
    settings_path = Path(path) if path is not None else default_settings_path()
    if not settings_path.is_file():
        return UserModelSettings()
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return UserModelSettings()
    if not isinstance(payload, dict) or payload.get("schema") != SETTINGS_SCHEMA:
        return UserModelSettings()
    try:
        api_key = _decode_secret(payload.get("api_key") or {})
        candidate = UserModelSettings(
            provider_preset=str(
                payload.get("provider_preset") or DEFAULT_PROVIDER_PRESET
            ),
            base_url=str(payload.get("base_url") or DEFAULT_BASE_URL),
            model=str(payload.get("model") or DEFAULT_MODEL),
            cohort_model=str(
                payload.get("cohort_model") or DEFAULT_COHORT_MODEL
            ),
            supports_thinking=bool(payload.get("supports_thinking", True)),
            api_key=api_key,
        )
        return validate_settings(candidate)
    except (OSError, UserSettingsError, ValueError, TypeError, AttributeError):
        return UserModelSettings()


def load_effective_settings(
    path: str | Path | None = None,
    *,
    prefer_saved: bool = False,
) -> UserModelSettings:
    saved = load_saved_settings(path)
    if prefer_saved and saved.configured:
        return saved
    return replace(
        saved,
        api_key=(
            os.getenv("BME_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or saved.api_key
        ),
        base_url=(
            os.getenv("BME_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or saved.base_url
        ),
        model=(
            os.getenv("BME_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or saved.model
        ),
        cohort_model=(
            os.getenv("BME_COHORT_MODEL")
            or os.getenv("DEEPSEEK_COHORT_MODEL")
            or saved.cohort_model
        ),
        supports_thinking=_environment_bool(
            "BME_LLM_SUPPORTS_THINKING", saved.supports_thinking
        ),
    )


def settings_from_payload(
    payload: dict[str, Any],
    *,
    current: UserModelSettings,
) -> UserModelSettings:
    supplied_key = str(payload.get("api_key") or "").strip()
    provider_preset = str(
        payload.get("provider_preset")
        or current.provider_preset
        or DEFAULT_PROVIDER_PRESET
    ).strip()
    if provider_preset not in SUPPORTED_PROVIDER_PRESETS:
        raise UserSettingsError("暂不支持这个模型接口类型。")
    changed_provider = provider_preset != current.provider_preset
    defaults = PROVIDER_DEFAULTS.get(provider_preset, {})
    base_url = str(
        payload.get("base_url")
        or (defaults.get("base_url", "") if changed_provider else current.base_url)
    ).strip()
    changed_interface = (
        changed_provider
        or base_url.rstrip("/") != current.base_url.rstrip("/")
    )
    if changed_interface and not supplied_key:
        raise UserSettingsError(
            "更换供应商或接口地址时，请填写这个接口的 API Key。",
            code="api_key_missing",
        )
    model = str(
        payload.get("model")
        or (defaults.get("model", "") if changed_provider else current.model)
    ).strip()
    cohort_model = str(
        payload.get("cohort_model")
        or (
            defaults.get("cohort_model", model)
            if changed_interface
            else current.cohort_model
        )
    ).strip()
    return validate_settings(
        UserModelSettings(
            provider_preset=provider_preset,
            base_url=base_url,
            model=model,
            cohort_model=cohort_model,
            supports_thinking=_payload_bool(
                payload.get("supports_thinking"),
                defaults.get("supports_thinking", False)
                if changed_provider
                else current.supports_thinking,
            ),
            api_key=supplied_key or current.api_key,
        )
    )


def validate_settings(settings: UserModelSettings) -> UserModelSettings:
    if settings.provider_preset not in SUPPORTED_PROVIDER_PRESETS:
        raise UserSettingsError("暂不支持这个模型接口类型。")
    endpoint = urlparse(settings.base_url)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise UserSettingsError("模型接口地址必须是完整的 http 或 https 地址。")
    if (
        settings.provider_preset == "openai"
        and settings.base_url.rstrip("/") != PROVIDER_DEFAULTS["openai"]["base_url"]
    ):
        raise UserSettingsError(
            "OpenAI 预设使用官方接口；其他地址请选择“其他兼容接口”。"
        )
    for label, value in (
        ("主模型", settings.model),
        ("数字人生成模型", settings.cohort_model),
    ):
        if not MODEL_NAME_PATTERN.fullmatch(value):
            raise UserSettingsError(f"{label}名称包含不支持的字符。")
    key = settings.api_key.strip()
    if not key:
        raise UserSettingsError("请填写 API Key。", code="api_key_missing")
    if len(key) < 8 or len(key) > 4096 or any(char.isspace() for char in key):
        raise UserSettingsError("API Key 的格式不正确。", code="api_key_invalid")
    return replace(
        settings,
        base_url=settings.base_url.rstrip("/"),
        api_key=key,
    )


def save_settings(
    settings: UserModelSettings,
    path: str | Path | None = None,
) -> Path:
    validated = validate_settings(settings)
    settings_path = Path(path) if path is not None else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SETTINGS_SCHEMA,
        "provider_preset": validated.provider_preset,
        "base_url": validated.base_url,
        "model": validated.model,
        "cohort_model": validated.cohort_model,
        "supports_thinking": validated.supports_thinking,
        "api_key": _encode_secret(validated.api_key),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="settings.",
        suffix=".tmp",
        dir=settings_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        os.replace(temporary_path, settings_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return settings_path


def apply_settings_to_environment(
    settings: UserModelSettings,
    *,
    overwrite: bool = True,
) -> None:
    mapping = {
        "BME_LLM_API_KEY": settings.api_key,
        "BME_LLM_PROVIDER_PRESET": settings.provider_preset,
        "BME_LLM_BASE_URL": settings.base_url,
        "BME_LLM_MODEL": settings.model,
        "BME_COHORT_MODEL": settings.cohort_model,
        "BME_LLM_SUPPORTS_THINKING": (
            "true" if settings.supports_thinking else "false"
        ),
    }
    for name, value in mapping.items():
        if overwrite or not os.getenv(name):
            os.environ[name] = value


def public_settings(
    settings: UserModelSettings,
    *,
    can_edit: bool,
    verified: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "configured": settings.configured,
        "provider_preset": settings.provider_preset,
        "base_url": settings.base_url,
        "model": settings.model,
        "cohort_model": settings.cohort_model,
        "supports_thinking": settings.supports_thinking,
        "api_key_masked": _masked_key(settings.api_key),
        "can_edit": can_edit,
    }
    if verified is not None:
        payload["verified"] = verified
    return payload


def probe_provider_credentials(
    settings: UserModelSettings,
    *,
    timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    """Validate a key through the provider's non-billable model-list endpoint."""

    validated = validate_settings(settings)
    request = urllib.request.Request(
        f"{validated.base_url}/models",
        headers={
            "Authorization": f"Bearer {validated.api_key}",
            "Accept": "application/json",
            "User-Agent": "BlindMenElephantModel/setup",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            body = response.read(1_000_000)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise UserSettingsError(
                "API Key 没有通过模型接口验证，请检查后重试。",
                code="api_key_rejected",
            ) from exc
        if exc.code in {404, 405} and (
            validated.provider_preset == "openai-compatible"
        ):
            return {
                "ready": True,
                "verified": False,
                "reason": "model_list_not_supported",
            }
        raise UserSettingsError(
            f"模型接口暂时返回 HTTP {exc.code}，请稍后重试。",
            code="provider_unavailable",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UserSettingsError(
            "暂时无法连接模型接口，请检查网络或接口地址。",
            code="provider_unreachable",
        ) from exc

    if status < 200 or status >= 300:
        raise UserSettingsError(
            "模型接口暂时不可用，请稍后重试。",
            code="provider_unavailable",
        )
    available_models: list[str] = []
    try:
        decoded = json.loads(body.decode("utf-8"))
        available_models = [
            str(item.get("id"))
            for item in (decoded.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    if not available_models:
        if validated.provider_preset != "openai-compatible":
            raise UserSettingsError(
                "模型接口未返回可用模型，暂时无法确认所选模型。",
                code="model_list_invalid",
            )
        return {
            "ready": True,
            "verified": False,
            "reason": "model_list_empty_or_invalid",
            "model_available": None,
        }
    if validated.provider_preset in {"deepseek", "openai"}:
        accepted_models = set(available_models)
        if validated.provider_preset == "deepseek" and "deepseek-flash" in accepted_models:
            accepted_models.add("deepseek-v4-flash")
        for label, model in (
            ("主模型", validated.model),
            ("数字人生成模型", validated.cohort_model),
        ):
            if model not in accepted_models:
                raise UserSettingsError(
                    f"{label} {model} 未出现在此 Key 可用的模型列表中，请换一个模型。",
                    code="model_unavailable",
                )
    return {
        "ready": True,
        "verified": True,
        "reason": None,
        "model_available": (
            None
            if not available_models
            else validated.model in available_models
        ),
    }


def _masked_key(api_key: str) -> str:
    if not api_key:
        return ""
    return f"****{api_key[-4:]}"


def _payload_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _environment_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    return fallback if value is None else _payload_bool(value, fallback)


def _encode_secret(secret: str) -> dict[str, str]:
    raw = secret.encode("utf-8")
    if sys.platform == "win32":
        protected = _windows_protect(raw)
        return {
            "protection": "windows-dpapi-current-user",
            "value": base64.b64encode(protected).decode("ascii"),
        }
    return {
        "protection": "user-file",
        "value": base64.b64encode(raw).decode("ascii"),
    }


def _decode_secret(payload: dict[str, Any]) -> str:
    value = str(payload.get("value") or "")
    if not value:
        return ""
    raw = base64.b64decode(value, validate=True)
    protection = payload.get("protection")
    if protection == "windows-dpapi-current-user":
        raw = _windows_unprotect(raw)
    elif protection != "user-file":
        raise ValueError("unsupported secret protection")
    return raw.decode("utf-8")


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _windows_protect(raw: bytes) -> bytes:
    if sys.platform != "win32":
        raise OSError("Windows DPAPI is unavailable")
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw),
        ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    target = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    succeeded = crypt32.CryptProtectData(
        ctypes.byref(source),
        "Blind Men Elephant Model API Key",
        None,
        None,
        None,
        0x1,
        ctypes.byref(target),
    )
    if not succeeded:
        raise OSError(ctypes.get_last_error(), "Windows could not protect the API key")
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def _windows_unprotect(raw: bytes) -> bytes:
    if sys.platform != "win32":
        raise OSError("Windows DPAPI is unavailable")
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(
        len(raw),
        ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    target = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    succeeded = crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(target),
    )
    if not succeeded:
        raise OSError(ctypes.get_last_error(), "Windows could not read the saved API key")
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)
