from __future__ import annotations

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bme_model.providers.deepseek import DeepSeekClient
from bme_model.service import (
    BMEHTTPServer,
    ProviderConfigurationError,
    RunCoordinator,
)
from bme_model.user_settings import (
    UserModelSettings,
    UserSettingsError,
    load_effective_settings,
    load_saved_settings,
    public_settings,
    probe_provider_credentials,
    save_settings,
    settings_from_payload,
)


HOMEPAGE = Path(__file__).resolve().parents[1] / "homepage"
EMPTY_PROVIDER_ENV = {
    "BME_LLM_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "BME_LLM_BASE_URL": "",
    "DEEPSEEK_BASE_URL": "",
    "BME_LLM_MODEL": "",
    "DEEPSEEK_MODEL": "",
    "BME_COHORT_MODEL": "",
    "DEEPSEEK_COHORT_MODEL": "",
    "BME_LLM_SUPPORTS_THINKING": "",
}


class UserSettingsTests(unittest.TestCase):
    def test_windows_one_click_entrypoints_are_present(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        installer = (project_root / "tools" / "install_windows.ps1").read_text(
            encoding="utf-8"
        )
        launcher = (project_root / "tools" / "start_web.ps1").read_text(
            encoding="utf-8"
        )

        self.assertTrue((project_root / "install-windows.cmd").is_file())
        self.assertTrue((project_root / "start-windows.cmd").is_file())
        self.assertIn(".venv\\Scripts\\python.exe", installer)
        self.assertIn("$productName.lnk", installer)
        self.assertIn("[char]0x76F2", installer)
        self.assertIn("blind-men-elephant-model.ico", installer)
        self.assertIn("2026-09-17-model-choice-v20", launcher)
        self.assertIn("$health.project_root", launcher)
        self.assertNotIn("/api/health?probe=provider", launcher)

    def test_saved_key_round_trips_without_public_disclosure(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "settings.json"
            settings = UserModelSettings(api_key="sk-test-123456789")
            save_settings(settings, path)
            loaded = load_saved_settings(path)

            self.assertEqual(loaded.api_key, settings.api_key)
            public = public_settings(loaded, can_edit=True)
            self.assertNotIn("api_key", public)
            self.assertEqual(public["api_key_masked"], "****6789")
            self.assertNotIn(settings.api_key, json.dumps(public))
            if sys.platform == "win32":
                self.assertNotIn(
                    settings.api_key,
                    path.read_text(encoding="utf-8"),
                )

    def test_malformed_settings_do_not_block_first_run(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "settings.json"
            for value in ('[]', '{"schema":"bme.user-settings.v1","api_key":"bad"}'):
                path.write_text(value, encoding="utf-8")
                self.assertFalse(load_saved_settings(path).configured)

    def test_blank_key_preserves_the_existing_saved_key(self) -> None:
        current = UserModelSettings(api_key="sk-existing-123456")
        updated = settings_from_payload(
            {"api_key": "", "model": "another-model"},
            current=current,
        )
        self.assertEqual(updated.api_key, current.api_key)
        self.assertEqual(updated.model, "another-model")

    def test_switching_provider_requires_its_own_key_and_defaults(self) -> None:
        current = UserModelSettings(api_key="sk-deepseek-123456")
        with self.assertRaises(UserSettingsError) as raised:
            settings_from_payload(
                {"provider_preset": "openai"}, current=current
            )
        self.assertEqual(raised.exception.code, "api_key_missing")

        switched = settings_from_payload(
            {
                "provider_preset": "openai",
                "api_key": "sk-openai-123456",
            },
            current=current,
        )
        self.assertEqual(switched.base_url, "https://api.openai.com/v1")
        self.assertEqual(switched.model, "gpt-4.1")
        self.assertEqual(switched.cohort_model, "gpt-4.1-mini")
        self.assertFalse(switched.supports_thinking)
        self.assertEqual(switched.api_key, "sk-openai-123456")

        chosen_model = settings_from_payload(
            {"model": "gpt-4o", "cohort_model": "gpt-4.1-mini"},
            current=switched,
        )
        self.assertEqual(chosen_model.model, "gpt-4o")

        with TemporaryDirectory() as root:
            path = Path(root) / "settings.json"
            save_settings(switched, path)
            self.assertEqual(load_saved_settings(path), switched)
            with patch.dict(
                os.environ,
                {
                    "BME_LLM_API_KEY": "sk-stale-generic-123456",
                    "DEEPSEEK_API_KEY": "sk-stale-deepseek-123456",
                    "BME_LLM_BASE_URL": "https://api.deepseek.com",
                    "BME_LLM_MODEL": "deepseek-v4-pro",
                },
            ):
                self.assertEqual(
                    load_effective_settings(path, prefer_saved=True), switched
                )

    def test_changing_custom_endpoint_cannot_reuse_old_key(self) -> None:
        current = UserModelSettings(
            provider_preset="openai-compatible",
            base_url="https://first.example/v1",
            model="first-model",
            cohort_model="first-model",
            supports_thinking=False,
            api_key="first-provider-key",
        )
        with self.assertRaises(UserSettingsError) as raised:
            settings_from_payload(
                {"base_url": "https://second.example/v1"},
                current=current,
            )
        self.assertEqual(raised.exception.code, "api_key_missing")

        switched = settings_from_payload(
            {
                "base_url": "https://second.example/v1",
                "api_key": "second-provider-key",
                "model": "second-model",
            },
            current=current,
        )
        self.assertEqual(switched.api_key, "second-provider-key")
        self.assertEqual(switched.model, "second-model")
        self.assertEqual(switched.cohort_model, "second-model")

    def test_invalid_key_is_rejected_before_it_can_be_saved(self) -> None:
        with self.assertRaises(UserSettingsError) as raised:
            settings_from_payload(
                {"api_key": "bad key"},
                current=UserModelSettings(),
            )
        self.assertEqual(raised.exception.code, "api_key_invalid")

    def test_client_reads_runtime_endpoint_instead_of_import_time_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BME_LLM_API_KEY": "sk-runtime-123456",
                "BME_LLM_BASE_URL": "https://provider.example/v1",
                "BME_LLM_MODEL": "runtime-model",
            },
        ):
            client = DeepSeekClient()
        self.assertEqual(client.base_url, "https://provider.example/v1")
        self.assertEqual(client.model, "runtime-model")

    def test_openai_client_does_not_send_deepseek_only_fields(self) -> None:
        client = DeepSeekClient(
            api_key="sk-openai-123456",
            base_url="https://api.openai.com/v1",
            model="gpt-4.1",
            user_id="isolated-run",
            supports_thinking=False,
        )
        with patch.object(client, "_send_chat_request") as send:
            client.chat(
                [{"role": "user", "content": "test"}],
                thinking=True,
                reasoning_effort="high",
            )
        payload = send.call_args.args[0]
        self.assertEqual(payload["model"], "gpt-4.1")
        self.assertNotIn("user_id", payload)
        self.assertNotIn("thinking", payload)
        self.assertNotIn("reasoning_effort", payload)

    def test_provider_probe_uses_the_nonbillable_models_endpoint(self) -> None:
        captured: dict[str, str] = {}

        class ModelListHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
                captured["path"] = self.path
                captured["authorization"] = self.headers.get("Authorization", "")
                body = json.dumps(
                    {"data": [{"id": "fixture-model"}, {"id": "deepseek-flash"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelListHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = probe_provider_credentials(
                UserModelSettings(
                    api_key="sk-probe-123456789",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="fixture-model",
                ),
                timeout_seconds=2,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(captured["path"], "/models")
        self.assertEqual(
            captured["authorization"],
            "Bearer sk-probe-123456789",
        )
        self.assertTrue(result["verified"])
        self.assertTrue(result["model_available"])

    def test_trusted_preset_rejects_missing_cohort_model(self) -> None:
        class ModelListResponse:
            status = 200

            def __enter__(self) -> "ModelListResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b'{"data":[{"id":"deepseek-v4-pro"}]}'

        with patch(
            "bme_model.user_settings.urllib.request.urlopen",
            return_value=ModelListResponse(),
        ):
            with self.assertRaises(UserSettingsError) as raised:
                probe_provider_credentials(
                    UserModelSettings(api_key="sk-deepseek-123456")
                )
        self.assertEqual(raised.exception.code, "model_unavailable")

    def test_empty_model_list_is_not_treated_as_verified(self) -> None:
        class EmptyModelListResponse:
            status = 200

            def __enter__(self) -> "EmptyModelListResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b'{"data":[]}'

        with patch(
            "bme_model.user_settings.urllib.request.urlopen",
            return_value=EmptyModelListResponse(),
        ):
            with self.assertRaises(UserSettingsError) as raised:
                probe_provider_credentials(
                    UserModelSettings(api_key="sk-deepseek-123456")
                )
            custom = probe_provider_credentials(
                UserModelSettings(
                    provider_preset="openai-compatible",
                    base_url="https://custom.example/v1",
                    model="custom-model",
                    cohort_model="custom-model",
                    supports_thinking=False,
                    api_key="sk-custom-123456",
                )
            )
        self.assertEqual(raised.exception.code, "model_list_invalid")
        self.assertFalse(custom["verified"])

    def test_real_provider_run_is_blocked_until_key_is_configured(self) -> None:
        with TemporaryDirectory() as root, patch.dict(
            os.environ, EMPTY_PROVIDER_ENV
        ):
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="three-layer",
            )
            try:
                with self.assertRaises(ProviderConfigurationError):
                    coordinator.submit("未配置密钥时能否错误地启动运行？")
            finally:
                coordinator.close(wait=True)

    def test_settings_api_verifies_saves_and_never_echoes_key(self) -> None:
        api_key = "sk-web-setup-123456789"
        with TemporaryDirectory() as root, patch.dict(
            os.environ, EMPTY_PROVIDER_ENV
        ), patch(
            "bme_model.service.probe_provider_credentials",
            return_value={
                "ready": True,
                "verified": True,
                "reason": None,
                "model_available": True,
            },
        ):
            settings_path = Path(root) / "private" / "settings.json"
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="three-layer",
                model="deepseek-v4-pro",
                cohort_model="deepseek-v4-flash",
            )
            server = BMEHTTPServer(
                ("127.0.0.1", 0),
                coordinator,
                HOMEPAGE,
                settings_path,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                before = _get_json(base + "/api/settings")
                self.assertFalse(before["configured"])

                saved = _post_json(
                    base + "/api/settings",
                    {
                        "api_key": api_key,
                        "provider_preset": "deepseek",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "cohort_model": "deepseek-v4-flash",
                        "supports_thinking": True,
                    },
                )
                after = _get_json(base + "/api/settings")

                self.assertTrue(saved["configured"])
                self.assertTrue(saved["verified"])
                self.assertTrue(after["configured"])
                self.assertEqual(after["api_key_masked"], "****6789")
                self.assertNotIn(api_key, json.dumps(saved))
                self.assertNotIn(api_key, json.dumps(after))
                self.assertEqual(
                    load_saved_settings(settings_path).api_key,
                    api_key,
                )

                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    _post_json(
                        base + "/api/settings",
                        {"provider_preset": "openai"},
                    )
                self.assertEqual(rejected.exception.code, 400)

                switched = _post_json(
                    base + "/api/settings",
                    {
                        "provider_preset": "openai",
                        "api_key": "sk-openai-web-123456",
                    },
                )
                self.assertEqual(switched["provider_preset"], "openai")
                self.assertEqual(switched["model"], "gpt-4.1")
                self.assertEqual(switched["cohort_model"], "gpt-4.1-mini")
                self.assertEqual(coordinator.model, "gpt-4.1")
                self.assertEqual(DeepSeekClient().base_url, "https://api.openai.com/v1")
                self.assertEqual(DeepSeekClient().api_key, "sk-openai-web-123456")
                self.assertEqual(load_saved_settings(settings_path).api_key, "sk-openai-web-123456")
            finally:
                server.shutdown()
                server.server_close()
                coordinator.close(wait=True)
                thread.join(timeout=3)

    def test_settings_cannot_change_while_a_run_is_active(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def fake_run(*_args: object, **_kwargs: object) -> dict:
            started.set()
            release.wait(timeout=3)
            return {"run_quality": {"status": "passed"}}

        with TemporaryDirectory() as root, patch(
            "bme_model.service.run_resilient_batch", side_effect=fake_run
        ):
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="mock",
                run_workers=1,
            )
            try:
                coordinator.submit("运行过程中是否应允许替换模型设置？")
                self.assertTrue(started.wait(timeout=2))
                with self.assertRaises(UserSettingsError) as raised:
                    coordinator.apply_provider_settings(
                        UserModelSettings(api_key="sk-test-123456")
                    )
                self.assertEqual(raised.exception.code, "run_active")
            finally:
                release.set()
                coordinator.close(wait=True)

    def test_failed_save_does_not_activate_an_unsaved_key(self) -> None:
        with TemporaryDirectory() as root, patch.dict(
            os.environ, EMPTY_PROVIDER_ENV
        ), patch(
            "bme_model.service.save_settings",
            side_effect=OSError("disk unavailable"),
        ):
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="three-layer",
                model="original-model",
                cohort_model="original-cohort",
            )
            try:
                with self.assertRaises(OSError):
                    coordinator.apply_provider_settings(
                        UserModelSettings(api_key="sk-unsaved-123456"),
                        settings_path=Path(root) / "settings.json",
                    )
                self.assertEqual(coordinator.model, "original-model")
                self.assertEqual(coordinator.cohort_model, "original-cohort")
                self.assertFalse(os.getenv("BME_LLM_API_KEY"))
            finally:
                coordinator.close(wait=True)

    def test_local_service_rejects_foreign_host_and_origin(self) -> None:
        with TemporaryDirectory() as root:
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs", provider="mock"
            )
            server = BMEHTTPServer(
                ("127.0.0.1", 0), coordinator, HOMEPAGE
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                self.assertEqual(_get_json(base + "/api/health")["status"], "ok")
                forged_host = urllib.request.Request(
                    base + "/api/settings",
                    headers={"Host": f"attacker.example:{server.server_port}"},
                )
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(forged_host, timeout=3)
                self.assertEqual(rejected.exception.code, 403)

                foreign_origin = urllib.request.Request(
                    base + "/api/runs",
                    data=b'{"question":"should this run?"}',
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "http://attacker.example",
                    },
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(foreign_origin, timeout=3)
                self.assertEqual(rejected.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                coordinator.close(wait=True)
                thread.join(timeout=3)


def _get_json(url: str) -> dict:
    return json.loads(
        urllib.request.urlopen(url, timeout=3).read().decode("utf-8")
    )


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return json.loads(
            urllib.request.urlopen(request, timeout=3).read().decode("utf-8")
        )
    except urllib.error.HTTPError:
        raise
