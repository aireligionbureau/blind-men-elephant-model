from __future__ import annotations

import json
import struct
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bme_model.pipeline import PipelineIncompleteError
from bme_model.service import BMEHTTPServer, RunCoordinator
from bme_model.runtime import RunJournal, atomic_write_json


HOMEPAGE = Path(__file__).resolve().parents[1] / "homepage"


class ServiceTests(unittest.TestCase):
    def test_health_and_homepage_are_served(self) -> None:
        with TemporaryDirectory() as root:
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="mock",
                run_workers=1,
                pipeline_mode="streaming_v2",
                synthesis_mode="optimized_v2",
            )
            server = BMEHTTPServer(("127.0.0.1", 0), coordinator, HOMEPAGE)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                health = json.loads(
                    urllib.request.urlopen(base + "/api/health", timeout=3)
                    .read()
                    .decode("utf-8")
                )
                provider_health = json.loads(
                    urllib.request.urlopen(
                        base + "/api/health?probe=provider", timeout=3
                    )
                    .read()
                    .decode("utf-8")
                )
                settings = json.loads(
                    urllib.request.urlopen(base + "/api/settings", timeout=3)
                    .read()
                    .decode("utf-8")
                )
                with urllib.request.urlopen(base + "/", timeout=3) as response:
                    homepage = response.read().decode("utf-8")
                    self.assertEqual(
                        response.headers["X-Content-Type-Options"],
                        "nosniff",
                    )
                    self.assertIn(
                        "connect-src 'self'",
                        response.headers["Content-Security-Policy"],
                    )
                self.assertEqual(health["status"], "ok")
                self.assertEqual(health["execution_profile"], "candidate")
                self.assertEqual(health["pipeline_mode"], "streaming_v2")
                self.assertEqual(health["synthesis_mode"], "optimized_v2")
                self.assertEqual(health["runs_namespace"], "runs")
                self.assertTrue(provider_health["provider_ready"])
                self.assertEqual(
                    provider_health["provider_probe"],
                    {"ready": True, "reason": None},
                )
                self.assertTrue(settings["configured"])
                self.assertFalse(settings["api_key_required"])
                self.assertNotIn("api_key", settings)
                self.assertIn("盲人摸象模型", homepage)
                self.assertIn('id="setupForm"', homepage)
                self.assertIn('id="puzzlePieces"', homepage)
                self.assertIn('class="phase-track"', homepage)
                self.assertIn("blind-men-elephant-model.ico", homepage)
                self.assertIn('rel="manifest"', homepage)

                request = urllib.request.Request(
                    base + "/api/runs",
                    data=json.dumps({"question": "短"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=3)
                try:
                    self.assertEqual(raised.exception.code, 400)
                finally:
                    raised.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                coordinator.close(wait=True)
                thread.join(timeout=3)

    def test_brand_icon_assets_cover_web_and_windows(self) -> None:
        assets = HOMEPAGE / "assets"
        expected_pngs = {
            "favicon-32.png": (32, 32),
            "apple-touch-icon.png": (180, 180),
            "icon-192.png": (192, 192),
            "icon-512.png": (512, 512),
            "blind-men-elephant-model-icon.png": (1024, 1024),
        }
        for filename, expected_size in expected_pngs.items():
            payload = (assets / filename).read_bytes()
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            dimensions = struct.unpack(">II", payload[16:24])
            self.assertEqual(dimensions, expected_size)

        ico_header = (assets / "blind-men-elephant-model.ico").read_bytes()[:6]
        reserved, image_type, image_count = struct.unpack("<HHH", ico_header)
        self.assertEqual((reserved, image_type), (0, 1))
        self.assertGreaterEqual(image_count, 7)

        manifest = json.loads(
            (HOMEPAGE / "site.webmanifest").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "盲人摸象模型")
        self.assertEqual(
            {item["sizes"] for item in manifest["icons"]},
            {"192x192", "512x512"},
        )

    def test_duplicate_active_submission_reuses_persisted_job(self) -> None:
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
                first, first_deduplicated = coordinator.submit(
                    "同一个复杂问题是否会被重复计费？"
                )
                self.assertTrue(started.wait(timeout=2))
                second, second_deduplicated = coordinator.submit(
                    "同一个复杂问题是否会被重复计费？"
                )
                self.assertFalse(first_deduplicated)
                self.assertTrue(second_deduplicated)
                self.assertEqual(first["run_id"], second["run_id"])
                self.assertEqual(first["pipeline_mode"], "sequential")
                self.assertEqual(first["synthesis_mode"], "legacy_sequential")
                queued = (
                    Path(root)
                    / "runs"
                    / first["run_id"]
                    / "queued_job.json"
                )
                self.assertTrue(queued.exists())
                queued_payload = json.loads(queued.read_text(encoding="utf-8"))
                self.assertEqual(queued_payload["pipeline_mode"], "sequential")
                self.assertEqual(
                    queued_payload["synthesis_mode"], "legacy_sequential"
                )
            finally:
                release.set()
                coordinator.close(wait=True)

    def test_baseline_and_optimized_jobs_have_distinct_deduplication_keys(self) -> None:
        with TemporaryDirectory() as root:
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="mock",
                run_workers=1,
            )
            try:
                baseline = coordinator._job_key(  # noqa: SLF001 - invariant test.
                    "同一个复杂问题",
                    pipeline_mode="sequential",
                    synthesis_mode="legacy_sequential",
                )
                optimized = coordinator._job_key(  # noqa: SLF001 - invariant test.
                    "同一个复杂问题",
                    pipeline_mode="streaming_v2",
                    synthesis_mode="optimized_v2",
                )
            finally:
                coordinator.close(wait=True)
            self.assertNotEqual(baseline, optimized)

    def test_resume_before_manifest_restarts_the_durable_queued_job(self) -> None:
        started = threading.Event()

        def fake_run(*_args: object, **_kwargs: object) -> dict:
            started.set()
            return {"run_quality": {"status": "passed"}}

        with TemporaryDirectory() as root, patch(
            "bme_model.service.run_resilient_batch", side_effect=fake_run
        ) as fresh_run, patch(
            "bme_model.service.resume_resilient_batch"
        ) as checkpoint_resume:
            runs_dir = Path(root) / "runs"
            run_id = "queued-before-manifest"
            run_dir = runs_dir / run_id
            atomic_write_json(
                run_dir / "queued_job.json",
                {
                    "schema": "bme.queued-job.v1",
                    "run_id": run_id,
                    "question": "服务刚排队就重启时能否自动恢复？",
                    "provider": "mock",
                    "model": "fixture-model",
                    "profile": "standard",
                },
            )
            coordinator = RunCoordinator(
                runs_dir=runs_dir,
                provider="mock",
                model="fixture-model",
                run_workers=1,
            )
            try:
                status, deduplicated = coordinator.resume(run_id)
                self.assertFalse(deduplicated)
                self.assertEqual(status["run_id"], run_id)
                self.assertTrue(started.wait(timeout=2))
            finally:
                coordinator.close(wait=True)
            fresh_run.assert_called_once()
            checkpoint_resume.assert_not_called()

    def test_retryable_stage_failure_auto_resumes_without_user_action(self) -> None:
        def fake_run(question: str, **kwargs: object) -> dict:
            run_dir = Path(kwargs["run_dir"])
            journal = RunJournal.create(
                run_dir,
                question=question,
                config={"provider_name": "mock", "model": "fixture-model"},
                stage_names=["cohort"],
            )
            journal.start_stage("cohort")
            error = RuntimeError("temporary cohort failure")
            journal.fail_stage("cohort", error, retryable=True)
            raise PipelineIncompleteError(run_dir, "cohort", error)

        def fake_resume(run_dir: Path, **_kwargs: object) -> dict:
            journal = RunJournal.load(run_dir)
            journal.begin_resume()
            journal.start_stage("cohort")
            journal.complete_stage("cohort")
            journal.finish()
            return {"run_quality": {"status": "passed"}}

        with TemporaryDirectory() as root, patch(
            "bme_model.service.run_resilient_batch", side_effect=fake_run
        ) as fresh_run, patch(
            "bme_model.service.resume_resilient_batch", side_effect=fake_resume
        ) as checkpoint_resume:
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="mock",
                model="fixture-model",
                run_workers=1,
                auto_resume_delay_seconds=0,
            )
            submitted, _deduplicated = coordinator.submit(
                "临时生成波动是否会由系统自动续跑？"
            )
            coordinator.close(wait=True)

            status = coordinator.status(submitted["run_id"])
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["current_stage"], None)
            self.assertIsNone(status["error"])
            fresh_run.assert_called_once()
            checkpoint_resume.assert_called_once()
            self.assertFalse(
                checkpoint_resume.call_args.kwargs["renew_time_budget"]
            )

    def test_auto_resume_budget_is_scoped_per_stage(self) -> None:
        resume_calls = 0

        def fake_run(question: str, **kwargs: object) -> dict:
            run_dir = Path(kwargs["run_dir"])
            journal = RunJournal.create(
                run_dir,
                question=question,
                config={"provider_name": "mock", "model": "fixture-model"},
                stage_names=["cohort", "semantic_verdicts"],
            )
            journal.start_stage("cohort")
            error = RuntimeError("temporary cohort failure")
            journal.fail_stage("cohort", error, retryable=True)
            raise PipelineIncompleteError(run_dir, "cohort", error)

        def fake_resume(run_dir: Path, **_kwargs: object) -> dict:
            nonlocal resume_calls
            resume_calls += 1
            journal = RunJournal.load(run_dir)
            journal.begin_resume()
            if resume_calls == 1:
                journal.start_stage("cohort")
                journal.complete_stage("cohort")
                journal.start_stage("semantic_verdicts")
                error = RuntimeError("temporary semantic failure")
                journal.fail_stage("semantic_verdicts", error, retryable=True)
                raise PipelineIncompleteError(
                    run_dir, "semantic_verdicts", error
                )
            journal.start_stage("semantic_verdicts")
            journal.complete_stage("semantic_verdicts")
            journal.finish()
            return {"run_quality": {"status": "passed"}}

        with TemporaryDirectory() as root, patch(
            "bme_model.service.run_resilient_batch", side_effect=fake_run
        ), patch(
            "bme_model.service.resume_resilient_batch", side_effect=fake_resume
        ):
            coordinator = RunCoordinator(
                runs_dir=Path(root) / "runs",
                provider="mock",
                model="fixture-model",
                run_workers=1,
                auto_resume_attempts=1,
                auto_resume_delay_seconds=0,
            )
            submitted, _deduplicated = coordinator.submit(
                "不同阶段的临时失败是否各自拥有恢复机会？"
            )
            coordinator.close(wait=True)

            status = coordinator.status(submitted["run_id"])
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(resume_calls, 2)

    def test_startup_recovery_isolates_failures_between_saved_runs(self) -> None:
        visited: list[str] = []

        def fake_resume(run_dir: Path, **_kwargs: object) -> dict:
            run_id = Path(run_dir).name
            visited.append(run_id)
            if run_id == "recover-a":
                raise RuntimeError("damaged checkpoint")
            return {"run_quality": {"status": "passed"}}

        with TemporaryDirectory() as root, patch(
            "bme_model.service.resume_resilient_batch", fake_resume
        ):
            runs_dir = Path(root) / "runs"
            for run_id, question in (
                ("recover-a", "第一个中断任务能否恢复？"),
                ("recover-b", "第二个中断任务是否仍会继续？"),
            ):
                journal = RunJournal.create(
                    runs_dir / run_id,
                    question=question,
                    config={
                        "provider_name": "mock",
                        "model": "fixture-model",
                        "profile_name": "standard",
                    },
                    stage_names=["preflight"],
                )
                journal.start_stage("preflight")
                journal.fail_stage("preflight", "interrupted", retryable=True)

            coordinator = RunCoordinator(
                runs_dir=runs_dir,
                provider="mock",
                model="fixture-model",
                run_workers=1,
            )
            coordinator.recover_startup_jobs(limit=10)
            coordinator.close(wait=True)

            self.assertEqual(visited, ["recover-a", "recover-b"])
            self.assertEqual(coordinator.status("recover-a")["status"], "failed")
            self.assertEqual(coordinator.status("recover-b")["status"], "succeeded")

    def test_status_counts_reused_cohort_from_saved_artifact(self) -> None:
        with TemporaryDirectory() as root:
            runs_dir = Path(root) / "runs"
            run_dir = runs_dir / "reused-cohort"
            journal = RunJournal.create(
                run_dir,
                question="韩国股市会不会崩盘",
                config={"provider_name": "mock", "model": "fixture-model"},
                stage_names=["cohort", "retrieval"],
            )
            cohort_path = run_dir / "cohort.json"
            atomic_write_json(
                cohort_path,
                {"personas": [{"id": f"p{index:02d}"} for index in range(24)]},
            )
            journal.start_stage("cohort")
            journal.complete_stage("cohort", artifacts=[cohort_path], reused=True)
            journal.start_stage("retrieval")

            coordinator = RunCoordinator(
                runs_dir=runs_dir,
                provider="mock",
                model="fixture-model",
                run_workers=1,
            )
            try:
                status = coordinator.status("reused-cohort")
            finally:
                coordinator.close(wait=True)

            self.assertEqual(status["current_stage"], "retrieval")
            self.assertEqual(status["cohort"], {"completed": 24, "total": 24})
            self.assertEqual(
                status["analysis_layers"]["forensic"]["status"], "running"
            )

    def test_status_exposes_detective_and_contour_checkpoints(self) -> None:
        with TemporaryDirectory() as root:
            runs_dir = Path(root) / "runs"
            run_dir = runs_dir / "layer-checkpoints"
            journal = RunJournal.create(
                run_dir,
                question="复杂问题如何形成当前真相轮廓？",
                config={"provider_name": "mock", "model": "fixture-model"},
                stage_names=[
                    "diagnosis",
                    "semantic_verdicts",
                    "shadow_puzzle",
                    "relational_synthesis",
                    "finalize",
                ],
            )
            for stage in ("diagnosis", "semantic_verdicts", "shadow_puzzle"):
                journal.start_stage(stage)
                journal.complete_stage(stage)
            journal.start_stage("relational_synthesis")
            journal.set_progress(
                "relational_synthesis",
                {
                    "detective_status": "succeeded",
                    "contour_status": "running",
                    "checkpointed": True,
                },
            )

            coordinator = RunCoordinator(
                runs_dir=runs_dir,
                provider="mock",
                model="fixture-model",
                run_workers=1,
            )
            try:
                status = coordinator.status("layer-checkpoints")
            finally:
                coordinator.close(wait=True)

            self.assertEqual(
                status["analysis_layers"]["forensic"]["status"], "succeeded"
            )
            self.assertEqual(
                status["analysis_layers"]["detective"]["status"], "succeeded"
            )
            self.assertEqual(
                status["analysis_layers"]["contour"]["status"], "running"
            )
            self.assertEqual(status["synthesis"]["detective_status"], "succeeded")
            self.assertEqual(status["synthesis"]["contour_status"], "running")
            self.assertTrue(status["synthesis"]["checkpointed"])


if __name__ == "__main__":
    unittest.main()
