from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.run_activation_candidate import candidate_preflight, load_case


class ActivationRunnerTests(unittest.TestCase):
    def test_preflight_never_runs_the_pipeline_or_exposes_the_key(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            baseline = root_path / "baseline"
            baseline.mkdir()
            (baseline / "run.json").write_text(
                json.dumps({"question": "候选链能否按要求运行？"}, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = root_path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "case-one",
                                "domain": "测试",
                                "question": "候选链能否按要求运行？",
                                "baseline_run": str(baseline),
                                "candidate_run": None,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "secret-not-for-output"},
            ), patch(
                "tools.run_activation_candidate.choose_regional_provider"
            ) as provider, patch(
                "tools.run_activation_candidate.run_resilient_batch"
            ) as live_run:
                provider.return_value = type("LiveSearch", (), {})()
                result = candidate_preflight(manifest, "case-one")
            serialized = json.dumps(result, ensure_ascii=False)
        live_run.assert_not_called()
        self.assertEqual(result["external_calls"], 0)
        self.assertTrue(result["ready"])
        self.assertNotIn("secret-not-for-output", serialized)

    def test_case_id_must_be_unique(self) -> None:
        with TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"case_id": "duplicate"},
                            {"case_id": "duplicate"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_case(manifest, "duplicate")


if __name__ == "__main__":
    unittest.main()
