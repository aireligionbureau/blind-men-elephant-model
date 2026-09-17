from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.audit_pipeline_upgrade import (
    QUALITY_DIMENSIONS,
    compare_cross_domain_suite,
    compare_runs,
    create_blind_review_packet,
)


def _audit_fixture(*, optimized: bool) -> dict:
    return {
        "question": "同一个复杂问题",
        "pipeline_mode": "streaming_v2" if optimized else "sequential",
        "synthesis_mode": (
            "optimized_v2" if optimized else "legacy_sequential"
        ),
        "timing": {"submission_to_result_seconds": 599 if optimized else 900},
        "quality": {
            "expected_people": 24,
            "succeeded_people": 24,
            "retrieval_external_coverage": 1.0,
            "semantic_fallback_count": 0,
            "cognitive_chain_score": 100.0,
            "semantic_verdict_score": 100.0,
            "semantic_live_person_count": 24,
            "semantic_adjudicated_person_count": 24,
            "semantic_inference_mode": (
                "validated_fast_v1" if optimized else None
            ),
            "meta_model_score": 100.0,
            "detector_coverage": {"complete_ten_detector_people": 24},
            "detective_passed": True,
            "accepted_traceability_rate": 1.0,
            "accepted_classic_traceability_rate": 1.0,
            "forced_relation_count": 0,
            "accepted_evidence_independent_relation_count": 1,
            "accepted_directional_support_relation_count": 0,
            "independence_overclaim_count": 0,
            "classic_world_evidence_count": 0,
            "contour_classic_world_evidence_count": 0,
            "contour_provenance_coverage": 1.0,
            "evidence_binding_contract": (
                "sentence_level.v1" if optimized else "legacy_statement_level"
            ),
            "sentence_binding_count": 5 if optimized else 0,
            "sentence_binding_coverage": 1.0 if optimized else 0.0,
            "verified_observation_count": 0,
            "directional_judgment_count": 0,
            "verified_accepted_evidence_count": 0,
            "final_contour_payload_valid": True,
            "direct_answer_present": True,
            "main_contour_present": True,
            "prompt_packet_coverage": "passed" if optimized else None,
            "detective_prompt_packet_coverage": (
                "passed" if optimized else None
            ),
            "counter_mode": (
                "speculative_parallel_v1" if optimized else None
            ),
            "speculative_draft_status": "succeeded" if optimized else None,
            "speculative_drafts_are_world_evidence": (
                False if optimized else None
            ),
            "speculative_final_uses_only_adjudicated_relations": (
                True if optimized else None
            ),
        },
    }


def _passing_review() -> dict:
    return {
        "reviewer": "blind-reviewer",
        "blind_order": True,
        "dimensions": {
            name: {"not_worse": True, "reason": f"{name} passed review"}
            for name in QUALITY_DIMENSIONS
        },
    }


class PipelineUpgradeGateTests(unittest.TestCase):
    def test_machine_pass_does_not_replace_blind_semantic_review(self) -> None:
        result = compare_runs(
            _audit_fixture(optimized=False),
            _audit_fixture(optimized=True),
        )
        self.assertTrue(result["machine_passed"])
        self.assertFalse(result["activation_ready"])
        self.assertEqual(
            result["semantic_review"]["status"], "pending_or_failed"
        )

    def test_all_six_dimensions_and_review_are_required_for_activation(self) -> None:
        result = compare_runs(
            _audit_fixture(optimized=False),
            _audit_fixture(optimized=True),
            semantic_review=_passing_review(),
        )
        self.assertEqual(set(result["quality_dimensions"]), set(QUALITY_DIMENSIONS))
        self.assertTrue(result["machine_passed"])
        self.assertTrue(result["activation_ready"])

        degraded = _audit_fixture(optimized=True)
        degraded["quality"]["semantic_verdict_score"] = 99.0
        failed = compare_runs(
            _audit_fixture(optimized=False),
            degraded,
            semantic_review=_passing_review(),
        )
        self.assertFalse(failed["quality_dimensions"]["shadow_specificity"]["passed"])
        self.assertFalse(failed["activation_ready"])

    def test_blind_preferences_are_unblinded_against_candidate_label(self) -> None:
        review = {
            "reviewer": "blind-reviewer",
            "blind_order": True,
            "_candidate_label": "B",
            "dimensions": {
                name: {"preferred": "tie", "reason": "两者在本维度相当"}
                for name in QUALITY_DIMENSIONS
            },
        }
        review["dimensions"]["truth_contour_quality"] = {
            "preferred": "B",
            "reason": "B 的条件边界更清楚",
        }
        result = compare_runs(
            _audit_fixture(optimized=False),
            _audit_fixture(optimized=True),
            semantic_review=review,
        )
        self.assertTrue(result["activation_ready"])

        review["dimensions"]["shadow_specificity"] = {
            "preferred": "A",
            "reason": "A 更具体",
        }
        failed = compare_runs(
            _audit_fixture(optimized=False),
            _audit_fixture(optimized=True),
            semantic_review=review,
        )
        self.assertFalse(failed["activation_ready"])

    def test_cross_domain_suite_stays_closed_while_candidates_are_missing(self) -> None:
        with TemporaryDirectory() as root:
            manifest_path = Path(root) / "suite.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": f"case-{index}",
                                "domain": f"domain-{index}",
                                "question": f"question-{index}",
                                "baseline_run": f"baseline-{index}",
                                "candidate_run": None,
                            }
                            for index in range(3)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = compare_cross_domain_suite(manifest_path)
        self.assertTrue(result["suite_shape_valid"])
        self.assertFalse(result["activation_ready"])
        self.assertFalse(result["default_switch_allowed"])
        self.assertTrue(
            all(item["status"] == "pending_candidate" for item in result["cases"])
        )

    def test_blind_review_packet_keeps_run_identity_in_separate_key(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            runs = []
            for name, answer in (("baseline-secret", "旧答案"), ("candidate-secret", "新答案")):
                run_dir = root_path / name
                run_dir.mkdir()
                (run_dir / "run.json").write_text(
                    json.dumps({"question": "同一个复杂问题"}, ensure_ascii=False),
                    encoding="utf-8",
                )
                (run_dir / "truth_contour.json").write_text(
                    json.dumps(
                        {"direct_answer": answer, "main_contour": answer},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (run_dir / "diagnosis.json").write_text("{}", encoding="utf-8")
                (run_dir / "detective.json").write_text("{}", encoding="utf-8")
                runs.append(run_dir)
            output_dir = root_path / "review"
            create_blind_review_packet(runs[0], runs[1], output_dir)
            packet_text = (output_dir / "review-packet.json").read_text(
                encoding="utf-8"
            )
            key = json.loads(
                (output_dir / "review-key.json").read_text(encoding="utf-8")
            )
        self.assertNotIn("baseline-secret", packet_text)
        self.assertNotIn("candidate-secret", packet_text)
        self.assertIn(key["candidate_label"], {"A", "B"})
        self.assertNotEqual(key["candidate_label"], key["baseline_label"])


if __name__ == "__main__":
    unittest.main()
