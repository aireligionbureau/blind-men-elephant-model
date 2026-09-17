from __future__ import annotations

import sys
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bme_model.contour import (
    build_contour_constraints,
    build_truth_contour,
    run_live_contour,
    run_provisional_contour_drafts,
)
from bme_model.contour.schemas import validate_truth_contour
from bme_model.contour.solver import (
    _downgrade_group_quantifiers,
    _structural_payload_is_safe,
)
from bme_model.contour.live import (
    CONTOUR_PIPELINE_VERSION,
    COUNTER_PROMPT_REVISION,
    FINAL_PROMPT_REVISION,
    _cached_contour_stage_matches,
    _call_validated_stage,
    _contour_level_errors,
    _contour_stage_fingerprint,
    _validate_compact_final,
    build_counter_pressure_packets,
    build_contour_case,
    build_contour_decision_packet,
    build_contour_prompt_packet,
    _clean_user_text,
    _repair_graph_contradictions,
    _revalidate_cached_raw_stage,
    _validate_candidate,
    _validate_final,
    validate_counter_pressure_packets,
    validate_contour_decision_packet,
    validate_contour_prompt_packet,
)
from bme_model.detective import (
    build_detective_output,
    materialize_puzzle_pieces,
    run_live_detective,
)
from bme_model.detective.live import (
    _adjudicate_candidates,
    _adjudication_messages,
    _apply_classic_reviews,
    _call_grouped_proposal,
    _call_sharded_adjudication,
    _cached_proposal_payload,
    _candidate_classic_check_ids,
    _proposal_piece_selection,
    _recover_complete_relation_array,
    build_detective_prompt_packet,
    build_detective_case,
    restore_detective_prompt_packet,
    validate_detective_prompt_packet,
)
from bme_model.detective.materializer import _build_source_catalog
from bme_model.detective.relations import _attach_relation_knowledge_provenance
from bme_model.detective.schemas import (
    PUZZLE_SCHEMA_VERSION,
    question_id,
    validate_detective_output,
    validate_puzzle_materials,
    stable_id,
)


QUESTION = "复杂问题会怎样发展"


def piece(
    piece_id: str,
    person_id: str,
    kind: str,
    text: str,
    *,
    families: list[str] | None = None,
) -> dict:
    return {
        "piece_id": piece_id,
        "question_id": question_id(QUESTION),
        "person_id": person_id,
        "person_name": person_id,
        "piece_kind": kind,
        "text": text,
        "normalized_claim": "".join(text.split()).lower(),
        "content_status": "candidate_retained_not_truth",
        "coordinates": {
            "epistemic_mode": "forecast",
            "answer_direction": "conditional",
            "scope": "整体",
            "time_horizon": "未来一年",
            "definitions": [],
            "condition_markers": [],
        },
        "source_ids": [],
        "source_families": families or [],
        "anchor_ids": ["C1"],
        "shadow_operators": [],
        "diagnostic_ids": [],
        "conclusion_leverage": "high",
        "relevance_status": "not_applicable",
        "parent_conclusion": text,
        "provenance": {"origin": "test", "person_id": person_id},
    }


def materials(*pieces: dict) -> dict:
    return {
        "schema_version": PUZZLE_SCHEMA_VERSION,
        "question": QUESTION,
        "question_id": question_id(QUESTION),
        "pieces": list(pieces),
        "source_catalog": [],
        "material_counts": {},
    }


def classic_fixture(person_ids: list[str]) -> tuple[dict, list[dict]]:
    shadows = []
    attributions = []
    differentials = []
    inversions = []
    scans = []
    outputs = []
    for index, person_id in enumerate(person_ids, start=1):
        shadow_id = f"shadow_{person_id}"
        chain = {
            "detector_id": "argument_forensics",
            "observed_evidence": [
                {
                    "stage": "reasoning",
                    "fact": "从短期样本直接推到长期整体",
                    "excerpt": "短期指标变化，因此长期整体必然变化",
                }
            ],
            "plain_language_diagnosis": "短期材料越过了长期推理桥",
            "diagnostic_confidence": 0.82,
            "evidence_sufficiency": {"level": "strong", "missing": []},
        }
        shadows.append(
            {
                "id": shadow_id,
                "person_id": person_id,
                "shadow_type": "distorted",
                "description": "短期材料被外推成长期整体结论",
                "source_stage": "reasoning",
                "attribution": {
                    "classic_lenses": ["argument_forensics"],
                    "likely_causes": ["缺少时间桥"],
                    "related_parameters": ["reasoning_path"],
                },
                "estimated_bias_vector": {
                    "direction": "夸大长期确定性",
                    "strength": "high",
                    "uncertainty": "medium",
                },
                "puzzle_use": "constraint_band",
                "detector_id": "argument_forensics",
                "chain_diagnosis": chain,
            }
        )
        attributions.append(
            {
                "shadow_id": shadow_id,
                "person_id": person_id,
                "classic_lenses": ["argument_forensics"],
                "mechanism_chains": [
                    {
                        "lens_id": "argument_forensics",
                        "core_idea": "检查证据与结论之间缺失的桥",
                        "mechanism_chain": ["局部证据", "桥梁缺失", "整体结论"],
                    }
                ],
                "micro_lens_candidates": [
                    {
                        "micro_lens_id": "argument_scope_jump",
                        "parent_lens": "argument_forensics",
                        "name": "范围跳跃",
                        "mechanism": "证据范围小于结论范围",
                        "differentiates_from": ["ecological_rationality"],
                        "inversion_hint": "缩回证据实际覆盖范围",
                        "misuse_warning": "合理外推不能仅因跨尺度就判错",
                    }
                ],
                "shadow_mechanism_candidates": [],
            }
        )
        differentials.append(
            {
                "shadow_id": shadow_id,
                "person_id": person_id,
                "rules": [
                    {
                        "rule_id": "argument_vs_ecology",
                        "status": "watch_for_confusion",
                        "competing_lenses": [
                            "argument_forensics",
                            "ecological_rationality",
                        ],
                        "confusion_pattern": "外推也可能来自有效经验",
                        "decision_questions": ["原环境是否仍然相同？"],
                        "coexist_when": ["域内有效但跨域越界"],
                        "misdiagnosis_risk": "误杀局部经验",
                    }
                ],
            }
        )
        inversions.append(
            {
                "shadow_id": shadow_id,
                "person_id": person_id,
                "puzzle_use": "constraint_band",
                "inversion_rules": ["把结论缩回证据覆盖的时间范围"],
                "boundary_conditions": ["只有确实缺少时间桥时适用"],
                "anti_misuse_rules": ["不能把一切外推都判成错误"],
                "resulting_material": "范围约束",
            }
        )
        scans.append(
            {
                "person_id": person_id,
                "semantic_verdict": {
                    "validation": {"valid": True},
                    "verdicts": [
                        {
                            "issue": "短期材料被外推成长期整体结论",
                            "missing_bridge": "缺少短期变化持续到长期的机制",
                            "preserved_fragment": "短期指标确实发生变化",
                            "from_claim": "短期指标变化",
                            "to_conclusion": "长期整体必然变化",
                            "impact_on_conclusion": "结论应缩回短期",
                            "competing_explanation": "短期变化可能来自稳定机制",
                            "falsification_test": "补足连续长期证据后撤销诊断",
                            "evidence_refs": ["R1", "C1"],
                            "source_detector_ids": ["argument_forensics"],
                            "conclusion_leverage": "high",
                            "evidence_validation": {
                                "valid": True,
                                "specificity_score": 1.0,
                                "invalid_refs": [],
                            },
                        }
                    ],
                    "unresolved": [],
                },
            }
        )
        outputs.append(
            {
                "person_id": person_id,
                "person": {"id": person_id, "name": person_id},
                "output": {
                    "conclusion": f"长期整体结论{index}",
                    "answer_position": {
                        "direction": "likely_yes",
                        "scope": "整体",
                    },
                    "core_assumptions": [],
                    "value_judgements": [],
                    "evidence_ledger": [],
                    "what_i_underweighted": [],
                },
            }
        )
    diagnosis = {
        "classic_grounding": [
            {
                "lens_id": "argument_forensics",
                "source_works": ["《学会提问》"],
                "core_idea": "逐段检查论证桥",
                "mechanism_chain": ["前提", "证据", "推理桥", "结论"],
            }
        ],
        "shadow_registry": shadows,
        "bias_attribution": attributions,
        "differential_diagnosis": differentials,
        "inversion_plan": inversions,
        "cognitive_chain_scans": scans,
        "collective_blind_spots": [],
    }
    return diagnosis, outputs


class FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)

    def chat(self, *_args, **_kwargs):
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            content=json.dumps(payload, ensure_ascii=False),
            raw={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(payload, ensure_ascii=False)
                        }
                    }
                ]
            },
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class DetectiveContourTests(unittest.TestCase):
    def test_structural_uncertainty_accepts_reliable_qualification(self) -> None:
        raw = {
            "direct_answer": {"text": "当前证据下无法可靠确定未来方向。"},
            "main_contour": {"text": "方向取决于需求与融资条件。"},
        }
        errors = _contour_level_errors(
            raw,
            {
                "inference_capability": {"structural_only": True},
                "topic_anchor_groups": [],
                "question": "复杂问题会怎样发展",
            },
        )

        self.assertNotIn(
            "structural-only direct_answer must be undetermined or explicitly conditional",
            errors,
        )
        self.assertTrue(_structural_payload_is_safe(raw))

    def test_structural_uncertainty_accepts_definition_branch_wording(self) -> None:
        raw = {
            "direct_answer": {
                "text": (
                    "当前材料无法给出这个问题的确定答案。"
                    "在功能主义定义下结论偏向一种分支，在神学定义下偏向另一分支。"
                )
            },
            "main_contour": {"text": "最终方向仍取决于定义和前提。"},
        }
        errors = _contour_level_errors(
            raw,
            {
                "inference_capability": {"structural_only": True},
                "topic_anchor_groups": [],
                "question": "AI会不会像人一样产生自己的宗教",
            },
        )

        self.assertNotIn(
            "structural-only direct_answer must be undetermined or explicitly conditional",
            errors,
        )
        self.assertNotIn(
            "structural-only direct_answer makes an unsupported yes/no claim",
            errors,
        )

    def test_structural_uncertainty_accepts_no_support_for_either_direction(self) -> None:
        raw = {
            "direct_answer": {
                "text": (
                    "当前证据无法支持AI会产生自己的宗教，或AI不会产生任何形式宗教，"
                    "这两个方向性结论。"
                )
            },
            "main_contour": {"text": "方向取决于宗教定义和未经检验的前提。"},
        }
        errors = _contour_level_errors(
            raw,
            {
                "inference_capability": {"structural_only": True},
                "topic_anchor_groups": [],
                "question": "AI会不会像人一样产生自己的宗教",
            },
        )

        self.assertNotIn(
            "structural-only direct_answer must be undetermined or explicitly conditional",
            errors,
        )
        self.assertNotIn(
            "structural-only direct_answer makes an unsupported yes/no claim",
            errors,
        )

    def test_failed_raw_contour_can_be_revalidated_before_another_call(self) -> None:
        fingerprint = "same-input"
        cached = {
            "pipeline_version": CONTOUR_PIPELINE_VERSION,
            "audit": {
                "stages": [
                    {
                        "stage": "final_from_provisional_drafts",
                        "status": "failed",
                        "input_fingerprint": fingerprint,
                        "errors": ["old validator rejected an equivalent phrase"],
                        "raw": {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {"answer": "无法可靠确定"},
                                            ensure_ascii=False,
                                        )
                                    }
                                }
                            ]
                        },
                    }
                ]
            },
        }

        payload, stage = _revalidate_cached_raw_stage(
            cached,
            "final_from_provisional_drafts",
            fingerprint,
            lambda value: (value, []) if value.get("answer") else ({}, ["missing"]),
        )

        self.assertEqual(payload["answer"], "无法可靠确定")
        self.assertIsNotNone(stage)
        self.assertEqual(stage["status"], "reused")
        self.assertEqual(stage["cache_source"], "revalidated_previous_raw_response")

    def test_failed_proposal_checkpoint_cannot_be_reused_as_complete(self) -> None:
        fingerprint = "same-proposal-input"
        cached = {
            "pipeline_version": "bme.detective-live.v13",
            "candidate_relations": [
                {
                    "proposal_mode": "content",
                    "relation_type": "conditional_complement",
                    "piece_ids": ["a", "b"],
                    "plain_language_explanation": "只有一个成功分片。",
                }
            ],
            "audit": {
                "stages": [
                    {
                        "stage": "content_proposal",
                        "status": "failed",
                        "input_fingerprint": fingerprint,
                        "raw": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "relations": [
                                                    {
                                                        "relation_type": "conditional_complement",
                                                        "piece_ids": ["a", "b"],
                                                    }
                                                ]
                                            }
                                        )
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        }

        self.assertEqual(
            _cached_proposal_payload(
                cached,
                "content_proposal",
                "content",
                input_fingerprint=fingerprint,
            ),
            {},
        )

    def test_detective_modes_change_scheduling_not_relation_semantics(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "需求上升时系统更可能扩张"),
            piece("b", "p2", "retained_local", "融资收紧时系统更可能收缩"),
        )

        class ModeClient:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0

            def chat(self, messages, **_kwargs):
                content = messages[-1]["content"]
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.04)
                with self.lock:
                    self.active -= 1
                if "逐条裁决下列候选" in content:
                    case_json = content.split("CASE_JSON:\n", 1)[1].split(
                        "\nOUTPUT_SCHEMA:\n", 1
                    )[0]
                    case_payload = json.loads(case_json)
                    if case_payload.get("schema") == "bme.detective-prompt-packet.v1":
                        case_payload = restore_detective_prompt_packet(
                            case_payload
                        )
                    candidates = case_payload["candidates"]
                    payload = {
                        "decisions": [
                            {
                                "candidate_id": item["candidate_id"],
                                "decision": "accepted",
                                "plain_language_explanation": "两块材料说明不同条件会推动相反方向。",
                                "competing_explanation": "两项条件也可能同时出现。",
                                "falsification_test": "若对象或时间范围不同就撤销关系。",
                                "source_independence": "unknown",
                                "common_error_risk": "low",
                                "confidence_profile": {
                                    "semantic_alignment": "high",
                                    "scope_compatibility": "high",
                                    "source_independence": "unknown",
                                    "diagnostic_certainty": "high",
                                },
                            }
                            for item in candidates
                        ]
                    }
                elif "只寻找内容关系" in content:
                    payload = {
                        "relations": [
                            {
                                "relation_type": "conditional_complement",
                                "piece_ids": ["a", "b"],
                                "plain_language_explanation": "两块材料描述不同条件下的相反方向。",
                                "shared_coordinate": {
                                    "same_proposition": "系统扩张还是收缩",
                                    "definition": "兼容",
                                    "scope": "兼容",
                                    "time": "兼容",
                                    "condition": "需求或融资成为主导约束",
                                },
                                "source_independence": "unknown",
                                "common_error_risk": "low",
                                "competing_explanation": "两项条件可能同时出现。",
                                "falsification_test": "对象不同则撤销。",
                            }
                        ]
                    }
                else:
                    payload = {"relations": []}
                encoded = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=encoded,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": encoded},
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )

        legacy_client = ModeClient()
        legacy, legacy_record = run_live_detective(
            QUESTION,
            data,
            model="fixture-model",
            retries=0,
            client_factory=lambda: legacy_client,
            execution_mode="legacy_sequential",
        )
        optimized_client = ModeClient()
        optimized, optimized_record = run_live_detective(
            QUESTION,
            data,
            model="fixture-model",
            retries=0,
            client_factory=lambda: optimized_client,
            execution_mode="optimized_v2",
        )
        self.assertEqual(legacy_client.peak, 1)
        self.assertGreaterEqual(optimized_client.peak, 2)
        self.assertEqual(legacy_record["status"], "succeeded")
        self.assertEqual(optimized_record["status"], "succeeded")
        self.assertEqual(
            [item["relation_id"] for item in legacy["relation_certificates"]],
            [item["relation_id"] for item in optimized["relation_certificates"]],
        )

    def test_adjudication_shards_run_concurrently_and_merge_in_order(self) -> None:
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        class ConcurrentClient:
            def chat(self, messages, **_kwargs):
                user_content = messages[-1]["content"]
                case_json = user_content.split("CASE_JSON:\n", 1)[1].split(
                    "\nOUTPUT_SCHEMA:\n", 1
                )[0]
                shard = json.loads(case_json)["candidates"]
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with lock:
                    state["active"] -= 1
                payload = {
                    "decisions": [
                        {
                            "candidate_id": item["candidate_id"],
                            "decision": "unresolved",
                            "plain_language_explanation": "材料仍不足以完成关系证明。",
                            "competing_explanation": "两块材料可能只是表面相似。",
                            "falsification_test": "补充同一命题与来源谱系后重审。",
                            "source_independence": "unknown",
                            "common_error_risk": "unknown",
                            "confidence_profile": {
                                "semantic_alignment": "unknown",
                                "scope_compatibility": "unknown",
                                "source_independence": "unknown",
                                "diagnostic_certainty": "unknown",
                            },
                        }
                        for item in shard
                    ]
                }
                content = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=content,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )

        candidates = [
            {
                "candidate_id": f"candidate-{index:02d}",
                "relation_type": "conditional_complement",
                "proposal_mode": "content",
                "piece_ids": ["a", "b"],
                "plain_language_explanation": "两块材料描述不同条件。",
            }
            for index in range(13)
        ]
        case = {
            "question": QUESTION,
            "piece_index": [
                {"piece_id": "a", "person_id": "p1", "text": "材料甲"},
                {"piece_id": "b", "person_id": "p2", "text": "材料乙"},
            ],
            "source_dependency_clusters": [],
            "source_lineage_rules": {},
            "mechanism_index": {},
        }
        payload, record = _call_sharded_adjudication(
            lambda: ConcurrentClient(),
            case,
            candidates,
            max_tokens=8192,
            retries=0,
        )
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["shard_count"], 5)
        self.assertGreaterEqual(state["peak"], 2)
        self.assertEqual(
            [item["candidate_id"] for item in payload["decisions"]],
            [item["candidate_id"] for item in candidates],
        )

    def test_adjudication_recovers_only_failed_shard_candidates(self) -> None:
        calls: list[tuple[str, ...]] = []
        failed_multi = False

        class TargetedRecoveryClient:
            def chat(self, messages, **_kwargs):
                nonlocal failed_multi
                user_content = messages[-1]["content"]
                case_json = user_content.split("CASE_JSON:\n", 1)[1].split(
                    "\nOUTPUT_SCHEMA:\n", 1
                )[0]
                shard = json.loads(case_json)["candidates"]
                candidate_ids = tuple(item["candidate_id"] for item in shard)
                calls.append(candidate_ids)
                if "candidate-03" in candidate_ids and len(candidate_ids) > 1:
                    failed_multi = True
                    content = json.dumps({"decisions": []})
                else:
                    content = json.dumps(
                        {
                            "decisions": [
                                {
                                    "candidate_id": item["candidate_id"],
                                    "decision": "unresolved",
                                }
                                for item in shard
                            ]
                        }
                    )
                return SimpleNamespace(
                    content=content,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            }
                        ]
                    },
                    usage={"total_tokens": 1},
                )

        candidates = [
            {
                "candidate_id": f"candidate-{index:02d}",
                "relation_type": "conditional_complement",
                "piece_ids": ["a", "b"],
                "plain_language_explanation": "两块材料描述不同条件。",
            }
            for index in range(7)
        ]
        case = {
            "question": QUESTION,
            "piece_index": [
                {"piece_id": "a", "person_id": "p1", "text": "材料甲"},
                {"piece_id": "b", "person_id": "p2", "text": "材料乙"},
            ],
            "source_dependency_clusters": [],
            "source_lineage_rules": {},
            "mechanism_index": {},
        }
        payload, record = _call_sharded_adjudication(
            lambda: TargetedRecoveryClient(),
            case,
            candidates,
            max_tokens=8192,
            retries=1,
        )

        self.assertTrue(failed_multi)
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["targeted_recovery_shard_count"], 1)
        self.assertEqual(
            record["targeted_recovery_attempted_shard_count"], 1
        )
        self.assertEqual(record["transport_unresolved_candidate_ids"], [])
        self.assertEqual(
            [item["candidate_id"] for item in payload["decisions"]],
            [item["candidate_id"] for item in candidates],
        )
        multi_calls = [item for item in calls if len(item) > 1]
        self.assertEqual(len(multi_calls), len(set(multi_calls)))
        self.assertTrue(any(item == ("candidate-03",) for item in calls))

    def test_adjudication_transport_gap_stays_unresolved(self) -> None:
        class PartiallyBrokenClient:
            def chat(self, messages, **_kwargs):
                user_content = messages[-1]["content"]
                case_json = user_content.split("CASE_JSON:\n", 1)[1].split(
                    "\nOUTPUT_SCHEMA:\n", 1
                )[0]
                shard = json.loads(case_json)["candidates"]
                if any(
                    item["candidate_id"] == "candidate-00" for item in shard
                ):
                    content = json.dumps({"decisions": []})
                else:
                    content = json.dumps(
                        {
                            "decisions": [
                                {
                                    "candidate_id": item["candidate_id"],
                                    "decision": "accepted",
                                }
                                for item in shard
                            ]
                        }
                    )
                return SimpleNamespace(
                    content=content,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            }
                        ]
                    },
                    usage={"total_tokens": 1},
                )

        candidates = [
            {
                "candidate_id": f"candidate-{index:02d}",
                "relation_type": "conditional_complement",
                "proposal_mode": "content",
                "piece_ids": ["a", "b"],
                "plain_language_explanation": "两块材料描述不同条件。",
            }
            for index in range(4)
        ]
        case = {
            "question": QUESTION,
            "piece_index": [
                {"piece_id": "a", "person_id": "p1", "text": "材料甲"},
                {"piece_id": "b", "person_id": "p2", "text": "材料乙"},
            ],
            "source_dependency_clusters": [],
            "source_lineage_rules": {},
            "mechanism_index": {},
        }
        payload, record = _call_sharded_adjudication(
            lambda: PartiallyBrokenClient(),
            case,
            candidates,
            max_tokens=8192,
            retries=1,
        )
        decisions = {
            item["candidate_id"]: item for item in payload["decisions"]
        }
        relations = _adjudicate_candidates(payload, candidates)
        relation_by_candidate = {
            (item.get("provenance") or {}).get("candidate_id"): item
            for item in relations
        }

        self.assertEqual(
            record["transport_unresolved_candidate_ids"], ["candidate-00"]
        )
        self.assertEqual(decisions["candidate-00"]["decision"], "unresolved")
        self.assertTrue(decisions["candidate-00"]["transport_fallback"])
        self.assertEqual(
            relation_by_candidate["candidate-00"]["status"], "unresolved"
        )
        self.assertTrue(
            relation_by_candidate["candidate-00"]["provenance"][
                "adjudication_transport_fallback"
            ]
        )

    def test_validated_stage_repairs_prior_output_without_full_regeneration(self) -> None:
        calls: list[dict] = []
        repair_inputs: list[tuple[dict, list[str]]] = []

        class RepairClient:
            def chat(self, messages, **kwargs):
                calls.append({"messages": messages, **kwargs})
                payload = {"valid": len(calls) == 2}
                content = json.dumps(payload)
                return SimpleNamespace(
                    content=content,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            }
                        ]
                    },
                    usage={"total_tokens": 1},
                )

        def repair_factory(payload, errors):
            repair_inputs.append((payload, errors))
            return [
                {"role": "system", "content": "repair"},
                {"role": "user", "content": "REPAIR_PACKET"},
            ]

        payload, record = _call_validated_stage(
            lambda: RepairClient(),
            [{"role": "user", "content": "FULL_CONTEXT"}],
            stage="final_from_provisional_drafts",
            max_tokens=12000,
            validator=lambda value: (
                (value, [])
                if value.get("valid")
                else ({}, ["missing concrete anchor"])
            ),
            retries=1,
            thinking_mode="disabled",
            repair_messages_factory=repair_factory,
            repair_max_tokens=8192,
            repair_thinking_mode="enabled",
        )

        self.assertEqual(payload, {"valid": True})
        self.assertEqual(record["status"], "succeeded")
        self.assertTrue(record["repair_attempted"])
        self.assertEqual(
            record["attempt_modes"],
            ["validated_fast_generation", "targeted_repair"],
        )
        self.assertEqual(repair_inputs[0][0], {"valid": False})
        self.assertEqual(calls[1]["messages"][-1]["content"], "REPAIR_PACKET")
        self.assertEqual(calls[1]["max_tokens"], 8192)
        self.assertEqual(calls[1]["thinking"], {"type": "enabled"})

    def test_contour_does_not_deny_an_accepted_independent_convergence(self) -> None:
        repaired = _repair_graph_contradictions(
            {
                "direct_answer": {"text": "尚不能确定。"},
                "main_contour": {"text": "整体缺乏独立会合或决定性证据。"},
            },
            {
                "accepted_relations": [
                    {
                        "status": "accepted",
                        "relation_type": "independent_convergence",
                    }
                ]
            },
        )
        self.assertIn(
            "现有独立会合不足以决定方向",
            repaired["main_contour"]["text"],
        )
        self.assertNotIn(
            "缺乏独立会合", repaired["main_contour"]["text"]
        )

    def test_detective_prompt_packet_is_reversible_and_tamper_evident(self) -> None:
        repeated = [
            piece(
                f"piece-{index:02d}",
                f"person-{index % 3}",
                "retained_local",
                f"同一结构下的局部材料 {index}",
            )
            for index in range(24)
        ]
        payload = {
            "question": QUESTION,
            "pieces": repeated,
            "source_dependency_clusters": [],
            "source_lineage_rules": {"same_family_is_dependent": True},
        }
        packet = build_detective_prompt_packet(payload)
        self.assertEqual(restore_detective_prompt_packet(packet), payload)
        self.assertEqual(validate_detective_prompt_packet(packet, payload), [])
        self.assertLess(
            len(json.dumps(packet, ensure_ascii=False)),
            len(json.dumps(payload, ensure_ascii=False)),
        )

        broken = json.loads(json.dumps(packet, ensure_ascii=False))
        coordinate_ref = next(
            iter(broken["reference_indexes"]["coordinates"])
        )
        broken["reference_indexes"]["coordinates"].pop(coordinate_ref)
        self.assertTrue(validate_detective_prompt_packet(broken, payload))

    def test_contour_cache_fingerprints_follow_stage_dependencies(self) -> None:
        case = {"question": QUESTION, "relations": ["r1"]}
        candidate = {"main_contour": {"text": "候选甲"}}
        counter = {"counter_contour": {"text": "反方甲"}}
        counter_fingerprint = _contour_stage_fingerprint(
            COUNTER_PROMPT_REVISION,
            {"case": case, "candidate": candidate},
        )
        final_fingerprint = _contour_stage_fingerprint(
            FINAL_PROMPT_REVISION,
            {"case": case, "candidate": candidate, "counter": counter},
        )
        cached = {
            "pipeline_version": CONTOUR_PIPELINE_VERSION,
            "audit": {
                "stages": [
                    {
                        "stage": "strongest_counter_contour",
                        "input_fingerprint": counter_fingerprint,
                    },
                    {
                        "stage": "final_adjudication",
                        "input_fingerprint": final_fingerprint,
                    },
                ]
            },
        }
        self.assertTrue(
            _cached_contour_stage_matches(
                cached,
                "strongest_counter_contour",
                counter_fingerprint,
            )
        )
        changed_candidate_fingerprint = _contour_stage_fingerprint(
            COUNTER_PROMPT_REVISION,
            {
                "case": case,
                "candidate": {"main_contour": {"text": "候选乙"}},
            },
        )
        self.assertFalse(
            _cached_contour_stage_matches(
                cached,
                "strongest_counter_contour",
                changed_candidate_fingerprint,
            )
        )
        changed_counter_fingerprint = _contour_stage_fingerprint(
            FINAL_PROMPT_REVISION,
            {
                "case": case,
                "candidate": candidate,
                "counter": {"counter_contour": {"text": "反方乙"}},
            },
        )
        self.assertFalse(
            _cached_contour_stage_matches(
                cached,
                "final_adjudication",
                changed_counter_fingerprint,
            )
        )

    def test_counter_pressure_packets_are_collectively_lossless(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求上升推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        relation_input = {
            "relation_type": "conditional_complement",
            "piece_ids": ["a", "b"],
            "status": "accepted",
            "plain_language_explanation": "需求上升与融资收紧会在不同条件下推动相反方向。",
            "source_independence": "unknown",
            "common_error_risk": "low",
            "competing_explanation": "两种条件也可能同时出现。",
            "falsification_test": "若对象或时间不同就撤销关系。",
        }
        detective = build_detective_output(
            QUESTION, data, semantic_relations=[relation_input]
        )
        constraints = build_contour_constraints(QUESTION, detective)
        case = build_contour_case(QUESTION, detective, constraints)
        packet = build_contour_prompt_packet(case)
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )

        def cited(text: str) -> dict:
            return {
                "text": text,
                "piece_ids": ["a", "b"],
                "relation_ids": [relation["relation_id"]],
                "constraint_ids": [constraint["constraint_id"]],
            }

        candidate = {
            "direct_answer": cited("方向取决于需求扩张与融资收紧谁占主导。"),
            "main_contour": cited("需求推动扩张，融资收紧推动收缩，二者的条件强弱决定方向。"),
            "key_conditions": [],
            "stable_parts": [],
            "boundary_conditions": [],
            "important_unknowns": [],
        }
        pressure_packets = build_counter_pressure_packets(packet, candidate)
        self.assertEqual(
            validate_counter_pressure_packets(
                pressure_packets, packet, candidate
            ),
            [],
        )
        self.assertEqual(len(pressure_packets), 3)
        broken = json.loads(json.dumps(pressure_packets, ensure_ascii=False))
        broken[0]["material"]["relation_index"].pop(
            relation["relation_id"]
        )
        self.assertTrue(
            validate_counter_pressure_packets(broken, packet, candidate)
        )

    def test_final_decision_packet_is_lossless_and_hydrates_provenance(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求扩张推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        detective = build_detective_output(
            QUESTION,
            data,
            semantic_relations=[
                {
                    "relation_type": "conditional_complement",
                    "piece_ids": ["a", "b"],
                    "status": "accepted",
                    "plain_language_explanation": "需求扩张与融资收紧会在不同条件下推动相反方向。",
                    "source_independence": "unknown",
                    "common_error_risk": "low",
                    "competing_explanation": "两种条件也可能同时出现。",
                    "falsification_test": "若对象或时间不同就撤销关系。",
                }
            ],
        )
        constraints = build_contour_constraints(QUESTION, detective)
        case = build_contour_case(QUESTION, detective, constraints)
        packet = build_contour_prompt_packet(case)
        decision_packet = build_contour_decision_packet(packet, case)
        self.assertEqual(
            validate_contour_decision_packet(decision_packet, packet, case),
            [],
        )
        broken = json.loads(json.dumps(decision_packet, ensure_ascii=False))
        broken["piece_table"]["rows"][0][4] = "被篡改的材料"
        self.assertTrue(
            validate_contour_decision_packet(broken, packet, case)
        )

        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )

        def compact_statement(text: str) -> dict:
            return {
                "text": text,
                "relation_ids": [relation["relation_id"]],
                "constraint_ids": [constraint["constraint_id"]],
                "extra_piece_ids": [],
            }

        payload = {
            "direct_answer": compact_statement(
                "走向取决于需求扩张和融资收紧谁先主导。"
            ),
            "main_contour": compact_statement(
                "需求扩张推动扩张，融资收紧推动收缩，条件强弱决定方向。"
            ),
            "key_conditions": [],
            "stable_parts": [],
            "boundary_conditions": [],
            "important_unknowns": [],
            "confidence_statement": "当前只能确认条件结构。",
            "why_this_contour": "它同时解释了两条方向相反的材料。",
            "strongest_counter_contour": "两种力量可能都不是主要驱动。",
            "counter_contour_disposition": "保留为待验证的替代解释。",
        }
        validated, errors = _validate_compact_final(payload, case)
        self.assertEqual(errors, [])
        self.assertEqual(
            set(validated["direct_answer"]["piece_ids"]), {"a", "b"}
        )
        self.assertEqual(
            set(validated["main_contour"]["piece_ids"]), {"a", "b"}
        )

    def test_sentence_binding_drops_overreach_and_keeps_readable_conditional_claim(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求扩张推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        detective = build_detective_output(
            QUESTION,
            data,
            semantic_relations=[
                {
                    "relation_type": "conditional_complement",
                    "piece_ids": ["a", "b"],
                    "status": "accepted",
                    "plain_language_explanation": "两种力量在不同条件下推动相反方向。",
                    "source_independence": "unknown",
                    "common_error_risk": "low",
                    "competing_explanation": "两种条件也可能同时出现。",
                    "falsification_test": "若对象或时间不同就撤销关系。",
                }
            ],
        )
        constraints = build_contour_constraints(QUESTION, detective)
        case = build_contour_case(QUESTION, detective, constraints)
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )

        def sentence(text: str, role: str) -> dict:
            return {
                "text": text,
                "claim_role": role,
                "relation_ids": [relation["relation_id"]],
                "constraint_ids": [constraint["constraint_id"]],
                "extra_piece_ids": [],
            }

        payload = {
            "direct_answer": {
                "sentences": [
                    sentence("未来一年很可能持续扩张。", "directional_judgment"),
                    sentence(
                        "如果需求扩张压过融资收紧，系统才会继续扩张。",
                        "conditional_inference",
                    ),
                ]
            },
            "main_contour": {
                "sentences": [
                    sentence(
                        "如果融资收紧先占主导，当前扩张方向就会被改写。",
                        "conditional_inference",
                    )
                ]
            },
            "key_conditions": [],
            "stable_parts": [],
            "boundary_conditions": [],
            "important_unknowns": [],
            "confidence_statement": "当前只能确认条件结构。",
            "why_this_contour": "它同时保留两条相反机制。",
            "strongest_counter_contour": "两种力量都不是主要驱动。",
            "counter_contour_disposition": "保留为待验证解释。",
        }
        validated, errors = _validate_compact_final(payload, case)
        self.assertEqual(errors, [])
        self.assertEqual(
            validated["direct_answer"]["text"],
            "如果需求扩张压过融资收紧，系统才会继续扩张。",
        )
        self.assertEqual(
            len(validated["direct_answer"]["sentence_validation_drops"]), 1
        )
        self.assertEqual(
            validated["direct_answer"]["sentence_bindings"][0]["claim_role"],
            "conditional_inference",
        )
        contour = build_truth_contour(
            QUESTION,
            detective,
            constraints,
            adjudicated_payload=validated,
        )
        self.assertEqual(
            contour["evidence_binding_contract"], "sentence_level.v1"
        )
        self.assertEqual(
            contour["contour_evaluation"]["sentence_binding_coverage"], 1.0
        )
        self.assertEqual(
            validate_truth_contour(
                contour,
                question=QUESTION,
                detective=detective,
                constraints=constraints,
            ),
            [],
        )
        malformed = json.loads(json.dumps(contour, ensure_ascii=False))
        malformed["contour_evaluation"]["sentence_binding_coverage"] = "not-a-number"
        malformed_errors = validate_truth_contour(
            malformed,
            question=QUESTION,
            detective=detective,
            constraints=constraints,
        )
        self.assertIn(
            "sentence-level evidence binding coverage is not numeric",
            malformed_errors,
        )

    def test_grouped_proposals_cover_relation_families_concurrently(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧"),
        )
        case = build_detective_case(QUESTION, data)
        lock = threading.Lock()
        active = 0
        peak = 0

        def fake_stage(*_args, **kwargs):
            nonlocal active, peak
            stage = kwargs["stage"]
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            relation_type = (
                "shared_assumption"
                if "frames_and_values" in stage
                else "conditional_complement"
            )
            payload = {
                "relations": [
                    {
                        "relation_type": relation_type,
                        "piece_ids": ["a", "b"],
                        "plain_language_explanation": "两个人的材料在同一问题上形成可检查的连接。",
                        "discovery_route": "native",
                        "connection_hypothesis_ids": [],
                    }
                ]
            }
            return payload, {
                "stage": stage,
                "status": "succeeded",
                "attempts": 1,
                "duration_seconds": 0.05,
                "errors": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "retryable": True,
            }

        started = time.perf_counter()
        with patch(
            "bme_model.detective.live._call_json_stage",
            side_effect=fake_stage,
        ):
            payload, record = _call_grouped_proposal(
                lambda: None,
                case,
                "content",
                {
                    "independent_convergence",
                    "direct_conflict",
                    "apparent_conflict",
                    "conditional_complement",
                    "scope_refinement",
                    "causal_relay",
                    "shared_assumption",
                    "definition_branch",
                    "value_branch",
                    "blind_spot_fill",
                    "non_comparable",
                },
                max_tokens=8192,
                retries=0,
                material_mode="lossless_packet_v2",
                parent_input_fingerprint="parent",
            )
        elapsed = time.perf_counter() - started

        self.assertEqual(peak, 2)
        self.assertLess(elapsed, 0.12)
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["shard_count"], 2)
        self.assertEqual(len(payload["relations"]), 2)
        self.assertEqual(
            record["prompt_packet"]["capacity_preserved"],
            {"native": 12, "classic_assisted": 4},
        )

    def test_contour_rejects_named_person_without_cited_piece(self) -> None:
        first = piece("a", "run_p01", "retained_local", "需求推动扩张")
        second = piece("b", "run_p02", "retained_local", "融资推动收缩")
        third = piece("c", "run_p03", "retained_local", "监管改变边界")
        first["person_name"] = "甲方"
        second["person_name"] = "乙方"
        third["person_name"] = "丙方"
        data = materials(first, second, third)
        detective = build_detective_output(
            QUESTION,
            data,
            semantic_relations=[
                {
                    "relation_type": "conditional_complement",
                    "piece_ids": ["a", "b"],
                    "status": "accepted",
                    "plain_language_explanation": "需求与融资在不同条件下推动相反方向。",
                    "source_independence": "unknown",
                    "common_error_risk": "low",
                    "competing_explanation": "也可能不是同一时间窗口。",
                    "falsification_test": "若时间不同则撤销。",
                },
                {
                    "relation_type": "scope_refinement",
                    "piece_ids": ["a", "c"],
                    "status": "accepted",
                    "plain_language_explanation": "监管材料把需求判断限制在特定边界内。",
                    "source_independence": "unknown",
                    "common_error_risk": "low",
                    "competing_explanation": "监管也可能尚未生效。",
                    "falsification_test": "若监管不适用则撤销。",
                },
            ],
        )
        constraints = build_contour_constraints(QUESTION, detective)
        case = build_contour_case(QUESTION, detective, constraints)
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )
        cited = {
            "piece_ids": ["a", "b"],
            "relation_ids": [relation["relation_id"]],
            "constraint_ids": [constraint["constraint_id"]],
        }
        payload = {
            "direct_answer": {
                "text": "甲方与丙方说明需求和融资谁占主导仍有条件。",
                **cited,
            },
            "main_contour": {
                "text": "需求推动扩张、融资推动收缩，条件强弱决定方向。",
                **cited,
            },
        }
        _, errors = _validate_candidate(payload, case)
        self.assertTrue(
            any("names people absent from cited pieces" in item for item in errors)
        )

    def test_parallel_counter_pressures_run_concurrently(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求上升推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        detective = build_detective_output(
            QUESTION,
            data,
            semantic_relations=[
                {
                    "relation_type": "conditional_complement",
                    "piece_ids": ["a", "b"],
                    "status": "accepted",
                    "plain_language_explanation": "需求上升与融资收紧在不同条件下推动相反方向。",
                    "source_independence": "unknown",
                    "common_error_risk": "low",
                    "competing_explanation": "两种条件也可能同时出现。",
                    "falsification_test": "若对象或时间不同就撤销关系。",
                }
            ],
        )
        constraints = build_contour_constraints(QUESTION, detective)
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )

        def cited(text: str) -> dict:
            return {
                "text": text,
                "piece_ids": ["a", "b"],
                "relation_ids": [relation["relation_id"]],
                "constraint_ids": [constraint["constraint_id"]],
            }

        candidate = {
            "direct_answer": cited("方向取决于需求扩张与融资收紧谁占主导。"),
            "main_contour": cited("需求推动扩张，融资收紧推动收缩，条件强弱决定方向。"),
            "key_conditions": [],
            "stable_parts": [],
            "boundary_conditions": [],
            "important_unknowns": [],
            "confidence_statement": "当前只对条件结构有信心。",
            "why_this_contour": "它保留了两种相反条件。",
            "strongest_counter_contour": "",
            "counter_contour_disposition": "",
        }
        counter = {
            "counter_contour": cited("需求与融资也可能只是共同因素的伴随现象。"),
            "why_it_may_fit_better": cited("现有材料没有锁定因果方向。"),
            "attacks": [cited("候选把条件关系写得太像因果关系。")],
            "decisive_tests": [cited("需要区分条件相关与因果驱动。")],
        }
        final = {
            **candidate,
            "strongest_counter_contour": "需求与融资可能只是共同因素的伴随现象。",
            "counter_contour_disposition": "保留为重要未知，因为因果方向仍未锁定。",
        }
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        class ParallelContourClient:
            def chat(self, messages, **_kwargs):
                content = messages[-1]["content"]
                if "COUNTER_PRESSURE_BUNDLE" in content:
                    payload = final
                elif "本轮专门负责" in content:
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    time.sleep(0.04)
                    with lock:
                        state["active"] -= 1
                    payload = counter
                else:
                    payload = candidate
                encoded = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=encoded,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": encoded},
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )

        contour, record = run_live_contour(
            QUESTION,
            detective,
            constraints,
            model="fixture-model",
            retries=0,
            client_factory=lambda: ParallelContourClient(),
            material_mode="lossless_packet_v2",
            counter_mode="parallel_pressure_v1",
        )
        self.assertEqual(record["status"], "succeeded")
        self.assertGreaterEqual(state["peak"], 2)
        self.assertEqual(
            len(record["counter_contour"]["pressure_tests"]), 3
        )
        pressure_stage = next(
            item
            for item in record["audit"]["stages"]
            if item["stage"] == "parallel_counter_pressure"
        )
        self.assertEqual(pressure_stage["pressure_stage_count"], 3)
        self.assertIn("需求", contour["direct_answer"])

    def test_speculative_drafts_overlap_and_final_uses_only_accepted_case(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求上升推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        detective_case = build_detective_case(QUESTION, data)
        candidate_id = "candidate-speculative-01"
        candidates = [
            {
                "candidate_id": candidate_id,
                "relation_type": "conditional_complement",
                "piece_ids": ["a", "b"],
                "plain_language_explanation": "需求与融资在不同条件下推动相反方向。",
                "shared_coordinate": {"same_proposition": "系统扩张还是收缩"},
                "source_independence": "unknown",
                "common_error_risk": "low",
                "competing_explanation": "两种条件可能同时出现。",
                "falsification_test": "若对象不同则撤销。",
            }
        ]
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        class DraftClient:
            def chat(self, messages, **_kwargs):
                content = messages[-1]["content"]
                case_payload = json.loads(
                    content.split("CASE_JSON:\n", 1)[1].split(
                        "\nOUTPUT_SCHEMA:\n", 1
                    )[0]
                )
                selected = case_payload["candidates"][0]
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.04)
                with lock:
                    state["active"] -= 1
                payload = {
                    "draft_id": "draft",
                    "direct_answer_hypothesis": "方向取决于需求与融资谁占主导。",
                    "contour_hypothesis": "需求推动扩张，融资收紧推动收缩。",
                    "candidate_ids": [selected["candidate_id"]],
                    "piece_ids": selected["piece_ids"],
                    "critical_assumptions": ["两块材料讨论同一系统"],
                    "failure_conditions": ["两块材料的对象不同"],
                }
                encoded = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=encoded,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": encoded},
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )

        drafts, draft_record = run_provisional_contour_drafts(
            QUESTION,
            detective_case,
            candidates,
            model="fixture-model",
            retries=0,
            client_factory=lambda: DraftClient(),
        )
        self.assertEqual(draft_record["status"], "succeeded")
        self.assertEqual(len(drafts["rival_drafts"]), 3)
        self.assertGreaterEqual(state["peak"], 2)

        detective = build_detective_output(
            QUESTION,
            data,
            semantic_relations=[
                {
                    **candidates[0],
                    "status": "accepted",
                    "provenance": {
                        "mode": "live_adversarial_adjudication",
                        "candidate_id": candidate_id,
                    },
                }
            ],
        )
        constraints = build_contour_constraints(QUESTION, detective)
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )

        def cited(text: str) -> dict:
            return {
                "text": text,
                "piece_ids": ["a", "b"],
                "relation_ids": [relation["relation_id"]],
                "constraint_ids": [constraint["constraint_id"]],
            }

        final = {
            "direct_answer": cited("方向取决于需求扩张与融资收紧谁占主导。"),
            "main_contour": cited("需求推动扩张，融资收紧推动收缩，条件强弱决定方向。"),
            "key_conditions": [],
            "stable_parts": [],
            "boundary_conditions": [],
            "important_unknowns": [],
            "confidence_statement": "当前只对条件结构有信心。",
            "why_this_contour": "它只保留了裁决后仍成立的条件关系。",
            "strongest_counter_contour": "两项变化也可能只是伴随现象。",
            "counter_contour_disposition": "保留为因果方向未知。",
        }
        final_client = FakeClient([final])
        contour, contour_record = run_live_contour(
            QUESTION,
            detective,
            constraints,
            model="fixture-model",
            retries=0,
            client_factory=lambda: final_client,
            material_mode="lossless_packet_v2",
            contour_mode="speculative_parallel_v1",
            provisional_drafts=drafts,
        )
        self.assertEqual(contour_record["status"], "succeeded")
        self.assertEqual(
            contour_record["audit"]["mode"],
            "speculative_drafts_final_adjudication",
        )
        self.assertEqual(len(final_client.payloads), 0)
        self.assertIn("需求", contour["direct_answer"])

    def test_one_missing_acceleration_draft_does_not_force_full_contour_fallback(self) -> None:
        data = materials(
            piece("a", "run_p01", "retained_local", "需求上升推动系统扩张"),
            piece("b", "run_p02", "retained_local", "融资收紧推动系统收缩"),
        )
        detective_case = build_detective_case(QUESTION, data)
        candidates = [
            {
                "candidate_id": "candidate-partial-01",
                "relation_type": "conditional_complement",
                "piece_ids": ["a", "b"],
                "plain_language_explanation": "两种条件推动相反方向。",
                "shared_coordinate": {"same_proposition": "系统方向"},
                "source_independence": "unknown",
                "common_error_risk": "low",
                "competing_explanation": "两种条件可能同时出现。",
                "falsification_test": "若对象不同则撤销。",
            }
        ]

        class PartialDraftClient:
            def chat(self, messages, **_kwargs):
                content = messages[-1]["content"]
                case_payload = json.loads(
                    content.split("CASE_JSON:\n", 1)[1].split(
                        "\nOUTPUT_SCHEMA:\n", 1
                    )[0]
                )
                schema = json.loads(
                    content.split("\nOUTPUT_SCHEMA:\n", 1)[1]
                )
                if schema["draft_id"] == "evidence_lineage":
                    raise RuntimeError("one draft transport failure")
                selected = case_payload["candidates"][0]
                payload = {
                    "draft_id": schema["draft_id"],
                    "direct_answer_hypothesis": "方向取决于两个条件。",
                    "contour_hypothesis": "两个条件共同约束整体方向。",
                    "candidate_ids": [selected["candidate_id"]],
                    "piece_ids": selected["piece_ids"],
                    "critical_assumptions": ["材料讨论同一系统"],
                    "failure_conditions": ["材料对象不同"],
                }
                encoded = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=encoded,
                    raw={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": encoded},
                            }
                        ]
                    },
                    usage={"total_tokens": 1},
                )

        drafts, record = run_provisional_contour_drafts(
            QUESTION,
            detective_case,
            candidates,
            model="fixture-model",
            retries=0,
            client_factory=lambda: PartialDraftClient(),
        )

        self.assertEqual(record["status"], "succeeded_partial_acceleration")
        self.assertEqual(record["audit"]["usable_draft_count"], 3)
        self.assertEqual(
            record["audit"]["missing_draft_roles"], ["evidence_lineage"]
        )
        self.assertEqual(len(drafts["rival_drafts"]), 2)
        self.assertTrue(
            record["audit"]["final_still_receives_full_adjudicated_case"]
        )

    def test_classic_material_is_additive_to_native_detective_sample(self) -> None:
        sample = [
            piece("p1_observed", "p1", "observed_claim", "观察"),
            piece("p1_retained_1", "p1", "retained_local", "局部一"),
            piece("p1_retained_2", "p1", "retained_local", "局部二"),
            piece("p1_fracture", "p1", "fracture", "断点"),
            piece("p1_assumption", "p1", "assumption", "经典提示的假设"),
        ]
        allowed = {
            "observed_claim",
            "retained_local",
            "fracture",
            "assumption",
        }
        native = _proposal_piece_selection(
            sample,
            allowed,
            mode="content",
        )
        assisted = _proposal_piece_selection(
            sample,
            allowed,
            mode="content",
            preferred_piece_ids={"p1_assumption"},
            classic_source_piece_ids={"p1_assumption"},
        )
        native_ids = {item["piece_id"] for item in native}
        assisted_ids = {item["piece_id"] for item in assisted}
        self.assertEqual(len(native), 4)
        self.assertTrue(native_ids.issubset(assisted_ids))
        self.assertIn("p1_assumption", assisted_ids)
        self.assertEqual(len(assisted), len(native) + 1)

    def test_truncated_recovery_preserves_native_budget_before_classic_additions(self) -> None:
        relations = [
            {
                "relation_type": "conditional_complement",
                "piece_ids": [f"n{index}", f"m{index}"],
                "plain_language_explanation": f"原生连接{index}",
                "discovery_route": "native",
                "connection_hypothesis_ids": [],
            }
            for index in range(14)
        ]
        relations.extend(
            {
                "relation_type": "scope_refinement",
                "piece_ids": [f"c{index}", f"d{index}"],
                "plain_language_explanation": f"经典辅助连接{index}",
                "discovery_route": "classic_assisted",
                "connection_hypothesis_ids": [f"connector_{index}"],
            }
            for index in range(7)
        )
        recovered = _recover_complete_relation_array(
            json.dumps({"relations": relations}, ensure_ascii=False)
        )
        native = [
            item for item in recovered if item["discovery_route"] == "native"
        ]
        assisted = [
            item
            for item in recovered
            if item["discovery_route"] == "classic_assisted"
        ]
        self.assertEqual(len(native), 12)
        self.assertEqual(len(assisted), 4)
        self.assertTrue(
            all(item["discovery_route"] == "native" for item in recovered[:12])
        )

    def test_classic_post_check_cannot_override_material_decision(self) -> None:
        diagnosis, outputs = classic_fixture(["p1", "p2"])
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        case = build_detective_case(QUESTION, materialized)
        hypothesis = next(
            item
            for item in case["classic_connection_hypotheses"]
            if item.get("candidate_partner_piece_ids")
            and "scope_refinement" in item["allowed_relation_types"]
        )
        partner_id = hypothesis["candidate_partner_piece_ids"][0]
        candidate = {
            "candidate_id": "candidate_native_with_classic_check",
            "question_id": question_id(QUESTION),
            "proposal_mode": "content",
            "relation_type": "scope_refinement",
            "piece_ids": [hypothesis["source_piece_id"], partner_id],
            "plain_language_explanation": "一人的时间越界被另一人的短期材料限定。",
            "shared_coordinate": {},
            "source_independence": "unknown",
            "common_error_risk": "medium",
            "competing_explanation": "两块材料也可能讨论不同指标。",
            "falsification_test": "补齐同一指标的长期证据后重审。",
            "connection_hypothesis_ids": [],
            "rejected_connection_hypothesis_ids": [],
            "discovery_routes": ["native"],
            "discovery_route": "native",
        }
        checks = _candidate_classic_check_ids(candidate, case)
        self.assertTrue(checks)
        candidate["classic_check_hypothesis_ids"] = checks

        material_prompt = _adjudication_messages(case, [candidate])[1][
            "content"
        ]
        self.assertNotIn(checks[0], material_prompt)
        self.assertNotIn("classic_connection_hypotheses", material_prompt)
        self.assertNotIn("discovery_route", material_prompt)
        material_decision = {
            "decisions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "decision": "accepted",
                    "plain_language_explanation": "材料本身足以证明范围需要缩小。",
                    "competing_explanation": "两块材料可能使用不同指标。",
                    "falsification_test": "若指标不同则撤销。",
                    "source_independence": "unknown",
                    "common_error_risk": "medium",
                }
            ]
        }
        base_relation = _adjudicate_candidates(
            material_decision, [candidate]
        )[0]
        self.assertEqual(base_relation["status"], "accepted")
        self.assertEqual(
            base_relation["provenance"][
                "unreviewed_connection_hypothesis_ids"
            ],
            checks,
        )

        def review(check_decision: str) -> dict:
            return {
                "reviews": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "hypothesis_id": checks[0],
                        "decision": check_decision,
                        "reason": "只检查经典工具是否准确解释连接。",
                    }
                ]
            }

        rejected = _apply_classic_reviews(
            [base_relation], [candidate], review("rejected")
        )[0]
        self.assertEqual(rejected["status"], "accepted")
        self.assertEqual(
            rejected["plain_language_explanation"],
            base_relation["plain_language_explanation"],
        )
        self.assertEqual(
            rejected["provenance"]["connection_hypothesis_ids"], []
        )
        self.assertEqual(
            rejected["provenance"]["rejected_connection_hypothesis_ids"],
            checks,
        )

        validated = _apply_classic_reviews(
            [base_relation], [candidate], review("validated")
        )[0]
        self.assertEqual(validated["status"], "accepted")
        self.assertEqual(
            validated["plain_language_explanation"],
            base_relation["plain_language_explanation"],
        )
        self.assertEqual(
            validated["provenance"]["connection_hypothesis_ids"], checks
        )
        self.assertFalse(
            validated["provenance"]["classic_truth_weight_bonus"]
        )
        self.assertTrue(
            validated["provenance"][
                "classic_review_cannot_modify_relation"
            ]
        )

    def test_materializer_keeps_forensic_trace_and_does_not_promote_retained_fragment(self) -> None:
        diagnosis = {
            "cognitive_chain_scans": [
                {
                    "person_id": "p1",
                    "semantic_verdict": {
                        "verdicts": [
                            {
                                "issue": "它从一个短期样本跳到了长期整体结论。",
                                "missing_bridge": "缺少短期现象持续到长期的机制。",
                                "preserved_fragment": "短期指标确实出现了变化。",
                                "from_claim": "短期指标变化",
                                "to_conclusion": "长期整体必然变化",
                                "impact_on_conclusion": "结论应缩回短期范围。",
                                "competing_explanation": "短期变化可能会逆转。",
                                "falsification_test": "若长期机制被连续数据证实，这条批评应撤销。",
                                "evidence_refs": ["R1", "C1"],
                                "conclusion_leverage": "high",
                            }
                        ],
                        "unresolved": [],
                    },
                }
            ],
            "collective_blind_spots": [],
        }
        outputs = [
            {
                "person_id": "p1",
                "person": {"id": "p1", "name": "甲", "time_horizon": "一年"},
                "output": {
                    "conclusion": "长期整体必然变化",
                    "answer_position": {"direction": "likely_yes", "scope": "整体"},
                    "core_assumptions": [],
                    "value_judgements": [],
                    "evidence_ledger": [],
                    "what_i_underweighted": [],
                },
            }
        ]
        result = materialize_puzzle_pieces(QUESTION, diagnosis, outputs, [])
        retained = [item for item in result["pieces"] if item["piece_kind"] == "retained_local"]
        distortions = [item for item in result["pieces"] if item["piece_kind"] == "distortion"]
        fractures = [item for item in result["pieces"] if item["piece_kind"] == "fracture"]
        self.assertEqual(retained[0]["content_status"], "candidate_retained_not_truth")
        self.assertTrue(retained[0]["diagnostic_ids"])
        self.assertEqual(len(distortions), 1)
        self.assertEqual(len(fractures), 1)
        self.assertFalse(result["materialization_audit"]["numeric_inversion_performed"])

    def test_shared_source_is_dependency_not_independent_support(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "同一局部判断", families=["url:https://x.test/a"]),
            piece("b", "p2", "retained_local", "同一局部判断", families=["url:https://x.test/a"]),
        )
        result = build_detective_output(QUESTION, data)
        types = [item["relation_type"] for item in result["relation_certificates"]]
        self.assertIn("shared_source", types)
        self.assertNotIn("independent_convergence", types)

    def test_google_news_catalog_keeps_underlying_publisher_identity(self) -> None:
        catalog = _build_source_catalog(
            {
                "p1": {
                    "source_decisions": [
                        {
                            "decision": "accept",
                            "result": {
                                "id": "rss-1",
                                "title": "Market report",
                                "snippet": "Evidence",
                                "url": "https://news.google.com/rss/articles/abc",
                                "source_name": "Seoul Economic Daily",
                                "source_type": "媒体报道",
                                "retrieval_layer": "external_search",
                                "verification_status": "external_unverified",
                            },
                        }
                    ]
                }
            }
        )

        self.assertEqual(
            catalog["rss-1"]["publisher_family"],
            "publisher:seoul economic daily",
        )
        self.assertEqual(catalog["rss-1"]["source_name"], "Seoul Economic Daily")

    def test_syndicated_copies_share_one_origin_family(self) -> None:
        catalog = _build_source_catalog(
            {
                "p1": {
                    "source_decisions": [
                        {
                            "decision": "accept",
                            "result": {
                                "id": "wire-1",
                                "title": "同一份政策报告发布了新预测 - 媒体甲",
                                "snippet": "报告预计相关指标将在未来一年变化。",
                                "url": "https://a.example.test/story/1",
                                "source_name": "媒体甲",
                                "retrieval_layer": "external_search",
                                "verification_status": "page_verified",
                            },
                        }
                    ]
                },
                "p2": {
                    "source_decisions": [
                        {
                            "decision": "accept",
                            "result": {
                                "id": "wire-2",
                                "title": "同一份政策报告发布了新预测 - 媒体乙",
                                "snippet": "报告预计相关指标将在未来一年变化。",
                                "url": "https://b.example.test/copy/9",
                                "source_name": "媒体乙",
                                "retrieval_layer": "external_search",
                                "verification_status": "page_verified",
                            },
                        }
                    ]
                },
            }
        )
        self.assertNotEqual(
            catalog["wire-1"]["canonical_url_family"],
            catalog["wire-2"]["canonical_url_family"],
        )
        self.assertNotEqual(
            catalog["wire-1"]["publisher_family"],
            catalog["wire-2"]["publisher_family"],
        )
        self.assertEqual(
            catalog["wire-1"]["origin_family"],
            catalog["wire-2"]["origin_family"],
        )

    def test_exact_claim_with_independent_lineage_can_be_provisional_convergence(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "未来一年内该指标会持续缓慢上升", families=["url:https://a.test/1"]),
            piece("b", "p2", "retained_local", "未来一年内该指标会持续缓慢上升", families=["url:https://b.test/2"]),
        )
        result = build_detective_output(QUESTION, data)
        relation = next(
            item
            for item in result["relation_certificates"]
            if item["relation_type"] == "independent_convergence"
        )
        self.assertEqual(relation["status"], "accepted")
        self.assertEqual(relation["source_independence"], "independent_by_current_ledger")

    def test_same_origin_copies_cannot_authorize_directional_support(self) -> None:
        first = piece(
            "a",
            "p1",
            "retained_local",
            "未来一年内该指标会持续缓慢上升",
            families=["url:https://a.test/1"],
        )
        second = piece(
            "b",
            "p2",
            "retained_local",
            "未来一年内该指标会持续缓慢上升",
            families=["url:https://b.test/2"],
        )
        for item, publisher in ((first, "publisher:a"), (second, "publisher:b")):
            item["evidence_source_families"] = list(item["source_families"])
            item["publisher_families"] = [publisher]
            item["origin_families"] = ["origin:shared-wire-copy"]
            item["independence_profile"] = {
                "origin_families": ["origin:shared-wire-copy"],
                "publisher_families": [publisher],
                "verification_statuses": ["page_verified"],
            }
        result = build_detective_output(QUESTION, materials(first, second))
        self.assertFalse(
            any(
                item.get("relation_type") == "independent_convergence"
                and (item.get("independence_profile") or {}).get(
                    "directional_support_authorized"
                )
                for item in result["relation_certificates"]
            )
        )

    def test_shared_model_does_not_erase_external_evidence_independence(self) -> None:
        first = piece(
            "a",
            "p1",
            "retained_local",
            "未来一年内该指标会持续缓慢上升",
            families=["shared-base-model-prior", "url:https://a.test/1"],
        )
        second = piece(
            "b",
            "p2",
            "retained_local",
            "未来一年内该指标会持续缓慢上升",
            families=["shared-base-model-prior", "url:https://b.test/2"],
        )
        for item, evidence_family in (
            (first, "url:https://a.test/1"),
            (second, "url:https://b.test/2"),
        ):
            item["evidence_source_families"] = [evidence_family]
            item["model_source_families"] = ["shared-base-model-prior"]
            item["independence_profile"] = {
                "model_families": ["shared-base-model-prior"],
                "evidence_families": [evidence_family],
                "verification_statuses": ["external_unverified"],
                "lineage_status": "external_lineage_linked",
            }
        result = build_detective_output(QUESTION, materials(first, second))
        relation = next(
            item
            for item in result["relation_certificates"]
            if item["relation_type"] == "independent_convergence"
        )
        self.assertEqual(
            relation["source_independence"],
            "evidence_independent_model_correlated",
        )
        self.assertEqual(
            relation["independence_profile"]["evidence_origin_independence"],
            "independent_by_current_ledger",
        )
        self.assertEqual(
            relation["independence_profile"]["model_independence"], "shared"
        )
        self.assertFalse(
            relation["independence_profile"]["directional_support_authorized"]
        )

    def test_declared_independence_cannot_override_evidence_ledger(self) -> None:
        first = piece("a", "p1", "retained_local", "局部甲")
        second = piece("b", "p2", "retained_local", "局部乙")
        guarded = _attach_relation_knowledge_provenance(
            {
                "relation_type": "definition_branch",
                "piece_ids": ["a", "b"],
                "status": "accepted",
                "source_independence": "independent",
                "independence_profile": {
                    "evidence_origin_independence": "independent",
                    "directional_support_authorized": True,
                },
                "provenance": {"mode": "live_adversarial_adjudication"},
            },
            {"a": first, "b": second},
        )

        self.assertEqual(guarded["source_independence"], "unknown")
        self.assertEqual(
            guarded["independence_profile"]["evidence_origin_independence"],
            "unknown",
        )
        self.assertFalse(
            guarded["independence_profile"]["directional_support_authorized"]
        )

    def test_entry_id_links_output_evidence_to_external_source_lineage(self) -> None:
        outputs = [
            {
                "person_id": "p1",
                "person": {"id": "p1", "name": "测试者"},
                "output": {
                    "conclusion": "融资条件正在收紧",
                    "answer_position": {"direction": "likely_yes", "scope": "市场"},
                    "core_assumptions": [],
                    "value_judgements": [],
                    "evidence_ledger": [
                        {"entry_id": "R1", "claim": "融资利差连续扩大"}
                    ],
                    "what_i_underweighted": [],
                },
            }
        ]
        ledgers = [
            {
                "person_id": "p1",
                "source_decisions": [
                    {
                        "decision": "accepted",
                        "result": {
                            "id": "R1",
                            "title": "融资利差数据",
                            "url": "https://data.example.test/credit",
                            "retrieval_layer": "external_search",
                            "verification_status": "external_unverified",
                            "source_type": "market_data",
                        },
                    }
                ],
            }
        ]
        result = materialize_puzzle_pieces(
            "融资条件是否收紧",
            {"cognitive_chain_scans": []},
            outputs,
            ledgers,
            {
                "relevant_evidence_dimensions": [
                    "融资利差与信贷条件",
                    "企业盈利与现金流",
                ]
            },
        )
        evidence = next(
            item
            for item in result["pieces"]
            if item["piece_kind"] == "evidence_claim"
        )
        self.assertEqual(evidence["source_ids"], ["R1"])
        self.assertEqual(
            evidence["independence_profile"]["lineage_status"],
            "external_lineage_linked",
        )
        self.assertEqual(
            result["materialization_audit"]["source_lineage_complete_rate"],
            1.0,
        )
        self.assertFalse(result["mechanism_index"]["adds_model_calls"])
        self.assertTrue(evidence["mechanism_ids"])

    def test_legacy_source_label_only_links_on_unique_high_confidence_match(self) -> None:
        outputs = [
            {
                "person_id": "p1",
                "person": {"id": "p1", "name": "测试者"},
                "output": {
                    "conclusion": "跨资产相关性正在变化",
                    "answer_position": {"direction": "conditional", "scope": "市场"},
                    "core_assumptions": [],
                    "value_judgements": [],
                    "evidence_ledger": [
                        {
                            "claim": "股票与债券的滚动相关性发生变化",
                            "source": "BlackRock",
                        },
                        {
                            "claim": "来源说法过于模糊",
                            "source": "研究报告",
                        },
                    ],
                    "what_i_underweighted": [],
                },
            }
        ]
        ledgers = [
            {
                "person_id": "p1",
                "source_decisions": [
                    {
                        "decision": "accept",
                        "result": {
                            "id": "BR1",
                            "title": "2025 Investment Directions | BlackRock",
                            "url": "https://www.blackrock.com/insights/2025",
                            "snippet": "Rolling correlation between stocks and bonds.",
                            "retrieval_layer": "external_search",
                            "verification_status": "external_unverified",
                        },
                    }
                ],
            }
        ]
        result = materialize_puzzle_pieces(
            "跨资产相关性是否变化",
            {"cognitive_chain_scans": []},
            outputs,
            ledgers,
        )
        evidence = [
            item
            for item in result["pieces"]
            if item["piece_kind"] == "evidence_claim"
        ]
        self.assertEqual(evidence[0]["source_ids"], ["BR1"])
        self.assertEqual(evidence[1]["source_ids"], [])
        self.assertEqual(
            result["materialization_audit"]["source_fallback_match_count"], 1
        )
        self.assertEqual(
            result["materialization_audit"]["source_fallback_unresolved_count"],
            1,
        )

    def test_silence_does_not_automatically_become_collective_blind_spot(self) -> None:
        data = materials(
            piece("a", "p1", "blind_spot_candidate", "可能漏看了监管变化"),
            piece("b", "p2", "blind_spot_candidate", "可能漏看了监管变化"),
        )
        result = build_detective_output(QUESTION, data)
        self.assertFalse(
            any(
                item["relation_type"] == "collective_blind_spot"
                and item["status"] == "accepted"
                for item in result["relation_certificates"]
            )
        )

    def test_unknown_piece_reference_is_rejected(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "甲判断"),
            piece("b", "p2", "retained_local", "乙判断"),
        )
        detective = build_detective_output(QUESTION, data)
        detective["relation_certificates"] = [
            {
                "relation_id": "bad",
                "relation_type": "direct_conflict",
                "piece_ids": ["a", "missing"],
                "status": "accepted",
                "plain_language_explanation": "冲突",
                "competing_explanation": "也可能不是同一范围",
                "falsification_test": "补充范围后重判",
            }
        ]
        errors = validate_detective_output(detective, data)
        self.assertTrue(any("unknown piece" in item for item in errors))

    def test_opposite_distortion_never_performs_numeric_inversion(self) -> None:
        data = materials(
            piece("a", "p1", "distortion", "把风险夸大到必然失败"),
            piece("b", "p2", "distortion", "把风险缩小到绝不失败"),
        )
        semantic_relation = {
            "relation_type": "opposite_distortion",
            "piece_ids": ["a", "b"],
            "status": "accepted",
            "plain_language_explanation": "两个人分别把同一风险向相反方向推到极端。",
            "source_independence": "unknown",
            "common_error_risk": "medium",
            "competing_explanation": "两人也可能讨论的是不同时间范围。",
            "falsification_test": "若证明两人讨论不同时间范围，应撤销这条关系。",
        }
        detective = build_detective_output(
            QUESTION, data, semantic_relations=[semantic_relation]
        )
        constraints = build_contour_constraints(QUESTION, detective)
        directional = next(
            item
            for item in constraints["constraints"]
            if item["constraint_type"] == "directional_bound"
        )
        contour = build_truth_contour(QUESTION, detective, constraints)
        self.assertIn("禁止计算", directional["caution"])
        self.assertFalse(contour["contour_evaluation"]["numeric_bias_inversion_performed"])

    def test_majority_of_unconnected_claims_is_not_truth(self) -> None:
        data = materials(
            piece("a", "p1", "observed_claim", "答案是会"),
            piece("b", "p2", "observed_claim", "答案是会"),
            piece("c", "p3", "observed_claim", "答案是会"),
        )
        detective = build_detective_output(QUESTION, data)
        constraints = build_contour_constraints(QUESTION, detective)
        contour = build_truth_contour(QUESTION, detective, constraints)
        self.assertIn("不能从这些数字人的数量", contour["direct_answer"])
        self.assertEqual(contour["generation_mode"], "deterministic_conservative_fallback")

    def test_truth_contour_rejects_invented_provenance(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "局部甲"),
            piece("b", "p2", "retained_local", "局部乙"),
        )
        detective = build_detective_output(QUESTION, data)
        constraints = build_contour_constraints(QUESTION, detective)
        contour = build_truth_contour(QUESTION, detective, constraints)
        contour["provenance_index"][0]["piece_ids"] = ["invented"]
        errors = validate_truth_contour(
            contour,
            question=QUESTION,
            detective=detective,
            constraints=constraints,
        )
        self.assertTrue(any("unknown piece" in item for item in errors))

    def test_invalid_live_adjudication_cannot_silently_become_fallback(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "局部甲"),
            piece("b", "p2", "retained_local", "局部乙"),
        )
        detective = build_detective_output(QUESTION, data)
        constraints = build_contour_constraints(QUESTION, detective)

        with self.assertRaisesRegex(
            ValueError, "Adjudicated truth contour failed delivery validation"
        ):
            build_truth_contour(
                QUESTION,
                detective,
                constraints,
                adjudicated_payload={
                    "direct_answer": {"text": "方向已经成立。"},
                    "main_contour": {"text": "缺少任何来源引用。"},
                },
            )

    def test_live_detective_and_contour_complete_adversarial_cycle(self) -> None:
        data = materials(
            piece("a", "p1", "retained_local", "需求上升时系统更可能扩张"),
            piece("b", "p2", "retained_local", "融资收紧时系统更可能收缩"),
        )
        candidate_id = stable_id(
            "candidate",
            question_id(QUESTION),
            "conditional_complement",
            "a|b",
        )
        adjudicated_explanation = "两人讨论的是同一系统在需求上升与融资收紧两种条件下的不同走向。"
        relation_id = stable_id(
            "relation",
            question_id(QUESTION),
            "conditional_complement",
            "a|b",
            adjudicated_explanation,
        )
        detective_client = FakeClient(
            [
                {
                    "relations": [
                        {
                            "relation_type": "conditional_complement",
                            "piece_ids": ["a", "b"],
                            "plain_language_explanation": "两块材料描述了不同条件下的方向。",
                            "shared_coordinate": {"same_proposition": "系统会扩张还是收缩"},
                            "source_independence": "unknown",
                            "common_error_risk": "low",
                            "competing_explanation": "两人也可能讨论了不同系统。",
                            "falsification_test": "若对象不是同一系统，应撤销这条连接。",
                        }
                    ]
                },
                {"relations": []},
                {
                    "decisions": [
                        {
                            "candidate_id": candidate_id,
                            "decision": "accepted",
                            "plain_language_explanation": adjudicated_explanation,
                            "competing_explanation": "需求和融资也可能同时变化，不能把两者完全分开。",
                            "falsification_test": "若材料证明两人采用了不同对象或不同时间范围，应撤销这条关系。",
                            "source_independence": "unknown",
                            "common_error_risk": "low",
                            "confidence_profile": {
                                "semantic_alignment": "high",
                                "scope_compatibility": "high",
                                "source_independence": "unknown",
                                "diagnostic_certainty": "high",
                            },
                        }
                    ]
                },
            ]
        )
        detective, detective_record = run_live_detective(
            QUESTION,
            data,
            model="fixture-model",
            retries=0,
            client_factory=lambda: detective_client,
        )
        self.assertEqual(detective_record["status"], "succeeded")
        self.assertEqual(detective_record["audit"]["native_candidate_count"], 1)
        self.assertEqual(
            detective_record["audit"]["classic_assisted_candidate_count"], 0
        )
        self.assertFalse(
            detective_record["audit"]["classic_truth_weight_bonus"]
        )
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "conditional_complement"
        )
        self.assertEqual(relation["relation_id"], relation_id)
        self.assertEqual(relation["status"], "accepted")

        constraints = build_contour_constraints(QUESTION, detective)
        contour_case = build_contour_case(QUESTION, detective, constraints)
        prompt_packet = build_contour_prompt_packet(contour_case)
        self.assertEqual(
            validate_contour_prompt_packet(prompt_packet, contour_case), []
        )
        broken_packet = json.loads(json.dumps(prompt_packet, ensure_ascii=False))
        broken_packet["piece_index"].pop("a")
        self.assertTrue(
            validate_contour_prompt_packet(broken_packet, contour_case)
        )
        constraint_id = stable_id(
            "constraint",
            question_id(QUESTION),
            "conditional_split",
            relation_id,
        )

        def cited(text: str) -> dict:
            return {
                "text": text,
                "piece_ids": ["a", "b"],
                "relation_ids": [relation_id],
                "constraint_ids": [constraint_id],
            }

        def piece_only(text: str) -> dict:
            return {
                "text": text,
                "piece_ids": ["a"],
                "relation_ids": [],
                "constraint_ids": [],
            }

        candidate = {
            "direct_answer": cited("会怎样发展取决于需求与融资哪一项先成为主导约束。"),
            "main_contour": cited("多数数字人都认为需求推动扩张，融资收紧推动收缩，方向由两者的相对约束决定。"),
            "key_conditions": [cited("需求强而融资未收紧时，轮廓偏向扩张。")],
            "stable_parts": [
                cited("需求与融资都是方向性约束。"),
                piece_only("需求上升已经是稳定事实。"),
            ],
            "boundary_conditions": [],
            "important_unknowns": [cited("两种条件同时出现时谁更强仍未知。")],
            "confidence_statement": "只对条件结构有中等信心。",
            "why_this_contour": "它保留了两块材料各自成立的条件。",
            "strongest_counter_contour": "",
            "counter_contour_disposition": "",
        }
        counter = {
            "counter_contour": cited("系统可能主要受第三个共同因素驱动，需求和融资只是伴随现象。"),
            "why_it_may_fit_better": cited("现有关系只说明条件不同，没有证明因果方向。"),
            "attacks": [cited("主候选把条件关系写得太像因果关系。")],
            "decisive_tests": [
                cited("需要连续观察30天，区分条件相关与因果驱动的材料。")
            ],
        }
        final = {
            **candidate,
            "direct_answer": cited("目前最稳妥的回答是：走向取决于需求扩张力与融资收缩力谁先占主导。"),
            "strongest_counter_contour": "需求和融资都可能只是第三个共同因素的伴随现象。",
            "counter_contour_disposition": "作为重要未知保留，因为现有关系没有证明因果方向。",
        }
        contour_client = FakeClient([candidate, counter, final])
        contour, contour_record = run_live_contour(
            QUESTION,
            detective,
            constraints,
            model="fixture-model",
            retries=0,
            client_factory=lambda: contour_client,
        )
        self.assertEqual(contour_record["status"], "succeeded")
        self.assertEqual(
            contour_record["audit"]["prompt_packet"]["coverage_validation"],
            "passed",
        )
        self.assertFalse(
            contour_record["audit"]["prompt_packet"][
                "task_relevant_material_removed"
            ]
        )
        self.assertEqual(contour["generation_mode"], "live_adversarial_inference")
        self.assertIn("需求扩张力", contour["direct_answer"])
        self.assertIn("需求推动扩张", contour["main_contour"])
        self.assertNotIn("多数数字人", contour["main_contour"])
        self.assertEqual(contour["stable_parts"], ["需求与融资都是方向性约束。"])
        self.assertEqual(contour["contour_evaluation"]["provenance_coverage"], 1.0)
        self.assertEqual(contour_record["counter_contour"]["decisive_tests"], [])
        self.assertEqual(
            contour_record["counter_contour"]["validation_drops"][0]["field"],
            "decisive_tests",
        )
        generic = {
            **candidate,
            "direct_answer": cited("尚不能确定，当前只有定义、范围和条件结构。"),
            "main_contour": cited("当前轮廓只是定义、范围、条件、价值和共同先验的组合。"),
        }
        _, specificity_errors = _validate_candidate(
            generic,
            contour_record["case"],
        )
        self.assertTrue(
            any("too abstract for this question" in item for item in specificity_errors)
        )
        mismatched = json.loads(json.dumps(candidate, ensure_ascii=False))
        mismatched["direct_answer"]["piece_ids"] = ["a"]
        _, endpoint_errors = _validate_candidate(
            mismatched,
            contour_record["case"],
        )
        self.assertTrue(
            any("omits relation endpoint pieces" in item for item in endpoint_errors)
        )

        detective_checkpoints = []
        detective_reused, reused_detective_record = run_live_detective(
            QUESTION,
            data,
            model="fixture-model",
            retries=0,
            client_factory=lambda: FakeClient([]),
            cached_record=detective_record,
            checkpoint_callback=detective_checkpoints.append,
        )
        self.assertEqual(reused_detective_record["status"], "succeeded")
        self.assertTrue(detective_checkpoints)
        self.assertTrue(
            all(
                item["status"] in {"reused", "not_needed"}
                for item in reused_detective_record["audit"]["stages"]
            )
        )

        legacy_detective, legacy_detective_record = run_live_detective(
            QUESTION,
            data,
            model="fixture-model",
            retries=0,
            client_factory=lambda: FakeClient([]),
            cached_record=detective_record,
            execution_mode="legacy_sequential",
        )
        self.assertEqual(
            legacy_detective_record["audit"]["execution_mode"],
            "legacy_sequential",
        )
        self.assertEqual(
            [
                item["relation_id"]
                for item in legacy_detective["relation_certificates"]
            ],
            [
                item["relation_id"]
                for item in detective_reused["relation_certificates"]
            ],
        )

        reused_constraints = build_contour_constraints(
            QUESTION, detective_reused
        )
        contour_checkpoints = []
        reused_contour, reused_contour_record = run_live_contour(
            QUESTION,
            detective_reused,
            reused_constraints,
            model="fixture-model",
            retries=0,
            client_factory=lambda: FakeClient([]),
            cached_record=contour_record,
            checkpoint_callback=contour_checkpoints.append,
        )
        self.assertEqual(reused_contour_record["status"], "succeeded")
        self.assertEqual(
            [item["status"] for item in reused_contour_record["audit"]["stages"]],
            ["reused", "reused", "reused"],
        )
        self.assertTrue(contour_checkpoints)
        self.assertEqual(
            reused_contour["generation_mode"], "live_adversarial_inference"
        )

        legacy_constraints = build_contour_constraints(
            QUESTION, legacy_detective
        )
        legacy_contour, legacy_contour_record = run_live_contour(
            QUESTION,
            legacy_detective,
            legacy_constraints,
            model="fixture-model",
            retries=0,
            client_factory=lambda: FakeClient([]),
            cached_record=contour_record,
            material_mode="legacy_full_case",
        )
        packet_audit = legacy_contour_record["audit"]["prompt_packet"]
        self.assertEqual(packet_audit["material_mode"], "legacy_full_case")
        self.assertEqual(
            packet_audit["sent_material_characters"],
            packet_audit["full_case_characters"],
        )
        self.assertEqual(legacy_contour["direct_answer"], contour["direct_answer"])
        self.assertEqual(legacy_contour["main_contour"], contour["main_contour"])


    def test_classic_trace_reaches_pieces_relations_constraints_and_contour(self) -> None:
        diagnosis, outputs = classic_fixture(["p1", "p2"])
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        self.assertEqual(
            materialized["materialization_audit"]["classic_trace_link_rate"],
            1.0,
        )
        self.assertEqual(
            materialized["materialization_audit"][
                "classic_used_as_world_evidence_count"
            ],
            0,
        )
        self.assertTrue(materialized["classic_diagnostic_traces"])
        self.assertTrue(
            all(
                item["authorization"] == "provisional"
                for item in materialized["inversion_certificates"]
            )
        )

        retained = [
            item
            for item in materialized["pieces"]
            if item["piece_kind"] == "retained_local"
        ]
        semantic_relation = {
            "relation_type": "scope_refinement",
            "piece_ids": [item["piece_id"] for item in retained],
            "status": "accepted",
            "plain_language_explanation": "两块材料都只能支持短期范围",
            "source_independence": "unknown",
            "common_error_risk": "medium",
            "competing_explanation": "长期机制也可能已经存在",
            "falsification_test": "补入长期连续证据后重新裁决",
        }
        detective = build_detective_output(
            QUESTION,
            materialized,
            semantic_relations=[semantic_relation],
        )
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "scope_refinement"
        )
        self.assertEqual(relation["status"], "accepted")
        self.assertEqual(
            relation["knowledge_provenance"]["relation_use_mode"],
            "diagnostic_context_considered",
        )
        self.assertEqual(
            relation["knowledge_provenance"]["inference_permission"],
            "relationship_adjudication_required",
        )
        self.assertFalse(
            relation["knowledge_provenance"]["classic_as_world_evidence"]
        )

        constraints = build_contour_constraints(QUESTION, detective)
        constraint = next(
            item
            for item in constraints["constraints"]
            if item["source_relation_ids"] == [relation["relation_id"]]
        )
        self.assertEqual(
            set(constraint["knowledge_provenance"]["classic_trace_ids"]),
            set(relation["knowledge_provenance"]["classic_trace_ids"]),
        )
        contour = build_truth_contour(QUESTION, detective, constraints)
        self.assertTrue(
            all(
                item.get("epistemic_role")
                == "diagnostic_method_not_world_evidence"
                for item in contour["provenance_index"]
            )
        )
        self.assertEqual(
            contour["classic_knowledge_audit"][
                "classic_used_as_world_evidence_count"
            ],
            0,
        )

    def test_classic_connection_hypothesis_is_auditable_through_contour(self) -> None:
        diagnosis, outputs = classic_fixture(["p1", "p2"])
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        bundle = materialized["classic_connection_hypotheses"]
        self.assertGreater(bundle["hypothesis_count"], 0)
        self.assertFalse(bundle["adds_model_calls"])
        self.assertTrue(
            all(
                item["classic_as_world_evidence"] is False
                for item in bundle["hypotheses"]
            )
        )
        piece_people = {
            item["piece_id"]: item["person_id"]
            for item in materialized["pieces"]
        }
        self.assertTrue(
            all(item["candidate_partner_piece_ids"] for item in bundle["hypotheses"])
        )
        self.assertTrue(
            all(
                piece_people[partner_id] != item["source_person_id"]
                for item in bundle["hypotheses"]
                for partner_id in item["candidate_partner_piece_ids"]
            )
        )
        self.assertTrue(
            all(
                piece_people[piece_id] == item["source_person_id"]
                for item in bundle["hypotheses"]
                for piece_id in item["applicable_source_piece_ids"]
            )
        )
        shadow_crosscheck = next(
            item
            for item in bundle["hypotheses"]
            if item["operator_id"] == "shadow_mechanism_crosscheck"
        )
        self.assertIn(
            "correlated_error", shadow_crosscheck["allowed_relation_types"]
        )
        self.assertIn(
            "opposite_distortion", shadow_crosscheck["allowed_relation_types"]
        )
        self.assertFalse(shadow_crosscheck["classic_as_world_evidence"])
        case = build_detective_case(QUESTION, materialized)
        self.assertTrue(case["classic_connection_hypotheses"])

        hypothesis = next(
            item
            for item in bundle["hypotheses"]
            if "scope_refinement" in item["allowed_relation_types"]
        )
        source_piece = next(
            item
            for item in materialized["pieces"]
            if item["piece_id"] == hypothesis["source_piece_id"]
        )
        partner = next(
            item
            for item in materialized["pieces"]
            if item["person_id"] != source_piece["person_id"]
            and item["piece_kind"] in hypothesis["partner_piece_kinds"]
        )
        semantic_relation = {
            "relation_type": "scope_refinement",
            "piece_ids": [source_piece["piece_id"], partner["piece_id"]],
            "status": "accepted",
            "plain_language_explanation": "一块材料暴露了时间越界，另一块材料只支持短期，因此结论必须缩回短期。",
            "source_independence": "unknown",
            "common_error_risk": "medium",
            "competing_explanation": "两块材料也可能讨论不同指标。",
            "falsification_test": "补齐同一指标的长期连续证据后重新判断范围。",
            "provenance": {
                "mode": "semantic_adjudication",
                "connection_hypothesis_ids": [hypothesis["hypothesis_id"]],
            },
        }
        detective = build_detective_output(
            QUESTION,
            materialized,
            semantic_relations=[semantic_relation],
        )
        relation = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "scope_refinement"
            and hypothesis["hypothesis_id"]
            in (
                item.get("knowledge_provenance") or {}
            ).get("connection_hypothesis_ids", [])
        )
        self.assertEqual(
            relation["knowledge_provenance"]["relation_use_mode"],
            "classic_connection_operator_tested",
        )
        constraints = build_contour_constraints(QUESTION, detective)
        constraint = next(
            item
            for item in constraints["constraints"]
            if relation["relation_id"] in item["source_relation_ids"]
        )
        self.assertIn(
            hypothesis["hypothesis_id"],
            constraint["knowledge_provenance"]["connection_hypothesis_ids"],
        )
        contour = build_truth_contour(QUESTION, detective, constraints)
        self.assertIn(
            hypothesis["hypothesis_id"],
            contour["classic_knowledge_audit"]["connection_hypothesis_ids"],
        )

    def test_active_classic_differential_remains_restricted(self) -> None:
        diagnosis, outputs = classic_fixture(["p1"])
        diagnosis["differential_diagnosis"][0]["rules"][0][
            "status"
        ] = "active_competition"
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        self.assertTrue(
            all(
                item["authorization"] == "restricted"
                for item in materialized["inversion_certificates"]
            )
        )

    def test_user_text_replaces_person_aliases_and_partial_internal_ids(self) -> None:
        cleaned = _clean_user_text(
            "P02依据relation_c71b、，P05仍有不同判断（constraint_ab12）。",
            {"P02": "刘洋", "P05": "玛丽"},
        )
        self.assertEqual(cleaned, "刘洋依据，玛丽仍有不同判断。")
        self.assertNotIn("P02", cleaned)
        self.assertNotIn("relation_", cleaned)

        punctuation = _clean_user_text("忽略了家庭债务（relation_ab12），。")
        self.assertEqual(punctuation, "忽略了家庭债务。")
        self.assertEqual(
            _downgrade_group_quantifiers("多数分析者一致认为风险很低"),
            "部分分析者认为风险很低",
        )

    def test_diagnostic_material_cannot_create_world_convergence(self) -> None:
        diagnosis, outputs = classic_fixture(["p1", "p2"])
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        retained = [
            item
            for item in materialized["pieces"]
            if item["piece_kind"] == "retained_local"
        ]
        relation = {
            "relation_type": "independent_convergence",
            "piece_ids": [item["piece_id"] for item in retained],
            "status": "accepted",
            "plain_language_explanation": "两人保留了同一短期局部",
            "source_independence": "unknown",
            "common_error_risk": "medium",
            "competing_explanation": "相同判断可能来自同一模型先验",
            "falsification_test": "核对来源谱系后重新裁决",
        }
        detective = build_detective_output(
            QUESTION,
            materialized,
            semantic_relations=[relation],
        )
        guarded = next(
            item
            for item in detective["relation_certificates"]
            if item["relation_type"] == "independent_convergence"
        )
        self.assertEqual(guarded["status"], "unresolved")
        self.assertEqual(
            guarded["knowledge_provenance"]["guardrail_action"],
            "downgraded_diagnostic_material_not_world_evidence",
        )

    def test_classic_trace_cannot_be_marked_as_world_evidence(self) -> None:
        diagnosis, outputs = classic_fixture(["p1"])
        materialized = materialize_puzzle_pieces(
            QUESTION, diagnosis, outputs, []
        )
        diagnostic_piece = next(
            item for item in materialized["pieces"] if item["diagnostic_ids"]
        )
        diagnostic_piece["knowledge_trace"]["classic_as_world_evidence"] = True
        errors = validate_puzzle_materials(materialized, QUESTION)
        self.assertTrue(
            any("treats classics as evidence" in item for item in errors)
        )


if __name__ == "__main__":
    unittest.main()
