from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from bme_model.meta_model.evaluator import (
    evaluate_meta_model_overall,
    evaluate_semantic_verdict_strength,
)
from bme_model.meta_model.semantic_verdicts import (
    _numeric_facts,
    _run_one_semantic_verdict,
    attach_semantic_verdicts,
    build_fallback_semantic_verdict,
    build_semantic_verdict_case,
    load_semantic_verdict_records,
    render_semantic_verdict_messages,
    run_semantic_verdict_batch,
    semantic_case_fingerprint,
    validate_semantic_verdict_output,
)


QUESTION = "美股泡沫会不会很快破灭"


def test_semantic_prompt_lists_only_real_case_anchors_and_detectors():
    case = _case()
    content = render_semantic_verdict_messages(case)[1]["content"]

    assert "ALLOWED_EVIDENCE_REFS" in content
    assert '"C1"' in content
    assert '"argument_forensics"' in content
    assert "默认只输出一个最关键 verdict" in content


def _persona() -> dict:
    return {
        "id": "person_1",
        "name": "危机工程师",
        "role_summary": "用复杂系统安全分析市场风险。",
        "cognitive_frames": ["复杂系统", "失效分析"],
        "expertise_strong": ["工程事故", "系统安全"],
        "expertise_weak": ["资本市场"],
        "values": {"safety": 0.7, "efficiency": 0.3},
        "risk_attitude": "规避尾部风险",
        "information_filter": {
            "preferred_evidence": ["事故案例", "失效分析"],
            "ignored_evidence": ["估值数据", "流动性数据"],
        },
    }


def _ledger() -> dict:
    return {
        "person_id": "person_1",
        "search_strategy": {"preferred_queries": ["美股 系统性风险 连锁崩溃"]},
        "source_decisions": [],
        "entries": [],
        "accepted_sources": [],
        "rejected_sources": [],
        "ignored_sources": [],
    }


def _output() -> dict:
    return {
        "core_assumptions": ["工程事故中的小扰动会触发连锁故障。"],
        "reasoning_path": [
            "股市也是复杂系统。",
            "所以一次小扰动也会让整个市场连锁崩溃。",
        ],
        "value_judgements": ["宁可高估尾部风险。"],
        "conclusion": "因此美股泡沫会在六个月内破灭。",
        "confidence": 0.75,
        "what_i_underweighted": ["估值、杠杆和市场流动性。"],
        "what_would_change_my_mind": ["市场去杠杆后仍能维持估值。"],
    }


def _finding(detector_id: str = "argument_forensics") -> dict:
    return {
        "detector_id": detector_id,
        "detector_name": detector_id,
        "status": "detected",
        "diagnostic_confidence": 0.9,
        "observed_evidence": [
            {
                "stage": "reasoning",
                "fact": "把工程事故类比直接迁移到股市。",
                "excerpt": "股市也是复杂系统，所以一次小扰动也会让整个市场连锁崩溃。",
            }
        ],
        "plain_language_diagnosis": "中间缺少市场传导机制。",
        "mechanism": "跨域类比替代了因果桥梁。",
        "consequence_for_conclusion": "六个月内破灭的时间判断需要降权。",
        "competing_explanations": ["复杂系统之间可能存在共同机制。"],
        "questions_to_distinguish": ["杠杆、流动性和强平怎样完成传导？"],
        "falsification_test": "补齐可验证的市场传导链。",
        "preserved_valid_part": "保留对系统脆弱性的提醒。",
        "knowledge_basis": [],
    }


def _scan() -> dict:
    return {
        "person_id": "person_1",
        "person_name": "危机工程师",
        "chain_coverage": {"score": 100},
        "detector_results": [
            _finding("argument_forensics"),
            _finding("local_whole_overreach"),
        ],
    }


def _case() -> dict:
    return build_semantic_verdict_case(QUESTION, _persona(), _ledger(), _output(), _scan())


def _valid_payload(case: dict | None = None) -> dict:
    case = case or _case()
    return {
        "person_id": case["person_id"],
        "headline": "工程事故类比不能直接推出美股六个月内破灭。",
        "verdicts": [
            {
                "issue": "它把工程事故中小扰动触发连锁故障，直接当成美股泡沫六个月内破灭的根据。",
                "from_claim": "工程事故中的小扰动会触发连锁故障。",
                "to_conclusion": "因此美股泡沫会在六个月内破灭。",
                "missing_bridge": "没有说明杠杆、流动性或强平怎样把扰动传成市场崩溃。",
                "impact_on_conclusion": "系统脆弱可以保留，但六个月内破灭必须降级为待验证假设。",
                "preserved_fragment": "保留对尾部风险和系统脆弱性的提醒。",
                "competing_explanation": "不同复杂系统可能共享部分失效机制。",
                "falsification_test": "若能给出杠杆和流动性触发的连续证据链，本诊断应撤销。",
                "evidence_refs": ["A1", "R1", "C1"],
                "source_detector_ids": ["argument_forensics", "local_whole_overreach"],
                "conclusion_leverage": "high",
            }
        ],
        "unresolved": [],
    }


def test_case_contains_only_one_person_and_addressable_chain() -> None:
    case = _case()
    assert case["person_id"] == "person_1"
    assert {item["id"] for item in case["chain_anchors"]} >= {"A1", "R1", "C1", "C2"}
    assert "person_2" not in json.dumps(case, ensure_ascii=False)


def test_specific_verdict_passes_evidence_lock() -> None:
    verdict = validate_semantic_verdict_output(_valid_payload(), _case())
    assert verdict["validation"]["valid"]
    assert verdict["verdicts"][0]["evidence_validation"]["specificity_score"] >= 0.9
    assert "工程事故" in verdict["summary"]["headline"]


def test_generic_template_is_rejected() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["issue"] = "证据、假设和结论之间存在推理跳跃。"
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert any("通用模板" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_unknown_evidence_reference_is_rejected() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["evidence_refs"] = ["A404", "C1"]
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert any("不存在的证据编号" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_backwards_falsification_condition_is_rejected() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["falsification_test"] = "若市场传导证据仍然缺失，则本诊断成立。"
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert any("撤销或降级" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_accepted_falsification_is_canonicalized_against_logical_inversion() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["falsification_test"] = (
        "若展示泡沫可能继续维持的反证，而此人仍坚持六个月内破灭，本诊断应降级。"
    )
    verdict = validate_semantic_verdict_output(payload, _case())
    test = verdict["verdicts"][0]["falsification_test"]
    assert "仍支持原结论" in test
    assert "此人仍坚持" not in test


def test_one_good_verdict_cannot_hide_one_invalid_verdict() -> None:
    payload = _valid_payload()
    invalid = dict(payload["verdicts"][0])
    invalid["falsification_test"] = "若市场传导证据仍然缺失，则本诊断成立。"
    payload["verdicts"].append(invalid)
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert "不能用合格项遮盖" in " ".join(verdict["validation"]["errors"])


def test_unanchored_numeric_fact_is_rejected() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["competing_explanation"] += "例如2010年的市场事件。"
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert any("所引证据中不存在" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_numeric_fact_from_identity_context_is_allowed() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["impact_on_conclusion"] += "安全价值权重0.7本身不能充当概率。"
    payload["verdicts"][0]["evidence_refs"].append("I3")
    verdict = validate_semantic_verdict_output(payload, _case())
    assert verdict["validation"]["valid"]


def test_numeric_fact_cannot_borrow_from_an_uncited_anchor() -> None:
    payload = _valid_payload()
    payload["verdicts"][0]["impact_on_conclusion"] += "安全价值权重0.7本身不能充当概率。"
    verdict = validate_semantic_verdict_output(payload, _case())
    assert not verdict["validation"]["valid"]
    assert any("所引证据中不存在" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_numeric_range_normalizes_typographic_hyphens() -> None:
    assert _numeric_facts("1-3年") == _numeric_facts("1‑3年")


def test_year_facts_match_with_or_without_a_suffix_before_ascii_text() -> None:
    assert _numeric_facts("2015年A股与2022年A股") == _numeric_facts(
        "2015、2022年出现不同表现"
    )


def test_conclusion_copy_cannot_pose_as_non_conclusion_evidence() -> None:
    case = _case()
    case["chain_anchors"].append(
        {"id": "E99", "stage": "conclusion", "text": "因此美股泡沫会在六个月内破灭。"}
    )
    payload = _valid_payload(case)
    payload["verdicts"][0]["evidence_refs"] = ["E99", "C1"]
    verdict = validate_semantic_verdict_output(payload, case)
    assert not verdict["validation"]["valid"]
    assert any("非结论锚点" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_same_root_problem_is_merged_even_when_wording_differs() -> None:
    payload = _valid_payload()
    duplicate = dict(payload["verdicts"][0])
    duplicate["issue"] = "它把股市同属复杂系统，当成一次小扰动足以令美股六个月内崩溃的证明。"
    duplicate["source_detector_ids"] = ["local_whole_overreach"]
    payload["verdicts"].append(duplicate)
    verdict = validate_semantic_verdict_output(payload, _case())
    assert verdict["validation"]["valid"]
    assert verdict["validation"]["accepted_verdict_count"] == 1
    assert verdict["validation"]["dropped_verdicts"][0]["errors"] == ["与更高优先级问题重复"]


def test_same_starting_claim_is_merged_across_detector_roots() -> None:
    payload = _valid_payload()
    duplicate = dict(payload["verdicts"][0])
    duplicate["issue"] = "工程安全经验照亮了风险，但不能充当金融市场六个月内崩溃的专业证明。"
    duplicate["from_claim"] = "工程事故中的小扰动会触发连锁故障，因此股市这个复杂系统也会崩溃。"
    duplicate["source_detector_ids"] = ["expert_failure"]
    payload["verdicts"].append(duplicate)
    case = _case()
    case["detected_detector_ids"].append("expert_failure")
    verdict = validate_semantic_verdict_output(payload, case)
    assert verdict["validation"]["accepted_verdict_count"] == 1
    assert verdict["validation"]["dropped_verdicts"][0]["errors"] == ["与更高优先级问题重复"]


def test_user_facing_verdicts_are_capped_at_two_roots() -> None:
    payload = _valid_payload()
    second = dict(payload["verdicts"][0])
    second["issue"] = "它只采信支持美股六个月内破灭的材料，使结论信心超过证据。"
    second["from_claim"] = "宁可高估尾部风险。"
    second["evidence_refs"] = ["V1", "C1"]
    second["source_detector_ids"] = ["bias_heuristics"]
    third = dict(payload["verdicts"][0])
    third["issue"] = "它承认低估了估值、杠杆和市场流动性，却仍断定美股泡沫会在六个月内破灭。"
    third["from_claim"] = "估值、杠杆和市场流动性。"
    third["evidence_refs"] = ["U1", "C1"]
    third["source_detector_ids"] = ["emotion_reason"]
    payload["verdicts"].extend([second, third])
    case = _case()
    case["detected_detector_ids"].extend(["bias_heuristics", "emotion_reason"])
    verdict = validate_semantic_verdict_output(payload, case)
    assert verdict["validation"]["accepted_verdict_count"] == 2
    assert any("优先级低于前两个" in reason for item in verdict["validation"]["dropped_verdicts"] for reason in item["errors"])


def test_fallback_still_names_the_actual_conclusion() -> None:
    verdict = build_fallback_semantic_verdict(_case())
    assert verdict["validation"]["valid"]
    assert "美股泡沫" in verdict["headline"]
    assert verdict["verdicts"][0]["evidence_refs"][-1] == "C1" or "C1" in verdict["verdicts"][0]["evidence_refs"]


def test_no_candidate_case_records_no_finding_without_calling_model() -> None:
    case = _case()
    case["diagnostic_candidates"] = []
    case["detected_detector_ids"] = []

    record = _run_one_semantic_verdict(
        case,
        lambda: (_ for _ in ()).throw(AssertionError("model must not be called")),
        "deepseek-v4-pro",
        8192,
        2,
    )

    assert record["status"] == "no_finding"
    assert record["attempts"] == 0
    assert record["verdict"]["validation"]["valid"]
    assert not record["verdict"]["verdicts"]


def test_validated_fast_mode_disables_thinking_then_escalates_on_invalid_output() -> None:
    case = _case()
    valid = _valid_payload(case)
    calls: list[dict] = []

    class FakeClient:
        def chat(self, _messages, **kwargs):  # noqa: ANN001, ANN003
            calls.append(kwargs)
            payload = {} if len(calls) == 1 else valid
            return SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
                raw={"choices": [{"finish_reason": "stop"}]},
            )

    record = _run_one_semantic_verdict(
        case,
        FakeClient,
        "fixture-model",
        8192,
        0,
        "validated_fast_v1",
    )

    assert record["status"] == "succeeded"
    assert record["attempts"] == 2
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["thinking"] == {"type": "enabled"}
    assert calls[1]["reasoning_effort"] == "high"
    assert record["attempt_log"][0]["thinking_mode"] == "disabled"
    assert record["attempt_log"][1]["thinking_mode"] == "enabled"


def test_semantic_fast_mode_has_an_isolated_checkpoint_fingerprint() -> None:
    case = _case()
    assert semantic_case_fingerprint(case) != semantic_case_fingerprint(
        case,
        execution_mode="validated_fast_v1",
    )


def test_fallback_record_can_recover_after_validator_upgrade() -> None:
    case = _case()
    payload = _valid_payload(case)
    record = {
        "person_id": case["person_id"],
        "person_name": case["person_name"],
        "status": "fallback",
        "case": case,
        "verdict": build_fallback_semantic_verdict(case),
        "errors": ["旧校验器误判"],
        "raw": {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
    }
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "person_1.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        loaded = load_semantic_verdict_records(directory)
    assert loaded[0]["status"] == "succeeded"
    assert loaded[0]["recovered_after_validator_upgrade"]


def test_fake_live_batch_attaches_and_scores_high() -> None:
    case = _case()
    payload = _valid_payload(case)

    class FakeClient:
        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003 - lightweight protocol fixture.
            del messages, kwargs
            return SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False),
                usage={"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
                raw={"choices": [{"finish_reason": "stop"}]},
            )

    diagnosis = {
        "readiness_evaluation": {"score": 100},
        "cognitive_chain_evaluation": {"score": 100},
        "cognitive_chain_scans": [_scan()],
    }
    with tempfile.TemporaryDirectory() as directory:
        records = run_semantic_verdict_batch(
            QUESTION,
            [_persona()],
            [_ledger()],
            [{"person_id": "person_1", "output": _output()}],
            diagnosis,
            output_dir=Path(directory),
            model="fixture-model",
            concurrency=1,
            retries=0,
            client_factory=FakeClient,
        )
    assert len(records) == 1
    assert records[0]["status"] == "succeeded"
    attach_semantic_verdicts(diagnosis, records)
    diagnosis["semantic_verdict_evaluation"] = evaluate_semantic_verdict_strength(diagnosis)
    diagnosis["meta_model_overall_evaluation"] = evaluate_meta_model_overall(diagnosis)
    assert diagnosis["semantic_verdict_evaluation"]["score"] >= 95
    assert diagnosis["meta_model_overall_evaluation"]["score"] >= 95
