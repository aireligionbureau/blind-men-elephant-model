from __future__ import annotations

from copy import deepcopy

from bme_model.meta_model.forensics import DETECTOR_IDS, run_cognitive_chain_scan
from bme_model.meta_model.lens_registry import load_lenses


QUESTION = "当前证据是否支持扩大这项技术部署？"


def _base_person() -> dict:
    return {
        "id": "calibration_person",
        "name": "校准观察者",
        "role_summary": "在明确边界内比较多类证据。",
        "cognitive_frames": ["系统思维", "归纳"],
        "expertise_strong": ["政策评估", "统计分析"],
        "expertise_weak": ["临床医学"],
        "values": {"safety": 0.34, "efficiency": 0.33, "fairness": 0.33},
        "time_horizon": "3-5年",
        "risk_attitude": "风险中性",
        "analysis_levels": ["个体", "组织", "制度"],
        "hidden_biases": [],
        "information_filter": {
            "trusted_sources": ["同行评审论文", "官方统计"],
            "distrusted_sources": ["匿名帖子"],
            "preferred_evidence": ["长期统计", "对照研究"],
            "ignored_evidence": ["未经核验轶事"],
        },
        "temperament": "冷静",
    }


def _entry(index: int, *, source_type: str, evidence_type: str) -> dict:
    return {
        "claim": f"经核验材料 {index} 只支持一个有边界的局部观察。",
        "source": f"https://example.com/{index}",
        "source_title": f"材料 {index}",
        "source_type": source_type,
        "evidence_type": evidence_type,
        "retrieval_layer": "external_search",
        "verification_status": "verified_external",
        "trust_reason": "样本清楚、方法透明、原始数据可核验。",
        "used_for": "检验局部命题",
        "confidence": 0.72,
    }


def _decision(index: int, *, source_type: str, evidence_type: str, decision: str = "accept") -> dict:
    return {
        "decision": decision,
        "trust_score": 0.72 if decision == "accept" else 0.25,
        "reasons": ["样本清楚", "方法透明", "原始数据可核验"],
        "result": {
            "source_type": source_type,
            "evidence_type": evidence_type,
            "title": f"候选材料 {index}",
            "snippet": "这条材料说明适用边界，并报告反例。",
        },
    }


def _base_ledger() -> dict:
    specs = [
        ("同行评审论文", "对照研究"),
        ("官方统计", "长期统计"),
        ("审计报告", "成本数据"),
        ("案例数据库", "失败案例"),
    ]
    entries = [_entry(index, source_type=source, evidence_type=evidence) for index, (source, evidence) in enumerate(specs)]
    decisions = [_decision(index, source_type=source, evidence_type=evidence) for index, (source, evidence) in enumerate(specs)]
    return {
        "person_id": "calibration_person",
        "question": QUESTION,
        "search_strategy": {
            "preferred_queries": ["技术部署 长期统计 反例", "技术部署 成本 失败案例"],
        },
        "source_decisions": decisions,
        "accepted_sources": decisions,
        "rejected_sources": [],
        "ignored_sources": [],
        "entries": entries,
    }


def _base_output() -> dict:
    return {
        "person_id": "calibration_person",
        "core_assumptions": ["试点数据能够代表当前限定地区。", "制度条件在观察期内基本稳定。"],
        "reasoning_path": [
            "对照研究支持限定地区中的效率改善，但不直接支持全国推广。",
            "另一方面，失败案例说明执行质量会改变结果，因此结论取决于治理条件。",
            "替代解释是样本选择造成表面改善，现有审计材料只能部分排除。",
            "所以当前只形成有边界的试点判断，并保留反例和不确定性。",
        ],
        "value_judgements": ["安全、效率与公平分别评价，不把价值偏好写成事实。"],
        "conclusion": "当前证据只支持在相似条件下继续小规模试点，不能据此推出全面部署。",
        "confidence": 0.58,
        "what_i_underweighted": ["不同地区的执行差异仍需补充。"],
        "what_would_change_my_mind": ["如果连续两期官方统计显示失败率低于既定阈值，并且独立审计排除样本选择，我会提高置信度。"],
    }


def _scan(person: dict | None = None, ledger: dict | None = None, output: dict | None = None) -> dict:
    result, _ = run_cognitive_chain_scan(
        QUESTION,
        person or _base_person(),
        ledger or _base_ledger(),
        output or _base_output(),
        load_lenses(),
    )
    return result


def _status(scan: dict, detector_id: str) -> str:
    return next(
        item["status"]
        for item in scan["detector_results"]
        if item["detector_id"] == detector_id
    )


def _prior_ledger() -> dict:
    ledger = _base_ledger()
    entries = []
    decisions = []
    for index, evidence_type in enumerate(["事故案例", "历史类比", "系统失效分析"]):
        entry = _entry(index, source_type="模型先验", evidence_type=evidence_type)
        entry["retrieval_layer"] = "model_prior"
        entry["verification_status"] = "unverified_model_prior"
        entries.append(entry)
        decision = _decision(index, source_type="模型先验", evidence_type=evidence_type)
        decision["reasons"] = ["模型先验", "证据类型符合原有偏好"]
        decisions.append(decision)
    ledger.update(
        {
            "entries": entries,
            "source_decisions": decisions,
            "accepted_sources": decisions,
            "ignored_sources": [
                _decision(8, source_type="官方统计", evidence_type="市场数据", decision="ignore"),
                _decision(9, source_type="反方研究", evidence_type="基准率", decision="ignore"),
                _decision(10, source_type="独立审计", evidence_type="反例", decision="ignore"),
                _decision(11, source_type="长期追踪", evidence_type="长期统计", decision="ignore"),
            ],
        }
    )
    return ledger


def test_every_person_is_checked_by_all_ten_detectors():
    scan = _scan()

    assert [item["detector_id"] for item in scan["detector_results"]] == list(DETECTOR_IDS)
    assert scan["chain_coverage"]["score"] == 100.0
    assert "透镜" not in scan["plain_language_summary"]["headline"]
    assert "检测器" not in scan["plain_language_summary"]["headline"]


def test_clean_bounded_chain_is_not_forced_into_shadow_labels():
    scan = _scan()

    for detector_id in DETECTOR_IDS:
        expected = "insufficient_evidence" if detector_id == "noise" else "not_detected"
        assert _status(scan, detector_id) == expected


def test_bias_detector_requires_manifestation_not_identity_label():
    person = _base_person()
    person["hidden_biases"] = ["确认偏误", "过度自信"]
    output = _base_output()
    output["confidence"] = 0.86

    assert _status(_scan(person, _prior_ledger(), output), "bias_heuristics") == "detected"
    assert _status(_scan(person, _base_ledger(), _base_output()), "bias_heuristics") == "not_detected"


def test_noise_detector_refuses_single_run_and_detects_repeat_drift():
    assert _status(_scan(), "noise") == "insufficient_evidence"

    output = _base_output()
    output["repeat_judgments"] = [
        {"conclusion": "应该立即全面部署", "confidence": 0.85},
        {"conclusion": "应该暂停部署", "confidence": 0.42},
        {"conclusion": "只适合继续试点", "confidence": 0.61},
    ]
    assert _status(_scan(output=output), "noise") == "detected"


def test_emotion_detector_separates_human_cost_from_probability():
    person = _base_person()
    person["temperament"] = "悲观且恐惧风险"
    ledger = _base_ledger()
    ledger["ignored_sources"] = [
        _decision(20, source_type="工人访谈", evidence_type="尊严与伤害叙事", decision="ignore")
    ]
    ledger["source_decisions"].extend(ledger["ignored_sources"])
    output = _base_output()
    output["value_judgements"] = []
    output["what_i_underweighted"] = ["工人失业后的家庭伤害和尊严成本。"]

    scan, _ = run_cognitive_chain_scan(
        "自动化裁员会怎样影响工人和家庭？",
        person,
        ledger,
        output,
        load_lenses(),
    )
    assert _status(scan, "emotion_reason") == "detected"
    assert _status(_scan(), "emotion_reason") == "not_detected"


def test_interpreter_detector_needs_smooth_story_and_closed_exit():
    output = _base_output()
    output["reasoning_path"] = [
        "安全最重要。",
        "已有材料符合安全优先。",
        "所以安全结论是正确的。",
    ]
    output["what_would_change_my_mind"] = []
    output["conclusion"] = "safety 要求立即停止，结论已经很清楚。"
    output["confidence"] = 0.83

    assert _status(_scan(ledger=_prior_ledger(), output=output), "interpreter") == "detected"
    assert _status(_scan(), "interpreter") == "not_detected"


def test_social_detector_distinguishes_identity_trust_from_method_quality():
    ledger = _base_ledger()
    for item in ledger["accepted_sources"]:
        item["reasons"] = ["来源类型匹配信任列表", "权威机构发布"]
    rejected = _decision(30, source_type="匿名帖子", evidence_type="反方材料", decision="reject")
    rejected["reasons"] = ["来源属于预设低信任群体"]
    ledger["rejected_sources"] = [rejected]
    ledger["source_decisions"].append(rejected)

    assert _status(_scan(ledger=ledger), "social_influence") == "detected"
    assert _status(_scan(), "social_influence") == "not_detected"


def test_argument_detector_finds_missing_bridges_but_accepts_bounded_chain():
    output = _base_output()
    output["reasoning_path"] = ["数据显示增长。", "因此增长必然导致全面成功。", "所以结论确定。"]
    output["conclusion"] = "增长证明全面部署一定成功。"
    output["confidence"] = 0.88

    assert _status(_scan(ledger=_prior_ledger(), output=output), "argument_forensics") == "detected"
    assert _status(_scan(), "argument_forensics") == "not_detected"


def test_expert_detector_preserves_domain_skill_and_catches_domain_expansion():
    output = _base_output()
    output["conclusion"] = "政策统计模型已经证明这项技术将在所有地区长期全面成功。"
    output["confidence"] = 0.82

    assert _status(_scan(output=output), "expert_failure") == "detected"
    assert _status(_scan(), "expert_failure") == "not_detected"


def test_paradigm_detector_needs_closed_problem_definition():
    ledger = _base_ledger()
    ledger["entries"] = ledger["entries"][:1]
    ignored = [
        _decision(40, source_type="受影响者访谈", evidence_type="质性叙事", decision="ignore"),
        _decision(41, source_type="伦理审查", evidence_type="权利影响", decision="ignore"),
    ]
    ledger["ignored_sources"] = ignored
    ledger["source_decisions"] = ledger["accepted_sources"][:1] + ignored
    output = _base_output()
    output["value_judgements"] = []
    output["conclusion"] = "只要 efficiency 提高就应部署，效率就是唯一成功标准。"

    assert _status(_scan(ledger=ledger, output=output), "paradigm") == "detected"
    assert _status(_scan(), "paradigm") == "not_detected"


def test_ecology_detector_protects_adaptive_experience_and_catches_mismatch():
    mismatch = _base_output()
    mismatch["reasoning_path"] = [
        "历史案例与过去经验都表明这套一线规则有效。",
        "当前技术类似旧系统，所以直接照搬即可。",
    ]
    mismatch["conclusion"] = "过去经验已经足够，当前也应采用同一规则。"
    assert _status(_scan(output=mismatch), "ecological_rationality") == "detected"

    matched = _base_output()
    matched["reasoning_path"] = [
        "这条地方实践规则来自多年持续记录和反复验证的反馈。",
        "当前环境与原环境存在结构变化，因此明确限定边界和失效条件。",
    ]
    assert _status(_scan(output=matched), "ecological_rationality") == "not_detected"


def test_local_whole_detector_is_a_peer_and_checks_scope_boundary():
    output = _base_output()
    output["conclusion"] = "这些个案证明整体系统将在短期内必然失败。"
    output["confidence"] = 0.9

    positive = _scan(ledger=_prior_ledger(), output=output)
    clean = _scan()
    assert _status(positive, "local_whole_overreach") == "detected"
    assert _status(clean, "local_whole_overreach") == "not_detected"
    assert list(DETECTOR_IDS).index("local_whole_overreach") == 9


def test_detected_findings_are_grounded_falsifiable_and_plain():
    person = deepcopy(_base_person())
    person["hidden_biases"] = ["确认偏误", "过度自信"]
    output = _base_output()
    output["reasoning_path"] = ["历史案例很危险。", "因此整体系统必然很快失败。", "所以必须立即行动。"]
    output["conclusion"] = "整体系统很快必然失败。"
    output["confidence"] = 0.91
    scan = _scan(person, _prior_ledger(), output)

    detected = [item for item in scan["detector_results"] if item["status"] == "detected"]
    assert detected
    for finding in detected:
        assert finding["observed_evidence"]
        assert finding["mechanism"]
        assert finding["consequence_for_conclusion"]
        assert finding["competing_explanations"]
        assert finding["falsification_test"]
        assert finding["preserved_valid_part"]

    summary = scan["plain_language_summary"]
    assert 1 <= len(summary["key_problems"]) <= 3
    assert all("detector" not in str(item).lower() for item in summary["key_problems"])
