from __future__ import annotations

from collections import Counter
from typing import Any

from .lens_registry import LensCard


DETECTOR_IDS = (
    "bias_heuristics",
    "noise",
    "emotion_reason",
    "interpreter",
    "social_influence",
    "argument_forensics",
    "expert_failure",
    "paradigm",
    "ecological_rationality",
    "local_whole_overreach",
)

DETECTOR_NAMES = {
    "bias_heuristics": "偏差与启发式检测器",
    "noise": "判断噪声检测器",
    "emotion_reason": "情绪与价值标记检测器",
    "interpreter": "事后合理化检测器",
    "social_influence": "社会影响检测器",
    "argument_forensics": "论证法医检测器",
    "expert_failure": "专家失效检测器",
    "paradigm": "范式封闭检测器",
    "ecological_rationality": "生态理性与错配检测器",
    "local_whole_overreach": "局部整体越界检测器",
}

CHAIN_STAGES = (
    "problem_framing",
    "information_filter",
    "retrieval",
    "evidence_selection",
    "assumption",
    "reasoning",
    "value_evaluation",
    "conclusion",
    "self_correction",
)


def run_cognitive_chain_scan(
    question: str,
    persona: dict[str, Any],
    ledger: dict[str, Any],
    output: dict[str, Any],
    lenses: dict[str, LensCard],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Examine one observer without using any other observer's output."""
    case = _build_case(question, persona, ledger, output, lenses)
    findings = [
        _detect_bias_heuristics(case),
        _detect_noise(case),
        _detect_emotion_reason(case),
        _detect_interpreter(case),
        _detect_social_influence(case),
        _detect_argument_forensics(case),
        _detect_expert_failure(case),
        _detect_paradigm(case),
        _detect_ecological_rationality(case),
        _detect_local_whole_overreach(case),
    ]

    shadow_specs: list[dict[str, Any]] = []
    for finding in findings:
        shadow_spec = finding.pop("_shadow_spec", None)
        if finding["status"] == "detected" and shadow_spec:
            shadow_spec["chain_diagnosis"] = {
                key: finding[key]
                for key in (
                    "detector_id",
                    "detector_name",
                    "chain_locations",
                    "observed_evidence",
                    "plain_language_diagnosis",
                    "mechanism",
                    "consequence_for_conclusion",
                    "competing_explanations",
                    "questions_to_distinguish",
                    "falsification_test",
                    "preserved_valid_part",
                    "diagnostic_confidence",
                    "evidence_sufficiency",
                )
            }
            shadow_specs.append(shadow_spec)

    chain_trace = _build_chain_trace(case)
    detected = [item for item in findings if item["status"] == "detected"]
    insufficient = [item for item in findings if item["status"] == "insufficient_evidence"]
    breakpoints = [
        {
            "detector_id": item["detector_id"],
            "where": item["chain_locations"],
            "what_happened": item["plain_language_diagnosis"],
            "effect": item["consequence_for_conclusion"],
        }
        for item in detected
    ]

    chain_scan = {
        "person_id": persona["id"],
        "person_name": persona.get("name", persona["id"]),
        "scope_rule": "只检查该数字人自身的完整思考链；未使用其他数字人的判断。",
        "chain_coverage": chain_trace["coverage"],
        "chain_specimens": chain_trace["stages"],
        "detector_results": findings,
        "chain_breakpoints": breakpoints,
        "plain_language_summary": _build_plain_language_summary(findings),
        "scan_summary": {
            "detected_count": len(detected),
            "not_detected_count": sum(item["status"] == "not_detected" for item in findings),
            "insufficient_evidence_count": len(insufficient),
            "detected_detectors": [item["detector_id"] for item in detected],
            "unresolved_detectors": [item["detector_id"] for item in insufficient],
            "strongest_findings": [
                item["plain_language_diagnosis"]
                for item in sorted(
                    detected,
                    key=lambda candidate: candidate["diagnostic_confidence"],
                    reverse=True,
                )[:3]
            ],
        },
        "scan_limits": [
            "检出表示思考链中存在可观察的阴影机制，不等于该数字人的最终结论必然错误。",
            "暂未检出不等于机制不存在，只表示现有材料没有达到证明负担。",
            "判断噪声需要同一认知身份证的重复独立判断；单次输出不能硬判噪声。",
            "本报告不进行跨数字人串联，也不据此直接宣布真相轮廓。",
        ],
    }
    return chain_scan, shadow_specs


def build_chain_detector_audit(chain_scans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for detector_id in DETECTOR_IDS:
        results = [
            finding
            for chain_scan in chain_scans
            for finding in chain_scan.get("detector_results", [])
            if finding.get("detector_id") == detector_id
        ]
        statuses = Counter(item.get("status", "unknown") for item in results)
        grounded = sum(bool(item.get("observed_evidence")) for item in results)
        falsifiable = sum(bool(item.get("falsification_test")) for item in results)
        audit.append(
            {
                "detector_id": detector_id,
                "detector_name": DETECTOR_NAMES[detector_id],
                "examined_person_count": len(results),
                "status_counts": dict(statuses),
                "evidence_grounding_rate": _ratio(grounded, len(results)),
                "falsifiability_rate": _ratio(falsifiable, len(results)),
                "rule": "每个数字人都必须经过本检测器；证据不足必须明说，不得用标签补空白。",
            }
        )
    return audit


def _build_plain_language_summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
    detected = sorted(
        (item for item in findings if item["status"] == "detected"),
        key=lambda item: item["diagnostic_confidence"],
        reverse=True,
    )
    key_problems = []
    used_groups: set[str] = set()
    for finding in detected:
        display_group = _display_group(finding["detector_id"])
        if display_group in used_groups:
            continue
        observations = finding.get("observed_evidence", [])
        key_problems.append(
            {
                "problem": finding["plain_language_diagnosis"],
                "where_it_appeared": _where_summary(observations),
                "how_it_affected_thinking": finding["consequence_for_conclusion"],
                "what_can_still_be_kept": finding["preserved_valid_part"],
            }
        )
        used_groups.add(display_group)
        if len(key_problems) == 3:
            break

    unresolved = [
        item["detector_name"]
        for item in findings
        if item["status"] == "insufficient_evidence"
    ]
    if key_problems:
        headline = _short(
            f"{key_problems[0]['where_it_appeared']} 这说明：{key_problems[0]['problem']}",
            180,
        )
    elif unresolved:
        headline = "现有材料还不足以指出确定的思考问题；元模型选择保留未知，而不是硬贴标签。"
    else:
        headline = "现有材料没有检出足以改变结论权重的明显思考问题。"
    return {
        "headline": headline,
        "key_problems": key_problems,
        "unresolved_checks": unresolved,
        "display_rule": "用户层只展示最关键的具体问题；检测器名称、经典术语和完整审计留在后台。",
    }


def _display_group(detector_id: str) -> str:
    groups = {
        "bias_heuristics": "information_weighting",
        "social_influence": "information_weighting",
        "argument_forensics": "reasoning_integrity",
        "interpreter": "reasoning_integrity",
        "local_whole_overreach": "reasoning_integrity",
        "expert_failure": "frame_boundary",
        "paradigm": "frame_boundary",
        "ecological_rationality": "frame_boundary",
        "emotion_reason": "human_value_weighting",
        "noise": "judgment_stability",
    }
    return groups.get(detector_id, detector_id)


def _where_summary(observations: list[dict[str, str]]) -> str:
    facts = [str(item.get("fact", "")).strip() for item in observations if item.get("fact")]
    if not facts:
        return "现有记录没有留下足够清楚的发生位置。"
    return _short("；".join(facts[:2]), 280)


def _build_case(
    question: str,
    persona: dict[str, Any],
    ledger: dict[str, Any],
    output: dict[str, Any],
    lenses: dict[str, LensCard],
) -> dict[str, Any]:
    output = output if isinstance(output, dict) else {}
    entries = [item for item in ledger.get("entries", []) if isinstance(item, dict)]
    decisions = [item for item in ledger.get("source_decisions", []) if isinstance(item, dict)]
    accepted = [item for item in decisions if item.get("decision") == "accept"]
    rejected = [item for item in decisions if item.get("decision") == "reject"]
    ignored = [item for item in decisions if item.get("decision") == "ignore"]
    if not accepted:
        accepted = [item for item in ledger.get("accepted_sources", []) if isinstance(item, dict)]
    if not rejected:
        rejected = [item for item in ledger.get("rejected_sources", []) if isinstance(item, dict)]
    if not ignored:
        ignored = [item for item in ledger.get("ignored_sources", []) if isinstance(item, dict)]

    assumptions = _string_list(output.get("core_assumptions"))
    reasoning = _string_list(output.get("reasoning_path"))
    value_judgements = _string_list(output.get("value_judgements"))
    underweighted = _string_list(output.get("what_i_underweighted"))
    change_conditions = _string_list(output.get("what_would_change_my_mind"))
    conclusion = str(output.get("conclusion", "") or "").strip()
    confidence = output.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, int | float) else None
    retrieval_layers = [str(item.get("retrieval_layer", "")) for item in entries]
    verified_external = [
        item
        for item in entries
        if item.get("retrieval_layer") not in {"model_prior", "mock_search", None}
        and item.get("verification_status")
        not in {"unverified_model_prior", "synthetic_fixture", None}
    ]
    search_strategy = ledger.get("search_strategy", {})
    queries = _string_list(search_strategy.get("preferred_queries") if isinstance(search_strategy, dict) else [])
    information_filter = persona.get("information_filter", {})
    if not isinstance(information_filter, dict):
        information_filter = {}

    return {
        "question": question,
        "persona": persona,
        "ledger": ledger,
        "output": output,
        "lenses": lenses,
        "entries": entries,
        "decisions": decisions,
        "accepted": accepted,
        "rejected": rejected,
        "ignored": ignored,
        "assumptions": assumptions,
        "reasoning": reasoning,
        "value_judgements": value_judgements,
        "underweighted": underweighted,
        "change_conditions": change_conditions,
        "conclusion": conclusion,
        "confidence": confidence_value,
        "retrieval_layers": retrieval_layers,
        "verified_external": verified_external,
        "queries": queries,
        "information_filter": information_filter,
        "all_output_text": " ".join(
            assumptions + reasoning + value_judgements + underweighted + change_conditions + [conclusion]
        ),
    }


def _detect_bias_heuristics(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    persona = case["persona"]
    hidden_biases = _string_list(persona.get("hidden_biases"))
    confidence = case["confidence"]
    model_prior_count = case["retrieval_layers"].count("model_prior")
    entry_count = len(case["entries"])

    if confidence is not None and confidence >= 0.68 and len(case["verified_external"]) <= 1:
        observations.append(
            _observation(
                "conclusion",
                f"自报置信度为 {confidence:.2f}，但经核验的外部证据只有 {len(case['verified_external'])} 条。",
                case["conclusion"],
            )
        )
    if entry_count and model_prior_count / entry_count >= 0.6:
        observations.append(
            _observation(
                "evidence_selection",
                f"{entry_count} 条采信材料中有 {model_prior_count} 条是模型先验，先验占比过高。",
            )
        )
    if len(case["ignored"]) + len(case["rejected"]) > len(case["accepted"]):
        observations.append(
            _observation(
                "evidence_selection",
                f"采信 {len(case['accepted'])} 条，忽略或拒绝 {len(case['ignored']) + len(case['rejected'])} 条，选择压力明显。",
            )
        )
    base_rate_text = _decision_text(case["ignored"] + case["rejected"])
    if _contains(base_rate_text, ["基准率", "长期统计", "分布", "样本", "市场数据"]):
        observations.append(
            _observation(
                "information_filter",
                "被挡在推理链外的材料包含基准率、统计分布或市场数据。",
                base_rate_text,
            )
        )

    has_manifestation = len(observations) >= 2 or (
        bool(observations)
        and any(token in " ".join(hidden_biases) for token in ["过度自信", "确认偏误", "锚定", "可得性"])
    )
    output_present = bool(case["conclusion"] and case["reasoning"])
    status = "detected" if has_manifestation else ("not_detected" if output_present else "insufficient_evidence")
    diagnosis = (
        "它先相信的方向，后来又被自己的搜索和证据选择不断加强；同时，它给出的信心超过了外部证据能够支撑的程度。"
        if status == "detected"
        else _negative_diagnosis(status, "偏差与启发式在实际判断链中的稳定显影")
    )
    return _finding(
        case,
        "bias_heuristics",
        status=status,
        observations=observations,
        chain_locations=["information_filter", "evidence_selection", "conclusion"],
        diagnosis=diagnosis,
        mechanism="初始先验或显著线索先确定方向，后续搜索与推理再为该方向补充理由。",
        consequence="可能使结论方向稳定偏斜，并让置信度高于证据本身允许的程度。",
        competing=["来源差异也可能来自合理的质量筛选，而非确认偏误。", "信息不足时依赖先验并不自动构成错误。"],
        questions=["移除最早先验后，搜索词和结论是否改变？", "补入基准率与反向证据后，置信度是否下降？"],
        falsification="用相同身份证重跑一次，但先呈现基准率和最强反证；若结论方向与置信度基本不变且理由充分，本诊断应降级。",
        preserved="保留它对高显著性风险或机会的敏感度，但不把这种敏感度直接当作概率证据。",
        confidence=_diagnostic_confidence(observations, base=0.46, cap=0.9),
        missing=[] if output_present else ["完整推理路径", "结论与置信度"],
        shadow_type="distorted",
        source_stage="evidence_selection",
        likely_causes=hidden_biases or ["启发式替代", "先验锚定"],
        related_parameters=["hidden_biases", "entries", "ignored_sources", "confidence"],
        bias_direction="向先验和显著材料所支持的方向倾斜",
        puzzle_use="constraint_band",
    )


def _detect_noise(case: dict[str, Any]) -> dict[str, Any]:
    output = case["output"]
    repeats = output.get("repeat_judgments") or output.get("judgment_repeats") or []
    repeats = repeats if isinstance(repeats, list) else []
    observations: list[dict[str, str]] = []
    status = "insufficient_evidence"
    diagnosis = "只有一次判断，无法区分稳定偏差与随机漂移；此处必须留白，不能把无法解释的部分硬叫作噪声。"
    missing = ["同一认知身份证、相同材料下的独立重复判断", "随机化证据顺序后的复测结果"]

    if len(repeats) >= 2:
        repeat_texts = [str(item.get("conclusion", item)) if isinstance(item, dict) else str(item) for item in repeats]
        normalized = {"".join(text.lower().split()) for text in repeat_texts if text.strip()}
        confidences = [
            float(item["confidence"])
            for item in repeats
            if isinstance(item, dict) and isinstance(item.get("confidence"), int | float)
        ]
        confidence_span = max(confidences) - min(confidences) if len(confidences) >= 2 else 0.0
        observations.append(
            _observation(
                "conclusion",
                f"获得 {len(repeats)} 次独立复测，结论表述有 {len(normalized)} 种，置信度跨度为 {confidence_span:.2f}。",
            )
        )
        if len(normalized) >= 2 or confidence_span >= 0.25:
            status = "detected"
            diagnosis = "同一数字人在相近条件下发生明显漂移，这部分更像判断系统抖动，暂时不能反演成有方向的偏见。"
        else:
            status = "not_detected"
            diagnosis = "重复判断较稳定，现有复测没有检出明显判断噪声。"
        missing = []

    return _finding(
        case,
        "noise",
        status=status,
        observations=observations,
        chain_locations=["reasoning", "conclusion"],
        diagnosis=diagnosis,
        mechanism="相同判断条件下，随机初始线索、表达顺序或局部权重造成无稳定方向的变异。",
        consequence="若误把噪声当成有结构偏见，元模型会为随机抖动编造过深解释。",
        competing=["稳定的价值、范式或信息过滤差异不属于噪声。", "结论措辞不同但实质立场一致，不应算作漂移。"],
        questions=["重跑后差异是否仍沿同一方向？", "改变证据呈现顺序会不会让判断翻转？"],
        falsification="至少进行三次独立复测；若立场、关键理由和置信区间稳定重合，就不应诊断为噪声。",
        preserved="噪声本身不提供偏见方向，只提示扩大不确定区间并重复采样。",
        confidence=0.82 if status in {"detected", "not_detected"} else 0.2,
        missing=missing,
        shadow_type="distorted",
        source_stage="conclusion",
        likely_causes=["判断条件不稳定", "证据顺序效应", "随机权重漂移"],
        related_parameters=["repeat_judgments", "confidence"],
        bias_direction="无稳定方向，应扩大误差范围",
        puzzle_use="downweight",
    )


def _detect_emotion_reason(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    temperament = str(case["persona"].get("temperament", ""))
    ignored_text = _decision_text(case["ignored"] + case["rejected"]) + " " + " ".join(case["underweighted"])
    human_tokens = ["伤害", "尊严", "恐惧", "访谈", "工人", "患者", "家庭", "质性", "权利", "处境"]
    human_context = _contains(case["question"] + " " + ignored_text, human_tokens)
    strong_temperament = _contains(temperament, ["悲观", "乐观", "进攻", "怀疑", "道义", "恐惧"])
    if human_context and _contains(ignored_text, human_tokens):
        observations.append(
            _observation(
                "information_filter",
                "被忽略或自报低估的材料包含受影响者体验、伤害、尊严或质性证据。",
                ignored_text,
            )
        )
    if strong_temperament and case["conclusion"]:
        observations.append(
            _observation(
                "value_evaluation",
                f"气质基调为“{temperament}”，并参与了风险、机会或伤害的显著性排序。",
                case["conclusion"],
            )
        )
    if human_context and not case["value_judgements"]:
        observations.append(
            _observation(
                "value_evaluation",
                "问题涉及人类处境，但输出没有单独记录价值判断，事实判断与价值赋权可能混在一起。",
            )
        )

    output_present = bool(case["conclusion"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if output_present else "insufficient_evidence")
    diagnosis = (
        "情绪或气质正在改变哪些后果最醒目，或者相反，人类体验被当成低质量材料挡在了推理链外。"
        if status == "detected"
        else _negative_diagnosis(status, "情绪放大或人类成本被压低")
    )
    return _finding(
        case,
        "emotion_reason",
        status=status,
        observations=observations,
        chain_locations=["information_filter", "value_evaluation", "conclusion"],
        diagnosis=diagnosis,
        mechanism="情绪与身体性标记为后果赋予显著性；标记过强会放大风险，标记被压掉会吞没人类成本。",
        consequence="可能改变风险比例感，或让结论遗漏指标无法表达的真实代价。",
        competing=["质性材料可能确实缺少事实核验。", "悲观或乐观语气不等于概率判断已经失真。"],
        questions=["情绪承载的是事实主张还是价值信号？", "把受影响者体验补回后，行动建议是否改变？"],
        falsification="把事实概率与价值代价分层重写；若加入人类体验后权衡不变，且气质不改变概率估计，本诊断应降级。",
        preserved="保留情绪提示出的伤害、尊严或机会维度，但不把情绪强度直接换算成事实概率。",
        confidence=_diagnostic_confidence(observations, base=0.42, cap=0.86),
        missing=[] if output_present else ["价值判断", "结论"],
        shadow_type="value",
        source_stage="value_evaluation",
        likely_causes=["情绪显著性", "气质放大", "躯体标记缺失"],
        related_parameters=["temperament", "value_judgements", "ignored_sources", "what_i_underweighted"],
        bias_direction="放大符合气质的后果，或压低不可量化的人类成本",
        puzzle_use="constraint_band",
    )


def _detect_interpreter(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    reasoning_text = " ".join(case["reasoning"])
    has_counterweight = _contains(reasoning_text, ["但是", "然而", "反例", "替代", "另一方面", "不确定", "也可能"])
    change_testability = _change_condition_testability(case["change_conditions"])
    top_value = _top_value(case["persona"].get("values", {}))
    value_named = bool(top_value and _contains(case["all_output_text"], [top_value[0], str(top_value[1])]))

    if len(case["reasoning"]) >= 3 and not has_counterweight:
        observations.append(
            _observation(
                "reasoning",
                "推理路径很顺，但没有可见的反例、替代解释或真实权衡点。",
                reasoning_text,
            )
        )
    if case["conclusion"] and change_testability == "weak":
        observations.append(
            _observation(
                "self_correction",
                "改变主意的条件缺失、抽象或无法触发，叙事缺少真正出口。",
                " ".join(case["change_conditions"]),
            )
        )
    if value_named and len(case["verified_external"]) <= 1:
        observations.append(
            _observation(
                "value_evaluation",
                f"推理明确沿最高价值“{top_value[0]}”展开，而外部核验材料很少。",
                case["all_output_text"],
            )
        )

    output_present = bool(case["conclusion"] and case["reasoning"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if output_present else "insufficient_evidence")
    diagnosis = (
        "这条推理更像在为已经形成的方向组织理由：故事很完整，但反证入口和可触发的改判条件偏弱。"
        if status == "detected"
        else _negative_diagnosis(status, "结论先在、理由后补的事后合理化")
    )
    return _finding(
        case,
        "interpreter",
        status=status,
        observations=observations,
        chain_locations=["assumption", "reasoning", "self_correction"],
        diagnosis=diagnosis,
        mechanism="判断方向先由价值、身份或直觉形成，解释器随后把选择过的材料串成连贯故事。",
        consequence="会让叙事流畅度冒充证据强度，并使反证难以真正改变结论。",
        competing=["连贯叙事也可能只是高质量表达。", "价值与结论一致并不证明理由是事后编造。"],
        questions=["删掉这条理由，结论是否仍完全不变？", "改变主意条件能否被现实数据明确触发？"],
        falsification="先列最强反证和替代解释，再独立重做结论；若结论强度随证据合理更新，本诊断应降级。",
        preserved="保留其组织材料的解释框架，但不把叙事完整性当作真实性证明。",
        confidence=_diagnostic_confidence(observations, base=0.4, cap=0.84),
        missing=[] if output_present else ["完整推理路径", "可触发的改判条件"],
        shadow_type="reasoning",
        source_stage="reasoning",
        likely_causes=["事后合理化", "叙事完整性幻觉", "动机性推理"],
        related_parameters=["reasoning_path", "values", "what_would_change_my_mind"],
        bias_direction="让既有结论显得比实际证据更连贯、更稳固",
        puzzle_use="fracture_boundary",
    )


def _detect_social_influence(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    accepted_reason_texts = [" ".join(_string_list(item.get("reasons"))) for item in case["accepted"]]
    social_tokens = ["信任列表", "权威", "主流", "共识", "领先", "行业", "机构", "身份"]
    quality_tokens = ["样本", "方法", "可复现", "原始数据", "同行评审", "数据质量", "核验"]
    social_count = sum(_contains(text, social_tokens) for text in accepted_reason_texts)
    quality_count = sum(_contains(text, quality_tokens) for text in accepted_reason_texts)
    accepted_count = len(case["accepted"])
    distrusted = _string_list(case["information_filter"].get("distrusted_sources"))
    rejected_text = _decision_text(case["rejected"])

    if accepted_count >= 2 and social_count / accepted_count >= 0.6 and quality_count == 0:
        observations.append(
            _observation(
                "evidence_selection",
                f"{accepted_count} 条采信决定中有 {social_count} 条依赖信任身份或社会地位线索，未记录方法质量理由。",
                " ".join(accepted_reason_texts),
            )
        )
    matched_distrust = [source for source in distrusted if source and source in rejected_text]
    if matched_distrust:
        observations.append(
            _observation(
                "information_filter",
                f"被明确排斥的来源与预设低信任群体重合：{'、'.join(matched_distrust[:4])}。",
                rejected_text,
            )
        )
    if _contains(rejected_text, ["工会", "边缘", "监管", "企业", "创业", "受害者"]):
        observations.append(
            _observation(
                "information_filter",
                "被排斥材料带有明显群体身份，来源身份可能先于内容质量影响信任。",
                rejected_text,
            )
        )

    has_selection_data = bool(case["decisions"] or case["accepted"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if has_selection_data else "insufficient_evidence")
    diagnosis = (
        "来源为什么被相信，部分取决于它是谁、属于哪一群体或是否符合预设信任，而不只是它拿出了什么证据。"
        if status == "detected"
        else _negative_diagnosis(status, "权威、身份或同温层替代证据质量")
    )
    return _finding(
        case,
        "social_influence",
        status=status,
        observations=observations,
        chain_locations=["information_filter", "evidence_selection"],
        diagnosis=diagnosis,
        mechanism="权威、身份亲近和群体归属先改变注意与信任，内容随后接受不对称审查。",
        consequence="可能制造信息同温层，并让外群体提供的有效变量在进入推理前消失。",
        competing=["成熟领域的权威来源可能确有更高方法质量。", "低信任来源也可能存在真实的利益冲突。"],
        questions=["去掉来源名称后，采信决定是否改变？", "信任理由能否落到样本、方法和可核验性？"],
        falsification="盲化来源身份后重新评估同一材料；若采信结果稳定且能给出方法质量依据，本诊断应降级。",
        preserved="保留来源筛选带来的专业质量控制，但剥离仅由身份或地位产生的额外信任。",
        confidence=_diagnostic_confidence(observations, base=0.44, cap=0.88),
        missing=[] if has_selection_data else ["逐条来源采信理由", "被拒来源内容"],
        shadow_type="informational",
        source_stage="evidence_selection",
        likely_causes=["权威效应", "群体内偏爱", "身份一致压力"],
        related_parameters=["trusted_sources", "distrusted_sources", "source_decisions"],
        bias_direction="向身份亲近或高地位来源倾斜",
        puzzle_use="negative_space",
    )


def _detect_argument_forensics(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    reasoning_text = " ".join(case["reasoning"])
    conclusion = case["conclusion"]
    verified_count = len(case["verified_external"])
    strong_claim = bool(_scope_flags(conclusion, case["confidence"]))
    causal_claim = _contains(conclusion + " " + reasoning_text, ["导致", "证明", "因此", "所以", "必然", "不可避免"])
    bridge_language = _contains(reasoning_text, ["机制", "中介", "条件", "前提", "如果", "取决于", "反例", "替代解释"])

    if case["assumptions"] and verified_count == 0:
        observations.append(
            _observation(
                "assumption",
                f"列出了 {len(case['assumptions'])} 个核心假设，但没有经核验的外部证据说明哪些假设已成立。",
                " ".join(case["assumptions"]),
            )
        )
    if causal_claim and not bridge_language:
        observations.append(
            _observation(
                "reasoning",
                "出现强因果连接词，但推理中没有清楚写出中间机制、条件或替代解释。",
                reasoning_text,
            )
        )
    if strong_claim and len(case["entries"]) <= 2:
        observations.append(
            _observation(
                "conclusion",
                f"结论范围或确定性较强，但证据账本只有 {len(case['entries'])} 条采信材料。",
                conclusion,
            )
        )
    if case["reasoning"] and not _contains(reasoning_text, ["反例", "另一方面", "替代", "相反", "失败条件"]):
        observations.append(
            _observation(
                "reasoning",
                "推理链没有处理最强反例或替代解释。",
                reasoning_text,
            )
        )

    output_present = bool(conclusion and case["reasoning"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if output_present else "insufficient_evidence")
    diagnosis = (
        "证据、假设和结论之间至少有一处没有被桥接：推理看起来走到了终点，但中间台阶没有交代清楚。"
        if status == "detected"
        else _negative_diagnosis(status, "证据、假设、推理与结论之间的断裂")
    )
    return _finding(
        case,
        "argument_forensics",
        status=status,
        observations=observations,
        chain_locations=["assumption", "reasoning", "conclusion"],
        diagnosis=diagnosis,
        mechanism="隐含或未验证前提承担了推理重量，证据被用于支持比自身能力更强的因果或确定性判断。",
        consequence="断裂点之后的结论需要降权，但断裂点之前的观察不应被一并删除。",
        competing=["非形式化经验判断可能省略了文字桥梁，但并非没有真实机制。", "缺少外部核验与论证无效不是同一件事。"],
        questions=["哪一个前提一旦为假，结论就会坍塌？", "最强替代解释是什么，现有证据如何排除它？"],
        falsification="把每一步改写成‘证据支持前提、前提支持推论’并补入最强反例；若链条仍完整，本诊断应降级。",
        preserved="保留断裂点之前被材料直接支撑的观察；断裂点之后只保留为待检假设。",
        confidence=_diagnostic_confidence(observations, base=0.48, cap=0.92),
        missing=[] if output_present else ["核心假设", "逐步推理", "结论"],
        shadow_type="reasoning",
        source_stage="reasoning",
        likely_causes=["隐含假设", "因果跳跃", "反例处理不足", "证据不足"],
        related_parameters=["core_assumptions", "reasoning_path", "entries", "conclusion"],
        bias_direction="高估现有证据对因果和强结论的支撑",
        puzzle_use="fracture_boundary",
    )


def _detect_expert_failure(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    strong_domains = _string_list(case["persona"].get("expertise_strong"))
    weak_domains = _string_list(case["persona"].get("expertise_weak"))
    preferred = _string_list(case["information_filter"].get("preferred_evidence"))
    trusted = _string_list(case["information_filter"].get("trusted_sources"))
    accepted_text = _decision_text(case["accepted"]) + " " + " ".join(
        str(item.get("source_type", "")) + " " + str(item.get("evidence_type", ""))
        for item in case["entries"]
    )
    identity_channels = preferred + trusted
    matched_channels = [channel for channel in identity_channels if channel and channel in accepted_text]
    broad_claim = bool(_scope_flags(case["conclusion"], case["confidence"]))

    if identity_channels and len(matched_channels) >= max(1, len(identity_channels) // 2):
        observations.append(
            _observation(
                "evidence_selection",
                f"采信通道与自身专业偏好高度贴合：{'、'.join(matched_channels[:5])}。",
                accepted_text,
            )
        )
    if len(case["persona"].get("cognitive_frames", [])) <= 2 and broad_claim:
        observations.append(
            _observation(
                "reasoning",
                f"主要依靠 {len(case['persona'].get('cognitive_frames', []))} 个认知框架，却给出了跨范围或高确定性结论。",
                case["conclusion"],
            )
        )
    weak_text = " ".join(case["underweighted"] + _decision_result_values(case["ignored"], "evidence_type"))
    matched_weak = [domain for domain in weak_domains if domain and domain in weak_text]
    if matched_weak:
        observations.append(
            _observation(
                "self_correction",
                f"自报低估内容与自身弱专业域重合：{'、'.join(matched_weak[:4])}。",
                weak_text,
            )
        )

    has_domain_data = bool(strong_domains and case["conclusion"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if has_domain_data else "insufficient_evidence")
    diagnosis = (
        "专业能力确实照亮了一个局部，但同一套工具正被带出熟悉领域，开始替代其他变量和解释框架。"
        if status == "detected"
        else _negative_diagnosis(status, "专业局部高清向域外扩张")
    )
    return _finding(
        case,
        "expert_failure",
        status=status,
        observations=observations,
        chain_locations=["evidence_selection", "reasoning", "conclusion"],
        diagnosis=diagnosis,
        mechanism="专业训练提高某些变量的可见度，随后这些高可见变量被误当成关键变量或全部变量。",
        consequence="域内判断可能仍很有价值，但跨域因果、时间和行动建议需要单独降权。",
        competing=["来源与专业匹配可能代表真正的域内能力。", "多框架表达也不自动保证跨域判断可靠。"],
        questions=["这条判断的专业边界在哪里？", "跨出该边界后，依赖了哪些未经验证的域外前提？"],
        falsification="把结论拆成域内与域外两部分，并用域外证据单独检验；若两部分都得到独立支持，本诊断应降级。",
        preserved="保留专业域内直接受证据支持的局部判断，不把专业盲区误写成反专家结论。",
        confidence=_diagnostic_confidence(observations, base=0.45, cap=0.88),
        missing=[] if has_domain_data else ["专业边界", "完整结论"],
        shadow_type="framing",
        source_stage="reasoning",
        likely_causes=["专家隧道", "单一工具扩张", "跨域外推"],
        related_parameters=["expertise_strong", "expertise_weak", "cognitive_frames", "entries"],
        bias_direction="向自身专业工具能解释的变量倾斜",
        puzzle_use="constraint_band",
    )


def _detect_paradigm(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    frames = _string_list(case["persona"].get("cognitive_frames"))
    evidence_types = {str(item.get("evidence_type")) for item in case["entries"] if item.get("evidence_type")}
    ignored_types = set(_decision_result_values(case["ignored"] + case["rejected"], "evidence_type"))
    top_value = _top_value(case["persona"].get("values", {}))

    if len(frames) <= 2 and len(evidence_types) <= 1 and case["conclusion"]:
        observations.append(
            _observation(
                "problem_framing",
                f"问题主要通过 {'、'.join(frames) or '单一未明框架'} 处理，采信证据只覆盖 {', '.join(sorted(evidence_types)) or '未知类型'}。",
                " ".join(case["queries"] + case["assumptions"]),
            )
        )
    if ignored_types and len(ignored_types) >= 2:
        observations.append(
            _observation(
                "information_filter",
                f"至少两类可能承载其他问题定义的材料没有进入推理：{'、'.join(sorted(ignored_types)[:5])}。",
            )
        )
    if top_value and _contains(case["all_output_text"], [top_value[0]]) and not case["value_judgements"]:
        observations.append(
            _observation(
                "value_evaluation",
                f"最高价值“{top_value[0]}”进入了结论，但没有与事实判断分层，成功标准可能被默认锁定。",
                case["conclusion"],
            )
        )

    output_present = bool(case["conclusion"] and case["assumptions"])
    status = "detected" if len(observations) >= 2 else ("not_detected" if output_present else "insufficient_evidence")
    diagnosis = (
        "这个数字人并非只是在某个事实点上有偏差；它先用自己的框架决定了什么算问题、什么算证据、什么算成功。"
        if status == "detected"
        else _negative_diagnosis(status, "单一范式封闭了问题定义和证据标准")
    )
    return _finding(
        case,
        "paradigm",
        status=status,
        observations=observations,
        chain_locations=["problem_framing", "assumption", "value_evaluation"],
        diagnosis=diagnosis,
        mechanism="认知框架先定义问题对象和成功标准，证据随后只在这套语言内部获得意义。",
        consequence="可能让其他框架中的关键变量根本没有进入问题，而不是进入后被反驳。",
        competing=["证据类型单一可能只是检索失败。", "普通概念不清或价值分歧不应过早升级为范式封闭。"],
        questions=["换一种成功标准后，问题是否变成另一个问题？", "哪些变量在这套语言里根本无法被命名？"],
        falsification="要求该数字人用一种相反框架重新定义问题并保留同一事实；若关键变量与结论边界不变，本诊断应降级。",
        preserved="保留该范式最擅长揭示的变量，但把其问题定义和成功标准标成有边界的观察窗口。",
        confidence=_diagnostic_confidence(observations, base=0.38, cap=0.82),
        missing=[] if output_present else ["问题定义", "核心假设"],
        shadow_type="framing",
        source_stage="problem_framing",
        likely_causes=["范式加载", "问题定义锁定", "成功标准单一"],
        related_parameters=["cognitive_frames", "core_assumptions", "values", "ignored_sources"],
        bias_direction="使本范式可表达的变量更可见，其他变量更不可见",
        puzzle_use="negative_space",
    )


def _detect_ecological_rationality(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    evidence_text = " ".join(
        str(item.get("source_type", "")) + " " + str(item.get("evidence_type", ""))
        for item in case["entries"]
    )
    full_text = evidence_text + " " + case["all_output_text"]
    experiential = _contains(full_text, ["一线", "地方", "实践", "经验", "访谈", "事故案例", "历史案例"])
    feedback_history = _contains(full_text, ["多年", "长期实践", "反复验证", "反馈", "校准记录", "持续记录"])
    transfer = _contains(full_text, ["类比", "类似", "历史上", "过去经验", "迁移", "照搬"])
    current_environment_test = _contains(full_text, ["当前环境", "边界", "不同之处", "失效条件", "结构变化", "新环境"])

    if experiential:
        observations.append(
            _observation(
                "evidence_selection",
                "推理使用了地方经验、一线实践或历史案例等难以完全形式化的材料。",
                full_text,
            )
        )
    if transfer and not current_environment_test:
        observations.append(
            _observation(
                "reasoning",
                "经验或历史类比被迁移到当前问题，但没有检查两个环境的反馈结构和适用边界。",
                case["all_output_text"],
            )
        )
    if feedback_history:
        observations.append(
            _observation(
                "reasoning",
                "材料说明该规则来自持续反馈环境，存在真实生态适配的可能。",
                full_text,
            )
        )

    if experiential and transfer and not current_environment_test:
        status = "detected"
        diagnosis = "这条经验在旧环境里也许有用，但它拿来解释当前问题时，没有检查环境和起作用的条件是否已经变了。"
    elif experiential and feedback_history and current_environment_test:
        status = "not_detected"
        diagnosis = "现有材料显示经验规则有反馈历史，也检查了适用边界；暂不把它误判成低级偏见。"
    elif experiential:
        status = "insufficient_evidence"
        diagnosis = "看见了经验规则，但缺少反馈历史或环境匹配证据；现在既不能叫它偏见，也不能把它浪漫化成智慧。"
    else:
        status = "not_detected" if case["conclusion"] else "insufficient_evidence"
        diagnosis = _negative_diagnosis(status, "经验启发式从原环境迁移后失配")

    return _finding(
        case,
        "ecological_rationality",
        status=status,
        observations=observations,
        chain_locations=["information_filter", "reasoning"],
        diagnosis=diagnosis,
        mechanism="简单规则在原有高反馈环境中形成适应性，但环境结构变化或跨域迁移会使它失效。",
        consequence="它可能把旧环境里成立的规律当成当前仍然成立，从而低估结构变化和失效条件。",
        competing=["看似简单的规则可能有深厚反馈基础。", "地方经验也可能被利益、记忆偏差和小样本污染。"],
        questions=["这条规则在哪种环境中形成？", "当前环境的反馈速度、激励和边界与原环境有什么不同？"],
        falsification="提供该规则在当前环境中的连续反馈记录与失败案例；若仍稳定有效，应撤销错配诊断。",
        preserved="在环境适配未被否定前，保留经验规则揭示的地方变量和实践摩擦。",
        confidence=0.78 if status == "detected" else (0.7 if status == "not_detected" else 0.3),
        missing=[] if status != "insufficient_evidence" else ["反馈历史", "当前环境适配证据"],
        shadow_type="distorted",
        source_stage="reasoning",
        likely_causes=["生态错配", "旧经验迁移", "反馈结构变化"],
        related_parameters=["experience_evidence", "reasoning_path", "environment_boundary"],
        bias_direction="把原环境有效性过度延伸到当前环境",
        puzzle_use="constraint_band",
    )


def _detect_local_whole_overreach(case: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, str]] = []
    entries = case["entries"]
    source_types = {str(item.get("source_type")) for item in entries if item.get("source_type")}
    evidence_types = {str(item.get("evidence_type")) for item in entries if item.get("evidence_type")}
    retrieval_layers = set(case["retrieval_layers"])
    local_flags: list[str] = []
    if len(entries) <= 2:
        local_flags.append("采信材料不超过两条")
    if len(source_types) <= 1 and entries:
        local_flags.append("来源类型单一")
    if len(evidence_types) <= 1 and entries:
        local_flags.append("证据类型单一")
    if entries and retrieval_layers <= {"model_prior", "mock_search"}:
        local_flags.append("没有经核验的真实外部材料")
    if entries and case["retrieval_layers"].count("model_prior") / len(entries) >= 0.6:
        local_flags.append("模型先验占多数")

    whole_flags = _scope_flags(case["conclusion"], case["confidence"])
    if local_flags:
        observations.append(
            _observation(
                "evidence_selection",
                "证据目前只支持有限局部：" + "；".join(local_flags) + "。",
                _entry_excerpt(entries),
            )
        )
    if whole_flags:
        observations.append(
            _observation(
                "conclusion",
                "结论提出了更大范围的主张：" + "；".join(whole_flags) + "。",
                case["conclusion"],
            )
        )

    output_present = bool(case["conclusion"])
    status = "detected" if local_flags and whole_flags else ("not_detected" if output_present else "insufficient_evidence")
    missing_bridges = _missing_bridges(local_flags, whole_flags)
    diagnosis = (
        "它掌握的材料可以支持一个有限观察，但结论跨过了材料边界，把局部、短期、个案或先验扩大成了整体判断。"
        if status == "detected"
        else _negative_diagnosis(status, "局部材料被扩大成整体、长期或高确定性结论")
    )
    return _finding(
        case,
        "local_whole_overreach",
        lens_ids=["argument_forensics"],
        status=status,
        observations=observations,
        chain_locations=["evidence_selection", "reasoning", "conclusion"],
        diagnosis=diagnosis,
        mechanism="结论范围、时间尺度或确定性扩张得比证据覆盖更快，中间缺少可审计的转换桥梁。",
        consequence="越界部分需要切掉或降级，局部观察则可继续保留为待核验材料。",
        competing=["证据数量少不等于证据弱，关键还要看识别力与问题结构。", "强结论可能来自严密机制推演，但必须把桥梁明确交出来。"],
        questions=missing_bridges or ["证据实际支持到哪里？", "结论是否保持在同一范围与时间尺度？"],
        falsification="补齐样本、尺度、时间、因果和外部核验桥；若每座桥都能独立成立，本诊断应撤销。",
        preserved=_residual_fragment(case),
        confidence=_diagnostic_confidence(observations, base=0.5, cap=0.94),
        missing=[] if output_present else ["明确结论", "证据覆盖范围"],
        shadow_type="reasoning",
        source_stage="conclusion",
        likely_causes=["以偏概全", "尺度外推", "时间外推", "确定性膨胀"],
        related_parameters=["entries", "source_types", "evidence_types", "conclusion", "confidence"],
        bias_direction="高估局部材料对更大范围结论的支撑力",
        puzzle_use="fracture_boundary",
    )


def _finding(
    case: dict[str, Any],
    detector_id: str,
    *,
    status: str,
    observations: list[dict[str, str]],
    chain_locations: list[str],
    diagnosis: str,
    mechanism: str,
    consequence: str,
    competing: list[str],
    questions: list[str],
    falsification: str,
    preserved: str,
    confidence: float,
    missing: list[str],
    shadow_type: str,
    source_stage: str,
    likely_causes: list[str],
    related_parameters: list[str],
    bias_direction: str,
    puzzle_use: str,
    lens_ids: list[str] | None = None,
) -> dict[str, Any]:
    lens_ids = lens_ids or [detector_id]
    lens_cards = [case["lenses"].get(lens_id) for lens_id in lens_ids]
    basis = [
        {
            "lens_id": lens.id,
            "source_works": lens.source_works,
            "boundary_condition": lens.boundary_conditions[0] if lens.boundary_conditions else "",
            "anti_misuse_rule": lens.anti_misuse_rules[0] if lens.anti_misuse_rules else "",
        }
        for lens in lens_cards
        if lens is not None
    ]
    result: dict[str, Any] = {
        "detector_id": detector_id,
        "detector_name": DETECTOR_NAMES[detector_id],
        "status": status,
        "severity": _severity(status, confidence),
        "diagnostic_confidence": round(confidence, 2),
        "evidence_sufficiency": {
            "level": _sufficiency_level(confidence, status),
            "missing": missing,
        },
        "chain_locations": chain_locations,
        "observed_evidence": observations,
        "plain_language_diagnosis": diagnosis,
        "mechanism": mechanism,
        "consequence_for_conclusion": consequence,
        "competing_explanations": competing,
        "questions_to_distinguish": questions,
        "falsification_test": falsification,
        "preserved_valid_part": preserved,
        "knowledge_basis": basis,
    }
    if status == "detected":
        result["_shadow_spec"] = {
            "detector_id": detector_id,
            "shadow_type": shadow_type,
            "source_stage": source_stage,
            "description": diagnosis,
            "lenses": lens_ids,
            "likely_causes": likely_causes,
            "related_parameters": related_parameters,
            "bias_direction": bias_direction,
            "strength": _severity(status, confidence),
            "puzzle_use": puzzle_use,
            "plain_language": diagnosis,
        }
    return result


def _build_chain_trace(case: dict[str, Any]) -> dict[str, Any]:
    persona = case["persona"]
    stages = [
        _stage(
            "problem_framing",
            bool(case["queries"] or case["assumptions"]),
            ["检索词：" + "；".join(case["queries"][:2]), "核心假设：" + "；".join(case["assumptions"][:2])],
            ["检索词", "核心假设"],
        ),
        _stage(
            "information_filter",
            bool(case["information_filter"]),
            [
                "信任来源：" + "、".join(_string_list(case["information_filter"].get("trusted_sources"))[:4]),
                "忽略证据：" + "、".join(_string_list(case["information_filter"].get("ignored_evidence"))[:4]),
            ],
            ["信任来源", "低信任来源", "偏好证据", "忽略证据"],
        ),
        _stage(
            "retrieval",
            bool(case["queries"] and case["decisions"]),
            [f"查询 {len(case['queries'])} 组，获得 {len(case['decisions'])} 个候选来源。"],
            ["查询词", "候选来源"],
        ),
        _stage(
            "evidence_selection",
            bool(case["decisions"] or case["entries"]),
            [f"采信 {len(case['accepted'])}，拒绝 {len(case['rejected'])}，忽略 {len(case['ignored'])}。"],
            ["逐条采信决定", "采信理由"],
        ),
        _stage("assumption", bool(case["assumptions"]), case["assumptions"][:3], ["核心假设"]),
        _stage("reasoning", bool(case["reasoning"]), case["reasoning"][:4], ["逐步推理"]),
        _stage(
            "value_evaluation",
            bool(persona.get("values") or case["value_judgements"]),
            [
                "价值权重：" + _value_summary(persona.get("values", {})),
                "显性价值判断：" + "；".join(case["value_judgements"][:2]),
            ],
            ["价值权重", "显性价值判断"],
        ),
        _stage(
            "conclusion",
            bool(case["conclusion"]),
            [case["conclusion"], f"自报置信度：{case['confidence']}"],
            ["结论", "置信度"],
        ),
        _stage(
            "self_correction",
            bool(case["underweighted"] or case["change_conditions"]),
            [
                "自报低估：" + "；".join(case["underweighted"][:2]),
                "改变条件：" + "；".join(case["change_conditions"][:2]),
            ],
            ["自报低估", "可触发的改变条件"],
        ),
    ]
    present = sum(stage["status"] == "present" for stage in stages)
    return {
        "coverage": {
            "score": round(present / len(stages) * 100, 1),
            "present_stage_count": present,
            "total_stage_count": len(stages),
            "missing_stages": [stage["stage"] for stage in stages if stage["status"] != "present"],
        },
        "stages": stages,
    }


def _stage(stage: str, present: bool, observations: list[str], expected: list[str]) -> dict[str, Any]:
    cleaned = [_short(item, 220) for item in observations if item and item.strip(" ：;")]
    return {
        "stage": stage,
        "status": "present" if present else "missing",
        "observed": cleaned,
        "missing": [] if present else expected,
    }


def _scope_flags(text: str, confidence: float | None) -> list[str]:
    flags: list[str] = []
    groups = {
        "把范围扩大到整体或普遍": ["整体", "全部", "所有", "普遍", "全面", "系统性", "结构性", "全局"],
        "作出明确时间判断": ["很快", "马上", "立即", "短期内", "长期", "不远的将来", "最终", "必将"],
        "作出强因果或必然判断": ["导致", "证明", "必然", "一定", "不可避免", "注定"],
        "使用高确定性措辞": ["高度可能", "很可能", "确定", "显然", "无疑", "高于主流预期"],
    }
    for label, tokens in groups.items():
        if _asserts_any(text, tokens):
            flags.append(label)
    if confidence is not None and confidence >= 0.75:
        flags.append(f"自报置信度达到 {confidence:.2f}")
    return flags


def _missing_bridges(local_flags: list[str], whole_flags: list[str]) -> list[str]:
    if not local_flags or not whole_flags:
        return []
    questions = ["样本桥：这个局部是否具有代表性，基准率与反例是什么？"]
    joined = " ".join(whole_flags)
    if "范围" in joined or "整体" in joined:
        questions.append("尺度桥：从个案、部门或单一变量到整体系统，中间经过哪些机制？")
    if "时间" in joined:
        questions.append("时间桥：当前信号怎样在所声称的期限内传导到结论？")
    if "因果" in joined:
        questions.append("因果桥：替代原因怎样被排除，必要条件与触发条件是什么？")
    if any("核验" in item or "先验" in item for item in local_flags):
        questions.append("核验桥：哪些材料是外部事实，哪些只是模型先验或模拟材料？")
    return questions


def _residual_fragment(case: dict[str, Any]) -> str:
    evidence_types = [str(item.get("evidence_type")) for item in case["entries"] if item.get("evidence_type")]
    if evidence_types:
        dominant = Counter(evidence_types).most_common(1)[0][0]
        return f"删掉扩大后的结论，暂时保留它对“{dominant}”这一局部维度的观察与敏感性。"
    if case["assumptions"]:
        return "删掉扩大后的结论，只保留为待检假设：" + _short(case["assumptions"][0], 140)
    return "没有足够材料确认局部正确，只能保留为低置信待检假设。"


def _negative_diagnosis(status: str, target: str) -> str:
    if status == "insufficient_evidence":
        return f"现有材料不足以判断是否存在{target}，这里必须保留未知。"
    return f"现有思考链暂未检出{target}；这不是永久排除，只是当前证据没有达到触发标准。"


def _diagnostic_confidence(observations: list[dict[str, str]], *, base: float, cap: float) -> float:
    if not observations:
        return min(base, 0.35)
    return min(cap, base + 0.14 * len(observations))


def _sufficiency_level(confidence: float, status: str) -> str:
    if status == "insufficient_evidence":
        return "insufficient"
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.5:
        return "moderate"
    return "weak"


def _severity(status: str, confidence: float) -> str:
    if status != "detected":
        return "none"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _observation(stage: str, fact: str, excerpt: str = "") -> dict[str, str]:
    result = {"stage": stage, "fact": fact}
    if excerpt.strip():
        result["excerpt"] = _short(excerpt, 260)
    return result


def _change_condition_testability(items: list[str]) -> str:
    if not items:
        return "weak"
    text = " ".join(items)
    concrete_tokens = ["若", "如果", "当", "数据", "指标", "比例", "连续", "达到", "低于", "高于", "发生", "证据"]
    return "strong" if _contains(text, concrete_tokens) else "weak"


def _entry_excerpt(entries: list[dict[str, Any]]) -> str:
    return "；".join(
        _short(str(item.get("claim") or item.get("source_title") or item.get("evidence_type") or ""), 100)
        for item in entries[:3]
    )


def _decision_text(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
        parts.extend(
            [
                str(result.get("source_type", "")),
                str(result.get("evidence_type", "")),
                str(result.get("title", "")),
                str(result.get("snippet", "")),
                " ".join(_string_list(item.get("reasons"))),
            ]
        )
    return " ".join(part for part in parts if part)


def _decision_result_values(items: list[dict[str, Any]], key: str) -> list[str]:
    values = []
    for item in items:
        result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
        value = result.get(key)
        if value:
            values.append(str(value))
    return values


def _top_value(values: Any) -> tuple[str, float] | None:
    if not isinstance(values, dict) or not values:
        return None
    numeric = [(str(key), float(value)) for key, value in values.items() if isinstance(value, int | float)]
    return max(numeric, key=lambda item: item[1]) if numeric else None


def _value_summary(values: Any) -> str:
    if not isinstance(values, dict):
        return ""
    return "、".join(f"{key}={value}" for key, value in values.items())


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _contains(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _asserts_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    negations = ["不能", "并非", "不是", "不应", "无法", "没有", "不足以", "不可据此", "未能"]
    for token in tokens:
        needle = token.lower()
        start = 0
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            prefix = lowered[max(0, index - 10) : index]
            if not any(negation in prefix for negation in negations):
                return True
            start = index + len(needle)
    return False


def _short(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0
