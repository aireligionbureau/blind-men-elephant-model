from __future__ import annotations

from typing import Any


def evaluate_meta_model_strength(report: dict[str, Any]) -> dict[str, Any]:
    rubric = [
        (
            "classic_grounding",
            10,
            _score_count(report.get("classic_grounding", []), target=9),
            "经典透镜是否稳定进入诊断底层。",
        ),
        (
            "micro_lens_resolution",
            10,
            _score_count_value(report.get("micro_lens_count", 0), target=40),
            "是否能把大透镜拆成更细的微机制。",
        ),
        (
            "shadow_mechanism_ontology",
            18,
            _score_count_value(report.get("shadow_mechanism_count", 0), target=18),
            "是否有独立的思维阴影生成机制本体。",
        ),
        (
            "full_chain_capture",
            14,
            _score_count_value(report.get("capture_protocol_stage_count", 0), target=10),
            "是否覆盖从问题框定到元模型自审的完整链条。",
        ),
        (
            "diagnostic_attribution_depth",
            12,
            _score_attribution_depth(report),
            "每条阴影是否具备透镜、微透镜、机制候选和成因归因。",
        ),
        (
            "differential_diagnosis",
            12,
            _score_presence(report.get("differential_diagnosis")) * _score_count_value(
                report.get("lens_conflict_rule_count", 0),
                target=10,
            ),
            "是否能处理容易混淆的透镜，而不是贴标签。",
        ),
        (
            "calibration_and_misdiagnosis_audit",
            10,
            min(
                _score_count_value(report.get("calibration_case_count", 0), target=5),
                _score_presence(report.get("misdiagnosis_audit")),
                _score_presence(report.get("mechanism_audit")),
            ),
            "是否有校准案例和误诊/漏诊审计。",
        ),
        (
            "shadow_inversion",
            8,
            _score_count(report.get("inversion_plan", []), target=max(1, len(report.get("shadow_registry", [])))),
            "阴影是否能被转成拼图材料。",
        ),
        (
            "self_audit",
            6,
            _score_presence(report.get("meta_reflection", {}).get("anti_misuse_rules")),
            "元模型是否审计自己的过度理性和过度诊断风险。",
        ),
    ]

    items = []
    total = 0.0
    for name, weight, ratio, description in rubric:
        score = round(weight * max(0.0, min(1.0, ratio)), 2)
        total += score
        items.append(
            {
                "criterion": name,
                "score": score,
                "weight": weight,
                "description": description,
            }
        )
    total_score = round(total, 2)
    return {
        "score": total_score,
        "target": 90,
        "passed": total_score >= 90,
        "score_scope": "经典底层与工具完整度；不代表真实案例诊断准确率。",
        "items": items,
        "verdict": _verdict(total_score),
    }


def evaluate_cognitive_chain_strength(report: dict[str, Any]) -> dict[str, Any]:
    scans = report.get("cognitive_chain_scans", [])
    detected = [
        finding
        for scan in scans
        for finding in scan.get("detector_results", [])
        if finding.get("status") == "detected"
    ]
    all_findings = [
        finding
        for scan in scans
        for finding in scan.get("detector_results", [])
    ]
    valid_statuses = {"detected", "not_detected", "insufficient_evidence"}

    rubric = [
        (
            "complete_chain_coverage",
            15,
            _average([scan.get("chain_coverage", {}).get("score", 0) / 100 for scan in scans]),
            "是否逐个覆盖问题框定到自我修正的完整思考链。",
        ),
        (
            "ten_detector_coverage",
            15,
            _average(
                [
                    min(1.0, len(scan.get("detector_results", [])) / 10)
                    for scan in scans
                ]
            ),
            "10 个检测器是否对每个数字人逐一运行。",
        ),
        (
            "evidence_grounding",
            15,
            _ratio_complete(detected, lambda item: bool(item.get("observed_evidence"))),
            "每个检出是否钉在具体链条材料上。",
        ),
        (
            "mechanism_and_effect",
            15,
            _ratio_complete(
                detected,
                lambda item: bool(item.get("mechanism") and item.get("consequence_for_conclusion")),
            ),
            "是否说明问题如何生成并影响结论。",
        ),
        (
            "differential_restraint",
            10,
            min(
                _ratio_complete(detected, lambda item: bool(item.get("competing_explanations"))),
                _ratio_complete(all_findings, lambda item: item.get("status") in valid_statuses),
            ),
            "是否考虑竞争解释，并允许证据不足而不硬判。",
        ),
        (
            "falsifiability",
            10,
            _ratio_complete(detected, lambda item: bool(item.get("falsification_test"))),
            "每项诊断是否给出可推翻条件。",
        ),
        (
            "preserve_valid_fragment",
            8,
            _ratio_complete(detected, lambda item: bool(item.get("preserved_valid_part"))),
            "是否切分受污染部分和仍可保留的局部。",
        ),
        (
            "plain_language_compression",
            12,
            _plain_language_score(scans),
            "用户层是否只用简明语言呈现最关键的具体问题。",
        ),
    ]

    items = []
    total = 0.0
    for name, weight, ratio, description in rubric:
        score = round(weight * max(0.0, min(1.0, ratio)), 2)
        total += score
        items.append(
            {
                "criterion": name,
                "score": score,
                "weight": weight,
                "description": description,
            }
        )
    total_score = round(total, 2)
    return {
        "score": total_score,
        "target": 90,
        "passed": total_score >= 90,
        "score_scope": "本轮逐人认知链透视的执行质量；诊断准确率仍需正例、反例和混淆例校准。",
        "items": items,
        "verdict": _chain_verdict(total_score),
    }


def evaluate_semantic_verdict_strength(report: dict[str, Any]) -> dict[str, Any]:
    scans = report.get("cognitive_chain_scans", [])
    semantic_scans = [scan for scan in scans if isinstance(scan.get("semantic_verdict"), dict)]
    verdicts = [
        verdict
        for scan in semantic_scans
        for verdict in scan.get("semantic_verdict", {}).get("verdicts", [])
    ]
    adjudicated_ratio = _average(
        [
            1.0
            if scan.get("semantic_verdict", {}).get("execution_status") in {"succeeded", "no_finding"}
            else 0.0
            for scan in scans
        ]
    )
    semantic_coverage = len(semantic_scans) / len(scans) if scans else 0.0
    specificity = _average(
        [
            float(verdict.get("evidence_validation", {}).get("specificity_score", 0))
            for verdict in verdicts
        ]
    ) if verdicts else _unresolved_semantic_score(semantic_scans)
    deduplicated = _average([_final_verdicts_deduplicated(scan) for scan in semantic_scans])
    duplicate_proposals_removed = sum(
        any(
            "重复" in str(reason)
            for reason in dropped.get("errors", [])
        )
        for scan in semantic_scans
        for dropped in scan.get("semantic_verdict", {}).get("validation", {}).get("dropped_verdicts", [])
    )

    rubric = [
        (
            "single_person_adjudication_coverage",
            10,
            min(semantic_coverage, adjudicated_ratio),
            "是否逐人完成判读；无候选问题时明确暂未检出，而不是强行贴错或依赖失败回退。",
        ),
        (
            "evidence_lock",
            15,
            _ratio_complete(
                verdicts,
                lambda item: bool(item.get("evidence_validation", {}).get("valid"))
                and "C1" in item.get("evidence_refs", [])
                and len(item.get("evidence_refs", [])) >= 2,
            ),
            "每个问题是否同时钉住推理起点和实际结论。",
        ),
        (
            "problem_specificity",
            20,
            specificity,
            "表述是否只能属于这个数字人、这条推理和这个具体问题。",
        ),
        (
            "explicit_reasoning_bridge",
            15,
            _ratio_complete(
                verdicts,
                lambda item: bool(item.get("from_claim") and item.get("to_conclusion") and item.get("missing_bridge")),
            ),
            "是否明确指出从哪句话跨到哪句话，以及中间缺了什么。",
        ),
        (
            "conclusion_leverage",
            10,
            _ratio_complete(
                verdicts,
                lambda item: item.get("conclusion_leverage") in {"high", "medium", "low"}
                and bool(item.get("impact_on_conclusion")),
            ),
            "是否说明修正问题后，结论方向、时间、范围或信心怎样变化。",
        ),
        (
            "root_cause_deduplication",
            10,
            deduplicated,
            "同一根问题是否被合并，而不是换术语重复展示。",
        ),
        (
            "differential_restraint",
            8,
            _ratio_complete(verdicts, lambda item: bool(item.get("competing_explanation"))),
            "是否给出可能使诊断不成立的竞争解释。",
        ),
        (
            "falsifiability",
            5,
            _ratio_complete(
                verdicts,
                lambda item: "有可核验证据" in item.get("falsification_test", "")
                and "仍支持原结论" in item.get("falsification_test", "")
                and "这条批评应撤销或降级" in item.get("falsification_test", ""),
            ),
            "是否说明什么新材料会推翻诊断。",
        ),
        (
            "preserve_valid_fragment",
            7,
            _ratio_complete(verdicts, lambda item: bool(item.get("preserved_fragment"))),
            "是否切掉越界部分，同时保留仍有根据的局部。",
        ),
    ]
    items = []
    total = 0.0
    for name, weight, ratio, description in rubric:
        score = round(weight * max(0.0, min(1.0, ratio)), 2)
        total += score
        items.append({"criterion": name, "score": score, "weight": weight, "description": description})
    total_score = round(total, 2)
    return {
        "score": total_score,
        "target": 90,
        "passed": total_score >= 90,
        "score_scope": "逐人具体语义判读的执行质量；不代表数字人的预测正确，也不代表真相准确率。",
        "items": items,
        "verdict_count": len(verdicts),
        "duplicate_proposals_removed": duplicate_proposals_removed,
        "live_person_count": sum(
            scan.get("semantic_verdict", {}).get("execution_status") == "succeeded"
            for scan in semantic_scans
        ),
        "no_finding_person_count": sum(
            scan.get("semantic_verdict", {}).get("execution_status") == "no_finding"
            for scan in semantic_scans
        ),
        "adjudicated_person_count": sum(
            scan.get("semantic_verdict", {}).get("execution_status") in {"succeeded", "no_finding"}
            for scan in semantic_scans
        ),
        "verdict": _semantic_verdict(total_score),
    }


def evaluate_meta_model_overall(report: dict[str, Any]) -> dict[str, Any]:
    foundation = float(report.get("readiness_evaluation", {}).get("score", 0))
    chain = float(report.get("cognitive_chain_evaluation", {}).get("score", 0))
    semantic = float(report.get("semantic_verdict_evaluation", {}).get("score", 0))
    score = round(foundation * 0.2 + chain * 0.35 + semantic * 0.45, 2)
    return {
        "score": score,
        "target": 90,
        "passed": score >= 90 and min(foundation, chain, semantic) >= 85,
        "components": {
            "classic_foundation": foundation,
            "cognitive_chain_scan": chain,
            "specific_semantic_verdict": semantic,
        },
        "weights": {
            "classic_foundation": 0.2,
            "cognitive_chain_scan": 0.35,
            "specific_semantic_verdict": 0.45,
        },
        "score_scope": "经典底层、逐人完整检查和具体语义判读的综合执行能力；仍不是真相准确率。",
        "verdict": _overall_verdict(score, min(foundation, chain, semantic)),
    }


def _score_count(items: list[Any], *, target: int) -> float:
    if target <= 0:
        return 1.0
    return min(1.0, len(items) / target)


def _score_count_value(value: int, *, target: int) -> float:
    if target <= 0:
        return 1.0
    return min(1.0, value / target)


def _score_presence(value: Any) -> float:
    return 1.0 if value else 0.0


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio_complete(items: list[dict[str, Any]], predicate: Any) -> float:
    if not items:
        return 1.0
    return sum(bool(predicate(item)) for item in items) / len(items)


def _final_verdicts_deduplicated(scan: dict[str, Any]) -> float:
    semantic = scan.get("semantic_verdict", {})
    verdicts = semantic.get("verdicts", [])
    if not semantic.get("validation", {}).get("valid") or len(verdicts) > 2:
        return 0.0
    roots = [_semantic_root(item) for item in verdicts]
    roots = [item for item in roots if item]
    return 1.0 if len(roots) == len(set(roots)) else 0.0


def _semantic_root(verdict: dict[str, Any]) -> str:
    detector_ids = verdict.get("source_detector_ids", [])
    detector_id = str(detector_ids[0]) if detector_ids else ""
    return {
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
    }.get(detector_id, detector_id)


def _unresolved_semantic_score(scans: list[dict[str, Any]]) -> float:
    if not scans:
        return 0.0
    return _average(
        [
            1.0
            if scan.get("semantic_verdict", {}).get("validation", {}).get("valid")
            and scan.get("semantic_verdict", {}).get("unresolved")
            else 0.0
            for scan in scans
        ]
    )


def _plain_language_score(scans: list[dict[str, Any]]) -> float:
    if not scans:
        return 0.0
    forbidden = {
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
        "透镜",
        "检测器",
    }
    scores = []
    for scan in scans:
        summary = scan.get("plain_language_summary", {})
        headline = str(summary.get("headline", ""))
        problems = summary.get("key_problems", [])
        has_detected = any(
            item.get("status") == "detected"
            for item in scan.get("detector_results", [])
        )
        complete = bool(headline) and (bool(problems) or not has_detected)
        concise = len(headline) <= 180 and len(problems) <= 3
        text = headline + " " + " ".join(str(item) for item in problems)
        jargon_free = not any(token in text for token in forbidden)
        scores.append(sum([complete, concise, jargon_free]) / 3)
    return _average(scores)


def _score_attribution_depth(report: dict[str, Any]) -> float:
    attributions = report.get("bias_attribution", [])
    if not attributions:
        return 0.0
    complete = 0
    for item in attributions:
        if (
            item.get("classic_lenses")
            and item.get("mechanism_chains")
            and item.get("micro_lens_candidates")
            and item.get("shadow_mechanism_candidates")
            and item.get("likely_causes")
        ):
            complete += 1
    return complete / len(attributions)


def _verdict(score: float) -> str:
    if score >= 95:
        return "已接近强健元模型底层，但仍需真实案例校准。"
    if score >= 90:
        return "达到当前 90 分门槛，可以进入案例校准和二阶综合。"
    if score >= 80:
        return "结构已成型，但还需要继续深耕阴影机制和误诊审计。"
    return "仍偏原型，需要继续打经典和阴影机制地基。"


def _chain_verdict(score: float) -> str:
    if score >= 95:
        return "认知链透视执行完整、可审计且表达克制；仍需用真实案例验证准确率。"
    if score >= 90:
        return "达到认知链透视执行门槛，可以进入正反例校准。"
    if score >= 80:
        return "认知链透视已经成形，但仍有检查遗漏或表达冗余。"
    return "尚未形成稳定的逐人完整链条诊断能力。"


def _semantic_verdict(score: float) -> str:
    if score >= 95:
        return "已能把后台发现稳定翻译成具体、可证伪且不可随意套用的逐人判读。"
    if score >= 90:
        return "具体语义判读达到执行门槛，但仍应继续扩充跨领域校准样本。"
    if score >= 80:
        return "已经能引用具体链条，但仍有模板化、重复或结论撬动力不足。"
    return "判读仍偏通用模板，尚未把检测结果转成一针见血的具体问题。"


def _overall_verdict(score: float, weakest: float) -> str:
    if score >= 95 and weakest >= 90:
        return "元模型逐人诊断链已经强健；下一项风险是跨领域真实案例准确率，而不是流程缺失。"
    if score >= 90 and weakest >= 85:
        return "元模型整体执行能力达到门槛，但最弱环节仍需继续校准。"
    return "整体能力尚未过关；不能用某一项满分掩盖最弱环节。"
