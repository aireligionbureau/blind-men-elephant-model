from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from .evaluator import evaluate_cognitive_chain_strength, evaluate_meta_model_strength
from .forensics import build_chain_detector_audit, run_cognitive_chain_scan
from .lens_registry import (
    LensCard,
    LensConflictRule,
    MicroLens,
    load_calibration_cases,
    load_lens_conflicts,
    load_lenses,
    load_micro_lenses,
)
from .shadow_ontology import (
    CaptureProtocolStage,
    ShadowMechanism,
    load_capture_protocols,
    load_shadow_mechanisms,
)


def diagnose_cognitive_shadows(
    question: str,
    personas: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]],
    *,
    person_outputs: list[dict[str, Any]] | None = None,
    lenses: dict[str, LensCard] | None = None,
) -> dict[str, Any]:
    lens_map = lenses or load_lenses()
    micro_lens_map = load_micro_lenses()
    conflict_rules = load_lens_conflicts()
    calibration_cases = load_calibration_cases()
    shadow_mechanisms = load_shadow_mechanisms()
    capture_protocols = load_capture_protocols()
    ledgers_by_person = {ledger["person_id"]: ledger for ledger in evidence_ledgers}
    outputs_by_person = _index_person_outputs(person_outputs or [])

    shadow_registry: list[dict[str, Any]] = []
    bias_attribution: list[dict[str, Any]] = []
    inversion_plan: list[dict[str, Any]] = []
    cognitive_chain_scans: list[dict[str, Any]] = []

    for persona in personas:
        person_id = persona["id"]
        ledger = ledgers_by_person.get(person_id, _empty_ledger(person_id, question))
        output_record = outputs_by_person.get(person_id, {})
        output = output_record.get("output", output_record) if output_record else {}

        chain_scan, shadow_specs = run_cognitive_chain_scan(
            question,
            persona,
            ledger,
            output,
            lens_map,
        )
        person_shadows = [
            _shadow_from_chain_finding(persona, shadow_spec)
            for shadow_spec in shadow_specs
        ]

        shadow_registry.extend(person_shadows)
        cognitive_chain_scans.append(chain_scan)
        for shadow in person_shadows:
            bias_attribution.append(
                {
                    "shadow_id": shadow["id"],
                    "person_id": person_id,
                    "classic_lenses": shadow["attribution"]["classic_lenses"],
                    "mechanism_chains": _lens_mechanism_summaries(
                        shadow["attribution"]["classic_lenses"],
                        lens_map,
                    ),
                    "micro_lens_candidates": _select_micro_lenses(shadow, micro_lens_map),
                    "shadow_mechanism_candidates": _select_shadow_mechanisms(
                        shadow,
                        shadow_mechanisms,
                    ),
                    "likely_causes": shadow["attribution"]["likely_causes"],
                    "related_parameters": shadow["attribution"]["related_parameters"],
                }
            )
            inversion_plan.append(_build_inversion_item(shadow, lens_map))

    collective_blind_spots = _diagnose_collective_blind_spots(question, personas, evidence_ledgers)
    disagreement_scan = _scan_disagreement_potential(personas)
    output_disagreement = _scan_output_disagreement(question, personas, outputs_by_person)
    blind_spot_registry = _build_blind_spot_registry(collective_blind_spots, outputs_by_person)
    differential_diagnosis = _build_differential_diagnosis(shadow_registry, conflict_rules)

    report = {
        "question": question,
        "meta_modules": {
            "cognitive_chain_scan": "逐个透视问题框定、信息过滤、检索、证据选择、假设、推理、价值、结论和自我修正。",
            "ten_detector_panel": "让 9 张经典透镜与局部整体越界检测器平行运行，每项都承担相同证明负担。",
            "shadow_inverter": "把阴影转译为约束带、负空间洞、断裂边界或锚点候选。",
            "self_auditor": "检查元模型自身是否过度结构化、过度理性主义或偏好可计算阴影。",
        },
        "lens_count": len(lens_map),
        "micro_lens_count": len(micro_lens_map),
        "lens_conflict_rule_count": len(conflict_rules),
        "calibration_case_count": len(calibration_cases),
        "shadow_mechanism_count": len(shadow_mechanisms),
        "capture_protocol_stage_count": len(capture_protocols),
        "lenses_used": sorted(lens_map),
        "classic_grounding": _build_classic_grounding(lens_map),
        "shadow_mechanism_ontology": _build_shadow_mechanism_ontology(shadow_mechanisms),
        "capture_protocol": _build_capture_protocol(capture_protocols),
        "mechanism_coverage": _build_mechanism_coverage(shadow_registry, shadow_mechanisms),
        "shadow_registry": shadow_registry,
        "cognitive_chain_scans": cognitive_chain_scans,
        "chain_detector_audit": build_chain_detector_audit(cognitive_chain_scans),
        "bias_attribution": bias_attribution,
        "person_output_count": len(outputs_by_person),
        "disagreement_structure": {
            "identity_based": disagreement_scan,
            "output_based": output_disagreement,
        },
        "blind_spot_registry": blind_spot_registry,
        "inversion_plan": inversion_plan,
        "differential_diagnosis": differential_diagnosis,
        "misdiagnosis_audit": _build_misdiagnosis_audit(differential_diagnosis),
        "mechanism_audit": _build_mechanism_audit(shadow_registry, shadow_mechanisms),
        "detector_hardening_audit": _build_detector_hardening_audit(lens_map, shadow_registry),
        "calibration_reminders": _build_calibration_reminders(calibration_cases),
        "collective_blind_spots": collective_blind_spots,
        "disagreement_scan": disagreement_scan,
        "meta_reflection": _build_meta_reflection(lens_map),
    }
    report["readiness_evaluation"] = evaluate_meta_model_strength(report)
    report["cognitive_chain_evaluation"] = evaluate_cognitive_chain_strength(report)
    return report


def diagnose_one_cognitive_chain(
    question: str,
    persona: dict[str, Any],
    evidence_ledger: dict[str, Any],
    person_output: dict[str, Any],
    *,
    lenses: dict[str, LensCard] | None = None,
) -> dict[str, Any]:
    """Run the unchanged ten-detector scan as soon as one chain is ready."""

    lens_map = lenses or load_lenses()
    scan, _shadow_specs = run_cognitive_chain_scan(
        question,
        persona,
        evidence_ledger,
        person_output,
        lens_map,
    )
    return scan


def _diagnose_information_shadows(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    lenses: dict[str, LensCard],
) -> list[dict[str, Any]]:
    shadows: list[dict[str, Any]] = []
    rejected = ledger.get("rejected_sources", [])
    ignored = ledger.get("ignored_sources", [])
    accepted = ledger.get("accepted_sources", [])

    if ignored:
        shadows.append(
            _shadow(
                persona,
                "informational",
                "information_filter",
                f"该数字人有 {len(ignored)} 个候选来源处于低注意力区，可能形成信息摄入盲区。",
                lenses=["social_influence", "emotion_reason"],
                likely_causes=["信息源偏好", "低注意力过滤", "证据类型不合口味"],
                related_parameters=["information_filter", "preferred_evidence", "ignored_evidence"],
                bias_direction="低估被忽略来源所代表的变量",
                strength=_strength(len(ignored), len(accepted) + len(rejected) + len(ignored)),
                puzzle_use="negative_space",
            )
        )

    if rejected:
        shadows.append(
            _shadow(
                persona,
                "informational",
                "evidence_selection",
                f"该数字人明确排斥 {len(rejected)} 个候选来源，这些排斥本身构成可审计的信息阴影。",
                lenses=["social_influence", "bias_heuristics"],
                likely_causes=["低信任来源类型", "身份/专业同温层", "确认偏误"],
                related_parameters=["distrusted_sources", "hidden_biases"],
                bias_direction="向自身信任来源和专业同温层倾斜",
                strength=_strength(len(rejected), len(accepted) + len(rejected) + len(ignored)),
                puzzle_use="constraint_band",
            )
        )

    if accepted and not rejected and not ignored:
        shadows.append(
            _shadow(
                persona,
                "informational",
                "evidence_selection",
                "候选来源几乎全部被采信，可能说明检索空间过窄，尚未接触反向证据。",
                lenses=["argument_forensics", "noise"],
                likely_causes=["检索关键词过窄", "候选来源缺少异质性"],
                related_parameters=["search_strategy"],
                bias_direction="可能高估当前证据集完整性",
                strength="medium",
                puzzle_use="downweight",
            )
        )

    return shadows


def _diagnose_bias_shadows(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    lenses: dict[str, LensCard],
) -> list[dict[str, Any]]:
    del lenses
    hidden_biases = persona.get("hidden_biases", [])
    values = persona.get("values", {})
    shadows: list[dict[str, Any]] = []
    if hidden_biases:
        shadows.append(
            _shadow(
                persona,
                "distorted",
                "assumption",
                f"该数字人内置偏差为 {', '.join(hidden_biases)}，其判断可能沿这些方向稳定扭曲。",
                lenses=["bias_heuristics", "interpreter"],
                likely_causes=list(hidden_biases),
                related_parameters=["hidden_biases", "values", "risk_attitude"],
                bias_direction=_bias_direction_from_values(values),
                strength="medium",
                puzzle_use="constraint_band",
            )
        )
    if values:
        top_value = max(values.items(), key=lambda item: item[1])
        if top_value[1] >= 0.45:
            shadows.append(
                _shadow(
                    persona,
                    "value",
                    "value_evaluation",
                    f"价值函数明显偏向 {top_value[0]}，会系统性放大相关收益或代价。",
                    lenses=["bias_heuristics", "interpreter", "paradigm"],
                    likely_causes=["价值权重不均衡", "动机性推理"],
                    related_parameters=["values"],
                    bias_direction=f"向 {top_value[0]} 相关结论倾斜",
                    strength="high" if top_value[1] >= 0.55 else "medium",
                    puzzle_use="constraint_band",
                )
            )
    if "过度自信" in hidden_biases and len(ledger.get("entries", [])) < 3:
        shadows.append(
            _shadow(
                persona,
                "reasoning",
                "conclusion",
                "该数字人带有过度自信倾向，但证据账本采信条目较少，结论置信度需要下调。",
                lenses=["bias_heuristics", "argument_forensics"],
                likely_causes=["过度自信", "证据不足"],
                related_parameters=["hidden_biases", "entries"],
                bias_direction="高估结论确定性",
                strength="medium",
                puzzle_use="downweight",
            )
        )
    return shadows


def _diagnose_argument_shadows(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    lenses: dict[str, LensCard],
) -> list[dict[str, Any]]:
    del lenses
    entries = ledger.get("entries", [])
    shadows: list[dict[str, Any]] = []
    if len(entries) <= 1:
        shadows.append(
            _shadow(
                persona,
                "reasoning",
                "reasoning",
                "采信证据过少，后续结论若很强，应视为证据不足或推理跳跃候选。",
                lenses=["argument_forensics"],
                likely_causes=["证据不足", "从局部到整体的外推风险"],
                related_parameters=["entries", "reasoning_path"],
                bias_direction="高估局部证据的外推能力",
                strength="high",
                puzzle_use="fracture_boundary",
            )
        )
    source_types = {entry.get("source_type") for entry in entries}
    if len(source_types) == 1 and len(entries) >= 2:
        shadows.append(
            _shadow(
                persona,
                "informational",
                "evidence_selection",
                f"采信证据集中于单一来源类型：{next(iter(source_types))}，可能形成信息泡泡。",
                lenses=["argument_forensics", "expert_failure"],
                likely_causes=["证据类型单一", "专业同温层"],
                related_parameters=["entries", "expertise_strong"],
                bias_direction="高估单一证据通道的代表性",
                strength="medium",
                puzzle_use="downweight",
            )
        )
    return shadows


def _diagnose_local_whole_overreach(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    output: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = ledger.get("entries", [])
    conclusion = str(output.get("conclusion", ""))
    reasoning_text = " ".join(str(item) for item in output.get("reasoning_path", []) or [])
    assumptions_text = " ".join(str(item) for item in output.get("core_assumptions", []) or [])
    if not conclusion:
        return []

    source_types = {
        str(entry.get("source_type"))
        for entry in entries
        if entry.get("source_type")
    }
    evidence_types = {
        str(entry.get("evidence_type"))
        for entry in entries
        if entry.get("evidence_type")
    }
    retrieval_layers = {
        str(entry.get("retrieval_layer"))
        for entry in entries
        if entry.get("retrieval_layer")
    }
    verified_external_count = sum(
        1
        for entry in entries
        if entry.get("retrieval_layer") not in {"model_prior", None}
        and entry.get("verification_status") not in {"unverified_model_prior", "synthetic_fixture", None}
    )
    confidence = output.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, int | float) else None

    local_evidence_flags: list[str] = []
    if len(entries) <= 2:
        local_evidence_flags.append("采信证据数量很少")
    if len(source_types) <= 1 and len(entries) >= 2:
        local_evidence_flags.append("来源类型单一")
    if len(evidence_types) <= 1 and len(entries) >= 2:
        local_evidence_flags.append("证据类型单一")
    if "model_prior" in retrieval_layers and verified_external_count == 0:
        local_evidence_flags.append("主要依赖模型先验或未外部核验材料")

    scope_text = f"{conclusion} {reasoning_text} {assumptions_text}"
    whole_scope_flags = _whole_scope_flags(scope_text, confidence_value)
    if not local_evidence_flags or not whole_scope_flags:
        return []

    evidence_floor = _evidence_floor(entries, evidence_types, source_types, retrieval_layers)
    conclusion_ceiling = _conclusion_ceiling(conclusion, whole_scope_flags)
    missing_bridges = _missing_bridge_questions(local_evidence_flags, whole_scope_flags)
    residual_fragment = _residual_local_fragment(entries, output)

    strength = "high" if len(local_evidence_flags) >= 2 and len(whole_scope_flags) >= 2 else "medium"
    return [
        _shadow(
            persona,
            "reasoning",
            "conclusion",
            "局部整体越界：该数字人的证据能支撑一个局部观察，但结论把它推成了整体判断或时间判断。",
            lenses=["argument_forensics", "bias_heuristics", "expert_failure"],
            likely_causes=["局部证据外推", "结论范围越界", "代表性替代", "尺度翻译缺桥"],
            related_parameters=["entries", "evidence_types", "source_types", "retrieval_layers", "conclusion", "confidence"],
            bias_direction="高估局部材料对整体结论的支撑力",
            strength=strength,
            puzzle_use="fracture_boundary",
            detector_id="local_whole_overreach",
            plain_language="它摸到的局部可能有用，但它把这个局部说成了整只象。",
            evidence_pressure={
                "evidence_floor": evidence_floor,
                "conclusion_ceiling": conclusion_ceiling,
                "local_evidence_flags": local_evidence_flags,
                "whole_scope_flags": whole_scope_flags,
                "missing_bridges": missing_bridges,
                "residual_fragment_after_cutting_overreach": residual_fragment,
            },
        )
    ]


def _diagnose_expert_shadows(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    lenses: dict[str, LensCard],
) -> list[dict[str, Any]]:
    del lenses
    strong = set(persona.get("expertise_strong", []))
    accepted_text = " ".join(
        f"{entry.get('source_type', '')} {entry.get('evidence_type', '')} {entry.get('trust_reason', '')}"
        for entry in ledger.get("entries", [])
    )
    if not strong:
        return []
    if ledger.get("entries") and any(term in accepted_text for term in strong):
        return [
            _shadow(
                persona,
                "framing",
                "assumption",
                "该数字人的采信证据与自身强专业域高度贴合，可能出现专家隧道视野。",
                lenses=["expert_failure", "paradigm"],
                likely_causes=["专业知识域强化", "群体内偏爱", "刺猬型解释风险"],
                related_parameters=["expertise_strong", "information_filter", "cognitive_frames"],
                bias_direction="向自身专业域可解释变量倾斜",
                strength="medium",
                puzzle_use="constraint_band",
            )
        ]
    return []


def _diagnose_emotion_shadows(
    persona: dict[str, Any],
    ledger: dict[str, Any],
    lenses: dict[str, LensCard],
) -> list[dict[str, Any]]:
    del lenses
    temperament = persona.get("temperament", "")
    ignored_text = " ".join(
        f"{item.get('result', {}).get('evidence_type', '')} {item.get('result', {}).get('snippet', '')}"
        for item in ledger.get("ignored_sources", [])
    )
    shadows: list[dict[str, Any]] = []
    if any(token in temperament for token in ["悲观", "乐观", "同情", "怀疑"]):
        shadows.append(
            _shadow(
                persona,
                "value",
                "value_evaluation",
                f"气质基调为“{temperament}”，可能影响风险/机会/伤害的显著性。",
                lenses=["emotion_reason", "bias_heuristics"],
                likely_causes=["情绪基调", "躯体标记", "注意力分配"],
                related_parameters=["temperament", "risk_attitude"],
                bias_direction="改变风险、机会或伤害的显著性权重",
                strength="medium",
                puzzle_use="constraint_band",
            )
        )
    if any(token in ignored_text for token in ["质性叙事", "伤害", "访谈", "权利"]):
        shadows.append(
            _shadow(
                persona,
                "silence",
                "information_filter",
                "该数字人忽略了可能承载情绪、身体性或尊严成本的证据。",
                lenses=["emotion_reason", "ecological_rationality"],
                likely_causes=["低估情绪性证据", "偏好可量化材料"],
                related_parameters=["ignored_sources", "ignored_evidence"],
                bias_direction="低估人类处境与体验成本",
                strength="medium",
                puzzle_use="negative_space",
            )
        )
    return shadows


def _diagnose_collective_blind_spots(
    question: str,
    personas: list[dict[str, Any]],
    ledgers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del personas
    ignored_counter: Counter[str] = Counter()
    accepted_counter: Counter[str] = Counter()
    for ledger in ledgers:
        for item in ledger.get("ignored_sources", []):
            evidence_type = item.get("result", {}).get("evidence_type", "unknown")
            ignored_counter[evidence_type] += 1
        for entry in ledger.get("entries", []):
            accepted_counter[entry.get("evidence_type", "unknown")] += 1

    blind_spots = []
    for evidence_type, ignored_count in ignored_counter.most_common():
        accepted_count = accepted_counter.get(evidence_type, 0)
        if ignored_count >= 2 and ignored_count > accepted_count:
            blind_spots.append(
                {
                    "dimension": evidence_type,
                    "description": (
                        f"回答“{question}”时，{evidence_type} 被忽略次数高于采信次数。"
                        "这只是集体漏看候选；还要证明它与本题判断目标直接相关。"
                    ),
                    "ignored_count": ignored_count,
                    "accepted_count": accepted_count,
                    "relevance_status": "candidate_unverified",
                    "puzzle_use": "negative_space",
                    "recommended_probe": (
                        f"先验证 {evidence_type} 与本题的机制关系；相关后，再生成专门检索并检验它的反叛者数字人。"
                    ),
                }
            )
    return blind_spots


def _whole_scope_flags(text: str, confidence: float | None) -> list[str]:
    flags: list[str] = []
    scope_tokens = {
        "整体判断": ["整体", "全部", "所有", "普遍", "全面", "系统性", "结构性", "全局", "市场", "社会", "国家"],
        "时间判断": ["很快", "马上", "立即", "短期", "长期", "未来", "不远的将来", "最终", "必将"],
        "强因果判断": ["导致", "证明", "必然", "一定", "不可避免", "说明", "所以", "因此"],
        "高确定性判断": ["很可能", "高度可能", "确定", "显然", "无疑", "高于主流预期"],
    }
    for label, tokens in scope_tokens.items():
        if any(token in text for token in tokens):
            flags.append(label)
    if confidence is not None and confidence >= 0.68:
        flags.append("置信度偏高")
    return flags


def _evidence_floor(
    entries: list[dict[str, Any]],
    evidence_types: set[str],
    source_types: set[str],
    retrieval_layers: set[str],
) -> str:
    if not entries:
        return "没有可采信证据，最多只能形成待检假设。"
    evidence = "、".join(sorted(evidence_types)) or "未知证据类型"
    sources = "、".join(sorted(source_types)) or "未知来源类型"
    layers = "、".join(sorted(retrieval_layers)) or "未知检索层"
    return f"当前账本有 {len(entries)} 条采信材料，主要覆盖 {evidence}，来源类型为 {sources}，检索层为 {layers}。"


def _conclusion_ceiling(conclusion: str, whole_scope_flags: list[str]) -> str:
    flags = "、".join(whole_scope_flags)
    return f"结论正在声明“{_short(conclusion, 180)}”，触发 {flags}，已经超过单一局部材料可自然支撑的范围。"


def _missing_bridge_questions(local_flags: list[str], whole_flags: list[str]) -> list[str]:
    questions = [
        "从局部材料到整体判断，中间缺少哪一级机制？",
        "从短期信号到长期结论，中间缺少哪类时间尺度变量？",
        "这个局部案例是否有样本代表性、基准率或同类反例校准？",
    ]
    if any("模型先验" in flag for flag in local_flags):
        questions.append("模型先验被当成事实证据了吗？外部核验在哪里？")
    if any("来源类型单一" in flag for flag in local_flags):
        questions.append("同一来源类型之外的证据是否会改变结论范围？")
    if any("置信度" in flag for flag in whole_flags):
        questions.append("置信度是否由证据质量支撑，还是由叙事流畅度支撑？")
    return questions


def _residual_local_fragment(entries: list[dict[str, Any]], output: dict[str, Any]) -> str:
    if entries:
        evidence_types = [
            str(entry.get("evidence_type"))
            for entry in entries
            if entry.get("evidence_type")
        ]
        dominant = Counter(evidence_types).most_common(1)
        if dominant:
            return f"删掉整体化结论后，仍可保留它对“{dominant[0][0]}”这一局部维度的敏感性。"
    assumptions = output.get("core_assumptions", []) or []
    if assumptions:
        return f"删掉越界结论后，保留这个待检假设：{_short(str(assumptions[0]), 120)}"
    return "删掉越界结论后，仅保留为低置信待检假设。"


def _index_person_outputs(person_outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in person_outputs:
        person_id = (
            item.get("person_id")
            or item.get("person", {}).get("id")
            or item.get("output", {}).get("person_id")
        )
        if person_id:
            indexed[str(person_id)] = item
    return indexed


def _scan_output_disagreement(
    question: str,
    personas: list[dict[str, Any]],
    outputs_by_person: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stance_clusters: defaultdict[str, list[str]] = defaultdict(list)
    confidence_values: list[float] = []
    conclusion_samples: list[dict[str, Any]] = []

    for persona in personas:
        person_id = persona["id"]
        record = outputs_by_person.get(person_id)
        if not record:
            continue
        output = record.get("output", record)
        conclusion = str(output.get("conclusion", ""))
        stance = _classify_stance(conclusion, question, output.get("answer_position"))
        stance_clusters[stance].append(person_id)
        confidence = output.get("confidence")
        if isinstance(confidence, int | float):
            confidence_values.append(float(confidence))
        conclusion_samples.append(
            {
                "person_id": person_id,
                "stance": stance,
                "confidence": confidence,
                "conclusion": conclusion[:300],
            }
        )

    return {
        "stance_clusters": dict(stance_clusters),
        "confidence": _confidence_summary(confidence_values),
        "conclusion_samples": conclusion_samples,
        "diagnosis": "输出分歧用于补充身份分歧；若 stance 与价值/框架聚类不一致，应优先复查证据账本和推理链。",
    }


def _classify_stance(
    text: str,
    question: str = "",
    answer_position: dict[str, Any] | None = None,
) -> str:
    """Use the person's structured answer, with a topic-neutral legacy fallback."""
    if isinstance(answer_position, dict):
        directness = str(answer_position.get("directness", "")).strip().lower()
        direction = str(answer_position.get("direction", "")).strip().lower()
        if directness == "reframed":
            return "reframed"
        mapped = {
            "yes": "likely_yes",
            "no": "likely_no",
            "conditional": "conditional",
            "uncertain": "undetermined",
            "not_applicable": "reframed",
        }.get(direction)
        if mapped:
            return mapped

    if not text:
        return "missing"
    lowered = text.lower()
    compact = "".join(lowered.split())

    if any(token in compact for token in ["核心不在于", "本质上是", "真正的问题是", "问题应改为"]):
        return "reframed"
    if any(
        token in lowered
        for token in [
            "无法断言",
            "无法判断",
            "不能确定",
            "尚无定论",
            "证据不足以判断",
            "cannot determine",
            "cannot conclude",
            "insufficient evidence",
        ]
    ) or ("无法" in compact and "断言" in compact):
        return "undetermined"
    if any(token in lowered for token in ["取决于", "前提是", "只有在", "depends on", "conditional on"]):
        return "conditional"
    if any(token in lowered for token in ["不能排除", "并非不可能", "cannot rule out"]):
        return "possibility_open"

    normative = any(token in question for token in ["该不该", "应不应该", "是否应该", "要不要"])
    if normative and any(token in lowered for token in ["不应该", "反对", "暂停", "禁止", "oppose", "pause"]):
        return "oppose"
    if normative and any(token in lowered for token in ["应该", "支持", "赞成", "推进", "support"]):
        return "support"
    if any(token in compact for token in ["答案是否定", "结论是否定", "倾向于否定"]):
        return "likely_no"
    if any(token in compact for token in ["答案是肯定", "结论是肯定", "倾向于肯定"]):
        return "likely_yes"
    return "unclear"


def _confidence_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
    }


def _build_blind_spot_registry(
    collective_blind_spots: list[dict[str, Any]],
    outputs_by_person: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    registry = [
        {
            "source": "evidence_ledger",
            "dimension": item["dimension"],
            "description": item["description"],
            "relevance_status": item.get("relevance_status", "candidate_unverified"),
            "puzzle_use": item["puzzle_use"],
            "recommended_probe": item["recommended_probe"],
        }
        for item in collective_blind_spots
    ]
    underweighted_counter: Counter[str] = Counter()
    for person_id, record in outputs_by_person.items():
        output = record.get("output", record)
        for item in output.get("what_i_underweighted", []) or []:
            underweighted_counter[str(item)] += 1
    for dimension, count in underweighted_counter.most_common():
        if count >= 2:
            registry.append(
                {
                    "source": "person_self_report",
                    "dimension": dimension,
                    "description": f"{count} 个数字人自报低估该维度，应作为候选集体盲区复查。",
                    "relevance_status": "candidate_unverified",
                    "puzzle_use": "negative_space",
                    "recommended_probe": f"生成专门追问 {dimension} 的反叛者数字人。",
                }
            )
    return registry


def _scan_disagreement_potential(personas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_top_value: defaultdict[str, list[str]] = defaultdict(list)
    by_frame: defaultdict[str, list[str]] = defaultdict(list)
    for persona in personas:
        values = persona.get("values", {})
        if values:
            by_top_value[max(values.items(), key=lambda item: item[1])[0]].append(persona["id"])
        for frame in persona.get("cognitive_frames", []):
            by_frame[frame].append(persona["id"])
    return [
        {
            "type": "value_conflict_potential",
            "clusters": dict(by_top_value),
            "diagnosis": "价值最高权重不同的数字人之间，可能产生事实无法消解的价值分歧。",
        },
        {
            "type": "frame_conflict_potential",
            "clusters": {key: value for key, value in by_frame.items() if len(value) >= 2},
            "diagnosis": "认知框架集群显示潜在范式冲突或分析层次错位。",
        },
    ]


def _build_inversion_item(shadow: dict[str, Any], lenses: dict[str, LensCard]) -> dict[str, Any]:
    lens_ids = shadow["attribution"]["classic_lenses"]
    inversion_rules: list[str] = []
    boundary_conditions: list[str] = []
    anti_misuse_rules: list[str] = []
    for lens_id in lens_ids:
        lens = lenses.get(lens_id)
        if lens is not None:
            inversion_rules.extend(lens.inversion_rules[:2])
            boundary_conditions.extend(lens.boundary_conditions[:1])
            anti_misuse_rules.extend(lens.anti_misuse_rules[:1])
    return {
        "shadow_id": shadow["id"],
        "person_id": shadow["person_id"],
        "puzzle_use": shadow["puzzle_use"],
        "estimated_bias_vector": shadow["estimated_bias_vector"],
        "inversion_rules": list(dict.fromkeys(inversion_rules)),
        "boundary_conditions": list(dict.fromkeys(boundary_conditions)),
        "anti_misuse_rules": list(dict.fromkeys(anti_misuse_rules)),
        "resulting_material": _material_name(shadow["puzzle_use"]),
    }


def _select_micro_lenses(
    shadow: dict[str, Any],
    micro_lenses: dict[str, MicroLens],
    *,
    max_per_parent: int = 2,
) -> list[dict[str, Any]]:
    lens_ids = shadow["attribution"]["classic_lenses"]
    selected: list[dict[str, Any]] = []
    for lens_id in lens_ids:
        candidates = [lens for lens in micro_lenses.values() if lens.parent_lens == lens_id]
        ranked = sorted(candidates, key=lambda lens: _micro_lens_score(shadow, lens), reverse=True)
        for lens in ranked[:max_per_parent]:
            selected.append(
                {
                    "micro_lens_id": lens.id,
                    "parent_lens": lens.parent_lens,
                    "name": lens.name,
                    "mechanism": lens.mechanism,
                    "diagnostic_questions": lens.diagnostic_questions[:2],
                    "differentiates_from": lens.differentiates_from,
                    "inversion_hint": lens.inversion_hint,
                    "misuse_warning": lens.misuse_warning,
                }
            )
    return selected


def _select_shadow_mechanisms(
    shadow: dict[str, Any],
    mechanisms: dict[str, ShadowMechanism],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        mechanisms.values(),
        key=lambda mechanism: _shadow_mechanism_score(shadow, mechanism),
        reverse=True,
    )
    selected = []
    for mechanism in ranked[:limit]:
        score = _shadow_mechanism_score(shadow, mechanism)
        if score <= 0:
            continue
        selected.append(
            {
                "mechanism_id": mechanism.id,
                "name": mechanism.name,
                "shadow_family": mechanism.shadow_family,
                "core_mechanism": mechanism.core_mechanism,
                "generation_chain": mechanism.generation_chain,
                "capture_questions": mechanism.capture_questions[:2],
                "false_positive_risks": mechanism.false_positive_risks[:1],
                "false_negative_risks": mechanism.false_negative_risks[:1],
                "puzzle_material": mechanism.puzzle_material,
                "match_score": score,
            }
        )
    return selected


def _shadow_mechanism_score(shadow: dict[str, Any], mechanism: ShadowMechanism) -> int:
    score = 0
    if mechanism.shadow_family == shadow.get("shadow_type"):
        score += 4
    if shadow.get("source_stage") in mechanism.chain_stages:
        score += 3
    if mechanism.puzzle_material == shadow.get("puzzle_use"):
        score += 2
    lens_ids = set(shadow.get("attribution", {}).get("classic_lenses", []))
    score += len(lens_ids & set(mechanism.linked_lenses))
    haystack = " ".join(
        [
            shadow.get("description", ""),
            shadow.get("source_stage", ""),
            shadow.get("shadow_type", ""),
            shadow.get("puzzle_use", ""),
            " ".join(shadow.get("attribution", {}).get("likely_causes", [])),
            " ".join(shadow.get("attribution", {}).get("related_parameters", [])),
        ]
    )
    for text in [mechanism.name, mechanism.core_mechanism, *mechanism.visible_signals]:
        for token in _diagnostic_tokens(text):
            if token in haystack:
                score += 1
    return score


def _micro_lens_score(shadow: dict[str, Any], micro_lens: MicroLens) -> int:
    haystack = " ".join(
        [
            shadow.get("description", ""),
            shadow.get("source_stage", ""),
            shadow.get("shadow_type", ""),
            " ".join(shadow.get("attribution", {}).get("likely_causes", [])),
            " ".join(shadow.get("attribution", {}).get("related_parameters", [])),
            shadow.get("estimated_bias_vector", {}).get("direction", ""),
        ]
    ).lower()
    score = 0
    for text in [
        micro_lens.name,
        micro_lens.mechanism,
        micro_lens.inversion_hint,
        *micro_lens.trigger_signals,
    ]:
        for token in _diagnostic_tokens(text):
            if token in haystack:
                score += 1
    return score


def _diagnostic_tokens(text: str) -> list[str]:
    tokens = [
        "证据",
        "结论",
        "风险",
        "情绪",
        "价值",
        "身份",
        "专家",
        "范式",
        "噪声",
        "反例",
        "因果",
        "锚",
        "基准率",
        "来源",
        "置信",
        "外推",
        "同温层",
        "受影响者",
        "地方",
        "尺度",
        "概念",
        "权威",
        "社会",
        "解释",
        "局部",
        "整体",
        "越界",
        "范围",
        "样本",
        "代表性",
        "短期",
        "长期",
        "同源",
        "先验",
    ]
    return [token for token in tokens if token in text]


def _build_differential_diagnosis(
    shadows: list[dict[str, Any]],
    conflict_rules: list[LensConflictRule],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for shadow in shadows:
        lens_ids = set(shadow["attribution"]["classic_lenses"])
        matches: list[dict[str, Any]] = []
        for rule in conflict_rules:
            competing = set(rule.competing_lenses)
            overlap = competing & lens_ids
            if not overlap:
                continue
            status = "active_competition" if competing <= lens_ids else "watch_for_confusion"
            matches.append(
                {
                    "rule_id": rule.id,
                    "status": status,
                    "competing_lenses": rule.competing_lenses,
                    "confusion_pattern": rule.confusion_pattern,
                    "decision_questions": rule.decision_questions,
                    "prefer_first_when": rule.prefer_first_when[:2],
                    "prefer_second_when": rule.prefer_second_when[:2],
                    "coexist_when": rule.coexist_when,
                    "misdiagnosis_risk": rule.misdiagnosis_risk,
                }
            )
        if matches:
            result.append(
                {
                    "shadow_id": shadow["id"],
                    "person_id": shadow["person_id"],
                    "shadow_type": shadow["shadow_type"],
                    "candidate_lenses": list(lens_ids),
                    "rules": matches[:3],
                }
            )
    return result


def _build_misdiagnosis_audit(differential_diagnosis: list[dict[str, Any]]) -> dict[str, Any]:
    risk_counter: Counter[str] = Counter()
    active_competitions = 0
    for item in differential_diagnosis:
        for rule in item["rules"]:
            risk_counter[rule["rule_id"]] += 1
            if rule["status"] == "active_competition":
                active_competitions += 1
    return {
        "active_competition_count": active_competitions,
        "watch_item_count": len(differential_diagnosis),
        "most_common_confusions": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in risk_counter.most_common(5)
        ],
        "audit_questions": [
            "这条阴影是否有稳定方向，还是只是判断噪声？",
            "这个启发式是错误偏见，还是环境适配规则？",
            "这是普通论证漏洞，还是范式不可通约？",
            "这条推理是真的生成结论，还是事后包装？",
            "元模型是否因为透镜可用而过度诊断？",
        ],
    }


def _build_local_whole_overreach_audit(shadows: list[dict[str, Any]]) -> dict[str, Any]:
    overreach_shadows = [
        shadow
        for shadow in shadows
        if shadow.get("detector_id") == "local_whole_overreach"
    ]
    missing_bridge_counter: Counter[str] = Counter()
    for shadow in overreach_shadows:
        pressure = shadow.get("evidence_pressure", {})
        for question in pressure.get("missing_bridges", []):
            missing_bridge_counter[question] += 1

    return {
        "detector_id": "local_whole_overreach",
        "description": "专门审计局部精确如何伪装成整体真相。",
        "trigger_count": len(overreach_shadows),
        "person_ids": [shadow["person_id"] for shadow in overreach_shadows],
        "most_common_missing_bridges": [
            {"question": question, "count": count}
            for question, count in missing_bridge_counter.most_common(8)
        ],
        "hard_rules": [
            "结论声称整体时，必须指出证据实际只支持到哪里。",
            "允许保留局部看对的部分，但必须切掉越界结论。",
            "模型先验、类比、个案、短期信号不能直接升级为整体判断。",
            "任何“很快、必然、系统性、全面”的结论都必须交出尺度桥和时间桥。",
        ],
    }


def _build_detector_hardening_audit(
    lenses: dict[str, LensCard],
    shadows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trigger_counter: Counter[str] = Counter(
        lens_id
        for shadow in shadows
        for lens_id in shadow.get("attribution", {}).get("classic_lenses", [])
    )
    hardening_questions = {
        "bias_heuristics": "这个偏差有没有被具体钉到样本、基准率、先验、显著性或置信度上？",
        "noise": "这是稳定方向的阴影，还是元模型把随机漂移解释得太深？",
        "emotion_reason": "情绪是在扭曲概率，还是在保存指标看不见的人类成本？",
        "interpreter": "推理是真的生成结论，还是结论先在、理由后补？改变条件可触发吗？",
        "social_influence": "信任来自方法质量，还是权威、身份、同温层和可见性？",
        "argument_forensics": "证据实际支持到哪一步？哪一步发生因果、尺度、时间或概念跳跃？",
        "expert_failure": "它在哪个局部真的看得清？跨出哪里开始失真？",
        "paradigm": "双方是否共享问题对象、成功标准和证据规则？补事实后是否仍不收敛？",
        "ecological_rationality": "这个简单规则来自高反馈环境，还是旧环境错配？适用边界在哪里？",
    }
    return [
        {
            "lens_id": lens_id,
            "name": lens.name,
            "trigger_count": trigger_counter.get(lens_id, 0),
            "needs_hardening": trigger_counter.get(lens_id, 0) == 0,
            "proof_burden": "不能只贴透镜名；必须落到证据账本、推理步骤、价值权重或缺席维度。",
            "hard_question": hardening_questions.get(lens_id, "这条检测器是否有足够可观察证据？"),
            "failure_if_soft": "元模型会变成概念标签机，不能把阴影转成拼图材料。",
        }
        for lens_id, lens in sorted(lenses.items())
    ]


def _build_calibration_reminders(calibration_cases: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.id,
            "title": case.title,
            "expected_primary_lenses": case.expected_primary_lenses,
            "expected_confusions": case.expected_confusions,
            "calibration_questions": case.calibration_questions,
            "failure_mode": case.failure_mode,
        }
        for case in calibration_cases
    ]


def _build_shadow_mechanism_ontology(mechanisms: dict[str, ShadowMechanism]) -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": mechanism.id,
            "name": mechanism.name,
            "shadow_family": mechanism.shadow_family,
            "core_mechanism": mechanism.core_mechanism,
            "generation_chain": mechanism.generation_chain,
            "chain_stages": mechanism.chain_stages,
            "linked_lenses": mechanism.linked_lenses,
            "puzzle_material": mechanism.puzzle_material,
        }
        for mechanism in mechanisms.values()
    ]


def _build_capture_protocol(protocols: list[CaptureProtocolStage]) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage.stage,
            "touch_target": stage.touch_target,
            "what_to_inspect": stage.what_to_inspect,
            "shadow_questions": stage.shadow_questions,
            "evidence_needed": stage.evidence_needed,
            "failure_if_skipped": stage.failure_if_skipped,
        }
        for stage in protocols
    ]


def _build_mechanism_coverage(
    shadows: list[dict[str, Any]],
    mechanisms: dict[str, ShadowMechanism],
) -> dict[str, Any]:
    shadow_families = Counter(shadow["shadow_type"] for shadow in shadows)
    stage_counter = Counter(shadow["source_stage"] for shadow in shadows)
    ontology_families = Counter(mechanism.shadow_family for mechanism in mechanisms.values())
    ontology_stages = Counter(stage for mechanism in mechanisms.values() for stage in mechanism.chain_stages)
    return {
        "observed_shadow_families": dict(shadow_families),
        "observed_source_stages": dict(stage_counter),
        "ontology_shadow_families": dict(ontology_families),
        "ontology_chain_stages": dict(ontology_stages),
        "coverage_warning": _coverage_warning(shadow_families, stage_counter),
    }


def _coverage_warning(shadow_families: Counter[str], stage_counter: Counter[str]) -> list[str]:
    warnings = []
    for family in ["informational", "framing", "value", "reasoning", "silence", "distorted"]:
        if shadow_families.get(family, 0) == 0:
            warnings.append(f"本次尚未捕捉到 {family} 阴影，需确认是真缺席还是漏诊。")
    for stage in ["information_filter", "evidence_selection", "assumption", "reasoning", "value_evaluation", "conclusion"]:
        if stage_counter.get(stage, 0) == 0:
            warnings.append(f"链条阶段 {stage} 尚无阴影记录，需复查摸盲人是否完整。")
    return warnings


def _build_mechanism_audit(
    shadows: list[dict[str, Any]],
    mechanisms: dict[str, ShadowMechanism],
) -> dict[str, Any]:
    del shadows
    return {
        "false_positive_watch": list(
            dict.fromkeys(
                risk
                for mechanism in mechanisms.values()
                for risk in mechanism.false_positive_risks[:1]
            )
        )[:8],
        "false_negative_watch": list(
            dict.fromkeys(
                risk
                for mechanism in mechanisms.values()
                for risk in mechanism.false_negative_risks[:1]
            )
        )[:8],
        "core_audit_questions": [
            "这条阴影是从哪个链条阶段生成的？",
            "它是入口前缺席、证据选择、假设冻结、推理断裂、价值放大，还是元模型过度诊断？",
            "这条阴影有没有足够可见信号，还是只是元模型的解释欲？",
            "是否存在同一现象的竞争机制解释？",
        ],
    }


def _build_classic_grounding(lenses: dict[str, LensCard]) -> list[dict[str, Any]]:
    return [
        {
            "lens_id": lens.id,
            "source_works": lens.source_works,
            "core_idea": lens.core_idea,
            "mechanism_chain": lens.mechanism_chain,
            "model_training_notes": lens.model_training_notes,
        }
        for lens in lenses.values()
    ]


def _lens_mechanism_summaries(lens_ids: list[str], lenses: dict[str, LensCard]) -> list[dict[str, Any]]:
    summaries = []
    for lens_id in lens_ids:
        lens = lenses.get(lens_id)
        if lens is not None:
            summaries.append(
                {
                    "lens_id": lens.id,
                    "core_idea": lens.core_idea,
                    "mechanism_chain": lens.mechanism_chain[:3],
                }
            )
    return summaries


def _build_meta_reflection(lenses: dict[str, LensCard]) -> dict[str, Any]:
    reflections: list[str] = []
    anti_misuse_rules: list[str] = []
    for lens in lenses.values():
        reflections.extend(lens.self_reflection[:1])
        anti_misuse_rules.extend(lens.anti_misuse_rules[:1])
    return {
        "possible_meta_biases": list(dict.fromkeys(reflections)),
        "anti_misuse_rules": list(dict.fromkeys(anti_misuse_rules)),
        "underweighted_dimensions": ["情绪性证据", "地方知识", "不可形式化的文化象征", "随机噪声"],
        "confidence": 0.72,
    }


def _shadow_from_chain_finding(
    persona: dict[str, Any],
    shadow_spec: dict[str, Any],
) -> dict[str, Any]:
    return _shadow(
        persona,
        shadow_spec["shadow_type"],
        shadow_spec["source_stage"],
        shadow_spec["description"],
        lenses=shadow_spec["lenses"],
        likely_causes=shadow_spec["likely_causes"],
        related_parameters=shadow_spec["related_parameters"],
        bias_direction=shadow_spec["bias_direction"],
        strength=shadow_spec["strength"],
        puzzle_use=shadow_spec["puzzle_use"],
        detector_id=shadow_spec["detector_id"],
        plain_language=shadow_spec.get("plain_language"),
        chain_diagnosis=shadow_spec.get("chain_diagnosis"),
    )


def _shadow(
    persona: dict[str, Any],
    shadow_type: str,
    source_stage: str,
    description: str,
    *,
    lenses: list[str],
    likely_causes: list[str],
    related_parameters: list[str],
    bias_direction: str,
    strength: str,
    puzzle_use: str,
    detector_id: str | None = None,
    plain_language: str | None = None,
    evidence_pressure: dict[str, Any] | None = None,
    chain_diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_id = f"{persona['id']}|{shadow_type}|{source_stage}|{description}"
    shadow_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    result = {
        "id": f"shadow_{shadow_id}",
        "person_id": persona["id"],
        "shadow_type": shadow_type,
        "description": description,
        "source_stage": source_stage,
        "attribution": {
            "likely_causes": likely_causes,
            "related_parameters": related_parameters,
            "classic_lenses": lenses,
        },
        "estimated_bias_vector": {
            "direction": bias_direction,
            "strength": strength,
            "uncertainty": "medium",
        },
        "puzzle_use": puzzle_use,
    }
    if detector_id:
        result["detector_id"] = detector_id
    if plain_language:
        result["plain_language"] = plain_language
    if evidence_pressure:
        result["evidence_pressure"] = evidence_pressure
    if chain_diagnosis:
        result["chain_diagnosis"] = chain_diagnosis
    return result


def _strength(part: int, whole: int) -> str:
    if whole <= 0:
        return "unknown"
    ratio = part / whole
    if ratio >= 0.6:
        return "high"
    if ratio >= 0.25:
        return "medium"
    return "low"


def _bias_direction_from_values(values: dict[str, float]) -> str:
    if not values:
        return "未知方向"
    top_value = max(values.items(), key=lambda item: item[1])[0]
    return f"向 {top_value} 相关解释和行动建议倾斜"


def _material_name(puzzle_use: str) -> str:
    return {
        "anchor": "互锁锚点候选",
        "constraint_band": "偏矫正约束带",
        "negative_space": "负空间洞",
        "downweight": "降权片段",
        "fracture_boundary": "推理断裂边界",
        "unknown": "未知材料",
    }.get(puzzle_use, "未知材料")


def _short(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 1)]}…"


def _empty_ledger(person_id: str, question: str) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "question": question,
        "entries": [],
        "accepted_sources": [],
        "rejected_sources": [],
        "ignored_sources": [],
        "source_decisions": [],
        "information_shadow_hints": [],
    }
