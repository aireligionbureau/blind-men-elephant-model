from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .evidence_normalization import normalize_output_evidence_entries


def build_shadow_puzzle(
    question: str,
    diagnosis: dict[str, Any],
    *,
    person_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shadows = diagnosis.get("shadow_registry", [])
    inversion_plan = diagnosis.get("inversion_plan", [])
    blind_spots = diagnosis.get("blind_spot_registry") or diagnosis.get("collective_blind_spots", [])
    outputs = person_outputs or []

    return {
        "question": question,
        "puzzle_material_counts": _count_materials(inversion_plan),
        "interlocking_anchors": _build_interlocking_anchors(diagnosis, outputs),
        "bias_corrected_fragments": _build_bias_corrected_fragments(inversion_plan, shadows),
        "negative_space_holes": _build_negative_space_holes(blind_spots, shadows),
        "fracture_boundaries": _build_fracture_boundaries(inversion_plan, shadows),
        "puzzle_quality": _score_puzzle_quality(diagnosis, inversion_plan, blind_spots),
        "next_probe": _build_next_probe(diagnosis, blind_spots),
    }


def _count_materials(inversion_plan: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(item.get("puzzle_use", "unknown") for item in inversion_plan)
    return dict(counter)


def _build_interlocking_anchors(
    diagnosis: dict[str, Any],
    person_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    stance_clusters = (
        diagnosis.get("disagreement_structure", {})
        .get("output_based", {})
        .get("stance_clusters", {})
    )
    if stance_clusters:
        for stance, person_ids in stance_clusters.items():
            if stance != "missing" and len(person_ids) >= 2:
                anchors.append(
                    {
                        "type": "conclusion_overlap",
                        "description": f"{len(person_ids)} 个数字人输出呈现 {stance} 倾向，可作为低强度重合锚点候选。",
                        "person_ids": person_ids,
                        "confidence": "low" if stance in {"unclear", "conditional"} else "medium",
                        "caution": "结论重合不等于事实为真，需检查这些数字人是否共享同一信息过滤器或价值权重。",
                    }
                )

    evidence_counter: Counter[str] = Counter()
    evidence_people: defaultdict[str, set[str]] = defaultdict(set)
    for record in person_outputs:
        output = record.get("output", record)
        person_id = record.get("person_id") or record.get("person", {}).get("id") or output.get("person_id")
        ledger = normalize_output_evidence_entries(output.get("evidence_ledger"))
        for entry in ledger:
            claim = str(entry.get("claim", "")).strip()
            if claim:
                key = claim[:160]
                evidence_counter[key] += 1
                if person_id:
                    evidence_people[key].add(str(person_id))
    for claim, count in evidence_counter.most_common(8):
        if count >= 2:
            anchors.append(
                {
                    "type": "shared_evidence_claim",
                    "description": claim,
                    "support_count": count,
                    "person_ids": sorted(evidence_people[claim]),
                    "confidence": "medium",
                    "caution": "共享证据需要排除同源复制和同温层采信。",
                }
            )

    if not anchors:
        anchors.append(
            {
                "type": "insufficient_anchor",
                "description": "尚未形成可用互锁锚点；应优先补充异质数字人和反向证据。",
                "person_ids": [],
                "confidence": "low",
                "caution": "不要为了生成真相轮廓而强行制造锚点。",
            }
        )
    return anchors


def _build_bias_corrected_fragments(
    inversion_plan: list[dict[str, Any]],
    shadows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shadows_by_id = {shadow["id"]: shadow for shadow in shadows}
    fragments = []
    for item in inversion_plan:
        if item.get("puzzle_use") not in {"constraint_band", "downweight"}:
            continue
        shadow = shadows_by_id.get(item["shadow_id"], {})
        fragments.append(
            {
                "shadow_id": item["shadow_id"],
                "person_id": item["person_id"],
                "material_type": "bias_corrected_fragment",
                "source_shadow": shadow.get("description", ""),
                "estimated_bias_vector": item.get("estimated_bias_vector", {}),
                "correction_rules": item.get("inversion_rules", []),
                "boundary_conditions": item.get("boundary_conditions", []),
                "anti_misuse_rules": item.get("anti_misuse_rules", []),
                "confidence": _fragment_confidence(item),
            }
        )
    return fragments


def _build_negative_space_holes(
    blind_spots: list[dict[str, Any]],
    shadows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    holes = []
    for item in blind_spots:
        holes.append(
            {
                "source": item.get("source", "collective_blind_spot"),
                "dimension": item.get("dimension", "unknown"),
                "description": item.get("description", ""),
                "recommended_probe": item.get("recommended_probe", ""),
                "confidence": "medium",
            }
        )
    for shadow in shadows:
        if shadow.get("puzzle_use") == "negative_space":
            holes.append(
                {
                    "source": "shadow_registry",
                    "dimension": shadow.get("source_stage", "unknown"),
                    "description": shadow.get("description", ""),
                    "recommended_probe": "补充专门能看见该缺席维度的数字人或证据源。",
                    "confidence": shadow.get("estimated_bias_vector", {}).get("strength", "medium"),
                }
            )
    return holes


def _build_fracture_boundaries(
    inversion_plan: list[dict[str, Any]],
    shadows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shadows_by_id = {shadow["id"]: shadow for shadow in shadows}
    boundaries = []
    for item in inversion_plan:
        if item.get("puzzle_use") != "fracture_boundary":
            continue
        shadow = shadows_by_id.get(item["shadow_id"], {})
        boundaries.append(
            {
                "shadow_id": item["shadow_id"],
                "person_id": item["person_id"],
                "description": shadow.get("description", ""),
                "source_stage": shadow.get("source_stage", ""),
                "repair_needed": item.get("inversion_rules", []),
                "confidence": item.get("estimated_bias_vector", {}).get("strength", "medium"),
            }
        )
    return boundaries


def _score_puzzle_quality(
    diagnosis: dict[str, Any],
    inversion_plan: list[dict[str, Any]],
    blind_spots: list[dict[str, Any]],
) -> dict[str, Any]:
    readiness = diagnosis.get("readiness_evaluation", {}).get("score", 0)
    material_count = len(inversion_plan) + len(blind_spots)
    quality = min(100, round(readiness * 0.65 + min(material_count, 40) / 40 * 35, 2))
    return {
        "score": quality,
        "readiness_score": readiness,
        "material_count": material_count,
        "caution": "这是拼图材料质量分，不是真相准确率。真实准确率需要案例校准和外部事实核查。",
    }


def _build_next_probe(
    diagnosis: dict[str, Any],
    blind_spots: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_questions = diagnosis.get("misdiagnosis_audit", {}).get("audit_questions", [])
    return {
        "blind_spot_probes": [
            item.get("recommended_probe")
            for item in blind_spots
            if item.get("recommended_probe")
        ][:8],
        "diagnostic_stress_tests": audit_questions[:5],
        "recommended_new_personas": [
            "反叛者数字人：专门信任被忽略证据类型。",
            "尺度翻译者数字人：检查个体、组织、制度、长期系统之间的外推。",
            "元模型审计者数字人：专门攻击元模型过度诊断。",
        ],
    }


def _fragment_confidence(item: dict[str, Any]) -> str:
    strength = item.get("estimated_bias_vector", {}).get("strength", "medium")
    if strength == "high":
        return "medium"
    if strength == "low":
        return "low"
    return "medium"
