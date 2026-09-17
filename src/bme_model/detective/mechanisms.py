from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schemas import compact_text, stable_id, unique_strings


MECHANISM_INDEX_VERSION = "bme.mechanism-index.v1"
MAX_MECHANISM_SLOTS = 10

_CONSTRUCTIVE_RELATION_TYPES = {
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
}

_STOP_TERMS = {
    "问题",
    "判断",
    "当前",
    "可能",
    "相关",
    "影响",
    "变化",
    "情况",
    "程度",
    "是否",
    "以及",
    "不同",
    "证据",
    "材料",
}


def build_mechanism_index(
    question: str,
    question_frame: dict[str, Any] | None,
    pieces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a cheap question-native coverage index without another model call."""

    frame = question_frame or {}
    dimensions = unique_strings(frame.get("relevant_evidence_dimensions") or [])
    source = "question_frame.relevant_evidence_dimensions"
    if len(dimensions) < 4:
        dimensions = unique_strings(
            [*dimensions, *(frame.get("relevant_knowledge_domains") or [])]
        )
        source = "question_frame.evidence_dimensions_and_domains"
    dimensions = dimensions[:MAX_MECHANISM_SLOTS]

    slots: list[dict[str, Any]] = []
    for position, description in enumerate(dimensions, start=1):
        label = _slot_label(description)
        slot_id = stable_id("mechanism", question, label, description, size=12)
        slots.append(
            {
                "slot_id": slot_id,
                "label": label,
                "description": description,
                "slot_type": _slot_type(description),
                "priority": "high" if position <= 6 else "medium",
                "source": source,
                "keywords": _keywords(description),
            }
        )

    assignments: defaultdict[str, list[str]] = defaultdict(list)
    for piece in pieces:
        ranked = _rank_slots(piece, slots)
        mechanism_ids = [slot_id for score, slot_id in ranked[:2] if score >= 0.1]
        piece["mechanism_ids"] = mechanism_ids
        for slot_id in mechanism_ids:
            assignments[slot_id].append(piece.get("piece_id"))

    coverage_slots = []
    for slot in slots:
        slot_pieces = [
            piece
            for piece in pieces
            if slot["slot_id"] in (piece.get("mechanism_ids") or [])
        ]
        people = unique_strings(piece.get("person_id") for piece in slot_pieces)
        evidence_families = unique_strings(
            family
            for piece in slot_pieces
            for family in piece.get("evidence_source_families", []) or []
        )
        if len(people) >= 2:
            material_status = "multi_view_material"
        elif slot_pieces:
            material_status = "single_view_material"
        else:
            material_status = "uncovered"
        coverage_slots.append(
            {
                **slot,
                "piece_ids": unique_strings(assignments.get(slot["slot_id"], [])),
                "piece_count": len(slot_pieces),
                "person_ids": people,
                "person_count": len(people),
                "external_evidence_family_count": len(evidence_families),
                "material_status": material_status,
            }
        )

    covered = sum(item["piece_count"] > 0 for item in coverage_slots)
    return {
        "schema_version": MECHANISM_INDEX_VERSION,
        "generation_mode": (
            "deterministic_from_existing_question_frame"
            if coverage_slots
            else "unavailable_without_question_frame"
        ),
        "adds_model_calls": False,
        "slots": coverage_slots,
        "slot_count": len(coverage_slots),
        "material_covered_slot_count": covered,
        "material_coverage_rate": (
            round(covered / len(coverage_slots), 4) if coverage_slots else 1.0
        ),
        "unmatched_piece_count": sum(
            not piece.get("mechanism_ids") for piece in pieces
        ),
        "forced_relation_count": 0,
    }


def finalize_mechanism_coverage(
    mechanism_index: dict[str, Any],
    pieces: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    piece_slots = {
        piece.get("piece_id"): set(piece.get("mechanism_ids") or [])
        for piece in pieces
    }
    accepted = [item for item in relations if item.get("status") == "accepted"]
    slots = []
    for slot in mechanism_index.get("slots", []) or []:
        slot_id = slot.get("slot_id")
        relation_ids = [
            relation.get("relation_id")
            for relation in accepted
            if any(
                slot_id in piece_slots.get(piece_id, set())
                for piece_id in relation.get("piece_ids", []) or []
            )
        ]
        constructive_relation_ids = [
            relation.get("relation_id")
            for relation in accepted
            if relation.get("relation_type") in _CONSTRUCTIVE_RELATION_TYPES
            and any(
                slot_id in piece_slots.get(piece_id, set())
                for piece_id in relation.get("piece_ids", []) or []
            )
        ]
        if relation_ids:
            status = "connected"
        elif slot.get("piece_count"):
            status = "material_only"
        else:
            status = "uncovered"
        slots.append(
            {
                **slot,
                "accepted_relation_ids": unique_strings(relation_ids),
                "constructive_relation_ids": unique_strings(
                    constructive_relation_ids
                ),
                "connection_status": status,
                "constructive_connection_status": (
                    "constructively_connected"
                    if constructive_relation_ids
                    else (
                        "structural_or_dependency_only"
                        if relation_ids
                        else status
                    )
                ),
            }
        )
    connected = sum(item["connection_status"] == "connected" for item in slots)
    constructive_connected = sum(
        item["constructive_connection_status"] == "constructively_connected"
        for item in slots
    )
    return {
        **mechanism_index,
        "slots": slots,
        "connected_slot_count": connected,
        "connection_coverage_rate": (
            round(connected / len(slots), 4) if slots else 1.0
        ),
        "constructively_connected_slot_count": constructive_connected,
        "constructive_connection_coverage_rate": (
            round(constructive_connected / len(slots), 4)
            if slots
            else 1.0
        ),
        "interpretation": (
            "覆盖账本区分建设性连接与仅有来源依赖或错误相关的结构连接；不强迫空白机制进入轮廓。"
        ),
    }


def mechanism_rarity(piece: dict[str, Any], counts: dict[str, int]) -> float:
    mechanism_ids = piece.get("mechanism_ids") or []
    if not mechanism_ids:
        return 0.0
    return max(1.0 / max(1, counts.get(slot_id, 1)) for slot_id in mechanism_ids)


def _rank_slots(
    piece: dict[str, Any], slots: list[dict[str, Any]]
) -> list[tuple[float, str]]:
    body = " ".join(
        compact_text(value)
        for value in (
            piece.get("text"),
            piece.get("parent_conclusion"),
            (piece.get("diagnostic_context") or {}).get("from_claim"),
            (piece.get("diagnostic_context") or {}).get("to_conclusion"),
        )
        if compact_text(value)
    )
    body_terms = _term_set(body)
    ranked = []
    for slot in slots:
        slot_terms = set(slot.get("keywords") or [])
        overlap = slot_terms & body_terms
        direct = sum(
            1 for keyword in slot_terms if len(keyword) >= 2 and keyword in body
        )
        denominator = max(4, len(slot_terms))
        score = len(overlap) / denominator + min(0.24, direct * 0.06)
        ranked.append((score, slot["slot_id"]))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def _slot_label(description: str) -> str:
    label = re.split(r"[:：;；]", description, maxsplit=1)[0]
    return compact_text(label)[:40] or compact_text(description)[:40]


def _slot_type(description: str) -> str:
    if any(marker in description for marker in ("定义", "范围", "口径")):
        return "definition_or_scope"
    if any(marker in description for marker in ("触发", "传导", "机制", "导火索")):
        return "causal_path"
    if any(marker in description for marker in ("指标", "数据", "流向", "情绪")):
        return "observable_signal"
    return "mechanism_or_evidence_dimension"


def _keywords(text: str) -> list[str]:
    return sorted(_term_set(text))[:80]


def _term_set(text: str) -> set[str]:
    lowered = compact_text(text).lower()
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_+.-]{1,}|\d+(?:\.\d+)?", lowered)
        if len(token) >= 2
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
        for size in (2, 3, 4):
            terms.update(
                chunk[index : index + size]
                for index in range(max(0, len(chunk) - size + 1))
            )
    return {term for term in terms if term not in _STOP_TERMS}
