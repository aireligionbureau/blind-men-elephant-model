from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import compact_text, stable_id, unique_strings


CONNECTOR_SCHEMA_VERSION = "bme.classic-connectors.v1"
_AUTHORIZATION_RANK = {"provisional": 0, "restricted": 1, "blocked": 2}
_STRUCTURAL_RELATIONS = {
    "apparent_conflict",
    "scope_refinement",
    "shared_assumption",
    "definition_branch",
    "value_branch",
    "diagnostic_challenge",
    "non_comparable",
}


def load_connection_operators(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    file_path = Path(path) if path is not None else _default_operator_path()
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [item for item in payload if isinstance(item, dict) and item.get("id")]


def build_connection_hypotheses(
    question: str,
    pieces: list[dict[str, Any]],
    certificates: list[dict[str, Any]],
    *,
    operators: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Turn classic diagnoses into testable relation searches, never facts."""

    operators = operators or load_connection_operators()
    by_lens: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for operator in operators:
        for lens_id in operator.get("lens_ids", []) or []:
            by_lens[compact_text(lens_id)].append(operator)
    certificate_by_id = {
        item.get("certificate_id"): item
        for item in certificates
        if item.get("certificate_id")
    }
    pieces_by_trace: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for piece in pieces:
        knowledge = piece.get("knowledge_trace") or {}
        for trace_id in knowledge.get("classic_trace_ids", []) or []:
            pieces_by_trace[trace_id].append(piece)

    hypotheses: list[dict[str, Any]] = []
    for trace_id, trace_pieces in sorted(pieces_by_trace.items()):
        source_piece = min(trace_pieces, key=_source_piece_rank)
        knowledge = source_piece.get("knowledge_trace") or {}
        certificate_ids = knowledge.get("inversion_certificate_ids", []) or []
        linked_certificates = [
            certificate_by_id[item]
            for item in certificate_ids
            if item in certificate_by_id
        ]
        authorization = _most_restrictive(
            [item.get("authorization") for item in linked_certificates]
        )
        if authorization == "blocked":
            continue
        candidate_operators = {
            operator["id"]: operator
            for lens_id in knowledge.get("lens_ids", []) or []
            for operator in by_lens.get(lens_id, [])
            if source_piece.get("piece_kind")
            in (operator.get("trigger_piece_kinds") or [])
        }
        for operator in sorted(
            candidate_operators.values(),
            key=lambda item: (-int(item.get("priority") or 0), item["id"]),
        )[:3]:
            relation_types = unique_strings(
                operator.get("candidate_relation_types") or []
            )
            if authorization == "restricted":
                relation_types = [
                    item for item in relation_types if item in _STRUCTURAL_RELATIONS
                ]
            if not relation_types:
                continue
            hypothesis_id = stable_id(
                "connector",
                question,
                trace_id,
                operator["id"],
                source_piece.get("piece_id"),
                size=14,
            )
            hypothesis = {
                "hypothesis_id": hypothesis_id,
                "operator_id": operator["id"],
                "classic_trace_ids": [trace_id],
                "inversion_certificate_ids": unique_strings(certificate_ids),
                "source_piece_id": source_piece.get("piece_id"),
                "applicable_source_piece_ids": unique_strings(
                    piece.get("piece_id")
                    for piece in trace_pieces
                    if piece.get("person_id")
                    == source_piece.get("person_id")
                    and piece.get("piece_kind")
                    in (operator.get("trigger_piece_kinds") or [])
                ),
                "source_person_id": source_piece.get("person_id"),
                "source_piece_kind": source_piece.get("piece_kind"),
                "source_excerpt": compact_text(source_piece.get("text"))[:180],
                "mechanism_ids": source_piece.get("mechanism_ids") or [],
                "authorization": authorization,
                "allowed_relation_types": relation_types,
                "partner_piece_kinds": unique_strings(
                    operator.get("partner_piece_kinds") or []
                ),
                "partner_must_be_different_person": True,
                "constructive_task": compact_text(
                    operator.get("constructive_task")
                ),
                "priority": int(operator.get("priority") or 0),
                "proof_obligations": unique_strings(
                    operator.get("proof_obligations") or []
                ),
                "forbidden_inferences": unique_strings(
                    operator.get("forbidden_inferences") or []
                ),
                "differential_statuses": knowledge.get(
                    "differential_statuses"
                )
                or [],
                "status": "ready_for_detective_test",
                "epistemic_role": "connection_method_not_world_evidence",
                "classic_as_world_evidence": False,
            }
            hypotheses.append(hypothesis)
            source_piece.setdefault(
                "classic_connection_hypothesis_ids", []
            ).append(hypothesis_id)

    for hypothesis in hypotheses:
        partner_ids = _candidate_partner_piece_ids(hypothesis, pieces)
        hypothesis["candidate_partner_piece_ids"] = partner_ids
        hypothesis["candidate_partner_person_ids"] = unique_strings(
            piece.get("person_id")
            for piece in pieces
            if piece.get("piece_id") in set(partner_ids)
        )
    hypotheses = sorted(hypotheses, key=lambda item: item["hypothesis_id"])
    return {
        "schema_version": CONNECTOR_SCHEMA_VERSION,
        "operators_loaded": len(operators),
        "hypotheses": hypotheses,
        "hypothesis_count": len(hypotheses),
        "blocked_trace_count": sum(
            _most_restrictive(
                [
                    certificate_by_id[item].get("authorization")
                    for item in (piece.get("knowledge_trace") or {}).get(
                        "inversion_certificate_ids", []
                    )
                    if item in certificate_by_id
                ]
            )
            == "blocked"
            for piece in pieces
            if (piece.get("knowledge_trace") or {}).get("classic_trace_ids")
        ),
        "adds_model_calls": False,
        "classic_as_world_evidence": False,
    }


def select_connection_hypotheses(
    bundle: dict[str, Any],
    selected_piece_ids: set[str],
    *,
    max_items: int = 36,
) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in bundle.get("hypotheses", []) or []
        if item.get("source_piece_id") in selected_piece_ids
    ]
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(
        eligible,
        key=lambda value: (
            -int(value.get("priority") or 0),
            _AUTHORIZATION_RANK.get(value.get("authorization"), 9),
            compact_text(value.get("source_person_id")),
            compact_text(value.get("hypothesis_id")),
        ),
    ):
        buckets[compact_text(item.get("operator_id"))].append(item)

    # Round-robin prevents one popular lens from consuming the whole prompt.
    selected: list[dict[str, Any]] = []
    operator_ids = sorted(
        buckets,
        key=lambda operator_id: (
            -max(int(item.get("priority") or 0) for item in buckets[operator_id]),
            operator_id,
        ),
    )
    while len(selected) < max_items and any(buckets.values()):
        for operator_id in operator_ids:
            if buckets[operator_id] and len(selected) < max_items:
                selected.append(buckets[operator_id].pop(0))
    return selected


def _source_piece_rank(piece: dict[str, Any]) -> tuple[int, int, str]:
    kind_rank = {"fracture": 0, "distortion": 1, "retained_local": 2}
    leverage_rank = 0 if piece.get("conclusion_leverage") == "high" else 1
    return (
        kind_rank.get(piece.get("piece_kind"), 9),
        leverage_rank,
        compact_text(piece.get("piece_id")),
    )


def _candidate_partner_piece_ids(
    hypothesis: dict[str, Any], pieces: list[dict[str, Any]]
) -> list[str]:
    source_person_id = hypothesis.get("source_person_id")
    source_mechanisms = set(hypothesis.get("mechanism_ids") or [])
    allowed_kinds = set(hypothesis.get("partner_piece_kinds") or [])
    ranked = []
    for piece in pieces:
        if (
            piece.get("person_id") == source_person_id
            or piece.get("piece_kind") not in allowed_kinds
            or not piece.get("piece_id")
        ):
            continue
        mechanism_overlap = len(
            source_mechanisms & set(piece.get("mechanism_ids") or [])
        )
        lineage = piece.get("independence_profile") or {}
        has_external_lineage = bool(
            piece.get("evidence_source_families")
            or lineage.get("evidence_families")
        )
        kind_rank = {
            "evidence_claim": 0,
            "retained_local": 1,
            "observed_claim": 2,
            "assumption": 3,
            "blind_spot_candidate": 4,
            "distortion": 5,
            "fracture": 6,
            "value_condition": 7,
        }.get(piece.get("piece_kind"), 9)
        ranked.append(
            (
                -mechanism_overlap,
                0 if has_external_lineage else 1,
                0 if piece.get("conclusion_leverage") == "high" else 1,
                kind_rank,
                compact_text(piece.get("person_id")),
                compact_text(piece.get("piece_id")),
                piece,
            )
        )
    output: list[str] = []
    seen_people: set[str] = set()
    for *_, piece in sorted(ranked):
        person_id = compact_text(piece.get("person_id"))
        if not person_id or person_id in seen_people:
            continue
        output.append(piece["piece_id"])
        seen_people.add(person_id)
        if len(output) >= 8:
            break
    return output


def _most_restrictive(values: list[Any]) -> str:
    valid = [compact_text(item) for item in values if compact_text(item) in _AUTHORIZATION_RANK]
    if not valid:
        return "blocked"
    return max(valid, key=_AUTHORIZATION_RANK.__getitem__)


def _default_operator_path() -> Path:
    return Path(__file__).resolve().parents[3] / "knowledge" / "connection_operators.json"
