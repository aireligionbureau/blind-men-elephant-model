from __future__ import annotations

import re
from typing import Any

from ..detective.schemas import (
    CLASSIC_EPISTEMIC_ROLE,
    compact_text,
    question_id,
)
from .epistemic import has_structural_uncertainty


CONSTRAINT_SCHEMA_VERSION = "bme.contour-constraints.v2"
TRUTH_CONTOUR_SCHEMA_VERSION = "bme.truth-contour.v2"

CONSTRAINT_TYPES = {
    "support_candidate",
    "dependency_penalty",
    "correlated_error_warning",
    "conditional_split",
    "conflict_boundary",
    "scope_boundary",
    "causal_bridge",
    "shared_assumption_dependency",
    "definition_branch",
    "value_branch",
    "directional_bound",
    "unknown_space",
    "non_comparable",
}

CONSTRUCTIVE_RELATION_TYPES = {
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

CONSTRUCTIVE_CONSTRAINT_TYPES = {
    "support_candidate",
    "conditional_split",
    "conflict_boundary",
    "scope_boundary",
    "causal_bridge",
    "shared_assumption_dependency",
    "definition_branch",
    "value_branch",
}

SENTENCE_CLAIM_ROLES = {
    "verified_observation",
    "directional_judgment",
    "conditional_inference",
    "structural_relation",
    "definition_boundary",
    "important_unknown",
    "value_condition",
}

VERIFIED_EVIDENCE_STATUSES = {
    "page_verified",
    "primary_verified",
    "cross_verified",
}


def validate_constraints(payload: dict[str, Any], detective: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != CONSTRAINT_SCHEMA_VERSION:
        errors.append("unexpected constraint schema version")
    if payload.get("question_id") != detective.get("question_id"):
        errors.append("constraint question_id mismatch")
    relation_ids = {
        item.get("relation_id")
        for item in detective.get("relation_certificates", []) or []
    }
    relation_by_id = {
        item.get("relation_id"): item
        for item in detective.get("relation_certificates", []) or []
    }
    trace_ids = {
        item.get("trace_id")
        for item in detective.get("classic_diagnostic_traces", []) or []
    }
    certificate_ids = {
        item.get("certificate_id")
        for item in detective.get("inversion_certificates", []) or []
    }
    bridge_ids = {
        item.get("bridge_id")
        for item in detective.get("diagnostic_bridges", []) or []
    }
    connector_ids = {
        item.get("hypothesis_id")
        for item in (
            detective.get("classic_connection_hypotheses") or {}
        ).get("hypotheses", [])
        or []
    }
    seen: set[str] = set()
    for item in payload.get("constraints", []) or []:
        constraint_id = compact_text(item.get("constraint_id"))
        if not constraint_id:
            errors.append("constraint_id is missing")
        elif constraint_id in seen:
            errors.append(f"duplicate constraint_id: {constraint_id}")
        seen.add(constraint_id)
        if item.get("constraint_type") not in CONSTRAINT_TYPES:
            errors.append(f"invalid constraint type: {constraint_id}")
        source_ids = item.get("source_relation_ids") or []
        if not source_ids:
            errors.append(f"constraint has no source relation: {constraint_id}")
        for relation_id in source_ids:
            if relation_id not in relation_ids:
                errors.append(f"constraint references unknown relation {relation_id}")
        knowledge = item.get("knowledge_provenance") or {}
        if knowledge.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
            errors.append(f"constraint lacks classic epistemic boundary: {constraint_id}")
        if knowledge.get("classic_as_world_evidence") is not False:
            errors.append(f"constraint treats classics as world evidence: {constraint_id}")
        expected_traces = {
            trace_id
            for relation_id in source_ids
            for trace_id in (
                relation_by_id.get(relation_id, {}).get("knowledge_provenance")
                or {}
            ).get("classic_trace_ids", [])
        }
        if not expected_traces <= set(knowledge.get("classic_trace_ids", []) or []):
            errors.append(f"constraint drops relation classic traces: {constraint_id}")
        _validate_knowledge_refs(
            knowledge,
            trace_ids=trace_ids,
            certificate_ids=certificate_ids,
            bridge_ids=bridge_ids,
            connector_ids=connector_ids,
            label=f"constraint {constraint_id}",
            errors=errors,
        )
    return errors


def validate_truth_contour(
    payload: dict[str, Any],
    *,
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != TRUTH_CONTOUR_SCHEMA_VERSION:
        errors.append("unexpected truth contour schema version")
    if compact_text(payload.get("question")) != compact_text(question):
        errors.append("truth contour question mismatch")
    if payload.get("question_id") != question_id(question):
        errors.append("truth contour question_id mismatch")
    if not compact_text(payload.get("direct_answer")):
        errors.append("direct_answer is missing")
    if not compact_text(payload.get("main_contour")):
        errors.append("main_contour is missing")

    piece_ids = {
        item.get("piece_id")
        for item in detective.get("puzzle_pieces", []) or []
    }
    relation_ids = {
        item.get("relation_id")
        for item in detective.get("relation_certificates", []) or []
    }
    constraint_ids = {
        item.get("constraint_id")
        for item in constraints.get("constraints", []) or []
    }
    trace_ids = {
        item.get("trace_id")
        for item in detective.get("classic_diagnostic_traces", []) or []
    }
    certificate_ids = {
        item.get("certificate_id")
        for item in detective.get("inversion_certificates", []) or []
    }
    bridge_ids = {
        item.get("bridge_id")
        for item in detective.get("diagnostic_bridges", []) or []
    }
    connector_ids = {
        item.get("hypothesis_id")
        for item in (
            detective.get("classic_connection_hypotheses") or {}
        ).get("hypotheses", [])
        or []
    }
    provenance = payload.get("provenance_index") or []
    if not provenance:
        errors.append("truth contour provenance is missing")
    for item in provenance:
        if not compact_text(item.get("statement")):
            errors.append("provenance statement is missing")
        refs = (
            list(item.get("piece_ids") or [])
            + list(item.get("relation_ids") or [])
            + list(item.get("constraint_ids") or [])
        )
        if not refs:
            errors.append("provenance entry has no refs")
        for value in item.get("piece_ids") or []:
            if value not in piece_ids:
                errors.append(f"truth contour references unknown piece {value}")
        for value in item.get("relation_ids") or []:
            if value not in relation_ids:
                errors.append(f"truth contour references unknown relation {value}")
        for value in item.get("constraint_ids") or []:
            if value not in constraint_ids:
                errors.append(f"truth contour references unknown constraint {value}")
        if item.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
            errors.append("truth contour provenance lacks classic epistemic boundary")
        if item.get("classic_as_world_evidence") is not False:
            errors.append("truth contour treats classics as world evidence")
        _validate_knowledge_refs(
            item,
            trace_ids=trace_ids,
            certificate_ids=certificate_ids,
            bridge_ids=bridge_ids,
            connector_ids=connector_ids,
            label=f"truth contour statement {item.get('statement_id')}",
            errors=errors,
        )

    if payload.get("evidence_binding_contract") == "sentence_level.v1":
        _validate_sentence_evidence_bindings(
            payload,
            detective=detective,
            constraints=constraints,
            piece_ids=piece_ids,
            relation_ids=relation_ids,
            constraint_ids=constraint_ids,
            trace_ids=trace_ids,
            certificate_ids=certificate_ids,
            bridge_ids=bridge_ids,
            connector_ids=connector_ids,
            errors=errors,
        )

    statements = [
        compact_text(payload.get("direct_answer")),
        compact_text(payload.get("main_contour")),
    ]
    for field in (
        "key_conditions",
        "stable_parts",
        "boundary_conditions",
        "important_unknowns",
    ):
        statements.extend(
            compact_text(item) for item in payload.get(field, []) or [] if compact_text(item)
        )
    traced_statements = {
        compact_text(item.get("statement")) for item in provenance if compact_text(item.get("statement"))
    }
    for statement in statements:
        if statement and statement not in traced_statements:
            errors.append("truth contour statement lacks provenance: " + statement[:80])
        if re.search(
            r"(?<![A-Za-z0-9_])(?:piece|relation|constraint)_[0-9a-f]{8,}(?![A-Za-z0-9_])",
            statement,
        ):
            errors.append("truth contour exposes internal ids in user-facing text")

    if payload.get("generation_mode") == "live_adversarial_inference":
        relation_by_id = {
            item.get("relation_id"): item
            for item in detective.get("relation_certificates", []) or []
        }
        constraint_by_id = {
            item.get("constraint_id"): item
            for item in constraints.get("constraints", []) or []
        }
        constructive_relation_ids = {
            relation_id
            for relation_id, item in relation_by_id.items()
            if item.get("status") == "accepted"
            and item.get("relation_type") in CONSTRUCTIVE_RELATION_TYPES
        }
        constructive_constraint_ids = {
            constraint_id
            for constraint_id, item in constraint_by_id.items()
            if item.get("constraint_type") in CONSTRUCTIVE_CONSTRAINT_TYPES
        }
        if not constructive_relation_ids:
            errors.append("live truth contour has no constructive accepted relation")
        by_statement_id = {
            item.get("statement_id"): item for item in provenance
        }
        for statement_id in ("direct_answer", "main_contour"):
            trace = by_statement_id.get(statement_id) or {}
            has_support = bool(
                set(trace.get("relation_ids") or []) & constructive_relation_ids
                or set(trace.get("constraint_ids") or []) & constructive_constraint_ids
            )
            if not has_support:
                errors.append(
                    f"{statement_id} lacks a constructive relation or constraint"
                )
        for statement_id, trace in by_statement_id.items():
            if not str(statement_id).startswith("stable_"):
                continue
            has_support = bool(
                set(trace.get("relation_ids") or []) & constructive_relation_ids
                or set(trace.get("constraint_ids") or [])
                & constructive_constraint_ids
            )
            if not has_support:
                errors.append(
                    f"{statement_id} promotes an unconnected candidate piece "
                    "to a stable contour statement"
                )
        directional_support = {
            relation_id
            for relation_id, item in relation_by_id.items()
            if item.get("status") == "accepted"
            and item.get("relation_type") == "independent_convergence"
            and (item.get("independence_profile") or {}).get(
                "directional_support_authorized", True
            )
        }
        if not directional_support:
            direct = compact_text(payload.get("direct_answer"))
            main = compact_text(payload.get("main_contour"))
            if not has_structural_uncertainty(direct):
                errors.append(
                    "structural-only live contour gives a directional direct answer"
                )
            hard_claims = (
                "可能性极低",
                "可能性极高",
                "绝对不可能",
                "必然会",
                "必然不会",
                "一致表明",
                "已经证明",
                "已经证实",
                "均未达到",
                "所有AI",
                "所有 AI",
            )
            if any(item in f"{direct} {main}" for item in hard_claims):
                errors.append(
                    "structural-only live contour exceeds its inference capability"
                )
    audit = payload.get("classic_knowledge_audit") or {}
    if audit.get("classic_used_as_world_evidence_count") not in {0, None}:
        errors.append("truth contour classic knowledge audit detected evidence misuse")
    return errors


def _validate_sentence_evidence_bindings(
    payload: dict[str, Any],
    *,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    piece_ids: set[Any],
    relation_ids: set[Any],
    constraint_ids: set[Any],
    trace_ids: set[Any],
    certificate_ids: set[Any],
    bridge_ids: set[Any],
    connector_ids: set[Any],
    errors: list[str],
) -> None:
    bindings = payload.get("sentence_evidence_bindings")
    if not isinstance(bindings, list) or not bindings:
        errors.append("sentence-level evidence binding is missing")
        return

    piece_by_id = {
        item.get("piece_id"): item
        for item in detective.get("puzzle_pieces", []) or []
        if item.get("piece_id")
    }
    relation_by_id = {
        item.get("relation_id"): item
        for item in detective.get("relation_certificates", []) or []
        if item.get("relation_id")
    }
    constraint_by_id = {
        item.get("constraint_id"): item
        for item in constraints.get("constraints", []) or []
        if item.get("constraint_id")
    }
    rendered = {
        "direct_answer": compact_text(payload.get("direct_answer")),
        "main_contour": compact_text(payload.get("main_contour")),
    }
    seen_claim_ids: set[str] = set()

    for binding in bindings:
        if not isinstance(binding, dict):
            errors.append("sentence evidence binding is not an object")
            continue
        claim_id = compact_text(binding.get("claim_id") or binding.get("statement_id"))
        if not claim_id:
            errors.append("sentence evidence binding has no claim_id")
        elif claim_id in seen_claim_ids:
            errors.append(f"duplicate sentence claim_id: {claim_id}")
        seen_claim_ids.add(claim_id)

        role = compact_text(binding.get("claim_role"))
        if role not in SENTENCE_CLAIM_ROLES:
            errors.append(f"invalid sentence claim role: {role or 'missing'}")
        field = compact_text(binding.get("rendered_in"))
        statement = compact_text(binding.get("statement"))
        if field not in rendered:
            errors.append(f"sentence binding has invalid rendered_in: {field or 'missing'}")
        elif not statement or statement not in rendered[field]:
            errors.append(f"sentence binding is absent from rendered {field}: {claim_id}")

        refs = (
            list(binding.get("piece_ids") or [])
            + list(binding.get("relation_ids") or [])
            + list(binding.get("constraint_ids") or [])
        )
        if not refs:
            errors.append(f"sentence binding has no refs: {claim_id}")
        for value in binding.get("piece_ids") or []:
            if value not in piece_ids:
                errors.append(f"sentence binding references unknown piece {value}")
        for value in binding.get("relation_ids") or []:
            if value not in relation_ids:
                errors.append(f"sentence binding references unknown relation {value}")
        for value in binding.get("constraint_ids") or []:
            if value not in constraint_ids:
                errors.append(f"sentence binding references unknown constraint {value}")
        if binding.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
            errors.append(f"sentence binding lacks classic epistemic boundary: {claim_id}")
        if binding.get("classic_as_world_evidence") is not False:
            errors.append(f"sentence binding treats classics as world evidence: {claim_id}")
        _validate_knowledge_refs(
            binding,
            trace_ids=trace_ids,
            certificate_ids=certificate_ids,
            bridge_ids=bridge_ids,
            connector_ids=connector_ids,
            label=f"sentence claim {claim_id}",
            errors=errors,
        )

        cited_pieces = [
            piece_by_id[value]
            for value in binding.get("piece_ids") or []
            if value in piece_by_id
        ]
        cited_relations = [
            relation_by_id[value]
            for value in binding.get("relation_ids") or []
            if value in relation_by_id
        ]
        cited_constraints = [
            constraint_by_id[value]
            for value in binding.get("constraint_ids") or []
            if value in constraint_by_id
        ]
        _validate_sentence_role(
            claim_id=claim_id,
            role=role,
            statement=statement,
            pieces=cited_pieces,
            relations=cited_relations,
            constraints=cited_constraints,
            errors=errors,
        )

    coverage = (payload.get("contour_evaluation") or {}).get(
        "sentence_binding_coverage"
    )
    if coverage is not None:
        try:
            numeric_coverage = float(coverage)
        except (TypeError, ValueError):
            errors.append("sentence-level evidence binding coverage is not numeric")
        else:
            if numeric_coverage < 1.0:
                errors.append("sentence-level evidence binding coverage is incomplete")


def _validate_sentence_role(
    *,
    claim_id: str,
    role: str,
    statement: str,
    pieces: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    errors: list[str],
) -> None:
    verification_statuses = {
        str(status)
        for piece in pieces
        for status in (
            piece.get("verification_statuses")
            or (piece.get("independence_profile") or {}).get(
                "verification_statuses", []
            )
            or []
        )
    }
    relation_types = {str(item.get("relation_type")) for item in relations}
    constraint_types = {str(item.get("constraint_type")) for item in constraints}
    constructive = any(
        item.get("status") == "accepted"
        and item.get("relation_type") in CONSTRUCTIVE_RELATION_TYPES
        for item in relations
    ) or bool(constraint_types.intersection(CONSTRUCTIVE_CONSTRAINT_TYPES))
    directional = any(
        item.get("status") == "accepted"
        and item.get("relation_type") == "independent_convergence"
        and (item.get("independence_profile") or {}).get(
            "directional_support_authorized", False
        )
        for item in relations
    )

    if role == "verified_observation" and not verification_statuses.intersection(
        VERIFIED_EVIDENCE_STATUSES
    ):
        errors.append(f"verified observation lacks verified evidence: {claim_id}")
    elif role == "directional_judgment" and not directional:
        errors.append(f"directional judgment lacks authorized support: {claim_id}")
    elif role == "conditional_inference":
        if not constructive:
            errors.append(f"conditional inference lacks a constructive link: {claim_id}")
        if not _has_condition_language(statement):
            errors.append(f"conditional inference omits its condition: {claim_id}")
    elif role == "structural_relation":
        if not constructive:
            errors.append(f"structural relation lacks a constructive link: {claim_id}")
        if _has_unconditional_direction(statement):
            errors.append(f"structural relation overstates a world direction: {claim_id}")
    elif role == "definition_boundary" and not (
        "definition_branch" in relation_types
        or "definition_branch" in constraint_types
    ):
        errors.append(f"definition boundary lacks a definition branch: {claim_id}")
    elif role == "important_unknown" and not _has_unknown_language(statement):
        errors.append(f"important unknown is written as known: {claim_id}")
    elif role == "value_condition" and not (
        "value_branch" in relation_types or "value_branch" in constraint_types
    ):
        errors.append(f"value condition lacks a value branch: {claim_id}")


def _has_condition_language(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "如果",
            "若",
            "取决于",
            "除非",
            "一旦",
            "只有",
            "条件下",
            "情况下",
            "时才",
        )
    )


def _has_unknown_language(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "未知",
            "尚不清楚",
            "尚未",
            "无法确定",
            "不能确定",
            "缺少",
            "没有足够",
            "仍待",
        )
    )


def _has_unconditional_direction(text: str) -> bool:
    if _has_condition_language(text):
        return False
    return any(
        marker in text
        for marker in (
            "概率较低",
            "概率较高",
            "风险极高",
            "风险很高",
            "很可能",
            "不太可能",
            "将会",
            "不会发生",
            "会发生",
            "已经发生",
            "已处于",
            "尚未进入",
        )
    )


def _validate_knowledge_refs(
    payload: dict[str, Any],
    *,
    trace_ids: set[Any],
    certificate_ids: set[Any],
    bridge_ids: set[Any],
    connector_ids: set[Any],
    label: str,
    errors: list[str],
) -> None:
    for trace_id in payload.get("classic_trace_ids", []) or []:
        if trace_id not in trace_ids:
            errors.append(f"{label} references unknown classic trace {trace_id}")
    for certificate_id in payload.get("inversion_certificate_ids", []) or []:
        if certificate_id not in certificate_ids:
            errors.append(
                f"{label} references unknown inversion certificate {certificate_id}"
            )
    for bridge_id in payload.get("diagnostic_bridge_ids", []) or []:
        if bridge_id not in bridge_ids:
            errors.append(f"{label} references unknown diagnostic bridge {bridge_id}")
    for connector_id in payload.get("connection_hypothesis_ids", []) or []:
        if connector_id not in connector_ids:
            errors.append(
                f"{label} references unknown connection hypothesis {connector_id}"
            )
