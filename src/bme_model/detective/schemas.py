from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


PUZZLE_SCHEMA_VERSION = "bme.puzzle-pieces.v2"
DETECTIVE_SCHEMA_VERSION = "bme.detective-output.v2"

CLASSIC_EPISTEMIC_ROLE = "diagnostic_method_not_world_evidence"
INVERSION_AUTHORIZATIONS = {"provisional", "restricted", "blocked"}

PIECE_KINDS = {
    "observed_claim",
    "assumption",
    "evidence_claim",
    "retained_local",
    "distortion",
    "fracture",
    "blind_spot_candidate",
    "value_condition",
}

RELATION_TYPES = {
    "independent_convergence",
    "shared_source",
    "shared_model_prior",
    "correlated_error",
    "direct_conflict",
    "apparent_conflict",
    "conditional_complement",
    "scope_refinement",
    "causal_relay",
    "shared_assumption",
    "definition_branch",
    "value_branch",
    "opposite_distortion",
    "blind_spot_fill",
    "collective_blind_spot",
    "diagnostic_challenge",
    "non_comparable",
}

RELATION_STATUSES = {"accepted", "rejected", "unresolved", "proposed"}


def stable_id(prefix: str, *parts: Any, size: int = 16) -> str:
    body = "\x1f".join(compact_text(part) for part in parts)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:size]
    return f"{prefix}_{digest}"


def question_id(question: str) -> str:
    return stable_id("question", question, size=12)


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [compact_text(item) for item in value if compact_text(item)]


def unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = compact_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def normalize_claim(text: Any) -> str:
    value = compact_text(text).lower()
    value = re.sub(r"[，。；：、！？,.!?;:\s]+", "", value)
    return value


def validate_puzzle_materials(payload: dict[str, Any], question: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != PUZZLE_SCHEMA_VERSION:
        errors.append("unexpected puzzle schema version")
    if compact_text(payload.get("question")) != compact_text(question):
        errors.append("puzzle question mismatch")
    expected_question_id = question_id(question)
    if payload.get("question_id") != expected_question_id:
        errors.append("puzzle question_id mismatch")

    traces = payload.get("classic_diagnostic_traces", []) or []
    certificates = payload.get("inversion_certificates", []) or []
    bridges = payload.get("diagnostic_bridges", []) or []
    trace_ids = _unique_id_set(traces, "trace_id", "classic trace", errors)
    certificate_ids = _unique_id_set(
        certificates, "certificate_id", "inversion certificate", errors
    )
    bridge_ids = _unique_id_set(
        bridges, "bridge_id", "diagnostic bridge", errors
    )
    source_catalog = payload.get("source_catalog", []) or []
    source_catalog_ids = _unique_id_set(
        source_catalog, "source_id", "source catalog entry", errors
    )

    for trace in traces:
        trace_id = compact_text(trace.get("trace_id"))
        if trace.get("question_id") != expected_question_id:
            errors.append(f"foreign question classic trace: {trace_id}")
        if trace.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
            errors.append(f"classic trace has invalid epistemic role: {trace_id}")
        if trace.get("classic_as_world_evidence") is not False:
            errors.append(f"classic trace is treated as world evidence: {trace_id}")
        certificate_id = compact_text(trace.get("inversion_certificate_id"))
        if certificate_id not in certificate_ids:
            errors.append(
                f"classic trace references unknown inversion certificate {certificate_id}: {trace_id}"
            )
        if trace.get("trace_status") == "complete" and not trace.get("lens_path"):
            errors.append(f"complete classic trace has no lens path: {trace_id}")

    for certificate in certificates:
        certificate_id = compact_text(certificate.get("certificate_id"))
        if certificate.get("question_id") != expected_question_id:
            errors.append(f"foreign question inversion certificate: {certificate_id}")
        if certificate.get("trace_id") not in trace_ids:
            errors.append(
                f"inversion certificate references unknown classic trace: {certificate_id}"
            )
        if certificate.get("authorization") not in INVERSION_AUTHORIZATIONS:
            errors.append(f"invalid inversion authorization: {certificate_id}")
        if certificate.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
            errors.append(f"inversion certificate has invalid role: {certificate_id}")
        if certificate.get("classic_as_world_evidence") is not False:
            errors.append(
                f"inversion certificate is treated as world evidence: {certificate_id}"
            )
        if not certificate.get("prohibited_uses"):
            errors.append(f"inversion certificate lacks prohibited uses: {certificate_id}")

    piece_ids = {
        compact_text(piece.get("piece_id"))
        for piece in payload.get("pieces", []) or []
        if compact_text(piece.get("piece_id"))
    }
    piece_person_by_id = {
        compact_text(piece.get("piece_id")): compact_text(piece.get("person_id"))
        for piece in payload.get("pieces", []) or []
        if compact_text(piece.get("piece_id"))
    }
    connector_bundle = payload.get("classic_connection_hypotheses") or {}
    connector_ids: set[str] = set()
    for hypothesis in connector_bundle.get("hypotheses", []) or []:
        hypothesis_id = compact_text(hypothesis.get("hypothesis_id"))
        if not hypothesis_id:
            errors.append("classic connection hypothesis id is missing")
        elif hypothesis_id in connector_ids:
            errors.append(f"duplicate classic connection hypothesis: {hypothesis_id}")
        connector_ids.add(hypothesis_id)
        if hypothesis.get("source_piece_id") not in piece_ids:
            errors.append(
                f"classic connection hypothesis references unknown piece: {hypothesis_id}"
            )
        applicable_source_piece_ids = hypothesis.get(
            "applicable_source_piece_ids"
        ) or [hypothesis.get("source_piece_id")]
        for applicable_piece_id in applicable_source_piece_ids:
            if applicable_piece_id not in piece_ids:
                errors.append(
                    f"classic connection hypothesis references unknown applicable piece: {hypothesis_id}"
                )
            elif piece_person_by_id.get(applicable_piece_id) != compact_text(
                hypothesis.get("source_person_id")
            ):
                errors.append(
                    f"classic connection hypothesis applicable piece changes person: {hypothesis_id}"
                )
        for partner_piece_id in hypothesis.get(
            "candidate_partner_piece_ids", []
        ) or []:
            if partner_piece_id not in piece_ids:
                errors.append(
                    f"classic connection hypothesis references unknown partner piece: {hypothesis_id}"
                )
            elif piece_person_by_id.get(partner_piece_id) == compact_text(
                hypothesis.get("source_person_id")
            ):
                errors.append(
                    f"classic connection hypothesis partner is same person: {hypothesis_id}"
                )
        if hypothesis.get("classic_as_world_evidence") is not False:
            errors.append(
                f"classic connection hypothesis treats classics as evidence: {hypothesis_id}"
            )
        for relation_type in hypothesis.get("allowed_relation_types", []) or []:
            if relation_type not in RELATION_TYPES:
                errors.append(
                    f"classic connection hypothesis has invalid relation type: {hypothesis_id}"
                )
        for trace_id in hypothesis.get("classic_trace_ids", []) or []:
            if trace_id not in trace_ids:
                errors.append(
                    f"classic connection hypothesis references unknown trace: {hypothesis_id}"
                )
        for certificate_id in hypothesis.get(
            "inversion_certificate_ids", []
        ) or []:
            if certificate_id not in certificate_ids:
                errors.append(
                    f"classic connection hypothesis references unknown certificate: {hypothesis_id}"
                )
    for bridge in bridges:
        bridge_id = compact_text(bridge.get("bridge_id"))
        if bridge.get("classic_as_world_evidence") is not False:
            errors.append(f"diagnostic bridge treats classics as evidence: {bridge_id}")
        for trace_id in bridge.get("classic_trace_ids", []) or []:
            if trace_id not in trace_ids:
                errors.append(
                    f"diagnostic bridge references unknown classic trace {trace_id}: {bridge_id}"
                )
        for certificate_id in bridge.get("inversion_certificate_ids", []) or []:
            if certificate_id not in certificate_ids:
                errors.append(
                    f"diagnostic bridge references unknown inversion certificate {certificate_id}: {bridge_id}"
                )
        for piece_id in bridge.get("output_piece_ids", []) or []:
            if piece_id not in piece_ids:
                errors.append(
                    f"diagnostic bridge references unknown output piece {piece_id}: {bridge_id}"
                )

    seen: set[str] = set()
    for piece in payload.get("pieces", []) or []:
        piece_id = compact_text(piece.get("piece_id"))
        if not piece_id:
            errors.append("piece_id is missing")
        elif piece_id in seen:
            errors.append(f"duplicate piece_id: {piece_id}")
        seen.add(piece_id)
        if piece.get("piece_kind") not in PIECE_KINDS:
            errors.append(f"invalid piece kind for {piece_id}")
        if piece.get("question_id") != expected_question_id:
            errors.append(f"foreign question piece: {piece_id}")
        if not compact_text(piece.get("text")):
            errors.append(f"piece text is missing: {piece_id}")
        provenance = piece.get("provenance") or {}
        if not compact_text(provenance.get("origin")):
            errors.append(f"piece provenance is missing: {piece_id}")
        for source_id in piece.get("source_ids", []) or []:
            if source_id not in source_catalog_ids:
                errors.append(
                    f"piece references unknown source {source_id}: {piece_id}"
                )
        profile = piece.get("independence_profile") or {}
        if piece.get("evidence_source_families") and not compact_text(
            profile.get("lineage_status")
        ):
            errors.append(f"external piece lacks lineage status: {piece_id}")
        knowledge = piece.get("knowledge_trace") or {}
        is_diagnostic_piece = bool(piece.get("diagnostic_ids")) or compact_text(
            provenance.get("origin")
        ).startswith("forensic.semantic_verdict.")
        if is_diagnostic_piece:
            if not knowledge:
                errors.append(f"diagnostic piece lacks classic knowledge trace: {piece_id}")
                continue
            if knowledge.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
                errors.append(f"diagnostic piece has invalid classic role: {piece_id}")
            if knowledge.get("classic_as_world_evidence") is not False:
                errors.append(f"diagnostic piece treats classics as evidence: {piece_id}")
            authorization = compact_text(
                knowledge.get("inversion_authorization")
            )
            if authorization not in INVERSION_AUTHORIZATIONS:
                errors.append(f"diagnostic piece has invalid authorization: {piece_id}")
            linked_trace_ids = knowledge.get("classic_trace_ids", []) or []
            if not linked_trace_ids and authorization != "blocked":
                errors.append(
                    f"unlinked diagnostic piece is not blocked from inversion: {piece_id}"
                )
            for trace_id in linked_trace_ids:
                if trace_id not in trace_ids:
                    errors.append(
                        f"piece references unknown classic trace {trace_id}: {piece_id}"
                    )
            for certificate_id in knowledge.get(
                "inversion_certificate_ids", []
            ) or []:
                if certificate_id not in certificate_ids:
                    errors.append(
                        f"piece references unknown inversion certificate {certificate_id}: {piece_id}"
                    )
            for bridge_id in knowledge.get("diagnostic_bridge_ids", []) or []:
                if bridge_id not in bridge_ids:
                    errors.append(
                        f"piece references unknown diagnostic bridge {bridge_id}: {piece_id}"
                    )
    return errors


def validate_detective_output(payload: dict[str, Any], materials: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != DETECTIVE_SCHEMA_VERSION:
        errors.append("unexpected detective schema version")
    if payload.get("question_id") != materials.get("question_id"):
        errors.append("detective question_id mismatch")

    piece_ids = {piece.get("piece_id") for piece in materials.get("pieces", []) or []}
    piece_by_id = {
        piece.get("piece_id"): piece
        for piece in materials.get("pieces", []) or []
        if piece.get("piece_id")
    }
    trace_ids = {
        item.get("trace_id")
        for item in materials.get("classic_diagnostic_traces", []) or []
    }
    certificate_ids = {
        item.get("certificate_id")
        for item in materials.get("inversion_certificates", []) or []
    }
    bridge_ids = {
        item.get("bridge_id")
        for item in materials.get("diagnostic_bridges", []) or []
    }
    connector_by_id = {
        item.get("hypothesis_id"): item
        for item in (
            materials.get("classic_connection_hypotheses") or {}
        ).get("hypotheses", [])
        or []
        if item.get("hypothesis_id")
    }
    relation_ids: set[str] = set()
    for relation in payload.get("relation_certificates", []) or []:
        relation_id = compact_text(relation.get("relation_id"))
        if not relation_id:
            errors.append("relation_id is missing")
        elif relation_id in relation_ids:
            errors.append(f"duplicate relation_id: {relation_id}")
        relation_ids.add(relation_id)
        if relation.get("relation_type") not in RELATION_TYPES:
            errors.append(f"invalid relation type: {relation_id}")
        if relation.get("status") not in RELATION_STATUSES:
            errors.append(f"invalid relation status: {relation_id}")
        endpoints = relation.get("piece_ids") or []
        if len(endpoints) < 2:
            errors.append(f"relation needs at least two endpoints: {relation_id}")
        for piece_id in endpoints:
            if piece_id not in piece_ids:
                errors.append(f"relation references unknown piece {piece_id}: {relation_id}")
        if not compact_text(relation.get("plain_language_explanation")):
            errors.append(f"relation explanation is missing: {relation_id}")
        if relation.get("status") == "accepted":
            if not compact_text(relation.get("competing_explanation")):
                errors.append(f"accepted relation lacks competing explanation: {relation_id}")
            if not compact_text(relation.get("falsification_test")):
                errors.append(f"accepted relation lacks falsification test: {relation_id}")
            knowledge = relation.get("knowledge_provenance") or {}
            if knowledge.get("epistemic_role") != CLASSIC_EPISTEMIC_ROLE:
                errors.append(
                    f"accepted relation lacks classic epistemic boundary: {relation_id}"
                )
            if knowledge.get("classic_as_world_evidence") is not False:
                errors.append(
                    f"accepted relation treats classics as world evidence: {relation_id}"
                )
            expected_traces = {
                trace_id
                for piece_id in endpoints
                for trace_id in (
                    piece_by_id.get(piece_id, {}).get("knowledge_trace") or {}
                ).get("classic_trace_ids", [])
            }
            if not expected_traces <= set(
                knowledge.get("classic_trace_ids", []) or []
            ):
                errors.append(
                    f"accepted relation drops endpoint classic traces: {relation_id}"
                )
        knowledge = relation.get("knowledge_provenance") or {}
        for trace_id in knowledge.get("classic_trace_ids", []) or []:
            if trace_id not in trace_ids:
                errors.append(
                    f"relation references unknown classic trace {trace_id}: {relation_id}"
                )
        for certificate_id in knowledge.get(
            "inversion_certificate_ids", []
        ) or []:
            if certificate_id not in certificate_ids:
                errors.append(
                    f"relation references unknown inversion certificate {certificate_id}: {relation_id}"
                )
        for bridge_id in knowledge.get("diagnostic_bridge_ids", []) or []:
            if bridge_id not in bridge_ids:
                errors.append(
                    f"relation references unknown diagnostic bridge {bridge_id}: {relation_id}"
                )
        for connector_id in knowledge.get(
            "connection_hypothesis_ids", []
        ) or []:
            connector = connector_by_id.get(connector_id)
            if not connector:
                errors.append(
                    f"relation references unknown connection hypothesis {connector_id}: {relation_id}"
                )
                continue
            applicable_source_piece_ids = set(
                connector.get("applicable_source_piece_ids")
                or [connector.get("source_piece_id")]
            )
            if not applicable_source_piece_ids.intersection(endpoints):
                errors.append(
                    f"relation drops connection hypothesis source piece {connector_id}: {relation_id}"
                )
            if relation.get("relation_type") not in (
                connector.get("allowed_relation_types") or []
            ):
                errors.append(
                    f"relation exceeds connection hypothesis authorization {connector_id}: {relation_id}"
                )
    return errors


def _unique_id_set(
    items: list[dict[str, Any]],
    field: str,
    label: str,
    errors: list[str],
) -> set[str]:
    output: set[str] = set()
    for item in items:
        value = compact_text(item.get(field))
        if not value:
            errors.append(f"{label} id is missing")
        elif value in output:
            errors.append(f"duplicate {label} id: {value}")
        output.add(value)
    return output
