from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .knowledge_trace import aggregate_piece_knowledge
from .mechanisms import finalize_mechanism_coverage
from .schemas import (
    DETECTIVE_SCHEMA_VERSION,
    RELATION_TYPES,
    compact_text,
    stable_id,
    unique_strings,
    validate_detective_output,
)


CONTENT_RELATIONS = {
    "independent_convergence",
    "direct_conflict",
    "apparent_conflict",
    "conditional_complement",
    "scope_refinement",
    "causal_relay",
    "shared_assumption",
    "definition_branch",
    "value_branch",
    "non_comparable",
}

SHADOW_RELATIONS = {
    "shared_source",
    "shared_model_prior",
    "correlated_error",
    "opposite_distortion",
    "collective_blind_spot",
    "diagnostic_challenge",
}

CROSS_LAYER_RELATIONS = {
    "blind_spot_fill",
    "scope_refinement",
    "opposite_distortion",
    "diagnostic_challenge",
}


def build_detective_output(
    question: str,
    materials: dict[str, Any],
    *,
    semantic_relations: list[dict[str, Any]] | None = None,
    semantic_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dual relation graph without drawing a truth contour."""

    pieces = list(materials.get("pieces", []) or [])
    piece_by_id = {piece["piece_id"]: piece for piece in pieces}
    deterministic = _source_dependency_relations(pieces)
    deterministic.extend(_exact_independent_convergence(pieces))
    relations = deterministic + _normalize_semantic_relations(
        semantic_relations or [],
        piece_by_id,
        question_id=materials["question_id"],
    )
    relations = _deduplicate_relations(relations)
    relations = [
        _attach_relation_knowledge_provenance(item, piece_by_id)
        for item in relations
    ]

    content_ids = [
        item["relation_id"]
        for item in relations
        if item["relation_type"] in CONTENT_RELATIONS
    ]
    shadow_ids = [
        item["relation_id"]
        for item in relations
        if item["relation_type"] in SHADOW_RELATIONS
    ]
    cross_ids = [
        item["relation_id"]
        for item in relations
        if item["relation_type"] in CROSS_LAYER_RELATIONS
    ]
    status_counts = Counter(item["status"] for item in relations)
    relation_type_counts = Counter(item["relation_type"] for item in relations)

    payload = {
        "schema_version": DETECTIVE_SCHEMA_VERSION,
        "question": compact_text(question),
        "question_id": materials["question_id"],
        "puzzle_schema_version": materials.get("schema_version"),
        "puzzle_pieces": pieces,
        "classic_diagnostic_traces": list(
            materials.get("classic_diagnostic_traces", []) or []
        ),
        "inversion_certificates": list(
            materials.get("inversion_certificates", []) or []
        ),
        "diagnostic_bridges": list(
            materials.get("diagnostic_bridges", []) or []
        ),
        "classic_connection_hypotheses": dict(
            materials.get("classic_connection_hypotheses") or {}
        ),
        "source_dependency_clusters": _source_dependency_clusters(pieces),
        "relation_certificates": relations,
        "content_relation_graph": {
            "node_ids": _graph_node_ids(relations, CONTENT_RELATIONS),
            "relation_ids": content_ids,
        },
        "shadow_relation_graph": {
            "node_ids": _graph_node_ids(relations, SHADOW_RELATIONS),
            "relation_ids": shadow_ids,
        },
        "cross_layer_relations": {
            "relation_ids": cross_ids,
        },
        "structural_patterns": _structural_patterns(relations),
        "mechanism_coverage": finalize_mechanism_coverage(
            materials.get("mechanism_index") or {},
            pieces,
            relations,
        ),
        "detective_evaluation": {
            "scope": "跨数字人材料连接的执行质量，不代表关系真实准确率或真相准确率。",
            "piece_count": len(pieces),
            "relation_count": len(relations),
            "status_counts": dict(status_counts),
            "relation_type_counts": dict(relation_type_counts),
            "accepted_traceability_rate": _accepted_traceability_rate(relations),
            "accepted_classic_traceability_rate": _accepted_classic_traceability_rate(
                relations
            ),
            "classic_connection_hypothesis_count": len(
                (
                    materials.get("classic_connection_hypotheses") or {}
                ).get("hypotheses", [])
                or []
            ),
            "classic_guided_relation_count": sum(
                bool(
                    (item.get("knowledge_provenance") or {}).get(
                        "connection_hypothesis_ids"
                    )
                )
                for item in relations
            ),
            "classic_guided_accepted_relation_count": sum(
                item.get("status") == "accepted"
                and bool(
                    (item.get("knowledge_provenance") or {}).get(
                        "connection_hypothesis_ids"
                    )
                )
                for item in relations
            ),
            "classic_discovered_relation_count": sum(
                bool(
                    (item.get("provenance") or {}).get(
                        "classic_discovery_hypothesis_ids"
                    )
                )
                for item in relations
            ),
            "classic_checked_relation_count": sum(
                bool(
                    (item.get("provenance") or {}).get("classic_check")
                )
                for item in relations
            ),
            "classic_validated_relation_count": sum(
                bool(
                    (item.get("knowledge_provenance") or {}).get(
                        "connection_hypothesis_ids"
                    )
                )
                for item in relations
            ),
            "classic_unreviewed_relation_count": sum(
                bool(
                    (item.get("knowledge_provenance") or {}).get(
                        "unreviewed_connection_hypothesis_ids"
                    )
                )
                for item in relations
            ),
            "classic_used_as_world_evidence_count": sum(
                bool(
                    (item.get("knowledge_provenance") or {}).get(
                        "classic_as_world_evidence"
                    )
                )
                for item in relations
            ),
            "forced_relation_count": 0,
            "passed": all(
                item.get("piece_ids")
                and item.get("plain_language_explanation")
                and item.get("competing_explanation")
                and item.get("falsification_test")
                and (item.get("knowledge_provenance") or {}).get(
                    "classic_as_world_evidence"
                )
                is False
                for item in relations
                if item.get("status") == "accepted"
            ),
        },
        "detective_self_audit": {
            "rules": [
                "数字人人数不是证据独立性。",
                "同方向结论不自动形成印证。",
                "相反结论不自动形成真实冲突。",
                "共同沉默只产生待核验盲区候选。",
                "经典连接假设只规定检验路线，不提供现实事实。",
                "没有可靠连接是合法结果。",
            ],
            "semantic_stage": semantic_audit or {
                "mode": "deterministic_foundation_only",
                "warning": "尚未经过跨人语义提出与对抗裁决；内容关系应保持克制。",
            },
        },
    }
    errors = validate_detective_output(payload, materials)
    if errors:
        raise ValueError("Invalid detective output: " + "; ".join(errors))
    return payload


def relation_certificate(
    *,
    question_id: str,
    relation_type: str,
    piece_ids: list[str],
    status: str,
    explanation: str,
    shared_coordinate: dict[str, Any] | None = None,
    source_independence: str = "unknown",
    common_error_risk: str = "unknown",
    competing_explanation: str,
    falsification_test: str,
    confidence_profile: dict[str, Any] | None = None,
    independence_profile: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    endpoints = sorted(unique_strings(piece_ids))
    relation_id = stable_id(
        "relation",
        question_id,
        relation_type,
        "|".join(endpoints),
        explanation,
    )
    return {
        "relation_id": relation_id,
        "question_id": question_id,
        "relation_type": relation_type,
        "piece_ids": endpoints,
        "status": status,
        "plain_language_explanation": compact_text(explanation),
        "shared_coordinate": shared_coordinate or {},
        "source_independence": source_independence,
        "independence_profile": independence_profile or {},
        "common_error_risk": common_error_risk,
        "competing_explanation": compact_text(competing_explanation),
        "falsification_test": compact_text(falsification_test),
        "confidence_profile": confidence_profile or {
            "semantic_alignment": "unknown",
            "scope_compatibility": "unknown",
            "source_independence": source_independence,
            "diagnostic_certainty": "unknown",
        },
        "provenance": provenance or {"mode": "deterministic"},
    }


def _source_dependency_relations(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for piece in pieces:
        if piece.get("piece_kind") not in {"observed_claim", "evidence_claim", "retained_local"}:
            continue
        for family in piece.get("source_families", []) or []:
            by_family[family].append(piece)

    relations: list[dict[str, Any]] = []
    for family, candidates in sorted(by_family.items()):
        people = {item.get("person_id") for item in candidates if item.get("person_id") != "collective"}
        if len(people) < 2:
            continue
        endpoints = _one_piece_per_person(candidates)
        relation_type = "shared_model_prior" if family == "shared-base-model-prior" else "shared_source"
        label = "共同的基础模型先验" if relation_type == "shared_model_prior" else "同一外部来源"
        relations.append(
            relation_certificate(
                question_id=endpoints[0]["question_id"],
                relation_type=relation_type,
                piece_ids=[item["piece_id"] for item in endpoints],
                status="accepted",
                explanation=f"{len(people)} 个数字人的这些材料依赖{label}，因此相似意见不能按 {len(people)} 份独立支持计算。",
                shared_coordinate={"source_family": family},
                source_independence="shared",
                common_error_risk="high" if relation_type == "shared_model_prior" else "medium",
                competing_explanation="即使来源相同，不同数字人也可能基于不同材料部分形成独立推理。",
                falsification_test="若能证明这些数字人的关键判断分别由不同的一手证据和独立推理产生，应拆分该依赖关系。",
                confidence_profile={
                    "semantic_alignment": "not_claimed",
                    "scope_compatibility": "not_applicable",
                    "source_independence": "shared",
                    "diagnostic_certainty": "high",
                },
                independence_profile={
                    "model_independence": (
                        "shared" if relation_type == "shared_model_prior" else "unknown"
                    ),
                    "evidence_origin_independence": (
                        "shared" if relation_type == "shared_source" else "not_assessed"
                    ),
                    "interpretation_independence": "correlated",
                },
                provenance={"mode": "exact_source_lineage", "source_family": family},
            )
        )
    return relations


def _exact_independent_convergence(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_claim: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for piece in pieces:
        if piece.get("piece_kind") not in {"retained_local", "evidence_claim"}:
            continue
        normalized = compact_text(piece.get("normalized_claim"))
        if len(normalized) >= 12:
            by_claim[normalized].append(piece)

    relations: list[dict[str, Any]] = []
    for normalized, candidates in sorted(by_claim.items()):
        people = {item.get("person_id") for item in candidates}
        if len(people) < 2:
            continue
        families = [_evidence_families(item) for item in candidates]
        if not families or not all(families):
            continue
        shared = set.intersection(*families)
        if shared:
            continue
        endpoints = _one_piece_per_person(candidates)
        shared_model = _shared_model_prior(endpoints)
        source_independence = (
            "evidence_independent_model_correlated"
            if shared_model
            else "independent_by_current_ledger"
        )
        relations.append(
            relation_certificate(
                question_id=endpoints[0]["question_id"],
                relation_type="independent_convergence",
                piece_ids=[item["piece_id"] for item in endpoints],
                status="accepted",
                explanation=f"{len(people)} 个数字人在可保留局部中给出相同命题，当前未发现共同来源，可作为独立会合候选。",
                shared_coordinate={"normalized_claim": normalized},
                source_independence=source_independence,
                common_error_risk=(
                    "shared_model_interpretation" if shared_model else "unknown"
                ),
                competing_explanation=(
                    "相同措辞可能来自共同训练语料或未被账本记录的共同来源。"
                    if shared_model
                    else "仍可能存在账本没有记录的共同来源或共同转述链。"
                ),
                falsification_test="若发现共同训练模板、共同引用链或命题只是在表面措辞上相同，应降级或撤销该会合。",
                confidence_profile={
                    "semantic_alignment": "high_exact_match",
                    "scope_compatibility": "unverified",
                    "source_independence": (
                        "evidence_high_model_low"
                        if shared_model
                        else "evidence_high_model_unknown"
                    ),
                    "diagnostic_certainty": "medium",
                },
                independence_profile={
                    "model_independence": (
                        "shared" if shared_model else "unknown"
                    ),
                    "evidence_origin_independence": "independent_by_current_ledger",
                    "publisher_independence": _publisher_independence(endpoints),
                    "interpretation_independence": (
                        "correlated" if shared_model else "unknown"
                    ),
                    "directional_support_authorized": _directional_support_authorized(
                        endpoints
                    ),
                },
                provenance={"mode": "exact_normalized_claim_match"},
            )
        )
    return relations


def _normalize_semantic_relations(
    relations: list[dict[str, Any]],
    piece_by_id: dict[str, dict[str, Any]],
    *,
    question_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in relations:
        relation_type = compact_text(item.get("relation_type"))
        if relation_type not in RELATION_TYPES:
            continue
        endpoints = unique_strings(item.get("piece_ids") or [])
        if len(endpoints) < 2 or any(piece_id not in piece_by_id for piece_id in endpoints):
            continue
        explanation = compact_text(item.get("plain_language_explanation") or item.get("explanation"))
        if not explanation:
            continue
        status = compact_text(item.get("status")) or "unresolved"
        if status == "accepted" and (
            not compact_text(item.get("competing_explanation"))
            or not compact_text(item.get("falsification_test"))
        ):
            status = "unresolved"
        output.append(
            relation_certificate(
                question_id=question_id,
                relation_type=relation_type,
                piece_ids=endpoints,
                status=status,
                explanation=explanation,
                shared_coordinate=item.get("shared_coordinate") or {},
                source_independence=compact_text(item.get("source_independence")) or "unknown",
                common_error_risk=compact_text(item.get("common_error_risk")) or "unknown",
                competing_explanation=compact_text(item.get("competing_explanation")) or "尚缺竞争解释，暂不接受该连接。",
                falsification_test=compact_text(item.get("falsification_test")) or "补充同范围、同定义的独立证据后重新裁决。",
                confidence_profile=item.get("confidence_profile") or {},
                independence_profile=item.get("independence_profile") or {},
                provenance=item.get("provenance") or {"mode": "semantic_adjudication"},
            )
        )
    return output


def _source_dependency_clusters(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: defaultdict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"person_ids": set(), "piece_ids": set()}
    )
    for piece in pieces:
        for family in piece.get("source_families", []) or []:
            by_family[family]["person_ids"].add(piece.get("person_id"))
            by_family[family]["piece_ids"].add(piece.get("piece_id"))
    clusters = []
    for family, values in sorted(by_family.items()):
        person_ids = sorted(item for item in values["person_ids"] if item)
        if len(person_ids) < 2:
            continue
        clusters.append(
            {
                "source_family": family,
                "person_ids": person_ids,
                "piece_ids": sorted(values["piece_ids"]),
                "effective_support_rule": "同一来源家族最多按一份来源支持计算。",
            }
        )
    return clusters


def _structural_patterns(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [item for item in relations if item.get("status") == "accepted"]
    by_type: defaultdict[str, list[str]] = defaultdict(list)
    for item in accepted:
        by_type[item["relation_type"]].append(item["relation_id"])
    labels = {
        "independent_convergence": "独立会合",
        "shared_source": "同源依赖",
        "shared_model_prior": "共同模型先验",
        "correlated_error": "相关错误",
        "direct_conflict": "真实冲突",
        "apparent_conflict": "表面冲突",
        "conditional_complement": "条件互补",
        "causal_relay": "因果接力",
        "definition_branch": "定义分叉",
        "value_branch": "价值分叉",
        "collective_blind_spot": "集体盲区",
    }
    return [
        {
            "pattern_type": relation_type,
            "plain_label": labels.get(relation_type, relation_type),
            "relation_ids": relation_ids,
            "count": len(relation_ids),
        }
        for relation_type, relation_ids in sorted(by_type.items())
    ]


def _graph_node_ids(relations: list[dict[str, Any]], allowed_types: set[str]) -> list[str]:
    return sorted(
        {
            piece_id
            for item in relations
            if item["relation_type"] in allowed_types
            for piece_id in item.get("piece_ids", [])
        }
    )


def _one_piece_per_person(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    priority = {"retained_local": 0, "observed_claim": 1, "evidence_claim": 2}
    for item in candidates:
        person_id = str(item.get("person_id"))
        current = selected.get(person_id)
        if current is None or priority.get(item.get("piece_kind"), 9) < priority.get(current.get("piece_kind"), 9):
            selected[person_id] = item
    return [selected[key] for key in sorted(selected)]


def _publisher_independence(pieces: list[dict[str, Any]]) -> str:
    families = [
        set(
            item.get("publisher_families", [])
            or (item.get("independence_profile") or {}).get(
                "publisher_families", []
            )
            or []
        )
        for item in pieces
    ]
    if not families or not all(families):
        return "unknown"
    return "shared" if set.intersection(*families) else "independent_by_current_ledger"


def _evidence_families(piece: dict[str, Any]) -> set[str]:
    origins = set(
        piece.get("origin_families", [])
        or (piece.get("independence_profile") or {}).get(
            "origin_families", []
        )
        or []
    )
    if origins:
        return origins
    explicit = set(piece.get("evidence_source_families", []) or [])
    if explicit:
        return explicit
    return {
        family
        for family in piece.get("source_families", []) or []
        if family != "shared-base-model-prior"
    }


def _directional_support_authorized(pieces: list[dict[str, Any]]) -> bool:
    statuses = {
        status
        for piece in pieces
        for status in (piece.get("independence_profile") or {}).get(
            "verification_statuses", []
        )
    }
    if not statuses:
        return False
    return any(
        status in {"page_verified", "primary_verified", "cross_verified"}
        for status in statuses
    )


def _deduplicate_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    status_rank = {"accepted": 3, "unresolved": 2, "proposed": 1, "rejected": 0}
    for relation in relations:
        key = (relation["relation_type"], tuple(sorted(relation.get("piece_ids", []))))
        current = chosen.get(key)
        if current is None or status_rank.get(relation.get("status"), -1) > status_rank.get(current.get("status"), -1):
            chosen[key] = relation
    return sorted(chosen.values(), key=lambda item: item["relation_id"])


def _accepted_traceability_rate(relations: list[dict[str, Any]]) -> float:
    accepted = [item for item in relations if item.get("status") == "accepted"]
    if not accepted:
        return 1.0
    traced = sum(
        1
        for item in accepted
        if item.get("piece_ids")
        and item.get("plain_language_explanation")
        and item.get("provenance")
    )
    return round(traced / len(accepted), 4)


def _attach_relation_knowledge_provenance(
    relation: dict[str, Any],
    piece_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    endpoint_pieces = [
        piece_by_id[piece_id]
        for piece_id in relation.get("piece_ids", []) or []
        if piece_id in piece_by_id
    ]
    knowledge = aggregate_piece_knowledge(endpoint_pieces)
    diagnostic_only = bool(endpoint_pieces) and len(
        knowledge.get("diagnostic_piece_ids") or []
    ) == len(endpoint_pieces)
    authorizations = set(knowledge.get("inversion_authorizations") or [])
    if "blocked" in authorizations:
        permission = "diagnostic_challenge_only"
    elif "restricted" in authorizations:
        permission = "structural_constraint_only"
    elif authorizations:
        permission = "relationship_adjudication_required"
    else:
        permission = "not_applicable"
    provenance_mode = compact_text((relation.get("provenance") or {}).get("mode"))
    connection_hypothesis_ids = unique_strings(
        (relation.get("provenance") or {}).get(
            "connection_hypothesis_ids"
        )
        or []
    )
    classic_discovery_hypothesis_ids = unique_strings(
        (relation.get("provenance") or {}).get(
            "classic_discovery_hypothesis_ids"
        )
        or []
    )
    rejected_connection_hypothesis_ids = unique_strings(
        (relation.get("provenance") or {}).get(
            "rejected_connection_hypothesis_ids"
        )
        or []
    )
    unreviewed_connection_hypothesis_ids = unique_strings(
        (relation.get("provenance") or {}).get(
            "unreviewed_connection_hypothesis_ids"
        )
        or []
    )
    classic_check = dict(
        (relation.get("provenance") or {}).get("classic_check") or {}
    )
    if connection_hypothesis_ids:
        use_mode = "classic_connection_operator_tested"
    elif knowledge.get("diagnostic_piece_ids") and provenance_mode in {
        "live_adversarial_adjudication",
        "semantic_adjudication",
    }:
        use_mode = "diagnostic_context_considered"
    elif knowledge.get("diagnostic_piece_ids"):
        use_mode = "lineage_carried_not_relation_basis"
    else:
        use_mode = "not_applicable"

    guarded = dict(relation)
    guarded["independence_profile"] = _merge_independence_profile(
        relation.get("independence_profile") or {},
        endpoint_pieces,
    )
    evidence_independence = guarded["independence_profile"].get(
        "evidence_origin_independence"
    )
    if evidence_independence == "independent_by_current_ledger":
        guarded["source_independence"] = (
            "evidence_independent_model_correlated"
            if guarded["independence_profile"].get("model_independence") == "shared"
            else "independent_by_current_ledger"
        )
    elif evidence_independence == "shared":
        guarded["source_independence"] = "shared"
    else:
        guarded["source_independence"] = "unknown"
    guardrail_action = "none"
    if guarded.get("status") == "accepted" and diagnostic_only:
        relation_type = guarded.get("relation_type")
        if relation_type in {"independent_convergence", "causal_relay"}:
            guarded["status"] = "unresolved"
            guardrail_action = "downgraded_diagnostic_material_not_world_evidence"
        elif permission == "diagnostic_challenge_only" and relation_type not in {
            "diagnostic_challenge",
            "non_comparable",
            "shared_source",
            "shared_model_prior",
            "correlated_error",
        }:
            guarded["status"] = "unresolved"
            guardrail_action = "downgraded_blocked_inversion"
    guarded["knowledge_provenance"] = {
        **knowledge,
        "connection_hypothesis_ids": connection_hypothesis_ids,
        "classic_discovery_hypothesis_ids": classic_discovery_hypothesis_ids,
        "rejected_connection_hypothesis_ids": rejected_connection_hypothesis_ids,
        "unreviewed_connection_hypothesis_ids": unreviewed_connection_hypothesis_ids,
        "classic_check": classic_check,
        "classic_review_cannot_modify_relation": bool(
            (relation.get("provenance") or {}).get(
                "classic_review_cannot_modify_relation"
            )
        ),
        "diagnostic_only_endpoints": diagnostic_only,
        "inference_permission": permission,
        "relation_use_mode": use_mode,
        "guardrail_action": guardrail_action,
    }
    return guarded


def _merge_independence_profile(
    declared: dict[str, Any],
    pieces: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_sets = [_evidence_families(piece) for piece in pieces]
    all_backed = bool(evidence_sets) and all(evidence_sets)
    shared_evidence = (
        set.intersection(*evidence_sets) if all_backed else set()
    )
    if all_backed and not shared_evidence:
        evidence_independence = "independent_by_current_ledger"
    elif shared_evidence:
        evidence_independence = "shared"
    else:
        evidence_independence = "unknown"
    model_sets = [
        set(piece.get("model_source_families", []) or [])
        or {
            family
            for family in piece.get("source_families", []) or []
            if family == "shared-base-model-prior"
        }
        for piece in pieces
    ]
    model_shared = bool(model_sets) and all(model_sets) and bool(
        set.intersection(*model_sets)
    )
    publisher_independence = _publisher_independence(pieces)
    directional = bool(
        evidence_independence == "independent_by_current_ledger"
        and publisher_independence == "independent_by_current_ledger"
        and _directional_support_authorized(pieces)
    )
    return {
        **declared,
        "model_independence": "shared" if model_shared else "unknown",
        "evidence_origin_independence": evidence_independence,
        "publisher_independence": publisher_independence,
        "interpretation_independence": (
            "correlated" if model_shared else "unknown"
        ),
        "directional_support_authorized": directional,
    }


def _shared_model_prior(pieces: list[dict[str, Any]]) -> bool:
    model_sets = [
        set(piece.get("model_source_families", []) or [])
        or {
            family
            for family in piece.get("source_families", []) or []
            if family == "shared-base-model-prior"
        }
        for piece in pieces
    ]
    return bool(model_sets) and all(model_sets) and bool(
        set.intersection(*model_sets)
    )


def _accepted_classic_traceability_rate(
    relations: list[dict[str, Any]],
) -> float:
    diagnostic = [
        item
        for item in relations
        if item.get("status") == "accepted"
        and (item.get("knowledge_provenance") or {}).get(
            "diagnostic_piece_ids"
        )
    ]
    if not diagnostic:
        return 1.0
    complete = sum(
        (item.get("knowledge_provenance") or {}).get("grounding_status")
        == "complete"
        for item in diagnostic
    )
    return round(complete / len(diagnostic), 4)
