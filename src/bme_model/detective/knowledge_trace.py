from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import compact_text, question_id, stable_id, unique_strings


CLASSIC_EPISTEMIC_ROLE = "diagnostic_method_not_world_evidence"
INVERSION_AUTHORIZATIONS = {"provisional", "restricted", "blocked"}


def build_classic_trace_bundle(
    question: str,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Turn meta-model lens use into auditable detective-layer lineage."""

    qid = question_id(question)
    grounding = {
        compact_text(item.get("lens_id")): item
        for item in diagnosis.get("classic_grounding", []) or []
        if compact_text(item.get("lens_id"))
    }
    attributions = {
        compact_text(item.get("shadow_id")): item
        for item in diagnosis.get("bias_attribution", []) or []
        if compact_text(item.get("shadow_id"))
    }
    differentials = {
        compact_text(item.get("shadow_id")): item
        for item in diagnosis.get("differential_diagnosis", []) or []
        if compact_text(item.get("shadow_id"))
    }
    inversions = {
        compact_text(item.get("shadow_id")): item
        for item in diagnosis.get("inversion_plan", []) or []
        if compact_text(item.get("shadow_id"))
    }

    traces: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    for shadow in diagnosis.get("shadow_registry", []) or []:
        shadow_id = compact_text(shadow.get("id"))
        person_id = compact_text(shadow.get("person_id"))
        if not shadow_id or not person_id:
            continue
        attribution = attributions.get(shadow_id, {})
        differential = differentials.get(shadow_id, {})
        inversion = inversions.get(shadow_id, {})
        chain = shadow.get("chain_diagnosis") or {}
        lens_ids = unique_strings(
            list((shadow.get("attribution") or {}).get("classic_lenses") or [])
            + list(attribution.get("classic_lenses") or [])
        )
        lens_path = _lens_path(lens_ids, grounding, attribution, inversion)
        differential_resolution = _differential_resolution(differential)
        evidence = [
            {
                "stage": compact_text(item.get("stage")),
                "fact": compact_text(item.get("fact")),
                "excerpt": compact_text(item.get("excerpt")),
            }
            for item in chain.get("observed_evidence", []) or []
            if compact_text(item.get("fact"))
        ]
        completeness = {
            "has_chain_evidence": bool(evidence),
            "has_classic_lens": bool(lens_path),
            "has_differential_check": bool(differential),
            "has_inversion_rule": bool(inversion.get("inversion_rules")),
            "has_boundary_rule": bool(inversion.get("boundary_conditions")),
            "has_anti_misuse_rule": bool(inversion.get("anti_misuse_rules")),
        }
        trace_status = _trace_status(chain, completeness)
        trace_id = stable_id("classic_trace", qid, shadow_id)
        certificate_id = stable_id("inversion", qid, shadow_id)
        authorization = _inversion_authorization(
            trace_status,
            differential_resolution,
            compact_text(shadow.get("puzzle_use")),
        )
        trace = {
            "trace_id": trace_id,
            "question_id": qid,
            "shadow_id": shadow_id,
            "person_id": person_id,
            "detector_id": compact_text(
                shadow.get("detector_id") or chain.get("detector_id")
            ),
            "source_stage": compact_text(shadow.get("source_stage")),
            "diagnosis": compact_text(
                shadow.get("plain_language")
                or chain.get("plain_language_diagnosis")
                or shadow.get("description")
            ),
            "chain_evidence": evidence,
            "evidence_refs": unique_strings(
                item.get("anchor_id")
                for item in evidence
                if item.get("anchor_id")
            ),
            "diagnostic_confidence": chain.get("diagnostic_confidence"),
            "evidence_sufficiency": chain.get("evidence_sufficiency") or {},
            "lens_path": lens_path,
            "micro_lens_candidates": _micro_lens_candidates(attribution),
            "shadow_mechanism_candidates": _shadow_mechanism_candidates(
                attribution
            ),
            "differential_diagnosis": _compact_differential(differential),
            "differential_resolution": differential_resolution,
            "inversion_certificate_id": certificate_id,
            "trace_status": trace_status,
            "completeness": completeness,
            "epistemic_role": CLASSIC_EPISTEMIC_ROLE,
            "classic_as_world_evidence": False,
        }
        certificate = {
            "certificate_id": certificate_id,
            "question_id": qid,
            "trace_id": trace_id,
            "shadow_id": shadow_id,
            "person_id": person_id,
            "material_role": compact_text(
                inversion.get("puzzle_use") or shadow.get("puzzle_use")
            )
            or "unknown",
            "resulting_material": compact_text(inversion.get("resulting_material")),
            "authorization": authorization,
            "authorization_reason": _authorization_reason(
                authorization,
                trace_status,
                differential_resolution,
            ),
            "inversion_rules": unique_strings(inversion.get("inversion_rules") or []),
            "boundary_conditions": unique_strings(
                inversion.get("boundary_conditions") or []
            ),
            "anti_misuse_rules": unique_strings(
                inversion.get("anti_misuse_rules") or []
            ),
            "estimated_bias_vector": inversion.get("estimated_bias_vector")
            or shadow.get("estimated_bias_vector")
            or {},
            "allowed_transformations": [
                "preserved_fragment_candidate",
                "diagnosed_distortion",
                "missing_bridge_boundary",
            ],
            "prohibited_uses": [
                "不得把经典著作或透镜本身当作现实事实证据",
                "不得把诊断材料计作独立来源或多数支持",
                "不得自动计算偏差中点、概率或数值修正",
                "不得绕过关系裁决直接宣布真相轮廓",
            ],
            "differential_status": differential_resolution["status"],
            "epistemic_role": CLASSIC_EPISTEMIC_ROLE,
            "classic_as_world_evidence": False,
        }
        traces.append(trace)
        certificates.append(certificate)

    return {
        "classic_diagnostic_traces": traces,
        "inversion_certificates": certificates,
        "trace_index": _trace_index(traces),
        "certificate_index": {
            item["certificate_id"]: item for item in certificates
        },
    }


def traces_for_verdict(
    bundle: dict[str, Any],
    person_id: str,
    detector_ids: list[Any],
) -> list[dict[str, Any]]:
    trace_ids: list[str] = []
    index = bundle.get("trace_index") or {}
    for detector_id in unique_strings(detector_ids):
        trace_ids.extend(index.get(f"{person_id}\x1f{detector_id}", []))
    by_id = {
        item["trace_id"]: item
        for item in bundle.get("classic_diagnostic_traces", []) or []
    }
    return [by_id[item] for item in unique_strings(trace_ids) if item in by_id]


def piece_knowledge_trace(
    traces: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    transformation: str,
) -> dict[str, Any]:
    certificates = bundle.get("certificate_index") or {}
    trace_ids = [item["trace_id"] for item in traces]
    certificate_ids = unique_strings(
        item.get("inversion_certificate_id") for item in traces
    )
    linked_certificates = [
        certificates[item]
        for item in certificate_ids
        if item in certificates
    ]
    authorizations = [
        compact_text(item.get("authorization"))
        for item in linked_certificates
    ]
    authorization = _most_restrictive_authorization(authorizations)
    return {
        "classic_trace_ids": trace_ids,
        "inversion_certificate_ids": certificate_ids,
        "lens_ids": unique_strings(
            lens.get("lens_id")
            for trace in traces
            for lens in trace.get("lens_path", []) or []
        ),
        "mechanism_summaries": unique_strings(
            lens.get("core_idea")
            for trace in traces
            for lens in trace.get("lens_path", []) or []
        )[:3],
        "differential_statuses": unique_strings(
            (trace.get("differential_resolution") or {}).get("status")
            for trace in traces
        ),
        "inversion_authorization": authorization,
        "material_roles": unique_strings(
            item.get("material_role") for item in linked_certificates
        ),
        "transformation": transformation,
        "trace_link_status": "linked" if trace_ids else "missing_classic_trace",
        "epistemic_role": CLASSIC_EPISTEMIC_ROLE,
        "classic_as_world_evidence": False,
    }


def aggregate_piece_knowledge(
    pieces: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic = [
        item
        for item in pieces
        if (item.get("knowledge_trace") or {}).get("trace_link_status")
    ]
    trace_ids = unique_strings(
        trace_id
        for item in diagnostic
        for trace_id in (item.get("knowledge_trace") or {}).get(
            "classic_trace_ids", []
        )
    )
    certificate_ids = unique_strings(
        certificate_id
        for item in diagnostic
        for certificate_id in (item.get("knowledge_trace") or {}).get(
            "inversion_certificate_ids", []
        )
    )
    bridge_ids = unique_strings(
        bridge_id
        for item in diagnostic
        for bridge_id in (item.get("knowledge_trace") or {}).get(
            "diagnostic_bridge_ids", []
        )
    )
    statuses = unique_strings(
        (item.get("knowledge_trace") or {}).get("inversion_authorization")
        for item in diagnostic
    )
    if not diagnostic:
        grounding_status = "not_applicable"
    elif all(
        (item.get("knowledge_trace") or {}).get("trace_link_status") == "linked"
        for item in diagnostic
    ):
        grounding_status = "complete"
    else:
        grounding_status = "partial"
    return {
        "diagnostic_piece_ids": [item.get("piece_id") for item in diagnostic],
        "classic_trace_ids": trace_ids,
        "inversion_certificate_ids": certificate_ids,
        "diagnostic_bridge_ids": bridge_ids,
        "inversion_authorizations": statuses,
        "grounding_status": grounding_status,
        "epistemic_role": CLASSIC_EPISTEMIC_ROLE,
        "classic_as_world_evidence": False,
    }


def _lens_path(
    lens_ids: list[str],
    grounding: dict[str, dict[str, Any]],
    attribution: dict[str, Any],
    inversion: dict[str, Any],
) -> list[dict[str, Any]]:
    mechanisms = {
        compact_text(item.get("lens_id")): item
        for item in attribution.get("mechanism_chains", []) or []
        if compact_text(item.get("lens_id"))
    }
    output = []
    for lens_id in lens_ids:
        base = grounding.get(lens_id, {})
        mechanism = mechanisms.get(lens_id, {})
        output.append(
            {
                "lens_id": lens_id,
                "source_works": unique_strings(base.get("source_works") or []),
                "core_idea": compact_text(
                    mechanism.get("core_idea") or base.get("core_idea")
                ),
                "mechanism_chain": unique_strings(
                    mechanism.get("mechanism_chain")
                    or base.get("mechanism_chain")
                    or []
                ),
                "boundary_conditions": unique_strings(
                    inversion.get("boundary_conditions") or []
                ),
                "anti_misuse_rules": unique_strings(
                    inversion.get("anti_misuse_rules") or []
                ),
            }
        )
    return output


def _micro_lens_candidates(attribution: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "micro_lens_id": compact_text(item.get("micro_lens_id")),
            "parent_lens": compact_text(item.get("parent_lens")),
            "name": compact_text(item.get("name")),
            "mechanism": compact_text(item.get("mechanism")),
            "differentiates_from": unique_strings(
                item.get("differentiates_from") or []
            ),
            "inversion_hint": compact_text(item.get("inversion_hint")),
            "misuse_warning": compact_text(item.get("misuse_warning")),
        }
        for item in attribution.get("micro_lens_candidates", []) or []
        if compact_text(item.get("micro_lens_id"))
    ]


def _shadow_mechanism_candidates(
    attribution: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "mechanism_id": compact_text(item.get("mechanism_id")),
            "name": compact_text(item.get("name")),
            "core_mechanism": compact_text(item.get("core_mechanism")),
            "generation_chain": unique_strings(item.get("generation_chain") or []),
            "false_positive_risks": unique_strings(
                item.get("false_positive_risks") or []
            ),
            "false_negative_risks": unique_strings(
                item.get("false_negative_risks") or []
            ),
            "puzzle_material": compact_text(item.get("puzzle_material")),
            "match_score": item.get("match_score"),
        }
        for item in attribution.get("shadow_mechanism_candidates", []) or []
        if compact_text(item.get("mechanism_id"))
    ]


def _compact_differential(differential: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": compact_text(item.get("rule_id")),
            "status": compact_text(item.get("status")),
            "competing_lenses": unique_strings(item.get("competing_lenses") or []),
            "confusion_pattern": compact_text(item.get("confusion_pattern")),
            "decision_questions": unique_strings(item.get("decision_questions") or []),
            "coexist_when": unique_strings(item.get("coexist_when") or []),
            "misdiagnosis_risk": compact_text(item.get("misdiagnosis_risk")),
        }
        for item in differential.get("rules", []) or []
        if compact_text(item.get("rule_id"))
    ]


def _differential_resolution(differential: dict[str, Any]) -> dict[str, Any]:
    rules = differential.get("rules", []) or []
    active = [
        compact_text(item.get("rule_id"))
        for item in rules
        if item.get("status") == "active_competition"
    ]
    watched = [
        compact_text(item.get("rule_id"))
        for item in rules
        if item.get("status") == "watch_for_confusion"
    ]
    if active:
        return {
            "status": "active_competition_unresolved",
            "rule_ids": active,
            "reason": "多个经典透镜都能解释当前阴影，现有记录尚未完成排他裁决。",
        }
    if watched:
        return {
            "status": "watch_for_confusion",
            "rule_ids": watched,
            "reason": "当前主诊断可暂用，但仍需防止与相邻机制混淆。",
        }
    return {
        "status": "no_recorded_competition",
        "rule_ids": [],
        "reason": "当前规则库没有记录需要优先裁决的透镜竞争。",
    }


def _trace_status(
    chain: dict[str, Any], completeness: dict[str, bool]
) -> str:
    sufficiency = compact_text(
        (chain.get("evidence_sufficiency") or {}).get("level")
    )
    if not completeness["has_chain_evidence"] or not completeness["has_classic_lens"]:
        return "blocked"
    if sufficiency in {"insufficient", "weak"}:
        return "blocked"
    if all(
        completeness[key]
        for key in (
            "has_inversion_rule",
            "has_boundary_rule",
            "has_anti_misuse_rule",
        )
    ):
        return "complete"
    return "partial"


def _inversion_authorization(
    trace_status: str,
    differential: dict[str, Any],
    puzzle_use: str,
) -> str:
    if trace_status == "blocked":
        return "blocked"
    if differential.get("status") == "active_competition_unresolved":
        return "restricted"
    if puzzle_use == "anchor" or trace_status == "partial":
        return "restricted"
    return "provisional"


def _authorization_reason(
    authorization: str,
    trace_status: str,
    differential: dict[str, Any],
) -> str:
    if authorization == "blocked":
        return "链条证据、经典依据或反演边界不完整，只能登记诊断缺口。"
    if differential.get("status") == "active_competition_unresolved":
        return "存在尚未裁决的透镜竞争，只能形成边界或待核验材料。"
    if authorization == "restricted":
        return "可以保留为候选材料，但不能升级为锚点或事实支持。"
    if differential.get("status") == "watch_for_confusion":
        return "主诊断暂可用于寻找连接，但关系裁决必须继续检查相邻机制，不能把它写成唯一原因。"
    return "可按反演规则形成暂定拼图材料，但仍须经过侦探关系裁决。"


def _trace_index(traces: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: defaultdict[str, list[str]] = defaultdict(list)
    for item in traces:
        key = f"{item['person_id']}\x1f{item.get('detector_id', '')}"
        index[key].append(item["trace_id"])
    return dict(index)


def _most_restrictive_authorization(values: list[str]) -> str:
    rank = {"provisional": 0, "restricted": 1, "blocked": 2}
    valid = [item for item in values if item in rank]
    if not valid:
        return "blocked"
    return max(valid, key=rank.__getitem__)
