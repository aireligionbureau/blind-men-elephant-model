from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..detective.schemas import compact_text, stable_id, unique_strings
from .epistemic import has_structural_uncertainty
from .schemas import (
    CONSTRAINT_SCHEMA_VERSION,
    TRUTH_CONTOUR_SCHEMA_VERSION,
    validate_constraints,
    validate_truth_contour,
)


RELATION_TO_CONSTRAINT = {
    "independent_convergence": "support_candidate",
    "shared_source": "dependency_penalty",
    "shared_model_prior": "dependency_penalty",
    "correlated_error": "correlated_error_warning",
    "direct_conflict": "conflict_boundary",
    "apparent_conflict": "conditional_split",
    "conditional_complement": "conditional_split",
    "scope_refinement": "scope_boundary",
    "causal_relay": "causal_bridge",
    "shared_assumption": "shared_assumption_dependency",
    "definition_branch": "definition_branch",
    "value_branch": "value_branch",
    "opposite_distortion": "directional_bound",
    "blind_spot_fill": "unknown_space",
    "collective_blind_spot": "unknown_space",
    "non_comparable": "non_comparable",
}


CONSTRAINT_RULES = {
    "support_candidate": "这些局部判断暂时可以互相印证，但仍需排除共同训练先验和未记账的共同来源。",
    "dependency_penalty": "这些声音不能按人数累加；它们共享同一来源或同一模型先验，最多算一束相关材料。",
    "correlated_error_warning": "相似结论可能由相同错误机制共同制造，不能把重复错误误当成共识。",
    "conditional_split": "看似相反的判断可能分别在不同条件下成立，轮廓必须写清条件，不能强行二选一。",
    "conflict_boundary": "双方在同一定义、范围和时间下无法同时成立；轮廓必须保留冲突，不能平均掉。",
    "scope_boundary": "局部判断只能约束其实际覆盖的对象、范围和时间，不能越界代表整体。",
    "causal_bridge": "这些材料可以组成一段候选因果链，但链条每一段仍需分别核验。",
    "shared_assumption_dependency": "多条结论依赖同一个前提；前提一旦失效，相关结论会一起松动。",
    "definition_branch": "分歧的一部分来自定义不同；轮廓必须先说明采用哪种定义。",
    "value_branch": "分歧的一部分来自价值取舍，而不是事实判断；主轮廓应把价值条件显式列出。",
    "directional_bound": "相反方向的扭曲只能提示真实情况可能位于两种夸张之间，不能据此计算数值中点。",
    "unknown_space": "这里是会改写结论的待核验空白；没有材料不能被写成已经证实的事实。",
    "non_comparable": "这些材料回答的不是同一个命题，暂时不能互相支持或反驳。",
}


def build_contour_constraints(
    question: str,
    detective: dict[str, Any],
) -> dict[str, Any]:
    """Translate accepted relations into rules that constrain contour inference."""

    constraints: list[dict[str, Any]] = []
    for relation in detective.get("relation_certificates", []) or []:
        if relation.get("status") != "accepted":
            continue
        relation_type = compact_text(relation.get("relation_type"))
        constraint_type = RELATION_TO_CONSTRAINT.get(relation_type)
        if not constraint_type:
            continue
        relation_id = relation["relation_id"]
        constraint_id = stable_id(
            "constraint",
            detective["question_id"],
            constraint_type,
            relation_id,
        )
        constraints.append(
            {
                "constraint_id": constraint_id,
                "question_id": detective["question_id"],
                "constraint_type": constraint_type,
                "source_relation_ids": [relation_id],
                "piece_ids": list(relation.get("piece_ids") or []),
                "rule": CONSTRAINT_RULES[constraint_type],
                "relation_explanation": compact_text(
                    relation.get("plain_language_explanation")
                ),
                "strength": _constraint_strength(relation, constraint_type),
                "scope": relation.get("shared_coordinate") or {},
                "caution": _constraint_caution(constraint_type),
                "knowledge_provenance": dict(
                    relation.get("knowledge_provenance") or {}
                ),
            }
        )

    counts = Counter(item["constraint_type"] for item in constraints)
    payload = {
        "schema_version": CONSTRAINT_SCHEMA_VERSION,
        "question": compact_text(question),
        "question_id": detective["question_id"],
        "constraints": sorted(constraints, key=lambda item: item["constraint_id"]),
        "constraint_counts": dict(sorted(counts.items())),
        "classic_knowledge_audit": _constraint_knowledge_audit(constraints),
        "solver_rules": [
            "人数不进入真相权重。",
            "同源材料先折叠，再讨论支持强度。",
            "局部保留项仍只是候选事实，不因被法医层保留就自动为真。",
            "未校准的相反偏差不做数值相减或取中点。",
            "共同沉默只形成未知空间，不自动形成反向事实。",
            "冲突若来自定义、范围、时间或价值条件，先分支再判断。",
        ],
    }
    errors = validate_constraints(payload, detective)
    if errors:
        raise ValueError("Invalid contour constraints: " + "; ".join(errors))
    return payload


def build_truth_contour(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    *,
    adjudicated_payload: dict[str, Any] | None = None,
    generation_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated truth contour, or an explicit conservative fallback."""

    if adjudicated_payload:
        if _requires_structural_cap(detective):
            adjudicated_payload = _structural_cap_payload(
                question,
                detective,
                constraints,
                adjudicated_payload,
            )
        payload = _normalize_adjudicated_contour(
            question,
            detective,
            constraints,
            adjudicated_payload,
            generation_audit or {},
        )
        errors = validate_truth_contour(
            payload,
            question=question,
            detective=detective,
            constraints=constraints,
        )
        if not errors:
            return payload
        raise ValueError(
            "Adjudicated truth contour failed delivery validation: "
            + "; ".join(errors)
        )

    payload = _conservative_contour(
        question,
        detective,
        constraints,
        generation_audit=generation_audit or {},
    )
    errors = validate_truth_contour(
        payload,
        question=question,
        detective=detective,
        constraints=constraints,
    )
    if errors:
        raise ValueError("Invalid truth contour: " + "; ".join(errors))
    return payload


def _requires_structural_cap(detective: dict[str, Any]) -> bool:
    directional = {
        item.get("relation_id")
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
        and item.get("relation_type") == "independent_convergence"
        and (item.get("independence_profile") or {}).get(
            "directional_support_authorized", True
        )
    }
    return not directional


def _structural_cap_payload(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    if _structural_payload_is_safe(raw):
        capped = dict(raw)
        for field in ("direct_answer", "main_contour"):
            statement = capped.get(field)
            if not isinstance(statement, dict):
                continue
            statement = dict(statement)
            bindings = statement.get("sentence_bindings") or []
            if isinstance(bindings, list) and bindings:
                repaired_bindings = [
                    {
                        **item,
                        "text": _downgrade_group_quantifiers(item.get("text")),
                    }
                    for item in bindings
                    if isinstance(item, dict) and compact_text(item.get("text"))
                ]
                statement["sentence_bindings"] = repaired_bindings
                statement["text"] = "".join(
                    compact_text(item.get("text")) for item in repaired_bindings
                )
            else:
                statement["text"] = _downgrade_group_quantifiers(
                    statement.get("text") or statement.get("statement")
                )
            capped[field] = statement
        capped["safety_cap"] = "structural_relations_only"
        return capped
    accepted = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
    ]
    constructive_types = {
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
    constructive = [
        item for item in accepted if item.get("relation_type") in constructive_types
    ]
    if not constructive:
        return raw

    relation_to_constraints: dict[str, list[str]] = {}
    for item in constraints.get("constraints", []) or []:
        for relation_id in item.get("source_relation_ids", []) or []:
            relation_to_constraints.setdefault(relation_id, []).append(
                item["constraint_id"]
            )

    all_constructive_refs = _refs_for_relations(
        constructive, relation_to_constraints
    )
    direct_sentences = [
        (
            f"当前材料还不能确定“{compact_text(question)}”的方向性答案。",
            "important_unknown",
        ),
        (
            "现有关系能够解释相反判断怎样受定义、条件或范围影响，却没有证明哪一方向为真。",
            "structural_relation",
        ),
    ]
    main_parts = [
        _structural_relation_summary(item.get("relation_type"))
        for item in constructive
    ]
    dependency_relations = [
        item
        for item in accepted
        if item.get("relation_type")
        in {"shared_source", "shared_model_prior", "correlated_error"}
    ]
    if dependency_relations:
        main_parts.append(
            "部分声音共享来源、模型先验或错误机制，不能按人数累加为独立支持"
        )
    main_sentences = [
        (
            "当前能拼出的轮廓是："
            + "；".join(dict.fromkeys(item for item in main_parts if item))
            + "。",
            "structural_relation",
        ),
        (
            "因此，本轮只能把问题收束到这些结构条件，不能越过它们宣布某一方向已经成立。",
            "structural_relation",
        ),
    ]
    main_refs = _refs_for_relations(
        constructive + dependency_relations,
        relation_to_constraints,
    )

    key_conditions = []
    stable_parts = []
    for relation in constructive[:4]:
        relation_type = relation.get("relation_type")
        key_conditions.append(
            {
                "text": _structural_condition(relation_type),
                **_refs_for_relations([relation], relation_to_constraints),
            }
        )
        stable_parts.append(
            {
                "text": _structural_stable_part(relation_type),
                **_refs_for_relations([relation], relation_to_constraints),
            }
        )

    boundaries = []
    for relation in dependency_relations[:3]:
        boundaries.append(
            {
                "text": _dependency_boundary(relation.get("relation_type")),
                **_refs_for_relations([relation], relation_to_constraints),
            }
        )
    unknown = (
        "在统一定义、条件和比较范围之后，哪一方向能获得来源真正独立的支持，当前关系图仍未回答。"
    )
    return {
        "direct_answer": _structural_bound_statement(
            "direct_answer", direct_sentences, all_constructive_refs
        ),
        "main_contour": _structural_bound_statement(
            "main_contour", main_sentences, main_refs
        ),
        "key_conditions": key_conditions,
        "stable_parts": stable_parts,
        "boundary_conditions": boundaries,
        "important_unknowns": [
            {"text": unknown, **all_constructive_refs}
        ],
        "confidence_statement": (
            "这里对分歧结构有把握，对问题的方向没有足够独立关系可以下注。"
        ),
        "why_this_contour": (
            "它只保留关系证书实际证明的内容：怎样的定义、条件、范围和来源依赖造成了现有分歧。"
        ),
        "strongest_counter_contour": (
            "仍可能存在一个明确的方向性答案，只是被当前的定义分歧和同源材料遮住了。"
        ),
        "counter_contour_disposition": (
            "作为待验证分支保留；在出现独立会合前，它不能替代当前的结构性轮廓。"
        ),
        "validation_drops": list(raw.get("validation_drops") or []),
        "safety_cap": "structural_relations_only",
    }


def _structural_bound_statement(
    field: str,
    sentences: list[tuple[str, str]],
    refs: dict[str, list[str]],
) -> dict[str, Any]:
    bindings = [
        {
            "claim_id": f"{field}_sentence_{index}",
            "claim_role": role,
            "text": compact_text(text),
            **refs,
        }
        for index, (text, role) in enumerate(sentences, start=1)
        if compact_text(text)
    ]
    return {
        "text": "".join(item["text"] for item in bindings),
        **refs,
        "sentence_bindings": bindings,
        "sentence_validation_drops": [],
        "bound_statement_validated": True,
    }


def _structural_payload_is_safe(raw: dict[str, Any]) -> bool:
    direct = compact_text((raw.get("direct_answer") or {}).get("text"))
    main = compact_text((raw.get("main_contour") or {}).get("text"))
    forbidden_directional_markers = (
        "必然会",
        "必然不会",
        "确定会",
        "确定不会",
        "大概率会",
        "大概率不会",
        "很可能在",
        "概率为",
        "方向已经成立",
        "已经证明会",
        "已经证明不会",
    )
    combined = f"{direct} {main}"
    return bool(
        direct
        and main
        and has_structural_uncertainty(direct)
        and not any(marker in combined for marker in forbidden_directional_markers)
    )


def _downgrade_group_quantifiers(value: Any) -> str:
    text = compact_text(value)
    replacements = {
        "大多数数字人一致认为": "部分数字人认为",
        "多数数字人一致认为": "部分数字人认为",
        "大多数分析者一致认为": "部分分析者认为",
        "多数分析者一致认为": "部分分析者认为",
        "大多数数字人": "部分数字人",
        "多数数字人": "部分数字人",
        "大多数分析者": "部分分析者",
        "多数分析者": "部分分析者",
        "所有数字人": "这些数字人",
        "全部数字人": "这些数字人",
        "所有分析者": "这些分析者",
        "全部分析者": "这些分析者",
        "数字人一致认为": "部分数字人认为",
        "数字人一致表明": "部分数字人的材料表明",
        "分析者一致认为": "部分分析者认为",
        "分析者一致表明": "部分分析者的材料表明",
    }
    pattern = "|".join(
        re.escape(value) for value in sorted(replacements, key=len, reverse=True)
    )
    return re.sub(pattern, lambda match: replacements[match.group(0)], text)


def _refs_for_relations(
    relations: list[dict[str, Any]],
    relation_to_constraints: dict[str, list[str]],
) -> dict[str, list[str]]:
    relation_ids = [item["relation_id"] for item in relations]
    return {
        "piece_ids": list(
            dict.fromkeys(
                piece_id
                for item in relations
                for piece_id in item.get("piece_ids", []) or []
            )
        ),
        "relation_ids": relation_ids,
        "constraint_ids": list(
            dict.fromkeys(
                constraint_id
                for relation_id in relation_ids
                for constraint_id in relation_to_constraints.get(relation_id, [])
            )
        ),
    }


def _structural_relation_summary(relation_type: Any) -> str:
    return {
        "definition_branch": "相反判断使用了不同定义，因此并未完全回答同一个命题",
        "conditional_complement": "不同判断可能分别在不同条件下成立，必须把条件写出来",
        "apparent_conflict": "表面冲突来自定义、范围、时间或条件错位，不能直接二选一",
        "direct_conflict": "同一命题上仍存在不能同时成立的判断，当前材料尚未裁定哪边为真",
        "scope_refinement": "局部判断只能约束其真实覆盖的范围，不能代表整体",
        "causal_relay": "若干局部可以组成候选因果链，但链条仍未成为已证事实",
        "shared_assumption": "多条结论依赖同一前提，前提失效时会一起松动",
        "value_branch": "部分分歧来自价值取舍，而不是事实本身",
        "blind_spot_fill": "一方遗漏的维度被另一方指出，但相关性仍需继续核验",
    }.get(str(relation_type), "现有材料形成了一条结构连接，但没有产生方向性证明")


def _structural_condition(relation_type: Any) -> str:
    return {
        "definition_branch": "只有先统一关键概念的定义，相反答案才真正具有可比性。",
        "conditional_complement": "只有写明各自成立的条件，局部判断才能进入同一幅轮廓。",
        "apparent_conflict": "只有对齐定义、范围、时间和条件，才能判断冲突是否真实。",
        "direct_conflict": "需要新的独立材料裁定同一命题上的真实冲突，不能用折中消掉。",
        "scope_refinement": "结论必须留在原材料实际覆盖的范围内。",
        "causal_relay": "因果链每一段都要有独立材料，才能从候选升级为支持。",
        "shared_assumption": "共同前提必须先被核验，相关结论才不会一起失效。",
        "value_branch": "事实判断和价值取舍必须分开表达。",
        "blind_spot_fill": "被补出的维度只有证明与原题直接相关后，才会改写轮廓。",
    }.get(str(relation_type), "这条结构关系需要进一步核验后才能改变方向性答案。")


def _structural_stable_part(relation_type: Any) -> str:
    return {
        "definition_branch": "这轮分歧至少有一部分来自关键概念定义不同，而不是同一事实上的直接冲突。",
        "conditional_complement": "至少有一部分相反判断可以被条件化，而不必强行互相否定。",
        "apparent_conflict": "至少有一部分冲突是比较坐标错位造成的。",
        "direct_conflict": "当前仍存在一处无法同时成立、也尚未被裁定的真实冲突。",
        "scope_refinement": "至少有一个局部判断曾越过自己能够支持的范围。",
        "causal_relay": "材料之间存在候选接力关系，但不能把接力自动写成完整因果链。",
        "shared_assumption": "若干判断共同依赖同一个尚需核验的前提。",
        "value_branch": "至少有一部分分歧属于价值排序，不应伪装成纯事实争议。",
        "blind_spot_fill": "另一视角指出了原判断遗漏的维度，但尚未证明其影响大小。",
    }.get(str(relation_type), "一条跨人结构关系已经成立，但它不决定答案方向。")


def _dependency_boundary(relation_type: Any) -> str:
    return {
        "shared_model_prior": "这些数字人共享同一基础模型先验，人数不能当作同等数量的独立证据。",
        "shared_source": "来自同一来源的相似意见最多算一束材料，不能重复计重。",
        "correlated_error": "重复出现的同类错误不能被误写成彼此印证。",
    }.get(str(relation_type), "相关材料存在依赖，不能按独立支持累加。")


def _normalize_adjudicated_contour(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    raw: dict[str, Any],
    generation_audit: dict[str, Any],
) -> dict[str, Any]:
    raw = _repair_adjudicated_raw_wording(raw, detective)
    allowed = _allowed_ids(detective, constraints)
    provenance: list[dict[str, Any]] = []

    direct = _statement(raw.get("direct_answer"), "direct_answer", allowed)
    main = _statement(raw.get("main_contour"), "main_contour", allowed)
    if direct:
        provenance.append(direct[1])
    if main:
        provenance.append(main[1])

    branches, branch_provenance = _statement_list(
        raw.get("key_conditions"), "branch", allowed, limit=5
    )
    stable, stable_provenance = _statement_list(
        raw.get("stable_parts"), "stable", allowed, limit=6
    )
    unknowns, unknown_provenance = _statement_list(
        raw.get("important_unknowns"), "unknown", allowed, limit=6
    )
    boundaries, boundary_provenance = _statement_list(
        raw.get("boundary_conditions"), "boundary", allowed, limit=5
    )
    provenance.extend(branch_provenance)
    provenance.extend(stable_provenance)
    provenance.extend(unknown_provenance)
    provenance.extend(boundary_provenance)
    provenance = _enrich_statement_knowledge_provenance(
        provenance,
        detective,
        constraints,
    )
    sentence_bindings = _sentence_evidence_bindings(
        raw,
        allowed,
        detective,
        constraints,
    )

    direct_text = direct[0] if direct else ""
    main_text = main[0] if main else ""
    expected_sentence_bindings = sum(
        len((raw.get(field) or {}).get("sentence_bindings", []) or [])
        for field in ("direct_answer", "main_contour")
        if isinstance(raw.get(field), dict)
    )
    sentence_binding_coverage = (
        round(len(sentence_bindings) / expected_sentence_bindings, 4)
        if expected_sentence_bindings
        else 0.0
    )
    return {
        "schema_version": TRUTH_CONTOUR_SCHEMA_VERSION,
        "question": compact_text(question),
        "question_id": detective["question_id"],
        "generation_mode": "live_adversarial_inference",
        "direct_answer": direct_text,
        "main_contour": main_text,
        "key_conditions": branches,
        "stable_parts": stable,
        "boundary_conditions": boundaries,
        "important_unknowns": unknowns,
        "confidence_statement": compact_text(raw.get("confidence_statement")),
        "why_this_contour": compact_text(raw.get("why_this_contour")),
        "strongest_counter_contour": compact_text(
            raw.get("strongest_counter_contour")
        ),
        "counter_contour_disposition": compact_text(
            raw.get("counter_contour_disposition")
        ),
        "validation_drops": list(raw.get("validation_drops") or []),
        **(
            {"evidence_binding_contract": "sentence_level.v1"}
            if sentence_bindings
            else {"evidence_binding_contract": "legacy_statement_level"}
        ),
        "sentence_evidence_bindings": sentence_bindings,
        "provenance_index": provenance,
        "classic_knowledge_audit": _contour_knowledge_audit(provenance),
        "contour_evaluation": {
            "uses_person_count_as_truth_weight": False,
            "numeric_bias_inversion_performed": False,
            "main_contour_present": bool(main_text),
            "direct_answer_present": bool(direct_text),
            "provenance_coverage": _provenance_coverage(
                [direct_text, main_text] + branches + stable + boundaries + unknowns,
                provenance,
            ),
            "sentence_binding_coverage": sentence_binding_coverage,
            "sentence_binding_count": len(sentence_bindings),
            "verified_observation_count": sum(
                item.get("claim_role") == "verified_observation"
                for item in sentence_bindings
            ),
            "directional_judgment_count": sum(
                item.get("claim_role") == "directional_judgment"
                for item in sentence_bindings
            ),
            "safety_cap": compact_text(raw.get("safety_cap")) or "none",
            "generation_audit": generation_audit,
        },
    }


def _repair_adjudicated_raw_wording(
    raw: dict[str, Any], detective: dict[str, Any]
) -> dict[str, Any]:
    repaired = dict(raw)
    for field in ("direct_answer", "main_contour"):
        statement = repaired.get(field)
        if not isinstance(statement, dict):
            continue
        repaired[field] = {
            **statement,
            "text": _repair_independent_convergence_wording(
                compact_text(statement.get("text")), detective
            ),
        }
        bindings = statement.get("sentence_bindings") or []
        if isinstance(bindings, list):
            repaired_bindings = []
            for item in bindings:
                if not isinstance(item, dict):
                    continue
                repaired_bindings.append(
                    {
                        **item,
                        "text": _repair_independent_convergence_wording(
                            compact_text(item.get("text")), detective
                        ),
                    }
                )
            repaired[field]["sentence_bindings"] = repaired_bindings
    return repaired


def _sentence_evidence_bindings(
    raw: dict[str, Any],
    allowed: dict[str, set[str]],
    detective: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for field in ("direct_answer", "main_contour"):
        statement = raw.get(field)
        if not isinstance(statement, dict):
            continue
        for index, item in enumerate(
            statement.get("sentence_bindings", []) or [], start=1
        ):
            if not isinstance(item, dict):
                continue
            text = compact_text(item.get("text"))
            refs = _valid_refs(item, allowed)
            if not text or not any(refs.values()):
                continue
            bindings.append(
                {
                    "claim_id": compact_text(item.get("claim_id"))
                    or f"{field}_sentence_{index}",
                    "statement_id": compact_text(item.get("claim_id"))
                    or f"{field}_sentence_{index}",
                    "rendered_in": field,
                    "claim_role": compact_text(item.get("claim_role")),
                    "statement": text,
                    **refs,
                }
            )
    return _enrich_statement_knowledge_provenance(
        bindings,
        detective,
        constraints,
    )


def _repair_independent_convergence_wording(
    text: str, detective: dict[str, Any]
) -> str:
    has_independent_convergence = any(
        item.get("status") == "accepted"
        and item.get("relation_type") == "independent_convergence"
        for item in detective.get("relation_certificates", []) or []
    )
    if not has_independent_convergence:
        return text
    replacements = {
        "缺乏独立会合或决定性证据": "现有独立会合不足以决定方向，也缺乏决定性证据",
        "没有独立会合或决定性证据": "现有独立会合不足以决定方向，也缺乏决定性证据",
        "缺乏任何独立会合": "现有独立会合不足以决定方向",
        "不存在独立会合": "现有独立会合不足以决定方向",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return compact_text(text)


def _conservative_contour(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    *,
    generation_audit: dict[str, Any],
) -> dict[str, Any]:
    accepted = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
    ]
    usable = [
        item
        for item in accepted
        if item.get("relation_type")
        not in {"shared_source", "shared_model_prior", "correlated_error"}
    ]
    refs = _fallback_refs(detective, constraints, accepted)
    if usable:
        direct = (
            "现有关系已经提供了一些彼此约束的局部线索，但尚未经过候选轮廓与反轮廓的对抗裁决，"
            "因此现在不能诚实地给出关于这个问题的实质性主结论。"
        )
        main = (
            "当前能确认的是哪些局部材料可以连接、哪些声音存在同源依赖、哪些分歧必须保留条件；"
            "它们还没有被反演成经得住最强替代解释的真相轮廓。"
        )
    else:
        direct = (
            "目前还不能从这些数字人的数量或相似结论推出这个问题的答案；"
            "可核验的跨人连接不足，给出明确倾向会把群体印象误写成真相。"
        )
        main = (
            "当前轮廓只有一条可靠边界：这些材料展示了观察者怎样判断，"
            "却还没有形成足以支持实质结论的独立会合、条件互补或因果接力。"
        )

    unknown = "尚未完成候选轮廓、最强反轮廓与最终裁决，因而不知道哪一种全局解释最能同时容纳现有材料。"
    provenance = [
        {
            "statement_id": "direct_answer",
            "statement": direct,
            **refs,
        },
        {
            "statement_id": "main_contour",
            "statement": main,
            **refs,
        },
        {
            "statement_id": "unknown_1",
            "statement": unknown,
            **refs,
        },
    ]
    provenance = _enrich_statement_knowledge_provenance(
        provenance,
        detective,
        constraints,
    )
    return {
        "schema_version": TRUTH_CONTOUR_SCHEMA_VERSION,
        "question": compact_text(question),
        "question_id": detective["question_id"],
        "generation_mode": "deterministic_conservative_fallback",
        "direct_answer": direct,
        "main_contour": main,
        "key_conditions": [],
        "stable_parts": [],
        "boundary_conditions": [],
        "important_unknowns": [unknown],
        "confidence_statement": "这里只对“证据不足以推出轮廓”有信心，不对问题本身的方向下注。",
        "why_this_contour": "它拒绝用人数、声量或同源重复代替关系证明。",
        "strongest_counter_contour": "尚未生成。",
        "counter_contour_disposition": "未裁决。",
        "provenance_index": provenance,
        "classic_knowledge_audit": _contour_knowledge_audit(provenance),
        "contour_evaluation": {
            "uses_person_count_as_truth_weight": False,
            "numeric_bias_inversion_performed": False,
            "main_contour_present": True,
            "direct_answer_present": True,
            "provenance_coverage": 1.0,
            "generation_audit": generation_audit,
        },
    }


def _statement(
    value: Any,
    statement_id: str,
    allowed: dict[str, set[str]],
) -> tuple[str, dict[str, Any]] | None:
    if isinstance(value, str):
        return None
    if not isinstance(value, dict):
        return None
    text = compact_text(value.get("text") or value.get("statement"))
    if not text:
        return None
    refs = _valid_refs(value, allowed)
    if not any(refs.values()):
        return None
    return text, {"statement_id": statement_id, "statement": text, **refs}


def _statement_list(
    values: Any,
    prefix: str,
    allowed: dict[str, set[str]],
    *,
    limit: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    provenance: list[dict[str, Any]] = []
    if not isinstance(values, list):
        return texts, provenance
    for index, item in enumerate(values[:limit], start=1):
        parsed = _statement(item, f"{prefix}_{index}", allowed)
        if not parsed:
            continue
        text, trace = parsed
        if prefix == "stable" and not (
            trace.get("relation_ids") or trace.get("constraint_ids")
        ):
            continue
        texts.append(text)
        provenance.append(trace)
    return texts, provenance


def _valid_refs(value: dict[str, Any], allowed: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        "piece_ids": _filter_ids(value.get("piece_ids"), allowed["piece_ids"]),
        "relation_ids": _filter_ids(value.get("relation_ids"), allowed["relation_ids"]),
        "constraint_ids": _filter_ids(
            value.get("constraint_ids"), allowed["constraint_ids"]
        ),
    }


def _allowed_ids(
    detective: dict[str, Any], constraints: dict[str, Any]
) -> dict[str, set[str]]:
    return {
        "piece_ids": {
            item.get("piece_id")
            for item in detective.get("puzzle_pieces", []) or []
            if item.get("piece_id")
        },
        "relation_ids": {
            item.get("relation_id")
            for item in detective.get("relation_certificates", []) or []
            if item.get("relation_id")
        },
        "constraint_ids": {
            item.get("constraint_id")
            for item in constraints.get("constraints", []) or []
            if item.get("constraint_id")
        },
    }


def _filter_ids(values: Any, allowed: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(item) for item in values if str(item) in allowed))


def _fallback_refs(
    detective: dict[str, Any],
    constraints: dict[str, Any],
    accepted: list[dict[str, Any]],
) -> dict[str, list[str]]:
    relation_ids = [item["relation_id"] for item in accepted[:3]]
    constraint_ids = [
        item["constraint_id"]
        for item in constraints.get("constraints", [])[:3]
    ]
    piece_ids = list(
        dict.fromkeys(
            piece_id
            for item in accepted[:3]
            for piece_id in item.get("piece_ids", [])
        )
    )[:6]
    if not piece_ids:
        piece_ids = [
            item["piece_id"]
            for item in detective.get("puzzle_pieces", [])[:3]
            if item.get("piece_id")
        ]
    return {
        "piece_ids": piece_ids,
        "relation_ids": relation_ids,
        "constraint_ids": constraint_ids,
    }


def _constraint_knowledge_audit(
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    traced = [
        item
        for item in constraints
        if (item.get("knowledge_provenance") or {}).get("classic_trace_ids")
    ]
    return {
        "constraint_count": len(constraints),
        "classic_traced_constraint_count": len(traced),
        "classic_trace_ids": unique_strings(
            trace_id
            for item in traced
            for trace_id in (item.get("knowledge_provenance") or {}).get(
                "classic_trace_ids", []
            )
        ),
        "connection_hypothesis_ids": unique_strings(
            connector_id
            for item in traced
            for connector_id in (item.get("knowledge_provenance") or {}).get(
                "connection_hypothesis_ids", []
            )
        ),
        "classic_used_as_world_evidence_count": sum(
            bool(
                (item.get("knowledge_provenance") or {}).get(
                    "classic_as_world_evidence"
                )
            )
            for item in constraints
        ),
        "epistemic_role": "diagnostic_method_not_world_evidence",
    }


def _enrich_statement_knowledge_provenance(
    provenance: list[dict[str, Any]],
    detective: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
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
    output = []
    for item in provenance:
        knowledge_records: list[dict[str, Any]] = []
        knowledge_records.extend(
            (piece_by_id[piece_id].get("knowledge_trace") or {})
            for piece_id in item.get("piece_ids", []) or []
            if piece_id in piece_by_id
        )
        knowledge_records.extend(
            (relation_by_id[relation_id].get("knowledge_provenance") or {})
            for relation_id in item.get("relation_ids", []) or []
            if relation_id in relation_by_id
        )
        knowledge_records.extend(
            (constraint_by_id[constraint_id].get("knowledge_provenance") or {})
            for constraint_id in item.get("constraint_ids", []) or []
            if constraint_id in constraint_by_id
        )
        output.append(
            {
                **item,
                "classic_trace_ids": unique_strings(
                    trace_id
                    for record in knowledge_records
                    for trace_id in record.get("classic_trace_ids", []) or []
                ),
                "inversion_certificate_ids": unique_strings(
                    certificate_id
                    for record in knowledge_records
                    for certificate_id in record.get(
                        "inversion_certificate_ids", []
                    )
                    or []
                ),
                "connection_hypothesis_ids": unique_strings(
                    connector_id
                    for record in knowledge_records
                    for connector_id in record.get(
                        "connection_hypothesis_ids", []
                    )
                    or []
                ),
                "diagnostic_bridge_ids": unique_strings(
                    bridge_id
                    for record in knowledge_records
                    for bridge_id in record.get("diagnostic_bridge_ids", [])
                    or []
                ),
                "inversion_authorizations": unique_strings(
                    authorization
                    for record in knowledge_records
                    for authorization in _authorization_values(record)
                ),
                "epistemic_role": "diagnostic_method_not_world_evidence",
                "classic_as_world_evidence": False,
            }
        )
    return output


def _authorization_values(record: dict[str, Any]) -> list[str]:
    values = list(record.get("inversion_authorizations", []) or [])
    single = compact_text(record.get("inversion_authorization"))
    if single:
        values.append(single)
    return values


def _contour_knowledge_audit(
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "statement_count": len(provenance),
        "classic_traced_statement_count": sum(
            bool(item.get("classic_trace_ids")) for item in provenance
        ),
        "classic_trace_ids": unique_strings(
            trace_id
            for item in provenance
            for trace_id in item.get("classic_trace_ids", []) or []
        ),
        "inversion_certificate_ids": unique_strings(
            certificate_id
            for item in provenance
            for certificate_id in item.get("inversion_certificate_ids", [])
            or []
        ),
        "connection_hypothesis_ids": unique_strings(
            connector_id
            for item in provenance
            for connector_id in item.get("connection_hypothesis_ids", [])
            or []
        ),
        "classic_used_as_world_evidence_count": sum(
            bool(item.get("classic_as_world_evidence")) for item in provenance
        ),
        "epistemic_role": "diagnostic_method_not_world_evidence",
    }


def _constraint_strength(relation: dict[str, Any], constraint_type: str) -> str:
    if constraint_type in {
        "dependency_penalty",
        "correlated_error_warning",
        "conflict_boundary",
        "definition_branch",
        "value_branch",
        "non_comparable",
    }:
        return "hard_guardrail"
    certainty = compact_text(
        (relation.get("confidence_profile") or {}).get("diagnostic_certainty")
    )
    return "provisional" if certainty not in {"high", "very_high"} else "strong_provisional"


def _constraint_caution(constraint_type: str) -> str:
    if constraint_type == "directional_bound":
        return "只能形成方向边界，禁止计算偏差大小、中心值或概率。"
    if constraint_type == "unknown_space":
        return "未知不是反向证据，必须由新材料填补。"
    if constraint_type == "support_candidate":
        return "独立性是当前账本下的暂定判断，不等于绝对独立。"
    return "只约束轮廓写法，不直接宣告事实为真。"


def _provenance_coverage(
    statements: list[str], provenance: list[dict[str, Any]]
) -> float:
    substantive = [compact_text(item) for item in statements if compact_text(item)]
    if not substantive:
        return 0.0
    traced = {compact_text(item.get("statement")) for item in provenance}
    return round(sum(item in traced for item in substantive) / len(substantive), 4)
