from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ..detective.schemas import compact_text
from ..json_utils import parse_json_content
from ..providers import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekTimeoutError,
    retry_delay_seconds,
    synthesis_request_timeout_seconds,
)
from .epistemic import has_structural_uncertainty
from .solver import build_truth_contour


CONTOUR_PIPELINE_VERSION = "bme.contour-live.v7"
CANDIDATE_PROMPT_REVISION = "bme.contour.candidate.v4"
COUNTER_PROMPT_REVISION = "bme.contour.counter.v3"
FINAL_PROMPT_REVISION = "bme.contour.final.v4"
COUNTER_PRESSURE_PROMPT_REVISION = "bme.contour.counter-pressure.v1"

COUNTER_PRESSURE_SPECS = (
    (
        "evidence_lineage",
        "证据与来源压力",
        "检查独立性是否被夸大、同源材料是否被重复计权、候选是否把局部保留项写成事实。",
    ),
    (
        "causal_conditions",
        "因果与条件压力",
        "检查因果方向、定义、范围、时间与成立条件，构造能解释同一批材料的替代机制。",
    ),
    (
        "shadow_unknowns",
        "阴影与未知压力",
        "检查共同盲区、相关错误、反演越界和被候选压扁的重要未知。",
    ),
)


CONTOUR_SYSTEM = """你是盲人摸象模型的轮廓层。侦探层已经把不同数字人的“看对、看错、没看见”加工成经过对抗裁决的关系证书；你要根据这些关系反演一个真相轮廓。

真相轮廓不是多数意见、折中意见或标准答案。它是当前材料下，能够同时解释最多可靠局部、分歧结构和认知阴影，同时违背最少硬约束的整体描述。

硬规则：
1. 第一眼必须直接回答用户原问题，然后给出唯一主轮廓。不能用方法论说明代替答案。
2. 人数、立场票数和措辞相似度不是真相权重。shared_source、shared_model_prior 和 correlated_error 只会降低独立性。
3. 只有 accepted 关系及其约束可支撑主张；未决关系只能进入未知或条件分支。
4. preserved/retained 局部仍是候选材料，不因法医层保留就自动为真。
5. 必须保留会真正改写主轮廓的关键条件分支，但不能堆成多份互不负责的答案。
6. 共同沉默是未知空间，不是反向事实。未校准的相反偏差不能计算中点、概率或数值边界。
7. 每个实质陈述必须引用输入中真实存在的 piece_id、relation_id 或 constraint_id；不得引入外部事实、数字、人物或来源。
8. 证据不足时要明确说到哪一步为止，但仍应尽量回答问题，而不是只讲模型局限。
9. 关系类型决定你能推出什么：definition_branch 只能证明答案受定义影响；correlated_error 只能削弱一类推理；shared_model_prior 只能降低独立性。它们都不能单独证明“会”“不会”或概率高低。
10. 正文只写给普通用户看的大白话，不得出现 piece_、relation_、constraint_、P02、p02 或长串 person_id 等内部编号；需要指人时使用 CASE_JSON 提供的 person_name。编号只能放在引用字段。
11. 直接答案和主轮廓必须说出本题材料里的具体对象、指标、机制或时间窗口。只写“定义、范围、条件、价值、分歧、共同先验”等抽象方法词，视为没有回答原题。
12. direct_answer 与 main_contour 的每个句子只能标一个 claim_role：verified_observation 需要已核验现实材料；directional_judgment 需要获准的独立方向支持；conditional_inference 必须明确写出条件；structural_relation 只能说明关系结构；definition_boundary、important_unknown、value_condition 分别只承担定义边界、未知或价值条件。不得用结构关系伪装现实事实。
只输出 JSON，不要输出 Markdown。"""


COUNTER_SYSTEM = """你是轮廓层的反方侦探。你的任务不是唱反调，而是用同一批已裁决材料构造最强替代轮廓。

检查主候选是否偷用了人数、把同源重复当独立证据、把局部保留项当事实、掩盖定义或价值分支、把未知写成事实、忽略能解释同样材料的另一套因果结构。替代轮廓必须引用已有 ID，不得增加外部事实。若实在无法形成更强替代，也要明确指出候选最脆弱的具体连接。只输出 JSON。"""


FINAL_SYSTEM = """你是轮廓层的最终裁决员。你将看到主候选、最强替代轮廓、关系证书和约束。

你必须逐项比较解释覆盖、来源独立性、条件兼容、对阴影的解释力、反证承受力和未知诚实度。最终只能交付一个主轮廓，并保留少量真正会改写它的条件分支。不得按人数、语气或候选写得是否自信来裁决。每个实质陈述都要引用已有 ID。只输出 JSON。"""


CLASSIC_CONTOUR_RULES = """
经典知识轨迹只说明法医诊断和反演材料是怎样产生的，不是关于用户问题的外部事实。
硬规则：
1. 经典著作、透镜、微透镜和机制名称不得增加事实权重，也不得替代搜索证据。
2. classic_as_world_evidence 必须保持 false。相同经典依据不构成独立会合，反而可能意味着诊断方法相关。
3. blocked 反演只能进入未知；restricted 只能形成边界或条件；provisional 仍必须依赖 accepted 关系和约束，不能单独支撑方向性答案。
4. 最终陈述引用关系或约束时，系统会自动继承其经典轨迹；你不得凭经典常识补写输入中不存在的事实。
"""


PROVISIONAL_DRAFT_SYSTEM = """你是轮廓层的推测草稿员。侦探正在并行裁决关系候选；你看到的 candidate_id 只是待审连接，不是已经成立的事实。

你的任务是提前构造一幅可被最终裁决推翻的轮廓假说，或一幅与之竞争的替代假说。草稿只能帮助最终裁决发现解释路线，不能获得事实权重。不得把候选数量当证据，不得引入输入外的事实、数字、人物或来源。每个草稿必须引用真实 candidate_id 和其完整端点 piece_id。只输出 JSON。"""


PROVISIONAL_DRAFT_PIPELINE_VERSION = "bme.provisional-contour-drafts.v2"
PROVISIONAL_DRAFT_PROMPT_REVISION = "bme.contour.provisional-draft.v2"
SPECULATIVE_FINAL_PROMPT_REVISION = "bme.contour.speculative-final.v6"


def run_provisional_contour_drafts(
    question: str,
    detective_case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int = 4096,
    retries: int = 2,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    cached_record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draft competing hypotheses while material relations are adjudicated."""

    started = time.perf_counter()
    factory = client_factory or (
        lambda: DeepSeekClient(
            model=model,
            timeout_seconds=synthesis_request_timeout_seconds(),
        )
    )
    case = _build_provisional_draft_case(question, detective_case, candidates)
    input_fingerprint = _contour_stage_fingerprint(
        PROVISIONAL_DRAFT_PROMPT_REVISION, case
    )
    cached_bundle = (
        (cached_record or {}).get("draft_bundle") or {}
        if (cached_record or {}).get("pipeline_version")
        == PROVISIONAL_DRAFT_PIPELINE_VERSION
        and (cached_record or {}).get("input_fingerprint")
        == input_fingerprint
        else {}
    )
    validated_cached, cached_errors = _validate_provisional_draft_bundle(
        cached_bundle, case
    )
    if validated_cached and not cached_errors:
        record = {
            "pipeline_version": PROVISIONAL_DRAFT_PIPELINE_VERSION,
            "status": "reused",
            "model": model,
            "input_fingerprint": input_fingerprint,
            "draft_bundle": validated_cached,
            "audit": {
                "status": "reused",
                "duration_seconds": 0.0,
                "stages": [],
                "usage": _empty_usage(),
                "provisional_only_not_world_evidence": True,
            },
            "usage": _empty_usage(),
        }
        return validated_cached, record
    cached_partial, cached_missing, _cached_partial_errors = (
        _salvage_provisional_draft_bundle(cached_bundle, case)
    )
    if cached_partial:
        record = {
            "pipeline_version": PROVISIONAL_DRAFT_PIPELINE_VERSION,
            "status": "reused_partial_acceleration",
            "model": model,
            "input_fingerprint": input_fingerprint,
            "draft_bundle": cached_partial,
            "audit": {
                "status": "reused_partial_acceleration",
                "duration_seconds": 0.0,
                "stages": [],
                "usage": _empty_usage(),
                "missing_draft_roles": cached_missing,
                "provisional_only_not_world_evidence": True,
                "final_still_receives_full_adjudicated_case": True,
            },
            "usage": _empty_usage(),
        }
        return cached_partial, record

    specs = [
        (
            "main_hypothesis",
            "主候选草稿",
            "构造当前待审连接能够支持的最强主轮廓假说，同时主动写出它最脆弱的前提。",
            case,
        )
    ]
    specs.extend(
        (
            focus_id,
            label,
            instruction,
            _provisional_focus_case(case, focus_id),
        )
        for focus_id, label, instruction in COUNTER_PRESSURE_SPECS
    )
    payloads: list[dict[str, Any] | None] = [None] * len(specs)
    records: list[dict[str, Any] | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = {}
        for index, (draft_id, label, instruction, material) in enumerate(specs):
            stage_fingerprint = _contour_stage_fingerprint(
                PROVISIONAL_DRAFT_PROMPT_REVISION,
                {
                    "draft_id": draft_id,
                    "instruction": instruction,
                    "case": material,
                },
            )
            future = executor.submit(
                _call_validated_stage,
                factory,
                _provisional_draft_messages(
                    material,
                    draft_id=draft_id,
                    label=label,
                    instruction=instruction,
                ),
                stage=f"provisional_draft_{draft_id}",
                validator=lambda value, allowed=material: _validate_provisional_draft(
                    value, allowed
                ),
                max_tokens=max(3072, min(max_tokens, 4096)),
                retries=retries,
                thinking_mode="disabled",
                input_fingerprint=stage_fingerprint,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            payloads[index], records[index] = future.result()

    complete_records = [item for item in records if item is not None]
    usage = _empty_usage()
    errors: list[str] = []
    for record in complete_records:
        usage = _merge_usage(usage, record.get("usage") or {})
        errors.extend(record.get("errors") or [])
    bundle = {
        "mode": "speculative_parallel_v1",
        "epistemic_status": "hypotheses_only_pending_relation_adjudication",
        "main_draft": payloads[0] or {},
        "rival_drafts": [
            {
                "focus_id": specs[index][0],
                "label": specs[index][1],
                "draft": payloads[index] or {},
            }
            for index in range(1, len(specs))
        ],
    }
    validated, bundle_errors = _validate_provisional_draft_bundle(bundle, case)
    errors.extend(bundle_errors)
    partial, missing_roles, partial_errors = (
        _salvage_provisional_draft_bundle(bundle, case)
    )
    succeeded = (
        len(complete_records) == len(specs)
        and all(item.get("status") == "succeeded" for item in complete_records)
        and bool(validated)
    )
    partial_usable = not succeeded and bool(partial)
    record_status = (
        "succeeded"
        if succeeded
        else "succeeded_partial_acceleration"
        if partial_usable
        else "failed"
    )
    delivered_bundle = validated if succeeded else partial if partial_usable else {}
    record = {
        "pipeline_version": PROVISIONAL_DRAFT_PIPELINE_VERSION,
        "status": record_status,
        "model": model,
        "input_fingerprint": input_fingerprint,
        "case": case,
        "draft_bundle": delivered_bundle,
        "audit": {
            "status": record_status,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "stage_count": len(specs),
            "stages": complete_records,
            "errors": errors,
            "partial_validation_errors": partial_errors,
            "missing_draft_roles": missing_roles,
            "usable_draft_count": (
                int(bool(delivered_bundle.get("main_draft")))
                + len(delivered_bundle.get("rival_drafts") or [])
            ),
            "usage": usage,
            "provisional_only_not_world_evidence": True,
            "full_candidate_coverage": "passed",
            "final_still_receives_full_adjudicated_case": True,
        },
        "usage": usage,
    }
    return delivered_bundle, record


def _build_provisional_draft_case(
    question: str,
    detective_case: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    endpoint_ids = {
        piece_id
        for candidate in candidates
        for piece_id in candidate.get("piece_ids", []) or []
    }
    material_candidates = [
        {
            key: value
            for key, value in candidate.items()
            if key
            in {
                "candidate_id",
                "relation_type",
                "piece_ids",
                "plain_language_explanation",
                "shared_coordinate",
                "source_independence",
                "common_error_risk",
                "competing_explanation",
                "falsification_test",
            }
        }
        for candidate in candidates
    ]
    return {
        "schema": "bme.provisional-contour-case.v1",
        "question": compact_text(question),
        "question_id": detective_case.get("question_id"),
        "epistemic_status": (
            "All relations are candidates pending adversarial adjudication. "
            "They are hypothesis prompts, never established facts."
        ),
        "pieces": [
            item
            for item in detective_case.get("piece_index", []) or []
            if item.get("piece_id") in endpoint_ids
        ],
        "candidates": material_candidates,
        "source_dependency_clusters": detective_case.get(
            "source_dependency_clusters"
        )
        or [],
        "source_lineage_rules": detective_case.get("source_lineage_rules")
        or {},
        "mechanism_index": detective_case.get("mechanism_index") or {},
        "allowed_ids": {
            "candidate_ids": sorted(
                item.get("candidate_id")
                for item in material_candidates
                if item.get("candidate_id")
            ),
            "piece_ids": sorted(endpoint_ids),
        },
    }


def _provisional_focus_case(
    case: dict[str, Any], focus_id: str
) -> dict[str, Any]:
    relation_focus = {
        "independent_convergence": "evidence_lineage",
        "shared_source": "evidence_lineage",
        "shared_model_prior": "evidence_lineage",
        "direct_conflict": "causal_conditions",
        "apparent_conflict": "causal_conditions",
        "conditional_complement": "causal_conditions",
        "scope_refinement": "causal_conditions",
        "causal_relay": "causal_conditions",
        "definition_branch": "causal_conditions",
        "value_branch": "causal_conditions",
    }
    selected = [
        item
        for item in case.get("candidates", []) or []
        if relation_focus.get(
            compact_text(item.get("relation_type")), "shadow_unknowns"
        )
        == focus_id
    ]
    if not selected:
        selected = list(case.get("candidates", []) or [])[:4]
    endpoint_ids = {
        piece_id
        for item in selected
        for piece_id in item.get("piece_ids", []) or []
    }
    return {
        **case,
        "pieces": [
            item
            for item in case.get("pieces", []) or []
            if item.get("piece_id") in endpoint_ids
        ],
        "candidates": selected,
        "allowed_ids": {
            "candidate_ids": sorted(
                item.get("candidate_id")
                for item in selected
                if item.get("candidate_id")
            ),
            "piece_ids": sorted(endpoint_ids),
        },
        "focus_id": focus_id,
        "collective_parent_case_is_lossless": True,
    }


def _provisional_draft_messages(
    case: dict[str, Any],
    *,
    draft_id: str,
    label: str,
    instruction: str,
) -> list[dict[str, str]]:
    schema = {
        "draft_id": draft_id,
        "direct_answer_hypothesis": "对原问题的暂定回答",
        "contour_hypothesis": "整体解释假说",
        "candidate_ids": ["existing candidate_id"],
        "piece_ids": ["complete endpoints of cited candidates"],
        "critical_assumptions": ["这幅假说依赖什么"],
        "failure_conditions": ["什么会使它失败"],
    }
    return [
        {"role": "system", "content": PROVISIONAL_DRAFT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"草稿职责：{label}。{instruction}\n"
                "最多引用 6 条真正关键的关系候选；引用 candidate_id 时必须列全它的 piece_ids。"
                "最多写 4 个关键假设和 4 个失败条件。不得使用 relation_id 或 constraint_id，"
                "因为关系裁决尚未完成。\nCASE_JSON:\n"
                + _prompt_json(case)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]


def _validate_provisional_draft(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["provisional draft is not an object"]
    errors: list[str] = []
    direct = compact_text(payload.get("direct_answer_hypothesis"))
    contour = compact_text(payload.get("contour_hypothesis"))
    if not direct or not contour:
        errors.append("provisional draft needs direct and contour hypotheses")
    allowed_candidates = set(
        (case.get("allowed_ids") or {}).get("candidate_ids") or []
    )
    allowed_pieces = set(
        (case.get("allowed_ids") or {}).get("piece_ids") or []
    )
    candidate_ids = list(
        dict.fromkeys(
            compact_text(item)
            for item in payload.get("candidate_ids", []) or []
        )
    )[:6]
    piece_ids = list(
        dict.fromkeys(
            compact_text(item)
            for item in payload.get("piece_ids", []) or []
        )
    )
    unknown_candidates = set(candidate_ids) - allowed_candidates
    unknown_pieces = set(piece_ids) - allowed_pieces
    if unknown_candidates:
        errors.append("draft cites unknown candidate_id")
    if unknown_pieces:
        errors.append("draft cites unknown piece_id")
    if not candidate_ids:
        errors.append("draft cites no relation candidate")
    candidate_map = {
        item.get("candidate_id"): item
        for item in case.get("candidates", []) or []
    }
    cited_pieces = set(piece_ids)
    for candidate_id in candidate_ids:
        endpoints = set(
            (candidate_map.get(candidate_id) or {}).get("piece_ids") or []
        )
        if not endpoints.issubset(cited_pieces):
            errors.append(
                f"draft omits endpoints for candidate {candidate_id}"
            )
    output = {
        "draft_id": compact_text(payload.get("draft_id")),
        "direct_answer_hypothesis": direct,
        "contour_hypothesis": contour,
        "candidate_ids": candidate_ids,
        "piece_ids": piece_ids,
        "critical_assumptions": [
            compact_text(item)
            for item in (payload.get("critical_assumptions") or [])[:4]
            if compact_text(item)
        ],
        "failure_conditions": [
            compact_text(item)
            for item in (payload.get("failure_conditions") or [])[:4]
            if compact_text(item)
        ],
    }
    return (output if not errors else {}), errors[:12]


def _validate_provisional_draft_bundle(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["provisional draft bundle is not an object"]
    main, errors = _validate_provisional_draft(
        payload.get("main_draft"), case
    )
    expected_focus_ids = [item[0] for item in COUNTER_PRESSURE_SPECS]
    rivals = payload.get("rival_drafts")
    if not isinstance(rivals, list):
        rivals = []
        errors.append("provisional bundle needs rival_drafts")
    actual_focus_ids = [compact_text(item.get("focus_id")) for item in rivals]
    if actual_focus_ids != expected_focus_ids:
        errors.append("provisional rival coverage mismatch")
    sanitized_rivals = []
    for item in rivals:
        focus_case = _provisional_focus_case(
            case, compact_text(item.get("focus_id"))
        )
        draft, draft_errors = _validate_provisional_draft(
            item.get("draft"), focus_case
        )
        errors.extend(draft_errors)
        if draft:
            sanitized_rivals.append(
                {
                    "focus_id": compact_text(item.get("focus_id")),
                    "label": compact_text(item.get("label")),
                    "draft": draft,
                }
            )
    output = {
        "mode": "speculative_parallel_v1",
        "epistemic_status": "hypotheses_only_pending_relation_adjudication",
        "main_draft": main,
        "rival_drafts": sanitized_rivals,
    }
    return (output if not errors else {}), errors[:20]


def _salvage_provisional_draft_bundle(
    payload: Any,
    case: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Keep valid non-authoritative drafts without treating gaps as evidence."""

    if not isinstance(payload, dict):
        return {}, ["main_hypothesis", *[item[0] for item in COUNTER_PRESSURE_SPECS]], [
            "provisional draft bundle is not an object"
        ]
    errors: list[str] = []
    missing_roles: list[str] = []
    main, main_errors = _validate_provisional_draft(
        payload.get("main_draft"), case
    )
    if not main:
        missing_roles.append("main_hypothesis")
        errors.extend(main_errors)

    raw_rivals = payload.get("rival_drafts")
    rivals = raw_rivals if isinstance(raw_rivals, list) else []
    rival_by_focus = {
        compact_text(item.get("focus_id")): item
        for item in rivals
        if isinstance(item, dict) and compact_text(item.get("focus_id"))
    }
    sanitized_rivals = []
    for focus_id, label, _instruction in COUNTER_PRESSURE_SPECS:
        raw = rival_by_focus.get(focus_id) or {}
        draft, draft_errors = _validate_provisional_draft(
            raw.get("draft"),
            _provisional_focus_case(case, focus_id),
        )
        if not draft:
            missing_roles.append(focus_id)
            errors.extend(draft_errors)
            continue
        sanitized_rivals.append(
            {
                "focus_id": focus_id,
                "label": compact_text(raw.get("label")) or label,
                "draft": draft,
            }
        )

    usable_count = int(bool(main)) + len(sanitized_rivals)
    if usable_count < 2:
        return {}, missing_roles, errors[:20]
    return (
        {
            "mode": "speculative_parallel_v1",
            "epistemic_status": (
                "hypotheses_only_pending_relation_adjudication"
            ),
            "main_draft": main,
            "rival_drafts": sanitized_rivals,
            "missing_draft_roles": missing_roles,
            "missing_role_instruction": (
                "Use the full adjudicated case to inspect these missing roles; "
                "never infer that a missing acceleration draft supports either side."
            ),
        },
        missing_roles,
        errors[:20],
    )


def run_live_contour(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 8192,
    retries: int = 2,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    cached_record: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    material_mode: str = "lossless_packet_v2",
    counter_mode: str = "single_counter",
    contour_mode: str = "post_adjudication",
    provisional_drafts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate, attack, and adjudicate one truth contour."""

    if material_mode not in {"legacy_full_case", "lossless_packet_v2"}:
        raise ValueError(
            "material_mode must be 'legacy_full_case' or 'lossless_packet_v2'"
        )
    if counter_mode not in {"single_counter", "parallel_pressure_v1"}:
        raise ValueError(
            "counter_mode must be 'single_counter' or 'parallel_pressure_v1'"
        )
    if contour_mode not in {
        "post_adjudication",
        "speculative_parallel_v1",
    }:
        raise ValueError(
            "contour_mode must be 'post_adjudication' or "
            "'speculative_parallel_v1'"
        )
    started = time.perf_counter()
    factory = client_factory or (
        lambda: DeepSeekClient(
            model=model,
            timeout_seconds=synthesis_request_timeout_seconds(),
        )
    )
    case = build_contour_case(question, detective, constraints)
    prompt_packet = build_contour_prompt_packet(case)
    packet_errors = validate_contour_prompt_packet(prompt_packet, case)
    if packet_errors:
        raise RuntimeError(
            "Contour prompt packet failed coverage validation: "
            + "; ".join(packet_errors)
        )
    decision_packet = build_contour_decision_packet(prompt_packet, case)
    decision_packet_errors = validate_contour_decision_packet(
        decision_packet, prompt_packet, case
    )
    if decision_packet_errors:
        raise RuntimeError(
            "Contour decision packet failed coverage validation: "
            + "; ".join(decision_packet_errors)
        )
    message_material = (
        case if material_mode == "legacy_full_case" else prompt_packet
    )
    packet_stats = {
        "schema": prompt_packet.get("schema"),
        "material_mode": material_mode,
        "full_case_characters": len(_prompt_json(case)),
        "prompt_packet_characters": len(
            _prompt_json(prompt_packet)
        ),
        "decision_packet_characters": len(
            _prompt_json(decision_packet)
        ),
        "character_reduction_ratio": round(
            1
            - len(_prompt_json(prompt_packet))
            / max(1, len(_prompt_json(case))),
            4,
        ),
        "sent_material_characters": len(
            _prompt_json(message_material)
        ),
        "coverage_validation": "passed",
        "decision_packet_coverage_validation": "passed",
        "task_relevant_material_removed": False,
    }
    usage = _empty_usage()
    stages = []

    if not case["constructive_relation_ids"]:
        audit = {
            "mode": "candidate_counter_adjudication",
            "counter_mode": counter_mode,
            "status": "fallback",
            "model": model,
            "counter_mode": counter_mode,
            "stages": [
                {
                    "stage": "constructive_relation_gate",
                    "status": "blocked",
                    "attempts": 0,
                    "errors": [
                        "只有来源依赖、相关错误或未知边界，尚无可支撑实质轮廓的关系。"
                    ],
                    "usage": _empty_usage(),
                }
            ],
            "usage": usage,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "candidate_present": False,
            "counter_contour_present": False,
            "final_adjudication_present": False,
            "prompt_packet": packet_stats,
        }
        contour = build_truth_contour(
            question,
            detective,
            constraints,
            generation_audit=audit,
        )
        return contour, {
            "pipeline_version": CONTOUR_PIPELINE_VERSION,
            "status": "fallback",
            "model": model,
            "case": case,
            "candidate_contour": {},
            "counter_contour": {},
            "final_adjudication": {},
            "audit": audit,
            "usage": usage,
        }

    if contour_mode == "speculative_parallel_v1" and provisional_drafts:
        return _run_speculative_final_contour(
            question,
            detective,
            constraints,
            case=case,
            message_material=(
                case
                if material_mode == "legacy_full_case"
                else decision_packet
            ),
            packet_stats=packet_stats,
            provisional_drafts=provisional_drafts,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            client_factory=factory,
            cached_record=cached_record or {},
            checkpoint_callback=checkpoint_callback,
            started=started,
        )

    candidate_input_fingerprint = _contour_stage_fingerprint(
        CANDIDATE_PROMPT_REVISION,
        {"case": case},
    )
    cached_candidate = (
        (cached_record or {}).get("candidate_contour") or {}
        if _cached_contour_stage_matches(
            cached_record or {},
            "candidate_contour",
            candidate_input_fingerprint,
        )
        else {}
    )
    candidate, candidate_errors = (
        _validate_candidate(cached_candidate, case)
        if cached_candidate
        else ({}, ["missing"])
    )
    if candidate and not candidate_errors:
        record = _reused_stage(
            "candidate_contour", candidate_input_fingerprint
        )
    else:
        candidate, record = _call_validated_stage(
            factory,
            _candidate_messages(case, message_material),
            stage="candidate_contour",
            validator=lambda payload: _validate_candidate(payload, case),
            max_tokens=max_tokens,
            retries=retries,
            thinking_mode="disabled",
            input_fingerprint=candidate_input_fingerprint,
        )
    usage = _merge_usage(usage, record.get("usage") or {})
    stages.append(record)
    _emit_contour_checkpoint(
        checkpoint_callback,
        model=model,
        case=case,
        candidate=candidate,
        counter={},
        final_payload={},
        stages=stages,
        usage=usage,
    )

    counter: dict[str, Any] = {}
    if candidate:
        reasoning_candidate = _contour_reasoning_payload(candidate)
        if counter_mode == "parallel_pressure_v1":
            counter_stage = "parallel_counter_pressure"
            counter_input_fingerprint = _contour_stage_fingerprint(
                COUNTER_PRESSURE_PROMPT_REVISION,
                {"case": case, "candidate": reasoning_candidate},
            )
            cached_counter = (
                (cached_record or {}).get("counter_contour") or {}
                if _cached_contour_stage_matches(
                    cached_record or {},
                    counter_stage,
                    counter_input_fingerprint,
                )
                else {}
            )
            counter, counter_errors = (
                _validate_counter_pressure_bundle(cached_counter, case)
                if cached_counter
                else ({}, ["missing"])
            )
            if counter and not counter_errors:
                record = _reused_stage(
                    counter_stage, counter_input_fingerprint
                )
            else:
                counter, record = _run_parallel_counter_pressure(
                    factory,
                    case,
                    reasoning_candidate,
                    prompt_packet,
                    max_tokens=max_tokens,
                    retries=retries,
                    input_fingerprint=counter_input_fingerprint,
                )
        else:
            counter_stage = "strongest_counter_contour"
            counter_input_fingerprint = _contour_stage_fingerprint(
                COUNTER_PROMPT_REVISION,
                {"case": case, "candidate": reasoning_candidate},
            )
            cached_counter = (
                (cached_record or {}).get("counter_contour") or {}
                if _cached_contour_stage_matches(
                    cached_record or {},
                    counter_stage,
                    counter_input_fingerprint,
                )
                else {}
            )
            counter, counter_errors = (
                _validate_counter(cached_counter, case)
                if cached_counter
                else ({}, ["missing"])
            )
            if counter and not counter_errors:
                record = _reused_stage(
                    counter_stage, counter_input_fingerprint
                )
            else:
                counter, record = _call_validated_stage(
                    factory,
                    _counter_messages(
                        case, reasoning_candidate, message_material
                    ),
                    stage=counter_stage,
                    validator=lambda payload: _validate_counter(
                        payload, case
                    ),
                    max_tokens=max(max_tokens, 8192),
                    retries=retries,
                    thinking_mode="enabled",
                    input_fingerprint=counter_input_fingerprint,
                )
        usage = _merge_usage(usage, record.get("usage") or {})
        stages.append(record)
    else:
        stages.append(
            _skipped_stage(
                "parallel_counter_pressure"
                if counter_mode == "parallel_pressure_v1"
                else "strongest_counter_contour"
            )
        )
    _emit_contour_checkpoint(
        checkpoint_callback,
        model=model,
        case=case,
        candidate=candidate,
        counter=counter,
        final_payload={},
        stages=stages,
        usage=usage,
    )

    final_payload: dict[str, Any] = {}
    if candidate and counter:
        reasoning_candidate = _contour_reasoning_payload(candidate)
        reasoning_counter = _contour_reasoning_payload(counter)
        final_input_fingerprint = _contour_stage_fingerprint(
            FINAL_PROMPT_REVISION,
            {
                "case": case,
                "candidate": reasoning_candidate,
                "counter": reasoning_counter,
            },
        )
        cached_final = (
            (cached_record or {}).get("final_adjudication") or {}
            if _cached_contour_stage_matches(
                cached_record or {},
                "final_adjudication",
                final_input_fingerprint,
            )
            else {}
        )
        final_payload, final_errors = (
            _validate_final(cached_final, case)
            if cached_final
            else ({}, ["missing"])
        )
        revalidated_record = None
        if not final_payload:
            final_payload, revalidated_record = _revalidate_cached_raw_stage(
                cached_record or {},
                "final_adjudication",
                final_input_fingerprint,
                lambda payload: _validate_final(payload, case),
            )
        if final_payload and not final_errors:
            record = revalidated_record or _reused_stage(
                "final_adjudication", final_input_fingerprint
            )
        elif final_payload and revalidated_record:
            record = revalidated_record
        else:
            final_payload, record = _call_validated_stage(
                factory,
                _final_messages(
                    case,
                    reasoning_candidate,
                    reasoning_counter,
                    message_material,
                ),
                stage="final_adjudication",
                validator=lambda payload: _validate_final(payload, case),
                max_tokens=max(max_tokens, 8192),
                retries=retries,
                thinking_mode="enabled",
                input_fingerprint=final_input_fingerprint,
            )
        usage = _merge_usage(usage, record.get("usage") or {})
        stages.append(record)
    else:
        stages.append(_skipped_stage("final_adjudication"))

    success = bool(final_payload)
    audit = {
        "mode": "candidate_counter_adjudication",
        "counter_mode": counter_mode,
        "status": "succeeded" if success else "fallback",
        "model": model,
        "stages": stages,
        "usage": usage,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "candidate_present": bool(candidate),
        "counter_contour_present": bool(counter),
        "final_adjudication_present": success,
        "prompt_packet": packet_stats,
    }
    contour = build_truth_contour(
        question,
        detective,
        constraints,
        adjudicated_payload=final_payload if success else None,
        generation_audit=audit,
    )
    record = {
        "pipeline_version": CONTOUR_PIPELINE_VERSION,
        "status": audit["status"],
        "model": model,
        "counter_mode": counter_mode,
        "case": case,
        "candidate_contour": candidate,
        "counter_contour": counter,
        "final_adjudication": final_payload,
        "audit": audit,
        "usage": usage,
    }
    if checkpoint_callback:
        checkpoint_callback(record)
    return contour, record


def _run_speculative_final_contour(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
    *,
    case: dict[str, Any],
    message_material: dict[str, Any],
    packet_stats: dict[str, Any],
    provisional_drafts: dict[str, Any],
    model: str,
    max_tokens: int,
    retries: int,
    client_factory: Callable[[], DeepSeekClient],
    cached_record: dict[str, Any],
    checkpoint_callback: Callable[[dict[str, Any]], None] | None,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolution_map = _candidate_resolution_map(detective)
    resolution_errors = _validate_candidate_resolution_map(
        resolution_map, detective
    )
    if resolution_errors:
        raise RuntimeError(
            "Candidate resolution map failed coverage validation: "
            + "; ".join(resolution_errors)
        )
    reasoning_drafts, draft_resolution_audit = (
        _resolved_provisional_reasoning_payload(
            provisional_drafts, resolution_map
        )
    )
    final_input_fingerprint = _contour_stage_fingerprint(
        SPECULATIVE_FINAL_PROMPT_REVISION,
        {
            "case": case,
            "provisional_drafts": reasoning_drafts,
            "candidate_resolution_map": resolution_map,
        },
    )
    cached_final = (
        cached_record.get("final_adjudication") or {}
        if _cached_contour_stage_matches(
            cached_record,
            "final_from_provisional_drafts",
            final_input_fingerprint,
        )
        else {}
    )
    final_payload, final_errors = (
        _validate_compact_final(cached_final, case)
        if cached_final
        else ({}, ["missing"])
    )
    revalidated_record = None
    if not final_payload:
        final_payload, revalidated_record = _revalidate_cached_raw_stage(
            cached_record,
            "final_from_provisional_drafts",
            final_input_fingerprint,
            lambda payload: _validate_compact_final(payload, case),
        )
    if final_payload and not final_errors:
        stage_record = revalidated_record or _reused_stage(
            "final_from_provisional_drafts", final_input_fingerprint
        )
    elif final_payload and revalidated_record:
        stage_record = revalidated_record
    else:
        final_payload, stage_record = _call_validated_stage(
            client_factory,
            _speculative_final_messages(
                case,
                reasoning_drafts,
                resolution_map,
                message_material,
            ),
            stage="final_from_provisional_drafts",
            validator=lambda payload: _validate_compact_final(payload, case),
            max_tokens=max(max_tokens, 12288),
            retries=retries,
            thinking_mode="disabled",
            input_fingerprint=final_input_fingerprint,
            repair_messages_factory=lambda payload, errors: (
                _compact_final_repair_messages(case, payload, errors)
            ),
            repair_max_tokens=min(max(max_tokens, 8192), 12288),
            repair_thinking_mode="disabled",
        )
    usage = _merge_usage(_empty_usage(), stage_record.get("usage") or {})
    success = bool(final_payload)
    audit = {
        "mode": "speculative_drafts_final_adjudication",
        "counter_mode": "speculative_parallel_v1",
        "status": "succeeded" if success else "fallback",
        "model": model,
        "stages": [stage_record],
        "usage": usage,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "candidate_present": bool(
            provisional_drafts.get("main_draft")
        ),
        "counter_contour_present": bool(
            provisional_drafts.get("rival_drafts")
        ),
        "final_adjudication_present": success,
        "final_inference_policy": (
            "validated_fast_then_compact_fast_repair_v2"
        ),
        "prompt_packet": packet_stats,
        "speculative_execution": {
            "drafts_are_world_evidence": False,
            "draft_candidate_ids_are_final_provenance": False,
            "final_case_uses_only_adjudicated_relations": True,
            "candidate_resolution_count": len(resolution_map),
            "candidate_resolution_coverage": "passed",
            "draft_resolution_coverage": "passed",
            "draft_resolution_audit": draft_resolution_audit,
        },
    }
    contour = build_truth_contour(
        question,
        detective,
        constraints,
        adjudicated_payload=final_payload if success else None,
        generation_audit=audit,
    )
    record = {
        "pipeline_version": CONTOUR_PIPELINE_VERSION,
        "status": audit["status"],
        "model": model,
        "counter_mode": "speculative_parallel_v1",
        "contour_mode": "speculative_parallel_v1",
        "case": case,
        "candidate_contour": provisional_drafts.get("main_draft") or {},
        "counter_contour": {
            "mode": "speculative_parallel_v1",
            "rival_drafts": provisional_drafts.get("rival_drafts") or [],
        },
        "provisional_drafts": provisional_drafts,
        "candidate_resolution_map": resolution_map,
        "final_adjudication": final_payload,
        "audit": audit,
        "usage": usage,
    }
    if checkpoint_callback:
        checkpoint_callback(record)
    return contour, record


def _candidate_resolution_map(
    detective: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for relation in detective.get("relation_certificates", []) or []:
        candidate_id = compact_text(
            (relation.get("provenance") or {}).get("candidate_id")
        )
        if not candidate_id:
            continue
        item = {
            "candidate_id": candidate_id,
            "decision": relation.get("status"),
        }
        if relation.get("status") == "accepted":
            item["accepted_relation_id"] = relation.get("relation_id")
        output.append(item)
    return sorted(output, key=lambda item: item["candidate_id"])


def _validate_candidate_resolution_map(
    resolution_map: list[dict[str, Any]], detective: dict[str, Any]
) -> list[str]:
    expected = {
        compact_text((relation.get("provenance") or {}).get("candidate_id")): {
            "decision": relation.get("status"),
            "accepted_relation_id": (
                relation.get("relation_id")
                if relation.get("status") == "accepted"
                else None
            ),
        }
        for relation in detective.get("relation_certificates", []) or []
        if compact_text(
            (relation.get("provenance") or {}).get("candidate_id")
        )
    }
    actual = {
        compact_text(item.get("candidate_id")): {
            "decision": item.get("decision"),
            "accepted_relation_id": item.get("accepted_relation_id"),
        }
        for item in resolution_map
        if compact_text(item.get("candidate_id"))
    }
    errors: list[str] = []
    if len(actual) != len(resolution_map):
        errors.append("candidate resolution map has missing or duplicate IDs")
    if set(actual) != set(expected):
        errors.append("candidate resolution coverage mismatch")
    for candidate_id in set(actual) & set(expected):
        if actual[candidate_id] != expected[candidate_id]:
            errors.append(
                f"candidate resolution mismatch: {candidate_id}"
            )
    return errors[:20]


def _resolved_provisional_reasoning_payload(
    provisional_drafts: dict[str, Any],
    resolution_map: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve draft references in code before asking the final model to judge."""

    cleaned = _contour_reasoning_payload(provisional_drafts)
    resolutions = {
        compact_text(item.get("candidate_id")): item
        for item in resolution_map
        if compact_text(item.get("candidate_id"))
    }
    classified_count = 0
    removed_piece_ref_count = 0

    def resolve_draft(draft: Any) -> dict[str, Any]:
        nonlocal classified_count, removed_piece_ref_count
        if not isinstance(draft, dict):
            return {}
        candidate_ids = list(
            dict.fromkeys(
                compact_text(item)
                for item in draft.get("candidate_ids", []) or []
                if compact_text(item)
            )
        )
        unknown = [item for item in candidate_ids if item not in resolutions]
        if unknown:
            raise RuntimeError(
                "Provisional draft cites candidates absent from the "
                "resolution map: "
                + ",".join(unknown[:4])
            )
        accepted_relation_ids = [
            compact_text(resolutions[item].get("accepted_relation_id"))
            for item in candidate_ids
            if resolutions[item].get("decision") == "accepted"
            and compact_text(
                resolutions[item].get("accepted_relation_id")
            )
        ]
        discarded_candidates = [
            {
                "candidate_id": item,
                "decision": resolutions[item].get("decision"),
            }
            for item in candidate_ids
            if resolutions[item].get("decision") != "accepted"
        ]
        classified_count += len(candidate_ids)
        removed_piece_ref_count += len(draft.get("piece_ids") or [])
        resolved = {
            key: value
            for key, value in draft.items()
            if key not in {"candidate_ids", "piece_ids"}
        }
        resolved["accepted_relation_ids"] = accepted_relation_ids
        resolved["discarded_candidates"] = discarded_candidates
        return resolved

    output = {
        key: value
        for key, value in cleaned.items()
        if key not in {"main_draft", "rival_drafts"}
    }
    output["main_draft"] = resolve_draft(cleaned.get("main_draft"))
    rivals = []
    for rival in cleaned.get("rival_drafts", []) or []:
        if not isinstance(rival, dict):
            continue
        rivals.append(
            {
                **{
                    key: value
                    for key, value in rival.items()
                    if key != "draft"
                },
                "draft": resolve_draft(rival.get("draft")),
            }
        )
    output["rival_drafts"] = rivals
    audit = {
        "draft_count": int(bool(output.get("main_draft"))) + len(rivals),
        "candidate_references_classified": classified_count,
        "mechanical_piece_references_removed": removed_piece_ref_count,
        "hypothesis_text_removed": False,
        "accepted_relations_remain_in_final_case": True,
        "rejected_or_unresolved_candidates_remain_non_evidence": True,
    }
    return output, audit


def _emit_contour_checkpoint(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    model: str,
    case: dict[str, Any],
    candidate: dict[str, Any],
    counter: dict[str, Any],
    final_payload: dict[str, Any],
    stages: list[dict[str, Any]],
    usage: dict[str, Any],
) -> None:
    if not callback:
        return
    callback(
        {
            "pipeline_version": CONTOUR_PIPELINE_VERSION,
            "status": "running",
            "model": model,
            "case": case,
            "candidate_contour": candidate,
            "counter_contour": counter,
            "final_adjudication": final_payload,
            "audit": {
                "status": "running",
                "model": model,
                "stages": stages,
                "usage": usage,
            },
            "usage": usage,
        }
    )


def _reused_stage(stage: str, input_fingerprint: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "reused",
        "attempts": 0,
        "duration_seconds": 0.0,
        "errors": [],
        "usage": _empty_usage(),
        "cache_source": "validated_previous_stage_payload",
        "input_fingerprint": input_fingerprint,
    }


def _cached_contour_stage_matches(
    cached_record: dict[str, Any],
    stage: str,
    input_fingerprint: str,
) -> bool:
    if cached_record.get("pipeline_version") != CONTOUR_PIPELINE_VERSION:
        return False
    return any(
        item.get("stage") == stage
        and item.get("input_fingerprint") == input_fingerprint
        for item in ((cached_record.get("audit") or {}).get("stages", []) or [])
        if isinstance(item, dict)
    )


def _revalidate_cached_raw_stage(
    cached_record: dict[str, Any],
    stage: str,
    input_fingerprint: str,
    validator: Callable[[Any], tuple[dict[str, Any], list[str]]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Recover a now-valid payload from a checkpointed raw model response.

    A validator can learn that two phrases or transport representations are
    semantically equivalent. When that happens, the previous model response is
    still the same evidence-locked answer and should be revalidated before we
    spend another long model call rewriting it.
    """
    if cached_record.get("pipeline_version") != CONTOUR_PIPELINE_VERSION:
        return {}, None
    stages = ((cached_record.get("audit") or {}).get("stages", []) or [])
    for previous in reversed(stages):
        if not isinstance(previous, dict):
            continue
        if previous.get("stage") != stage or previous.get(
            "input_fingerprint"
        ) != input_fingerprint:
            continue
        raw = previous.get("raw") or {}
        content = (
            ((((raw.get("choices") or [{}])[0]) or {}).get("message") or {}).get(
                "content"
            )
            if isinstance(raw, dict)
            else None
        )
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            parsed = parse_json_content(content)
            validated, errors = validator(parsed)
        except Exception:  # noqa: BLE001 - a bad cache falls through to a live call.
            continue
        if not validated:
            continue
        record = _reused_stage(stage, input_fingerprint)
        record.update(
            {
                "cache_source": "revalidated_previous_raw_response",
                "prior_stage_status": previous.get("status"),
                "prior_validation_error_count": len(previous.get("errors") or []),
                "revalidation_errors": errors,
            }
        )
        return validated, record
    return {}, None


def _contour_stage_fingerprint(revision: str, payload: Any) -> str:
    serialized = json.dumps(
        {"revision": revision, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _contour_reasoning_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove validator audit notes that cannot change contour reasoning."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key != "validation_drops"
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(payload)


def build_contour_case(
    question: str,
    detective: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    pieces = {item["piece_id"]: item for item in detective.get("puzzle_pieces", [])}
    accepted = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
    ]
    unresolved = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "unresolved"
    ]
    endpoint_ids = {
        piece_id for relation in accepted for piece_id in relation.get("piece_ids", [])
    }
    endpoint_pieces = [
        _compact_piece(pieces[piece_id])
        for piece_id in sorted(endpoint_ids)
        if piece_id in pieces
    ]
    constructive_types = {
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
    constructive_relation_ids = sorted(
        item["relation_id"]
        for item in accepted
        if item.get("relation_type") in constructive_types
    )
    constructive_constraint_types = {
        "support_candidate",
        "conditional_split",
        "conflict_boundary",
        "scope_boundary",
        "causal_bridge",
        "shared_assumption_dependency",
        "definition_branch",
        "value_branch",
    }
    constructive_constraint_ids = sorted(
        item["constraint_id"]
        for item in constraints.get("constraints", []) or []
        if item.get("constraint_type") in constructive_constraint_types
    )
    directional_support_relation_ids = sorted(
        item["relation_id"]
        for item in accepted
        if item.get("relation_type") == "independent_convergence"
        and (item.get("independence_profile") or {}).get(
            "directional_support_authorized", True
        )
    )
    topic_anchor_groups = _build_topic_anchor_groups(
        accepted,
        pieces,
        constructive_types,
    )
    return {
        "question": compact_text(question),
        "question_id": detective["question_id"],
        "person_aliases": _person_aliases(pieces),
        "pieces": endpoint_pieces,
        "accepted_relations": [_compact_relation(item) for item in accepted],
        "mechanism_coverage": {
            "adds_model_calls": False,
            "slots": [
                {
                    "slot_id": item.get("slot_id"),
                    "label": item.get("label"),
                    "description": compact_text(item.get("description"))[:120],
                    "connection_status": item.get("connection_status"),
                    "accepted_relation_ids": item.get("accepted_relation_ids") or [],
                }
                for item in (detective.get("mechanism_coverage") or {}).get(
                    "slots", []
                )
            ],
            "instruction": (
                "已连接机制可参与轮廓；只有材料但未连接的机制只能作为候选或未知；"
                "未覆盖机制不得被补写成事实。"
            ),
        },
        "unresolved_relation_signals": [
            {
                "relation_type": item.get("relation_type"),
                "explanation": compact_text(item.get("plain_language_explanation")),
            }
            for item in unresolved[:20]
        ],
        "constraints": [
            {
                "constraint_id": item.get("constraint_id"),
                "constraint_type": item.get("constraint_type"),
                "source_relation_ids": item.get("source_relation_ids") or [],
                "piece_ids": item.get("piece_ids") or [],
                "rule": item.get("rule"),
                "relation_explanation": item.get("relation_explanation"),
                "strength": item.get("strength"),
                "caution": item.get("caution"),
                "knowledge_provenance": item.get("knowledge_provenance")
                or {},
            }
            for item in constraints.get("constraints", []) or []
        ],
        "allowed_ids": {
            "piece_ids": sorted(endpoint_ids),
            "relation_ids": sorted(item["relation_id"] for item in accepted),
            "constraint_ids": sorted(
                item["constraint_id"]
                for item in constraints.get("constraints", []) or []
            ),
        },
        "constructive_relation_ids": constructive_relation_ids,
        "constructive_constraint_ids": constructive_constraint_ids,
        "inference_capability": {
            "directional_support_available": bool(
                directional_support_relation_ids
            ),
            "directional_support_relation_ids": directional_support_relation_ids,
            "structural_only": not directional_support_relation_ids,
            "rule": (
                "当前只有结构关系。直接回答和主轮廓必须保持未定或条件化；"
                "定义分支、相关错误和共同模型先验不能推出会、不会、概率高低或时间表。"
                if not directional_support_relation_ids
                else "存在暂定的独立会合，但仍只能给出可撤销的方向，不得写成确定事实。"
            ),
        },
        "specificity_requirements": {
            "rule": (
                "直接答案至少落到一个本题具体关系，主轮廓至少落到两个；"
                "必须使用所引用关系中的具体对象、指标、机制或时间窗口，不能只写抽象方法词。"
            ),
            "minimum_relation_groups": {
                "direct_answer": min(1, len(topic_anchor_groups)),
                "main_contour": min(2, len(topic_anchor_groups)),
            },
            "relation_anchor_groups": topic_anchor_groups,
        },
        "solver_rules": constraints.get("solver_rules") or [],
    }


def build_contour_prompt_packet(case: dict[str, Any]) -> dict[str, Any]:
    """Encode every contour-usable fact once and replace repetition with refs."""

    pieces = case.get("pieces", []) or []
    relations = case.get("accepted_relations", []) or []
    constraints = case.get("constraints", []) or []
    people: dict[str, str] = {}
    conclusions: dict[str, str] = {}
    classic_boundary_index: dict[str, dict[str, Any]] = {}
    classic_boundary_refs: dict[str, str] = {}

    def boundary_ref(payload: dict[str, Any]) -> str:
        boundary = _classic_boundary(payload)
        fingerprint = json.dumps(
            boundary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        existing = classic_boundary_refs.get(fingerprint)
        if existing:
            return existing
        ref = f"classic_boundary_{len(classic_boundary_index) + 1:02d}"
        classic_boundary_refs[fingerprint] = ref
        classic_boundary_index[ref] = boundary
        return ref

    for piece in pieces:
        person_id = compact_text(piece.get("person_id"))
        if person_id and piece.get("person_name"):
            people[person_id] = compact_text(piece.get("person_name"))
        conclusion = compact_text(piece.get("parent_conclusion"))
        if person_id and conclusion:
            conclusions.setdefault(person_id, conclusion)

    piece_index: dict[str, dict[str, Any]] = {}
    for piece in pieces:
        piece_id = compact_text(piece.get("piece_id"))
        person_id = compact_text(piece.get("person_id"))
        encoded = {
            "person_id": person_id,
            "piece_kind": piece.get("piece_kind"),
            "text": piece.get("text"),
            "content_status": piece.get("content_status"),
            "coordinates": piece.get("coordinates") or {},
            "source_families": piece.get("source_families") or [],
            "verification_statuses": piece.get("verification_statuses") or [],
            "classic_boundary_ref": boundary_ref(
                piece.get("classic_knowledge_trace") or {}
            ),
        }
        conclusion = compact_text(piece.get("parent_conclusion"))
        if conclusion and conclusions.get(person_id) == conclusion:
            encoded["parent_conclusion_ref"] = person_id
        elif conclusion:
            encoded["parent_conclusion"] = conclusion
        piece_index[piece_id] = encoded

    relation_index = {
        compact_text(relation.get("relation_id")): {
            "relation_type": relation.get("relation_type"),
            "piece_ids": relation.get("piece_ids") or [],
            "plain_language_explanation": relation.get(
                "plain_language_explanation"
            ),
            "shared_coordinate": relation.get("shared_coordinate") or {},
            "source_independence": relation.get("source_independence"),
            "common_error_risk": relation.get("common_error_risk"),
            "competing_explanation": relation.get("competing_explanation"),
            "falsification_test": relation.get("falsification_test"),
            "classic_boundary_ref": boundary_ref(
                relation.get("knowledge_provenance") or {}
            ),
        }
        for relation in relations
    }

    constraint_index: dict[str, dict[str, Any]] = {}
    for constraint in constraints:
        constraint_id = compact_text(constraint.get("constraint_id"))
        source_relation_ids = constraint.get("source_relation_ids") or []
        encoded = {
            "constraint_type": constraint.get("constraint_type"),
            "source_relation_ids": source_relation_ids,
            "rule": constraint.get("rule"),
            "strength": constraint.get("strength"),
            "caution": constraint.get("caution"),
            "classic_boundary_ref": boundary_ref(
                constraint.get("knowledge_provenance") or {}
            ),
        }
        relation_id = source_relation_ids[0] if len(source_relation_ids) == 1 else ""
        relation = relation_index.get(relation_id) or {}
        if relation and (constraint.get("piece_ids") or []) == (
            relation.get("piece_ids") or []
        ):
            encoded["piece_ids_ref"] = relation_id
        else:
            encoded["piece_ids"] = constraint.get("piece_ids") or []
        if relation and compact_text(constraint.get("relation_explanation")) == compact_text(
            relation.get("plain_language_explanation")
        ):
            encoded["relation_explanation_ref"] = relation_id
        else:
            encoded["relation_explanation"] = constraint.get(
                "relation_explanation"
            )
        constraint_index[constraint_id] = encoded

    specificity = case.get("specificity_requirements") or {}
    return {
        "schema": "bme.contour-prompt-packet.v1",
        "question": case.get("question"),
        "question_id": case.get("question_id"),
        "packet_rules": {
            "index_keys_are_valid_ids": True,
            "parent_conclusion_ref": "resolve in parent_conclusions",
            "piece_ids_ref": "resolve in relation_index[ref].piece_ids",
            "relation_explanation_ref": (
                "resolve in relation_index[ref].plain_language_explanation"
            ),
            "classic_boundary_ref": (
                "resolve in classic_boundary_index; every material keeps one"
            ),
            "classic_boundary_scope": (
                "Only inferential authorization is exposed here; complete classic audit "
                "IDs and review records remain in the detective artifact and never act "
                "as world evidence."
            ),
        },
        "people": people,
        "parent_conclusions": conclusions,
        "classic_boundary_index": classic_boundary_index,
        "piece_index": piece_index,
        "relation_index": relation_index,
        "constraint_index": constraint_index,
        "mechanism_coverage": case.get("mechanism_coverage") or {},
        "unresolved_relation_signals": case.get(
            "unresolved_relation_signals"
        )
        or [],
        "constructive_relation_ids": case.get("constructive_relation_ids") or [],
        "constructive_constraint_ids": case.get("constructive_constraint_ids")
        or [],
        "inference_capability": case.get("inference_capability") or {},
        "specificity_requirements": {
            "rule": specificity.get("rule"),
            "minimum_relation_groups": specificity.get(
                "minimum_relation_groups"
            )
            or {},
            "instruction": (
                "Use concrete objects, indicators, mechanisms, or time windows from "
                "the cited relation and piece indexes."
            ),
        },
        "solver_rules": case.get("solver_rules") or [],
    }


def validate_contour_prompt_packet(
    packet: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    """Prove that all material the contour is authorized to use is represented."""

    errors: list[str] = []
    if packet.get("question") != case.get("question"):
        errors.append("question mismatch")
    if packet.get("question_id") != case.get("question_id"):
        errors.append("question_id mismatch")
    pieces = {
        compact_text(item.get("piece_id")): item
        for item in case.get("pieces", []) or []
    }
    relations = {
        compact_text(item.get("relation_id")): item
        for item in case.get("accepted_relations", []) or []
    }
    constraints = {
        compact_text(item.get("constraint_id")): item
        for item in case.get("constraints", []) or []
    }
    piece_index = packet.get("piece_index") or {}
    relation_index = packet.get("relation_index") or {}
    constraint_index = packet.get("constraint_index") or {}
    classic_boundary_index = packet.get("classic_boundary_index") or {}
    if set(piece_index) != set(pieces):
        errors.append("piece coverage mismatch")
    if set(relation_index) != set(relations):
        errors.append("relation coverage mismatch")
    if set(constraint_index) != set(constraints):
        errors.append("constraint coverage mismatch")

    expected_people: dict[str, str] = {}
    expected_conclusions: dict[str, str] = {}
    for original in case.get("pieces", []) or []:
        person_id = compact_text(original.get("person_id"))
        person_name = compact_text(original.get("person_name"))
        conclusion = compact_text(original.get("parent_conclusion"))
        if person_id and person_name:
            expected_people[person_id] = person_name
        if person_id and conclusion:
            expected_conclusions.setdefault(person_id, conclusion)
    if packet.get("people") != expected_people:
        errors.append("people index mismatch")
    if packet.get("parent_conclusions") != expected_conclusions:
        errors.append("parent conclusion index mismatch")

    exact_top_level_fields = (
        "mechanism_coverage",
        "unresolved_relation_signals",
        "constructive_relation_ids",
        "constructive_constraint_ids",
        "inference_capability",
        "solver_rules",
    )
    for field in exact_top_level_fields:
        expected = case.get(field) or ([] if field in {
            "unresolved_relation_signals",
            "constructive_relation_ids",
            "constructive_constraint_ids",
            "solver_rules",
        } else {})
        if packet.get(field) != expected:
            errors.append(f"top-level material mismatch: {field}")

    expected_allowed_ids = {
        "piece_ids": sorted(piece_index),
        "relation_ids": sorted(relation_index),
        "constraint_ids": sorted(constraint_index),
    }
    if expected_allowed_ids != (case.get("allowed_ids") or {}):
        errors.append("allowed ids are not reconstructible from packet indexes")

    conclusions = packet.get("parent_conclusions") or {}
    for piece_id, original in pieces.items():
        encoded = piece_index.get(piece_id) or {}
        reconstructed_conclusion = encoded.get("parent_conclusion")
        if encoded.get("parent_conclusion_ref"):
            reconstructed_conclusion = conclusions.get(
                encoded["parent_conclusion_ref"]
            )
        checks = {
            "person_id": original.get("person_id"),
            "piece_kind": original.get("piece_kind"),
            "text": original.get("text"),
            "content_status": original.get("content_status"),
            "coordinates": original.get("coordinates") or {},
            "source_families": original.get("source_families") or [],
            "verification_statuses": original.get("verification_statuses") or [],
        }
        if any(encoded.get(key) != value for key, value in checks.items()):
            errors.append(f"piece material mismatch: {piece_id}")
        if compact_text(reconstructed_conclusion) != compact_text(
            original.get("parent_conclusion")
        ):
            errors.append(f"piece conclusion mismatch: {piece_id}")
        boundary = classic_boundary_index.get(
            compact_text(encoded.get("classic_boundary_ref"))
        )
        if boundary != _classic_boundary(
            original.get("classic_knowledge_trace") or {}
        ):
            errors.append(f"piece classic boundary mismatch: {piece_id}")

    for relation_id, original in relations.items():
        encoded = relation_index.get(relation_id) or {}
        for field in (
            "relation_type",
            "piece_ids",
            "plain_language_explanation",
            "shared_coordinate",
            "source_independence",
            "common_error_risk",
            "competing_explanation",
            "falsification_test",
        ):
            if field == "piece_ids":
                expected = original.get(field) or []
            elif field == "shared_coordinate":
                expected = original.get(field) or {}
            else:
                expected = original.get(field)
            if encoded.get(field) != expected:
                errors.append(f"relation material mismatch: {relation_id}/{field}")
        boundary = classic_boundary_index.get(
            compact_text(encoded.get("classic_boundary_ref"))
        )
        if boundary != _classic_boundary(
            original.get("knowledge_provenance") or {}
        ):
            errors.append(f"relation classic boundary mismatch: {relation_id}")

    for constraint_id, original in constraints.items():
        encoded = constraint_index.get(constraint_id) or {}
        relation_ref = compact_text(encoded.get("piece_ids_ref"))
        reconstructed_piece_ids = encoded.get("piece_ids")
        if relation_ref:
            reconstructed_piece_ids = (
                relation_index.get(relation_ref) or {}
            ).get("piece_ids")
        explanation_ref = compact_text(
            encoded.get("relation_explanation_ref")
        )
        reconstructed_explanation = encoded.get("relation_explanation")
        if explanation_ref:
            reconstructed_explanation = (
                relation_index.get(explanation_ref) or {}
            ).get("plain_language_explanation")
        if reconstructed_piece_ids != (original.get("piece_ids") or []):
            errors.append(f"constraint piece mismatch: {constraint_id}")
        if compact_text(reconstructed_explanation) != compact_text(
            original.get("relation_explanation")
        ):
            errors.append(f"constraint explanation mismatch: {constraint_id}")
        for field in (
            "constraint_type",
            "source_relation_ids",
            "rule",
            "strength",
            "caution",
        ):
            if field == "source_relation_ids":
                expected = original.get(field) or []
            else:
                expected = original.get(field)
            if encoded.get(field) != expected:
                errors.append(f"constraint material mismatch: {constraint_id}/{field}")
        boundary = classic_boundary_index.get(
            compact_text(encoded.get("classic_boundary_ref"))
        )
        if boundary != _classic_boundary(
            original.get("knowledge_provenance") or {}
        ):
            errors.append(f"constraint classic boundary mismatch: {constraint_id}")

    referenced_boundary_ids = {
        compact_text(encoded.get("classic_boundary_ref"))
        for index in (piece_index, relation_index, constraint_index)
        for encoded in index.values()
        if compact_text(encoded.get("classic_boundary_ref"))
    }
    if referenced_boundary_ids != set(classic_boundary_index):
        errors.append("classic boundary index has missing or unused entries")

    specificity = case.get("specificity_requirements") or {}
    packet_specificity = packet.get("specificity_requirements") or {}
    if packet_specificity.get("rule") != specificity.get("rule"):
        errors.append("specificity rule mismatch")
    if packet_specificity.get("minimum_relation_groups") != (
        specificity.get("minimum_relation_groups") or {}
    ):
        errors.append("specificity minimums mismatch")

    reconstructed_relations = []
    for relation_id, encoded in relation_index.items():
        reconstructed_relations.append(
            {
                **{
                    key: value
                    for key, value in encoded.items()
                    if key != "classic_boundary_ref"
                },
                "relation_id": relation_id,
            }
        )
    reconstructed_pieces = {
        piece_id: {
            **{
                key: value
                for key, value in encoded.items()
                if key
                not in {
                    "classic_boundary_ref",
                    "parent_conclusion_ref",
                    "parent_conclusion",
                }
            },
            "piece_id": piece_id,
        }
        for piece_id, encoded in piece_index.items()
    }
    constructive_types = {
        relation.get("relation_type")
        for relation in reconstructed_relations
        if relation.get("relation_id")
        in set(packet.get("constructive_relation_ids") or [])
    }
    reconstructed_anchor_groups = _build_topic_anchor_groups(
        reconstructed_relations,
        reconstructed_pieces,
        constructive_types,
    )
    if reconstructed_anchor_groups != (
        specificity.get("relation_anchor_groups") or []
    ):
        errors.append("question-specific anchor groups are not reconstructible")
    return errors[:20]


def build_contour_decision_packet(
    packet: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    """Re-encode the complete contour material for the final decision call.

    The final model needs every accepted relation, constraint, endpoint, and
    classic boundary, but it does not need verbose object keys repeated for
    every row. Tables and exact reference indexes reduce transport without
    dropping any reasoning material. The companion validator reconstructs the
    original indexes before a live call is allowed.
    """

    piece_fields = (
        "person_id",
        "piece_kind",
        "text",
        "content_status",
        "coordinates",
        "source_families",
        "verification_statuses",
        "classic_boundary_ref",
        "parent_conclusion_ref",
        "parent_conclusion",
    )
    relation_fields = (
        "relation_type",
        "piece_ids",
        "plain_language_explanation",
        "shared_coordinate",
        "source_independence",
        "common_error_risk",
        "competing_explanation",
        "falsification_test",
        "classic_boundary_ref",
    )
    constraint_fields = (
        "constraint_type",
        "source_relation_ids",
        "rule",
        "strength",
        "caution",
        "classic_boundary_ref",
        "piece_ids_ref",
        "piece_ids",
        "relation_explanation_ref",
        "relation_explanation",
    )
    return {
        "schema": "bme.contour-decision-packet.v1",
        "source_schema": packet.get("schema"),
        "question": packet.get("question"),
        "question_id": packet.get("question_id"),
        "packet_rules": {
            "table_rows": (
                "Each row is [id, presence_bitmap, values...]; fields are in "
                "the table's fields array and absent fields have a null slot."
            ),
            "index_keys_are_valid_ids": True,
            "parent_conclusion_ref": "resolve in parent_conclusions",
            "piece_ids_ref": "resolve in relation_table[ref].piece_ids",
            "relation_explanation_ref": (
                "resolve in relation_table[ref].plain_language_explanation"
            ),
            "classic_boundary_ref": "resolve in classic_boundary_index",
            "provenance_hydration": (
                "Cite relation_ids and constraint_ids. The program inherits "
                "their endpoint piece_ids after generation; extra_piece_ids "
                "are only for material not already reachable from those IDs."
            ),
            "classic_boundary_scope": (
                "Classic material authorizes or limits inference only; it is "
                "never world evidence."
            ),
        },
        "people": packet.get("people") or {},
        "person_aliases": case.get("person_aliases") or {},
        "parent_conclusions": packet.get("parent_conclusions") or {},
        "classic_boundary_index": packet.get("classic_boundary_index") or {},
        "piece_table": _contour_index_table(
            packet.get("piece_index") or {}, piece_fields
        ),
        "relation_table": _contour_index_table(
            packet.get("relation_index") or {}, relation_fields
        ),
        "constraint_table": _contour_index_table(
            packet.get("constraint_index") or {}, constraint_fields
        ),
        "mechanism_coverage": packet.get("mechanism_coverage") or {},
        "unresolved_relation_signals": packet.get(
            "unresolved_relation_signals"
        )
        or [],
        "constructive_relation_ids": packet.get(
            "constructive_relation_ids"
        )
        or [],
        "constructive_constraint_ids": packet.get(
            "constructive_constraint_ids"
        )
        or [],
        "inference_capability": packet.get("inference_capability") or {},
        "specificity_requirements": packet.get(
            "specificity_requirements"
        )
        or {},
        "solver_rules": packet.get("solver_rules") or [],
    }


def validate_contour_decision_packet(
    decision_packet: dict[str, Any],
    source_packet: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    """Prove that compact final-decision material is exactly reconstructible."""

    errors: list[str] = []
    if decision_packet.get("question") != source_packet.get("question"):
        errors.append("decision question mismatch")
    if decision_packet.get("question_id") != source_packet.get("question_id"):
        errors.append("decision question_id mismatch")
    if decision_packet.get("person_aliases") != (
        case.get("person_aliases") or {}
    ):
        errors.append("decision person aliases mismatch")
    table_pairs = (
        ("piece_table", "piece_index"),
        ("relation_table", "relation_index"),
        ("constraint_table", "constraint_index"),
    )
    for table_name, index_name in table_pairs:
        try:
            restored = _restore_contour_index_table(
                decision_packet.get(table_name) or {}
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{table_name} cannot be reconstructed: {exc}")
            continue
        if restored != (source_packet.get(index_name) or {}):
            errors.append(f"decision material mismatch: {index_name}")
    exact_fields = (
        "people",
        "parent_conclusions",
        "classic_boundary_index",
        "mechanism_coverage",
        "unresolved_relation_signals",
        "constructive_relation_ids",
        "constructive_constraint_ids",
        "inference_capability",
        "specificity_requirements",
        "solver_rules",
    )
    for field in exact_fields:
        expected = source_packet.get(field)
        if expected is None:
            expected = [] if field in {
                "unresolved_relation_signals",
                "constructive_relation_ids",
                "constructive_constraint_ids",
                "solver_rules",
            } else {}
        if decision_packet.get(field) != expected:
            errors.append(f"decision top-level material mismatch: {field}")
    reconstructed_ids = {
        "piece_ids": sorted(
            _restore_contour_index_table(
                decision_packet.get("piece_table") or {}
            )
        ),
        "relation_ids": sorted(
            _restore_contour_index_table(
                decision_packet.get("relation_table") or {}
            )
        ),
        "constraint_ids": sorted(
            _restore_contour_index_table(
                decision_packet.get("constraint_table") or {}
            )
        ),
    }
    if reconstructed_ids != (case.get("allowed_ids") or {}):
        errors.append("decision allowed IDs are not reconstructible")
    return errors[:20]


def _contour_index_table(
    index: dict[str, dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, Any]:
    rows = []
    for item_id, item in index.items():
        presence = sum(
            1 << position
            for position, field in enumerate(fields)
            if field in item
        )
        rows.append(
            [
                item_id,
                presence,
                *[item.get(field) for field in fields],
            ]
        )
    return {"fields": list(fields), "rows": rows}


def _restore_contour_index_table(
    table: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fields = table.get("fields")
    rows = table.get("rows")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise TypeError("table needs fields and rows")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields) + 2:
            raise ValueError("table row width mismatch")
        item_id = compact_text(row[0])
        if not item_id or item_id in output:
            raise ValueError("table row has a missing or duplicate id")
        presence = int(row[1])
        output[item_id] = {
            str(field): row[position + 2]
            for position, field in enumerate(fields)
            if presence & (1 << position)
        }
    return output


def build_counter_pressure_packets(
    packet: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Partition the full contour packet across complementary counter checks."""

    focus_ids = [item[0] for item in COUNTER_PRESSURE_SPECS]
    relation_focus = {
        "independent_convergence": "evidence_lineage",
        "shared_source": "evidence_lineage",
        "shared_model_prior": "evidence_lineage",
        "direct_conflict": "causal_conditions",
        "apparent_conflict": "causal_conditions",
        "conditional_complement": "causal_conditions",
        "scope_refinement": "causal_conditions",
        "causal_relay": "causal_conditions",
        "definition_branch": "causal_conditions",
        "value_branch": "causal_conditions",
        "correlated_error": "shadow_unknowns",
        "opposite_distortion": "shadow_unknowns",
        "blind_spot_fill": "shadow_unknowns",
        "collective_blind_spot": "shadow_unknowns",
        "diagnostic_challenge": "shadow_unknowns",
        "shared_assumption": "shadow_unknowns",
        "non_comparable": "shadow_unknowns",
    }
    core_refs = _collect_contour_refs(
        {
            "direct_answer": candidate.get("direct_answer"),
            "main_contour": candidate.get("main_contour"),
        }
    )
    relation_index = packet.get("relation_index") or {}
    constraint_index = packet.get("constraint_index") or {}
    piece_index = packet.get("piece_index") or {}
    relation_ids_by_focus = {focus_id: set() for focus_id in focus_ids}
    for relation_id, relation in relation_index.items():
        focus_id = relation_focus.get(
            compact_text(relation.get("relation_type")),
            "evidence_lineage",
        )
        relation_ids_by_focus[focus_id].add(relation_id)
    for focus_id in focus_ids:
        relation_ids_by_focus[focus_id].update(core_refs["relation_ids"])

    constraint_ids_by_focus = {focus_id: set() for focus_id in focus_ids}
    for index, (constraint_id, constraint) in enumerate(
        constraint_index.items()
    ):
        source_ids = set(constraint.get("source_relation_ids") or [])
        assigned = [
            focus_id
            for focus_id in focus_ids
            if source_ids & relation_ids_by_focus[focus_id]
        ]
        if constraint_id in core_refs["constraint_ids"]:
            assigned = list(focus_ids)
        if not assigned:
            assigned = [focus_ids[index % len(focus_ids)]]
        for focus_id in assigned:
            constraint_ids_by_focus[focus_id].add(constraint_id)
            relation_ids_by_focus[focus_id].update(source_ids)

    unresolved = list(packet.get("unresolved_relation_signals") or [])
    outputs = []
    for focus_index, (focus_id, label, instruction) in enumerate(
        COUNTER_PRESSURE_SPECS
    ):
        relation_ids = relation_ids_by_focus[focus_id]
        constraint_ids = constraint_ids_by_focus[focus_id]
        piece_ids = set(core_refs["piece_ids"])
        for relation_id in relation_ids:
            piece_ids.update(
                (relation_index.get(relation_id) or {}).get("piece_ids") or []
            )
        for constraint_id in constraint_ids:
            constraint = constraint_index.get(constraint_id) or {}
            piece_ref = compact_text(constraint.get("piece_ids_ref"))
            if piece_ref:
                relation_ids.add(piece_ref)
                piece_ids.update(
                    (relation_index.get(piece_ref) or {}).get("piece_ids") or []
                )
            else:
                piece_ids.update(constraint.get("piece_ids") or [])

        selected_relations = {
            relation_id: relation_index[relation_id]
            for relation_id in relation_index
            if relation_id in relation_ids
        }
        selected_constraints = {
            constraint_id: constraint_index[constraint_id]
            for constraint_id in constraint_index
            if constraint_id in constraint_ids
        }
        selected_pieces = {
            piece_id: piece_index[piece_id]
            for piece_id in piece_index
            if piece_id in piece_ids
        }
        selected_people_ids = {
            compact_text(item.get("person_id"))
            for item in selected_pieces.values()
            if compact_text(item.get("person_id"))
        }
        boundary_refs = {
            compact_text(item.get("classic_boundary_ref"))
            for index_payload in (
                selected_pieces,
                selected_relations,
                selected_constraints,
            )
            for item in index_payload.values()
            if compact_text(item.get("classic_boundary_ref"))
        }
        material = {
            "schema": packet.get("schema"),
            "question": packet.get("question"),
            "question_id": packet.get("question_id"),
            "packet_rules": packet.get("packet_rules") or {},
            "pressure_scope": {
                "focus_id": focus_id,
                "label": label,
                "instruction": instruction,
                "collectively_lossless_with_other_pressure_packets": True,
            },
            "people": {
                key: value
                for key, value in (packet.get("people") or {}).items()
                if key in selected_people_ids
            },
            "parent_conclusions": {
                key: value
                for key, value in (
                    packet.get("parent_conclusions") or {}
                ).items()
                if key in selected_people_ids
            },
            "classic_boundary_index": {
                key: value
                for key, value in (
                    packet.get("classic_boundary_index") or {}
                ).items()
                if key in boundary_refs
            },
            "piece_index": selected_pieces,
            "relation_index": selected_relations,
            "constraint_index": selected_constraints,
            "mechanism_coverage": packet.get("mechanism_coverage") or {},
            "unresolved_relation_signals": [
                item
                for index, item in enumerate(unresolved)
                if index % len(focus_ids) == focus_index
            ],
            "constructive_relation_ids": [
                item
                for item in packet.get("constructive_relation_ids") or []
                if item in relation_ids
            ],
            "constructive_constraint_ids": [
                item
                for item in packet.get("constructive_constraint_ids") or []
                if item in constraint_ids
            ],
            "inference_capability": packet.get("inference_capability") or {},
            "specificity_requirements": packet.get(
                "specificity_requirements"
            )
            or {},
            "solver_rules": packet.get("solver_rules") or [],
        }
        outputs.append(
            {
                "focus_id": focus_id,
                "label": label,
                "instruction": instruction,
                "material": material,
            }
        )
    return outputs


def validate_counter_pressure_packets(
    pressure_packets: list[dict[str, Any]],
    full_packet: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_focus_ids = [item[0] for item in COUNTER_PRESSURE_SPECS]
    actual_focus_ids = [item.get("focus_id") for item in pressure_packets]
    if actual_focus_ids != expected_focus_ids:
        errors.append("counter pressure focus coverage mismatch")
    core_refs = _collect_contour_refs(
        {
            "direct_answer": candidate.get("direct_answer"),
            "main_contour": candidate.get("main_contour"),
        }
    )
    union_fields = {
        "people": {},
        "parent_conclusions": {},
        "classic_boundary_index": {},
        "piece_index": {},
        "relation_index": {},
        "constraint_index": {},
    }
    unresolved_union: list[Any] = []
    constructive_relation_ids: set[str] = set()
    constructive_constraint_ids: set[str] = set()
    exact_fields = (
        "schema",
        "question",
        "question_id",
        "packet_rules",
        "mechanism_coverage",
        "inference_capability",
        "specificity_requirements",
        "solver_rules",
    )
    for pressure in pressure_packets:
        material = pressure.get("material") or {}
        for field in exact_fields:
            if material.get(field) != full_packet.get(field):
                errors.append(
                    f"counter pressure global material mismatch: {field}"
                )
        for field in union_fields:
            for key, value in (material.get(field) or {}).items():
                previous = union_fields[field].get(key)
                if previous is not None and previous != value:
                    errors.append(
                        f"counter pressure duplicate conflict: {field}/{key}"
                    )
                union_fields[field][key] = value
        unresolved_union.extend(
            material.get("unresolved_relation_signals") or []
        )
        constructive_relation_ids.update(
            material.get("constructive_relation_ids") or []
        )
        constructive_constraint_ids.update(
            material.get("constructive_constraint_ids") or []
        )
        for field in ("piece_ids", "relation_ids", "constraint_ids"):
            index_name = field.replace("_ids", "_index")
            missing = core_refs[field] - set(material.get(index_name) or {})
            if missing:
                errors.append(
                    f"counter pressure omits candidate refs: {field}"
                )
        boundary_ids = set(material.get("classic_boundary_index") or {})
        for index_name in (
            "piece_index",
            "relation_index",
            "constraint_index",
        ):
            for item in (material.get(index_name) or {}).values():
                ref = compact_text(item.get("classic_boundary_ref"))
                if ref and ref not in boundary_ids:
                    errors.append(
                        f"counter pressure has unresolved classic boundary: {ref}"
                    )
    for field, merged in union_fields.items():
        if merged != (full_packet.get(field) or {}):
            errors.append(f"counter pressure union coverage mismatch: {field}")
    if sorted(
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in unresolved_union
    ) != sorted(
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in full_packet.get("unresolved_relation_signals") or []
    ):
        errors.append("counter pressure unresolved-signal coverage mismatch")
    if constructive_relation_ids != set(
        full_packet.get("constructive_relation_ids") or []
    ):
        errors.append("counter pressure constructive-relation coverage mismatch")
    if constructive_constraint_ids != set(
        full_packet.get("constructive_constraint_ids") or []
    ):
        errors.append("counter pressure constructive-constraint coverage mismatch")
    return errors[:20]


def _collect_contour_refs(payload: Any) -> dict[str, set[str]]:
    refs = {
        "piece_ids": set(),
        "relation_ids": set(),
        "constraint_ids": set(),
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in refs and isinstance(item, list):
                    refs[key].update(compact_text(entry) for entry in item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return refs


def _validate_counter_pressure_bundle(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["counter pressure bundle is not an object"]
    tests = payload.get("pressure_tests")
    if not isinstance(tests, list):
        return {}, ["counter pressure bundle needs pressure_tests"]
    expected = [item[0] for item in COUNTER_PRESSURE_SPECS]
    actual = [compact_text(item.get("focus_id")) for item in tests]
    errors = []
    if actual != expected:
        errors.append("counter pressure result coverage mismatch")
    sanitized_tests = []
    for item in tests:
        validated, item_errors = _validate_counter(
            item.get("counter") if isinstance(item, dict) else None,
            case,
        )
        if item_errors:
            errors.extend(
                f"{compact_text(item.get('focus_id'))}: {error}"
                for error in item_errors
            )
        if validated:
            sanitized_tests.append(
                {
                    "focus_id": compact_text(item.get("focus_id")),
                    "label": compact_text(item.get("label")),
                    "counter": validated,
                }
            )
    output = {
        "mode": "parallel_pressure_v1",
        "collective_material_coverage": "lossless_validated",
        "pressure_tests": sanitized_tests,
    }
    return (output if not errors else {}), errors[:20]


def _run_parallel_counter_pressure(
    client_factory: Callable[[], DeepSeekClient],
    case: dict[str, Any],
    candidate: dict[str, Any],
    full_packet: dict[str, Any],
    *,
    max_tokens: int,
    retries: int,
    input_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pressure_packets = build_counter_pressure_packets(full_packet, candidate)
    coverage_errors = validate_counter_pressure_packets(
        pressure_packets, full_packet, candidate
    )
    if coverage_errors:
        raise RuntimeError(
            "Counter pressure packets failed coverage validation: "
            + "; ".join(coverage_errors)
        )
    started = time.perf_counter()
    records: list[dict[str, Any] | None] = [None] * len(pressure_packets)
    payloads: list[dict[str, Any] | None] = [None] * len(pressure_packets)
    with ThreadPoolExecutor(max_workers=len(pressure_packets)) as executor:
        futures = {}
        for index, pressure in enumerate(pressure_packets):
            stage_fingerprint = _contour_stage_fingerprint(
                COUNTER_PRESSURE_PROMPT_REVISION,
                {
                    "focus_id": pressure["focus_id"],
                    "material": pressure["material"],
                    "candidate": candidate,
                },
            )
            future = executor.submit(
                _call_validated_stage,
                client_factory,
                _counter_messages(
                    case,
                    candidate,
                    pressure["material"],
                    focus_label=pressure["label"],
                    focus_instruction=pressure["instruction"],
                ),
                stage=f"counter_pressure_{pressure['focus_id']}",
                validator=lambda value: _validate_counter(value, case),
                max_tokens=max(4096, min(max_tokens, 6144)),
                retries=retries,
                thinking_mode="enabled",
                input_fingerprint=stage_fingerprint,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            payloads[index], records[index] = future.result()

    complete_records = [item for item in records if item is not None]
    usage = _empty_usage()
    errors: list[str] = []
    for record in complete_records:
        usage = _merge_usage(usage, record.get("usage") or {})
        errors.extend(record.get("errors") or [])
    successful = len(complete_records) == len(pressure_packets) and all(
        item.get("status") == "succeeded" for item in complete_records
    )
    bundle = {
        "mode": "parallel_pressure_v1",
        "collective_material_coverage": "lossless_validated",
        "pressure_tests": [
            {
                "focus_id": pressure["focus_id"],
                "label": pressure["label"],
                "counter": payload,
            }
            for pressure, payload in zip(pressure_packets, payloads)
            if isinstance(payload, dict) and payload
        ],
    }
    validated, bundle_errors = _validate_counter_pressure_bundle(bundle, case)
    errors.extend(bundle_errors)
    successful = successful and bool(validated)
    return (validated if successful else {}), {
        "stage": "parallel_counter_pressure",
        "status": "succeeded" if successful else "failed",
        "attempts": sum(
            int(item.get("attempts") or 0) for item in complete_records
        ),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
        "usage": usage,
        "thinking_mode": "enabled_parallel",
        "retryable": all(
            item.get("retryable", True) for item in complete_records
        ),
        "input_fingerprint": input_fingerprint,
        "pressure_stage_count": len(pressure_packets),
        "pressure_stages": complete_records,
        "packet_coverage_validation": "passed",
        "full_packet_characters": len(
            json.dumps(full_packet, ensure_ascii=False)
        ),
        "largest_pressure_packet_characters": max(
            len(json.dumps(item["material"], ensure_ascii=False))
            for item in pressure_packets
        ),
        "total_pressure_packet_characters": sum(
            len(json.dumps(item["material"], ensure_ascii=False))
            for item in pressure_packets
        ),
    }


def _classic_boundary(payload: dict[str, Any]) -> dict[str, Any]:
    authorizations = payload.get("inversion_authorizations") or (
        [payload.get("inversion_authorization")]
        if payload.get("inversion_authorization")
        else []
    )
    classic_check = payload.get("classic_check") or {}
    return {
        "trace_present": bool(
            payload.get("classic_trace_ids")
            or payload.get("connection_hypothesis_ids")
            or payload.get("inversion_certificate_ids")
        ),
        "inversion_authorizations": list(dict.fromkeys(authorizations)),
        "grounding_status": payload.get("grounding_status"),
        "epistemic_role": payload.get("epistemic_role"),
        "classic_check_status": classic_check.get("decision"),
        "inference_permission": payload.get("inference_permission"),
        "guardrail_action": payload.get("guardrail_action"),
        "classic_as_world_evidence": False,
    }


def _candidate_messages(
    case: dict[str, Any],
    prompt_packet: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    schema = _final_schema()
    capability = case.get("inference_capability") or {}
    capability_instruction = (
        "本案没有任何独立会合关系，只有结构关系。direct_answer 必须明确写成尚不能确定或取决于定义；"
        "main_contour 只能描述定义分叉、推理漏洞和来源依赖，禁止判断会、不会、概率高低或时间表。\n"
        if capability.get("structural_only")
        else "本案存在暂定独立会合，方向仍须写成可撤销判断。\n"
    )
    return [
        {"role": "system", "content": CONTOUR_SYSTEM + CLASSIC_CONTOUR_RULES},
        {
            "role": "user",
            "content": (
                "先生成一份可被反方攻击的主候选。direct_answer 用一至三句话直接回答原题；"
                "main_contour 说明最能同时解释可靠局部和分歧的整体形状。最多保留 4 个关键条件、5 个稳定局部、5 个未知。\n"
                "引用任何包含 2 至 4 个端点的 relation_id 时，piece_ids 必须列全该关系的端点；"
                "只使用 person_aliases 给出的姓名，不得自行给人物改名。\n"
                "必须把 CASE_JSON.specificity_requirements 中至少规定数量的具体关系写进正文，"
                "并引用对应关系或约束；不得只复述方法论。\n"
                "同时使用 mechanism_coverage 检查轮廓是否过窄，但不得为追求覆盖率制造连接或事实。\n"
                + capability_instruction
                + "CASE_JSON:\n"
                + _prompt_json(prompt_packet or case)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]


def _counter_messages(
    case: dict[str, Any],
    candidate: dict[str, Any],
    prompt_packet: dict[str, Any] | None = None,
    *,
    focus_label: str = "",
    focus_instruction: str = "",
) -> list[dict[str, str]]:
    schema = {
        "counter_contour": _statement_schema("最强替代轮廓"),
        "why_it_may_fit_better": _statement_schema("它为何可能解释得更好"),
        "attacks": [_statement_schema("候选的具体薄弱点")],
        "decisive_tests": [_statement_schema("什么会区分两幅轮廓")],
    }
    return [
        {"role": "system", "content": COUNTER_SYSTEM + CLASSIC_CONTOUR_RULES},
        {
            "role": "user",
            "content": (
                "对主候选做钢人化反驳。优先攻击它最关键的连接，不要换一道题。\n"
                + (
                    f"本轮专门负责“{focus_label}”：{focus_instruction}；"
                    "其他压力方向会由并行审查者承担；仍可指出与本方向直接相连的问题。\n"
                    if focus_label
                    else ""
                )
                + (
                    "只保留最多 2 个互不重复的 attacks 和最多 2 个真正有区分力的 decisive_tests；"
                    if focus_label
                    else "只保留最多 4 个互不重复的 attacks 和最多 3 个真正有区分力的 decisive_tests；"
                )
                + "数量更少但更致命，胜过罗列相似质疑。\n"
                "引用任何包含 2 至 4 个端点的 relation_id 时，piece_ids 必须列全该关系的端点；"
                "只使用 person_aliases 给出的姓名，不得自行给人物改名。\n"
                "CASE_JSON:\n"
                + _prompt_json(prompt_packet or case)
                + "\nMAIN_CANDIDATE:\n"
                + _prompt_json(candidate)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]


def _speculative_final_messages(
    case: dict[str, Any],
    provisional_drafts: dict[str, Any],
    resolution_map: list[dict[str, Any]],
    prompt_packet: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FINAL_SYSTEM + CLASSIC_CONTOUR_RULES},
        {
            "role": "user",
            "content": (
                "完成最终裁决。PROVISIONAL_DRAFTS 是侦探裁决关系时并行生成的候选与反方假说，"
                "它们没有事实权力，可能依赖后来被拒绝或保持未决的 candidate_id。"
                "CANDIDATE_RESOLUTION_MAP 只告诉你这些候选后来怎样裁决；最终主张只能由 CASE_JSON 中"
                "现存的 accepted relation_id、constraint_id 和 piece_id 支撑。\n"
                "先逐一推翻草稿中依赖 rejected 或 unresolved 候选的部分，再从最终关系证书重新绘制唯一主轮廓。"
                "不得把草稿措辞、草稿数量或 candidate_id 当作证据，也不得在最终引用字段中输出 candidate_id。\n"
                "PROVENANCE: cite relation_ids and constraint_ids used for each statement. "
                "Do not copy their endpoint piece_ids; the program inherits and validates "
                "those endpoints automatically. Use extra_piece_ids only for a directly "
                "used piece that is not already reachable through a cited relation or constraint.\n"
                "DRAFT RESOLUTION: accepted_relation_ids are the only draft links that "
                "survived adjudication. discarded_candidates are retained only to explain "
                "which provisional route failed; never restore them as evidence.\n"
                "direct_answer 和 main_contour 必须直接回答本题，并写出具体对象、指标、机制或时间窗口。"
                "这两个字段必须由完整、可独立读懂的中文句子组成；每句话只承担一项可检查判断，"
                "并分别标明 claim_role 与自己的引用。后台会逐句验明证据资格，前台仍按原顺序合成自然段。"
                "最多保留 4 个关键条件、5 个稳定局部、4 个边界和 5 个重要未知。\n"
                "引用任何包含 2 至 4 个端点的 relation_id 时，piece_ids 必须列全该关系的端点；"
                "只使用 person_aliases 给出的姓名。\n"
                + (
                    "本案只有结构关系：直接答案必须保持未定或条件化，不能借草稿补出方向结论。\n"
                    if case.get("inference_capability", {}).get(
                        "structural_only"
                    )
                    else ""
                )
                + "CASE_JSON:\n"
                + _prompt_json(prompt_packet)
                + "\nPROVISIONAL_DRAFTS:\n"
                + _prompt_json(provisional_drafts)
                + "\nCANDIDATE_RESOLUTION_MAP:\n"
                + _prompt_json(resolution_map)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(_compact_final_schema())
            ),
        },
    ]


def _compact_final_repair_messages(
    case: dict[str, Any],
    previous_payload: Any,
    validation_errors: list[str],
) -> list[dict[str, str]]:
    constructive_relations = set(case.get("constructive_relation_ids") or [])
    constructive_constraints = set(
        case.get("constructive_constraint_ids") or []
    )
    repair_packet = {
        "question": case.get("question"),
        "person_aliases": case.get("person_aliases") or {},
        "inference_capability": case.get("inference_capability") or {},
        "specificity_requirements": case.get("specificity_requirements") or {},
        "constructive_relations": [
            _compact_relation(item)
            for item in case.get("accepted_relations", []) or []
            if item.get("relation_id") in constructive_relations
        ],
        "constructive_constraints": [
            {
                key: item.get(key)
                for key in (
                    "constraint_id",
                    "constraint_type",
                    "source_relation_ids",
                    "piece_ids",
                    "rule",
                    "relation_explanation",
                    "strength",
                    "caution",
                )
            }
            for item in case.get("constraints", []) or []
            if item.get("constraint_id") in constructive_constraints
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                FINAL_SYSTEM
                + CLASSIC_CONTOUR_RULES
                + "\n你现在只修复一份已经完成推理但未通过机器校验的轮廓。"
                "不得重新扩展问题、增加事实或改变证明标准。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请针对 VALIDATION_ERRORS 修复 PREVIOUS_OUTPUT，并返回完整 JSON。"
                "只能使用 REPAIR_PACKET 中列出的关系和约束；"
                "direct_answer 与 main_contour 必须直接说本题中的具体对象、定义分支和条件，"
                "每个句子都要有 claim_role，并引用自己的 relation_ids 或 constraint_ids。"
                "不要解释修复过程。\n"
                "VALIDATION_ERRORS:\n"
                + _prompt_json(validation_errors[:12])
                + "\nREPAIR_PACKET:\n"
                + _prompt_json(repair_packet)
                + "\nPREVIOUS_OUTPUT:\n"
                + _prompt_json(previous_payload)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(_compact_final_schema())
            ),
        },
    ]


def _final_messages(
    case: dict[str, Any],
    candidate: dict[str, Any],
    counter: dict[str, Any],
    prompt_packet: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    counter_is_bundle = (
        counter.get("mode") == "parallel_pressure_v1"
        and isinstance(counter.get("pressure_tests"), list)
    )
    return [
        {"role": "system", "content": FINAL_SYSTEM + CLASSIC_CONTOUR_RULES},
        {
            "role": "user",
            "content": (
                "完成最终裁决。可以改写主候选，也可以让反轮廓胜出，但不能把二者含糊拼接。"
                + (
                    "你会看到三路互补的反方压力审查；必须逐路比较，再选出真正最强的替代解释，不能按数量合并。"
                    if counter_is_bundle
                    else ""
                )
                + "why_this_contour 要说明胜出的具体关系理由；counter_contour_disposition 要说明反轮廓被保留、条件化或驳回的原因。\n"
                "direct_answer 和 main_contour 仍必须通过本题具体性门槛，明确写出所引用关系中的具体对象、指标、机制或时间窗口。\n"
                "这两个字段必须由完整、可独立读懂的中文句子组成；每句话只承担一项可检查判断，"
                "并分别标明 claim_role 与自己的引用。后台会逐句验明证据资格，前台仍按原顺序合成自然段。\n"
                "最多保留 4 个关键条件、5 个稳定局部、4 个边界和 5 个重要未知；只留会改变轮廓的项目。\n"
                "引用任何包含 2 至 4 个端点的 relation_id 时，piece_ids 必须列全该关系的端点；"
                "只使用 person_aliases 给出的姓名，不得自行给人物改名。\n"
                + (
                    "本案仍然只有结构关系：最终直接回答必须保持未定或条件化，禁止借候选或反轮廓补出方向结论。\n"
                    if case.get("inference_capability", {}).get("structural_only")
                    else ""
                )
                + "CASE_JSON:\n"
                + _prompt_json(prompt_packet or case)
                + "\nMAIN_CANDIDATE:\n"
                + _prompt_json(candidate)
                + (
                    "\nCOUNTER_PRESSURE_BUNDLE:\n"
                    if counter_is_bundle
                    else "\nSTRONGEST_COUNTER:\n"
                )
                + _prompt_json(counter)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(_final_schema())
            ),
        },
    ]


def _final_schema() -> dict[str, Any]:
    return {
        "direct_answer": _bound_statement_schema(
            "直接回答原问题", "1至3个完整句子"
        ),
        "main_contour": _bound_statement_schema(
            "唯一主轮廓", "2至7个完整句子"
        ),
        "key_conditions": [_statement_schema("条件变化会怎样改写主轮廓")],
        "stable_parts": [_statement_schema("在关键分支下仍较稳定的局部")],
        "boundary_conditions": [_statement_schema("这条轮廓不能越过的范围")],
        "important_unknowns": [_statement_schema("会显著改写结论的未知")],
        "confidence_statement": "置信只说明关系和证据边界，不给伪精确概率",
        "why_this_contour": "它为何比替代轮廓更能解释现有材料",
        "strongest_counter_contour": "最强替代轮廓的大白话摘要",
        "counter_contour_disposition": "替代轮廓被保留、条件化或驳回的原因",
    }


def _statement_schema(description: str) -> dict[str, Any]:
    return {
        "text": description,
        "piece_ids": ["existing piece_id"],
        "relation_ids": ["existing accepted relation_id"],
        "constraint_ids": ["existing constraint_id"],
    }


def _bound_statement_schema(
    description: str, sentence_count: str, *, compact: bool = False
) -> dict[str, Any]:
    sentence = {
        "text": "一个完整、具体、可独立读懂的中文句子",
        "claim_role": (
            "verified_observation|directional_judgment|conditional_inference|"
            "structural_relation|definition_boundary|important_unknown|value_condition"
        ),
        "relation_ids": ["existing accepted relation_id"],
        "constraint_ids": ["existing constraint_id"],
    }
    if compact:
        sentence["extra_piece_ids"] = [
            "only an existing piece_id not inherited through cited IDs"
        ]
    else:
        sentence["piece_ids"] = ["existing piece_id"]
    return {
        "purpose": description,
        "sentence_count": sentence_count,
        "sentences": [sentence],
    }


def _compact_final_schema() -> dict[str, Any]:
    def statement(description: str) -> dict[str, Any]:
        return {
            "text": description,
            "relation_ids": ["existing accepted relation_id"],
            "constraint_ids": ["existing constraint_id"],
            "extra_piece_ids": [
                "only an existing piece_id not inherited through cited IDs"
            ],
        }

    return {
        "direct_answer": _bound_statement_schema(
            "直接回答原问题", "1至3个完整句子", compact=True
        ),
        "main_contour": _bound_statement_schema(
            "唯一主轮廓", "2至7个完整句子", compact=True
        ),
        "key_conditions": [statement("条件变化会怎样改写主轮廓")],
        "stable_parts": [statement("在关键分支下仍较稳定的局部")],
        "boundary_conditions": [statement("这条轮廓不能越过的范围")],
        "important_unknowns": [statement("会显著改写结论的未知")],
        "confidence_statement": "置信只说明关系和证据边界，不给伪精确概率",
        "why_this_contour": "它为何比替代轮廓更能解释现有材料",
        "strongest_counter_contour": "最强替代轮廓的大白话摘要",
        "counter_contour_disposition": "替代轮廓被保留、条件化或驳回的原因",
    }


def _validate_candidate(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    return _sanitize_final_like(payload, case, require_counter_fields=False)


def _validate_final(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    return _sanitize_final_like(payload, case, require_counter_fields=True)


def _validate_compact_final(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Hydrate mechanical provenance before applying the strict validator."""

    return _validate_final(_hydrate_final_provenance(payload, case), case)


def _hydrate_final_provenance(
    payload: Any, case: dict[str, Any]
) -> Any:
    if not isinstance(payload, dict):
        return payload
    relation_map = {
        compact_text(item.get("relation_id")): item
        for item in case.get("accepted_relations", []) or []
        if compact_text(item.get("relation_id"))
    }
    constraint_map = {
        compact_text(item.get("constraint_id")): item
        for item in case.get("constraints", []) or []
        if compact_text(item.get("constraint_id"))
    }

    def hydrate_leaf(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        hydrated = dict(value)
        relation_ids = _raw_id_list(value.get("relation_ids"))
        constraint_ids = _raw_id_list(value.get("constraint_ids"))
        piece_ids = [
            *_raw_id_list(value.get("piece_ids")),
            *_raw_id_list(value.get("extra_piece_ids")),
        ]
        for relation_id in relation_ids:
            piece_ids.extend(
                (relation_map.get(relation_id) or {}).get("piece_ids") or []
            )
        for constraint_id in constraint_ids:
            constraint = constraint_map.get(constraint_id) or {}
            piece_ids.extend(constraint.get("piece_ids") or [])
            for relation_id in constraint.get("source_relation_ids") or []:
                piece_ids.extend(
                    (relation_map.get(str(relation_id)) or {}).get(
                        "piece_ids"
                    )
                    or []
                )
        hydrated["piece_ids"] = list(
            dict.fromkeys(str(item) for item in piece_ids if str(item))
        )
        hydrated["relation_ids"] = relation_ids
        hydrated["constraint_ids"] = constraint_ids
        hydrated.pop("extra_piece_ids", None)
        return hydrated

    def hydrate_statement(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        sentences = value.get("sentences")
        if not isinstance(sentences, list):
            return hydrate_leaf(value)
        hydrated_sentences = [
            hydrate_leaf(item) for item in sentences if isinstance(item, dict)
        ]
        return {
            "sentences": hydrated_sentences,
            "purpose": value.get("purpose"),
        }

    hydrated_payload = dict(payload)
    for field in ("direct_answer", "main_contour"):
        hydrated_payload[field] = hydrate_statement(payload.get(field))
    for field in (
        "key_conditions",
        "stable_parts",
        "boundary_conditions",
        "important_unknowns",
    ):
        values = payload.get(field)
        if isinstance(values, list):
            hydrated_payload[field] = [
                hydrate_statement(item) for item in values
            ]
    return hydrated_payload


def _raw_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item)))


def _sanitize_final_like(
    payload: Any,
    case: dict[str, Any],
    *,
    require_counter_fields: bool,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["output is not an object"]
    errors: list[str] = []
    dropped: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for field in ("direct_answer", "main_contour"):
        statement, statement_errors = _sanitize_statement(
            payload.get(field), case, field
        )
        errors.extend(statement_errors)
        if statement:
            output[field] = statement
            for item in statement.get("sentence_validation_drops", []) or []:
                dropped.append({"field": field, **item})
    for field, limit in (
        ("key_conditions", 5),
        ("stable_parts", 6),
        ("boundary_conditions", 5),
        ("important_unknowns", 6),
    ):
        output[field] = []
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values[:limit], start=1):
            statement, statement_errors = _sanitize_statement(
                item, case, f"{field}[{index}]"
            )
            if statement and not statement_errors:
                output[field].append(statement)
            if statement_errors:
                dropped.append(
                    {
                        "field": field,
                        "index": index,
                        "errors": statement_errors,
                    }
                )
    for field in (
        "confidence_statement",
        "why_this_contour",
        "strongest_counter_contour",
        "counter_contour_disposition",
    ):
        output[field] = _clean_user_text(
            payload.get(field), case.get("person_aliases") or {}
        )
    if "direct_answer" not in output or "main_contour" not in output:
        errors.append("direct_answer and main_contour need valid provenance")
    if require_counter_fields and (
        not output.get("strongest_counter_contour")
        or not output.get("counter_contour_disposition")
    ):
        errors.append("final adjudication did not dispose of the counter contour")
    output = _repair_graph_contradictions(output, case)
    errors.extend(_contour_level_errors(output, case))
    if dropped:
        output["validation_drops"] = dropped
    return (output if not errors else {}), errors


def _repair_graph_contradictions(
    output: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    has_independent_convergence = any(
        item.get("status") == "accepted"
        and item.get("relation_type") == "independent_convergence"
        for item in case.get("accepted_relations", []) or []
    )
    if not has_independent_convergence:
        return output
    replacements = {
        "缺乏独立会合或决定性证据": "现有独立会合不足以决定方向，也缺乏决定性证据",
        "没有独立会合或决定性证据": "现有独立会合不足以决定方向，也缺乏决定性证据",
        "缺乏任何独立会合": "现有独立会合不足以决定方向",
        "不存在独立会合": "现有独立会合不足以决定方向",
    }
    repaired = dict(output)
    for field in ("direct_answer", "main_contour"):
        statement = repaired.get(field)
        if not isinstance(statement, dict):
            continue
        text = compact_text(statement.get("text"))
        for source, target in replacements.items():
            text = text.replace(source, target)
        repaired[field] = {**statement, "text": text}
    return repaired


def _validate_counter(
    payload: Any, case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return {}, ["output is not an object"]
    errors = []
    dropped: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for field in ("counter_contour", "why_it_may_fit_better"):
        statement, statement_errors = _sanitize_statement(
            payload.get(field), case, field
        )
        errors.extend(statement_errors)
        if statement:
            output[field] = statement
    for field, limit in (("attacks", 4), ("decisive_tests", 3)):
        output[field] = []
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values[:limit], start=1):
            statement, statement_errors = _sanitize_statement(
                item, case, f"{field}[{index}]"
            )
            if statement and not statement_errors:
                output[field].append(statement)
            if statement_errors:
                dropped.append(
                    {
                        "field": field,
                        "index": index,
                        "errors": statement_errors,
                    }
                )
    if "counter_contour" not in output or "why_it_may_fit_better" not in output:
        errors.append("counter contour and its comparative reason are required")
    if dropped:
        output["validation_drops"] = dropped
    return (output if not errors else {}), errors


def _sanitize_statement(
    value: Any, case: dict[str, Any], label: str
) -> tuple[dict[str, Any], list[str]]:
    if isinstance(value, dict) and isinstance(value.get("sentences"), list):
        return _sanitize_bound_statement(value, case, label)
    if not isinstance(value, dict):
        return {}, [f"{label} is not a cited statement"]
    text = _clean_user_text(
        value.get("text") or value.get("statement"),
        case.get("person_aliases") or {},
    )
    if not text:
        return {}, [f"{label} text is missing"]
    limits = {
        "direct_answer": 320,
        "main_contour": 900,
    }
    limit = limits.get(label, 360)
    if len(text) > limit:
        return {}, [f"{label} exceeds {limit} characters"]
    allowed = case["allowed_ids"]
    refs = {}
    errors = []
    for field in ("piece_ids", "relation_ids", "constraint_ids"):
        raw_values = value.get(field)
        if not isinstance(raw_values, list):
            raw_values = []
        raw_ids = list(dict.fromkeys(str(item) for item in raw_values))
        valid = [item for item in raw_ids if item in set(allowed[field])]
        invalid = [item for item in raw_ids if item not in set(allowed[field])]
        if invalid:
            errors.append(f"{label} invented {field}: {','.join(invalid[:3])}")
        refs[field] = valid
    if not any(refs.values()):
        errors.append(f"{label} has no valid provenance")
    relation_map = {
        str(item.get("relation_id")): item
        for item in case.get("accepted_relations", [])
        if item.get("relation_id")
    }
    cited_piece_ids = set(refs["piece_ids"])
    for relation_id in refs["relation_ids"]:
        endpoints = set(
            str(item)
            for item in (relation_map.get(relation_id) or {}).get(
                "piece_ids", []
            )
        )
        if 2 <= len(endpoints) <= 4 and not endpoints.issubset(cited_piece_ids):
            missing = sorted(endpoints - cited_piece_ids)
            errors.append(
                f"{label} omits relation endpoint pieces for {relation_id}: "
                + ",".join(missing)
            )
    piece_map = {
        str(item.get("piece_id")): item
        for item in case.get("pieces", []) or []
        if item.get("piece_id")
    }
    cited_people = {
        compact_text((piece_map.get(piece_id) or {}).get("person_id"))
        for piece_id in cited_piece_ids
    }
    people_by_name: defaultdict[str, set[str]] = defaultdict(set)
    for item in piece_map.values():
        name = compact_text(item.get("person_name"))
        person_id = compact_text(item.get("person_id"))
        if len(name) >= 2 and person_id:
            people_by_name[name].add(person_id)
    uncited_names = sorted(
        name
        for name, person_ids in people_by_name.items()
        if name in text and cited_people.isdisjoint(person_ids)
    )
    if uncited_names:
        errors.append(
            f"{label} names people absent from cited pieces: "
            + ",".join(uncited_names[:3])
        )
    if label in {"direct_answer", "main_contour"}:
        supports_contour = bool(
            set(refs["relation_ids"]) & set(case["constructive_relation_ids"])
            or set(refs["constraint_ids"])
            & set(case["constructive_constraint_ids"])
        )
        if not supports_contour:
            errors.append(f"{label} cites no constructive relation or constraint")
    if label.startswith("stable_parts["):
        supports_stable_part = bool(
            set(refs["relation_ids"]) & set(case["constructive_relation_ids"])
            or set(refs["constraint_ids"])
            & set(case["constructive_constraint_ids"])
        )
        if not supports_stable_part:
            errors.append(
                f"{label} cites only unconnected candidate pieces; "
                "a retained local observation is not yet a stable fact"
            )
    unsupported_numbers = _numbers(text) - _numbers(_referenced_text(refs, case))
    if unsupported_numbers:
        errors.append(
            f"{label} adds numbers absent from its cited material: "
            + ",".join(sorted(unsupported_numbers)[:5])
        )
    return {"text": text, **refs}, errors


def _sanitize_bound_statement(
    value: dict[str, Any], case: dict[str, Any], label: str
) -> tuple[dict[str, Any], list[str]]:
    limits = {"direct_answer": 3, "main_contour": 7}
    sentence_limit = limits.get(label, 1)
    accepted: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for index, raw in enumerate(
        (value.get("sentences") or [])[:sentence_limit], start=1
    ):
        if not isinstance(raw, dict):
            drops.append(
                {"sentence_index": index, "errors": ["sentence is not an object"]}
            )
            continue
        claim_role = compact_text(raw.get("claim_role"))
        sentence, errors = _sanitize_statement(
            {key: item for key, item in raw.items() if key != "sentences"},
            case,
            f"{label}.sentence[{index}]",
        )
        if sentence:
            errors.extend(
                _sentence_epistemic_errors(
                    sentence,
                    case,
                    claim_role=claim_role,
                )
            )
        if errors or not sentence:
            drops.append({"sentence_index": index, "errors": errors or ["invalid"]})
            continue
        accepted.append(
            {
                "claim_id": f"{label}_sentence_{index}",
                "claim_role": claim_role,
                **sentence,
            }
        )

    minimum = 1
    if len(accepted) < minimum:
        return {}, [f"{label} has no sentence with admissible evidence"]

    refs = {
        field: list(
            dict.fromkeys(
                item_id
                for sentence in accepted
                for item_id in sentence.get(field, []) or []
            )
        )
        for field in ("piece_ids", "relation_ids", "constraint_ids")
    }
    text = _join_bound_sentences(
        [compact_text(item.get("text")) for item in accepted]
    )
    max_length = 320 if label == "direct_answer" else 900
    if len(text) > max_length:
        return {}, [f"{label} exceeds {max_length} characters after sentence binding"]
    if label in {"direct_answer", "main_contour"}:
        supports_contour = bool(
            set(refs["relation_ids"]) & set(case["constructive_relation_ids"])
            or set(refs["constraint_ids"])
            & set(case["constructive_constraint_ids"])
        )
        if not supports_contour:
            return {}, [f"{label} cites no constructive relation or constraint"]
    return {
        "text": text,
        **refs,
        "sentence_bindings": accepted,
        "sentence_validation_drops": drops,
        "bound_statement_validated": True,
    }, []


def _sentence_epistemic_errors(
    sentence: dict[str, Any],
    case: dict[str, Any],
    *,
    claim_role: str,
) -> list[str]:
    allowed_roles = {
        "verified_observation",
        "directional_judgment",
        "conditional_inference",
        "structural_relation",
        "definition_boundary",
        "important_unknown",
        "value_condition",
    }
    if claim_role not in allowed_roles:
        return [f"unknown claim_role: {claim_role or 'missing'}"]

    text = compact_text(sentence.get("text"))
    relation_ids = set(sentence.get("relation_ids") or [])
    constraint_ids = set(sentence.get("constraint_ids") or [])
    piece_ids = set(sentence.get("piece_ids") or [])
    pieces = {
        str(item.get("piece_id")): item for item in case.get("pieces", []) or []
    }
    relations = {
        str(item.get("relation_id")): item
        for item in case.get("accepted_relations", []) or []
    }
    constraints = {
        str(item.get("constraint_id")): item
        for item in case.get("constraints", []) or []
    }
    statuses = {
        str(status)
        for piece_id in piece_ids
        for status in (pieces.get(piece_id) or {}).get(
            "verification_statuses", []
        )
    }
    verified = bool(
        statuses.intersection(
            {"page_verified", "primary_verified", "cross_verified"}
        )
    )
    constructive = bool(
        relation_ids.intersection(case.get("constructive_relation_ids") or [])
        or constraint_ids.intersection(
            case.get("constructive_constraint_ids") or []
        )
    )
    directional = bool(
        relation_ids.intersection(
            (case.get("inference_capability") or {}).get(
                "directional_support_relation_ids", []
            )
            or []
        )
    )
    relation_types = {
        str((relations.get(relation_id) or {}).get("relation_type"))
        for relation_id in relation_ids
    }
    constraint_types = {
        str((constraints.get(constraint_id) or {}).get("constraint_type"))
        for constraint_id in constraint_ids
    }

    errors: list[str] = []
    if claim_role == "verified_observation" and not verified:
        errors.append("verified_observation lacks page, primary, or cross verification")
    elif claim_role == "directional_judgment" and not directional:
        errors.append("directional_judgment lacks authorized independent support")
    elif claim_role == "conditional_inference":
        if not constructive:
            errors.append("conditional_inference lacks a constructive relation")
        if not _has_condition_language(text):
            errors.append("conditional_inference does not state its condition")
    elif claim_role == "structural_relation":
        if not constructive:
            errors.append("structural_relation lacks a constructive relation")
        if _has_unconditional_direction(text):
            errors.append("structural_relation overstates a world direction")
    elif claim_role == "definition_boundary":
        if "definition_branch" not in relation_types and "definition_branch" not in constraint_types:
            errors.append("definition_boundary lacks a definition branch")
    elif claim_role == "important_unknown" and not _has_unknown_language(text):
        errors.append("important_unknown is written as if it were known")
    elif claim_role == "value_condition":
        if "value_branch" not in relation_types and "value_branch" not in constraint_types:
            errors.append("value_condition lacks a value branch")
    return errors


def _join_bound_sentences(sentences: list[str]) -> str:
    output = []
    for sentence in sentences:
        text = compact_text(sentence)
        if not text:
            continue
        if text[-1:] not in "。！？!?":
            text += "。"
        output.append(text)
    return "".join(output)


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


def _contour_level_errors(
    output: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    errors = _topic_specificity_errors(output, case)
    if not case.get("inference_capability", {}).get("structural_only"):
        return errors
    direct = compact_text((output.get("direct_answer") or {}).get("text"))
    main = compact_text((output.get("main_contour") or {}).get("text"))
    combined = f"{direct} {main}"
    if not has_structural_uncertainty(direct):
        errors.append("structural-only direct_answer must be undetermined or explicitly conditional")
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
    found = [item for item in hard_claims if item in combined]
    if found:
        errors.append(
            "structural relations cannot support directional or universal claims: "
            + ",".join(found)
        )
    if ("不会产生" in direct or "会产生意识" in direct) and not has_structural_uncertainty(
        direct
    ):
        errors.append("structural-only direct_answer makes an unsupported yes/no claim")
    return errors


def _build_topic_anchor_groups(
    relations: list[dict[str, Any]],
    pieces: dict[str, dict[str, Any]],
    constructive_types: set[str],
) -> list[dict[str, Any]]:
    groups = []
    for relation in relations:
        if relation.get("relation_type") not in constructive_types:
            continue
        anchors = _nested_strings(relation.get("shared_coordinate") or {})
        explanation = compact_text(relation.get("plain_language_explanation"))
        if explanation:
            anchors.append(explanation)
        for piece_id in relation.get("piece_ids", []) or []:
            piece_text = compact_text((pieces.get(piece_id) or {}).get("text"))
            if piece_text:
                anchors.append(_limit(piece_text, 180))
        anchors = list(dict.fromkeys(item for item in anchors if len(item) >= 2))[:8]
        if anchors:
            groups.append(
                {
                    "relation_id": relation.get("relation_id"),
                    "relation_type": relation.get("relation_type"),
                    "anchors": anchors,
                }
            )
    return groups


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            text
            for item in value.values()
            for text in _nested_strings(item)
        ]
    if isinstance(value, list):
        return [text for item in value for text in _nested_strings(item)]
    text = compact_text(value)
    return [text] if text else []


def _topic_specificity_errors(
    output: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    requirements = case.get("specificity_requirements") or {}
    groups = requirements.get("relation_anchor_groups") or []
    minimums = requirements.get("minimum_relation_groups") or {}
    if not groups:
        return []
    constraint_relations = {
        item.get("constraint_id"): set(item.get("source_relation_ids") or [])
        for item in case.get("constraints", []) or []
    }
    errors = []
    for field in ("direct_answer", "main_contour"):
        statement = output.get(field) or {}
        text = compact_text(statement.get("text"))
        supported_ids = set(statement.get("relation_ids") or [])
        for constraint_id in statement.get("constraint_ids") or []:
            supported_ids.update(constraint_relations.get(constraint_id, set()))
        matched = [
            group
            for group in groups
            if group.get("relation_id") in supported_ids
            and _anchor_group_matches(text, group.get("anchors") or [])
        ]
        required = int(minimums.get(field, 0) or 0)
        if len(matched) >= required:
            continue
        examples = "；".join(
            f"{group.get('relation_id')}={','.join((group.get('anchors') or [])[:2])}"
            for group in groups[:4]
        )
        errors.append(
            f"{field} is too abstract for this question: matched {len(matched)}/{required} "
            "cited topic-specific relation groups. Name concrete objects, indicators, "
            f"mechanisms, or time windows from cited material. Available anchors: {examples}"
        )
    return errors


def _anchor_group_matches(text: str, anchors: list[str]) -> bool:
    lowered = text.lower()
    if any(
        token.lower() in lowered
        for anchor in anchors
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", anchor)
    ):
        return True
    ignored = {
        "当前", "材料", "判断", "条件", "关系", "问题", "结论", "可能", "不同",
        "来源", "范围", "时间", "价值", "冲突", "结构", "定义", "事实", "证据",
        "模型", "数字", "盲区", "分歧", "共同", "影响", "分析", "指标", "说明",
    }
    informative_bigrams = {
        token
        for anchor in anchors
        for chunk in re.findall(r"[\u4e00-\u9fff]+", anchor)
        for token in (chunk[index : index + 2] for index in range(len(chunk) - 1))
        if token not in ignored and token in text
    }
    if len(informative_bigrams) >= 2:
        return True
    return any(
        chunk[index : index + 4] in text
        for anchor in anchors
        for chunk in re.findall(r"[\u4e00-\u9fff]+", anchor)
        for index in range(max(0, len(chunk) - 3))
    )


def _contains_internal_id(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])(?:piece|relation|constraint)_[0-9a-f]{8,}(?![A-Za-z0-9_])",
            text,
        )
    )


def _clean_user_text(
    value: Any, person_aliases: dict[str, str] | None = None
) -> str:
    text = compact_text(value)
    for alias, name in sorted(
        (person_aliases or {}).items(), key=lambda item: -len(item[0])
    ):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
            name,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(
        r"(?<![A-Za-z0-9_])[Pp]\d{1,3}(?![A-Za-z0-9_])",
        "一位数字人",
        text,
    )
    text = re.sub(
        r"[（(]?(?<![A-Za-z0-9_])(?:piece|relation|constraint)_[0-9a-z]{4,}(?![A-Za-z0-9_])[）)]?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[、，,]{2,}", "，", text)
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"[、，,；;:：]+\s*([。！？!?])", r"\1", text)
    text = re.sub(r"([（(])\s*[、，,；;:\s]*([）)])", "", text)
    text = re.sub(r"\s+([，。；：、,.!?])", r"\1", text)
    text = re.sub(r"([（(])\s*([）)])", "", text)
    return compact_text(text)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?%?", text))


def _referenced_text(refs: dict[str, list[str]], case: dict[str, Any]) -> str:
    pieces = {item.get("piece_id"): item for item in case.get("pieces", [])}
    relations = {
        item.get("relation_id"): item for item in case.get("accepted_relations", [])
    }
    constraints = {
        item.get("constraint_id"): item for item in case.get("constraints", [])
    }
    values = []
    for piece_id in refs.get("piece_ids", []):
        item = pieces.get(piece_id) or {}
        coordinates = item.get("coordinates") or {}
        values.extend(
            [
                item.get("text", ""),
                item.get("parent_conclusion", ""),
                coordinates.get("scope", ""),
                coordinates.get("time_horizon", ""),
                coordinates.get("direction", ""),
            ]
        )
    for relation_id in refs.get("relation_ids", []):
        item = relations.get(relation_id) or {}
        values.append(item.get("plain_language_explanation", ""))
    for constraint_id in refs.get("constraint_ids", []):
        item = constraints.get(constraint_id) or {}
        values.extend([item.get("rule", ""), item.get("relation_explanation", "")])
    return " ".join(str(item) for item in values if item)


def _call_validated_stage(
    client_factory: Callable[[], DeepSeekClient],
    messages: list[dict[str, str]],
    *,
    stage: str,
    validator: Callable[[Any], tuple[dict[str, Any], list[str]]],
    max_tokens: int,
    retries: int,
    thinking_mode: str,
    input_fingerprint: str = "",
    repair_messages_factory: Callable[
        [Any, list[str]], list[dict[str, str]]
    ]
    | None = None,
    repair_max_tokens: int = 4096,
    repair_thinking_mode: str = "disabled",
) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = _empty_usage()
    errors: list[str] = []
    last_raw: dict[str, Any] = {}
    started = time.perf_counter()
    attempts = 0
    retryable = True
    current_messages = list(messages)
    current_max_tokens = max_tokens
    current_thinking_mode = thinking_mode
    repair_attempted = False
    attempt_modes: list[str] = []
    for attempt in range(retries + 1):
        attempts = attempt + 1
        attempt_modes.append(
            "targeted_repair"
            if repair_attempted
            else (
                "validated_fast_generation"
                if current_thinking_mode == "disabled"
                else "full_reasoning"
            )
        )
        try:
            response = client_factory().chat(
                current_messages,
                temperature=0.15,
                max_tokens=current_max_tokens,
                response_format={"type": "json_object"},
                thinking={"type": current_thinking_mode},
                reasoning_effort=(
                    "high" if current_thinking_mode == "enabled" else None
                ),
            )
            usage = _merge_usage(usage, response.usage)
            last_raw = response.raw
            finish_reason = compact_text(
                (((response.raw.get("choices") or [{}])[0]) or {}).get(
                    "finish_reason"
                )
            )
            if finish_reason == "length":
                errors.append(
                    "generation stopped at max_tokens before a complete contour output"
                )
                break
            parsed = parse_json_content(response.content)
            validated, validation_errors = validator(parsed)
            if validated:
                return validated, {
                    "stage": stage,
                    "status": "succeeded",
                    "attempts": attempts,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "errors": errors,
                    "usage": usage,
                    "raw": last_raw,
                    "thinking_mode": thinking_mode,
                    "repair_attempted": repair_attempted,
                    "attempt_modes": attempt_modes,
                    "input_fingerprint": input_fingerprint,
                }
            errors.extend(validation_errors)
            if (
                repair_messages_factory is not None
                and not repair_attempted
                and attempt < retries
            ):
                current_messages = repair_messages_factory(
                    parsed, validation_errors
                )
                current_max_tokens = max(1024, int(repair_max_tokens))
                current_thinking_mode = repair_thinking_mode
                repair_attempted = True
            else:
                current_messages = list(messages) + [
                    {
                        "role": "assistant",
                        "content": json.dumps(parsed, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": "上一版未通过来源锁定校验，请修复："
                        + "；".join(validation_errors[:8]),
                    },
                ]
                current_max_tokens = max_tokens
                current_thinking_mode = thinking_mode
        except DeepSeekTimeoutError as exc:
            errors.append(str(exc))
            retryable = False
            break
        except DeepSeekAPIError as exc:
            errors.append(str(exc))
            if not exc.retryable:
                retryable = False
                break
            if attempt < retries:
                time.sleep(retry_delay_seconds(exc, attempt))
                continue
        except Exception as exc:  # noqa: BLE001 - deterministic fallback remains valid.
            errors.append(str(exc))
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    return {}, {
        "stage": stage,
        "status": "failed",
        "attempts": attempts,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
        "usage": usage,
        "raw": last_raw,
        "thinking_mode": thinking_mode,
        "repair_attempted": repair_attempted,
        "attempt_modes": attempt_modes,
        "retryable": retryable,
        "input_fingerprint": input_fingerprint,
    }


def _compact_piece(piece: dict[str, Any]) -> dict[str, Any]:
    independence = piece.get("independence_profile") or {}
    return {
        "piece_id": piece.get("piece_id"),
        "person_id": piece.get("person_id"),
        "person_name": re.sub(
            r"#\d+$", "", compact_text(piece.get("person_name"))
        ),
        "piece_kind": piece.get("piece_kind"),
        "text": _limit(piece.get("text"), 320),
        "content_status": piece.get("content_status"),
        "coordinates": piece.get("coordinates") or {},
        "source_families": piece.get("source_families") or [],
        "verification_statuses": independence.get("verification_statuses") or [],
        "parent_conclusion": _limit(piece.get("parent_conclusion"), 220),
        "classic_knowledge_trace": _compact_knowledge_provenance(
            piece.get("knowledge_trace") or {}
        ),
    }


def _person_aliases(
    pieces: dict[str, dict[str, Any]],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for piece in pieces.values():
        person_id = compact_text(piece.get("person_id"))
        person_name = re.sub(
            r"#\d+$", "", compact_text(piece.get("person_name"))
        )
        match = re.search(r"_p(\d+)$", person_id, flags=re.IGNORECASE)
        if match and person_name:
            aliases[f"P{int(match.group(1)):02d}"] = person_name
    return aliases


def _compact_relation(relation: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": relation.get("relation_id"),
        "relation_type": relation.get("relation_type"),
        "piece_ids": relation.get("piece_ids") or [],
        "plain_language_explanation": relation.get(
            "plain_language_explanation"
        ),
        "shared_coordinate": relation.get("shared_coordinate") or {},
        "source_independence": relation.get("source_independence"),
        "common_error_risk": relation.get("common_error_risk"),
        "competing_explanation": relation.get("competing_explanation"),
        "falsification_test": relation.get("falsification_test"),
        "knowledge_provenance": _compact_knowledge_provenance(
            relation.get("knowledge_provenance") or {}
        ),
    }


def _compact_knowledge_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "classic_trace_ids": payload.get("classic_trace_ids") or [],
        "connection_hypothesis_ids": payload.get(
            "connection_hypothesis_ids"
        )
        or [],
        "inversion_certificate_ids": payload.get(
            "inversion_certificate_ids"
        )
        or [],
        "inversion_authorizations": payload.get(
            "inversion_authorizations"
        )
        or (
            [payload.get("inversion_authorization")]
            if payload.get("inversion_authorization")
            else []
        ),
        "grounding_status": payload.get("grounding_status"),
        "epistemic_role": payload.get("epistemic_role"),
        "classic_as_world_evidence": False,
    }


def _skipped_stage(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "skipped",
        "attempts": 0,
        "errors": [],
        "usage": _empty_usage(),
    }


def _limit(value: Any, size: int) -> str:
    text = compact_text(value)
    return text if len(text) <= size else text[: size - 1] + "…"


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    prompt = int(left.get("prompt_tokens") or 0) + int(
        right.get("prompt_tokens") or right.get("input_tokens") or 0
    )
    completion = int(left.get("completion_tokens") or 0) + int(
        right.get("completion_tokens") or right.get("output_tokens") or 0
    )
    total = int(left.get("total_tokens") or 0) + int(right.get("total_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
    }
