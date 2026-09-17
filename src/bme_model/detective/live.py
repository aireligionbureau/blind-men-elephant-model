from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ..json_utils import parse_json_content
from ..providers import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekTimeoutError,
    retry_delay_seconds,
    synthesis_request_timeout_seconds,
)
from .classic_connectors import select_connection_hypotheses
from .mechanisms import mechanism_rarity
from .relations import build_detective_output
from .schemas import RELATION_TYPES, compact_text, stable_id


CONTENT_TYPES = {
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
    "non_comparable",
}

SHADOW_TYPES = {
    "correlated_error",
    "opposite_distortion",
    "blind_spot_fill",
    "collective_blind_spot",
    "diagnostic_challenge",
    "shared_assumption",
    "definition_branch",
    "value_branch",
    "non_comparable",
}

DETECTIVE_PIPELINE_VERSION = "bme.detective-live.v13"
DETECTIVE_PROMPT_PACKET_SCHEMA = "bme.detective-prompt-packet.v1"
PROPOSAL_PROMPT_REVISION = "bme.detective.proposal.v4"
ADJUDICATION_PROMPT_REVISION = "bme.detective.adjudication.v4"
CLASSIC_REVIEW_PROMPT_REVISION = "bme.detective.classic-review.v2"
PROPOSAL_CACHE_COMPATIBLE_VERSIONS = {DETECTIVE_PIPELINE_VERSION}
MAX_NATIVE_PROPOSALS = 12
MAX_CLASSIC_ASSISTED_PROPOSALS = 4
MAX_CLASSIC_MATERIAL_ADDITIONS = 12
MAX_CLASSIC_CHECKS_PER_CANDIDATE = 1
GROUPED_PROPOSAL_MIN_PIECES = 64

PROPOSAL_RELATION_LANES = {
    "content": (
        (
            "evidence_and_mechanism",
            frozenset(
                {
                    "independent_convergence",
                    "direct_conflict",
                    "apparent_conflict",
                    "conditional_complement",
                    "scope_refinement",
                    "causal_relay",
                    "blind_spot_fill",
                    "non_comparable",
                }
            ),
            9,
            3,
        ),
        (
            "frames_and_values",
            frozenset(
                {
                    "shared_assumption",
                    "definition_branch",
                    "value_branch",
                }
            ),
            3,
            1,
        ),
    ),
    "shadow": (
        (
            "distortion_and_challenge",
            frozenset(
                {
                    "correlated_error",
                    "opposite_distortion",
                    "diagnostic_challenge",
                    "non_comparable",
                }
            ),
            5,
            2,
        ),
        (
            "blind_spots_and_frames",
            frozenset(
                {
                    "blind_spot_fill",
                    "collective_blind_spot",
                    "shared_assumption",
                    "definition_branch",
                    "value_branch",
                }
            ),
            7,
            2,
        ),
    ),
}


DETECTIVE_SYSTEM = """你是盲人摸象模型的侦探层。法医层已经逐人检查完整思考链；你只负责让这些诊断材料彼此发生关系。

你的工作不是投票、总结观点或直接回答问题，而是建立可审计的跨人连接：哪些局部观察独立会合，哪些只是同源重复，哪些表面冲突来自定义、范围、时间、条件或价值不同，哪些错误彼此强化，哪些盲区被另一人的材料填补。

硬规则：
1. 人数、声量和相同结论都不是事实权重。同方向不自动算印证，相反方向不自动算冲突。
2. 只有在同一命题、同一定义、同一范围和兼容时间下，才可判断支持或冲突。
3. 独立性必须拆开判断：共同基础模型表示解释过程相关，不自动等于外部证据同源；相同 URL、相同底层发布者或转述链表示证据相关。搜索引擎只是发现路径，不是证据来源。
4. 外部证据家族彼此独立时，可以记录“证据独立、解释相关”；但未核验网页只能形成待核验会合，不能直接支撑方向性答案。
5. 法医层保留的局部内容仍只是候选材料，不是已经证明的事实。
6. 共同沉默只能成为待核验盲区，不能自动反演成与沉默相反的事实。
7. 未校准的相反偏差只能形成方向提示，禁止计算中点、概率或偏差大小。
8. 每条连接必须引用输入中真实存在的 piece_id，并用大白话说清这几块材料究竟怎样连接。
9. 宁可没有连接，也不要为了图看起来完整而硬连。
10. 不得引入输入以外的事实、数字、来源或人物。
只输出 JSON，不要输出 Markdown。"""


ADJUDICATOR_SYSTEM = """你是盲人摸象模型侦探层的对抗裁决员。另一模型提出了跨数字人关系候选，你必须逐条尝试推翻它们。

裁决重点：是否真在比较同一命题；定义、范围、时间和条件是否兼容；模型、外部证据和发布者三条血缘分别怎样；相似判断是否只是共同模型解释或同源材料；所谓盲区是否与问题直接相关；所谓冲突是否其实可以条件化；所谓互补是否只是把两句不相干的话拼在一起。

accepted 不是“我觉得有道理”，而是当前材料足以证明这条关系存在。accepted 必须同时给出最强竞争解释和可以撤销这条关系的检验。证据不足用 unresolved；关系被材料否定用 rejected。不得新增候选、piece_id 或外部事实。只输出 JSON。"""


CLASSIC_REVIEW_SYSTEM = """你是盲人摸象模型的经典工具复核员。侦探已经在完全看不到经典知识的情况下完成关系裁决；你无权修改 accepted、rejected、unresolved、置信度、关系解释或任何现实判断。

你只复核一件事：指定的经典连接工具是否准确解释了现有材料关系的诊断来路。validated 必须逐项满足 proof_obligations 且不触犯 forbidden_inferences；否则 rejected。经典不是现实证据，不能补写事实，也不能因为工具著名而通过。只输出 JSON。"""


CLASSIC_TRACE_RULES = """
经典知识轨迹是法医层使用诊断工具留下的审计记录。它只能帮助你理解一块阴影为何形成、适用边界在哪里、允许怎样反演；不能证明现实问题中的事实。
硬规则：
1. 不得因为两块材料使用同一经典透镜，就判定它们彼此支持、彼此冲突或形成独立会合。
2. classic_as_world_evidence 必须始终为 false；经典著作、透镜和机制名称不能充当外部来源。
3. inversion_authorization=blocked 的材料只能登记诊断缺口；restricted 只能形成边界、挑战或待核验连接；provisional 仍须通过关系证书裁决。
4. differential_statuses 含 active_competition_unresolved 时，必须保留误诊可能，不得把主透镜解释写成唯一原因。
5. 关系解释必须说清具体材料如何连接；经典机制只能解释这条连接的诊断来路。
6. 经典是查漏和反演助手，不是侦探的上级。经典连接没有候选优先权、证据权重加成或裁决优先级；没有经典轨迹但材料充分的原生连接，必须按完全相同的证明标准保留。
"""


def run_live_detective(
    question: str,
    materials: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 8192,
    retries: int = 2,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    cached_record: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    proposal_ready_callback: Callable[
        [dict[str, Any], list[dict[str, Any]]], None
    ]
    | None = None,
    execution_mode: str = "optimized_v2",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Propose and adversarially adjudicate cross-person relations."""

    if execution_mode not in {"legacy_sequential", "optimized_v2"}:
        raise ValueError(
            "execution_mode must be 'legacy_sequential' or 'optimized_v2'"
        )
    started = time.perf_counter()
    factory = client_factory or (
        lambda: DeepSeekClient(
            model=model,
            timeout_seconds=synthesis_request_timeout_seconds(),
        )
    )
    case = build_detective_case(question, materials)
    usage = _empty_usage()
    stage_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    proposal_specs = (
        ("content", CONTENT_TYPES),
        ("shadow", SHADOW_TYPES),
    )
    material_mode = (
        "legacy_full_case"
        if execution_mode == "legacy_sequential"
        else "lossless_packet_v2"
    )
    proposal_bundles = {
        mode: _proposal_prompt_bundle(
            case,
            mode,
            allowed_types,
            material_mode=material_mode,
        )
        for mode, allowed_types in proposal_specs
    }
    proposal_results: dict[
        str, tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    grouped_proposals = (
        execution_mode == "optimized_v2"
        and len(case.get("piece_index") or []) >= GROUPED_PROPOSAL_MIN_PIECES
    )
    if execution_mode == "legacy_sequential":
        for mode, allowed_types in proposal_specs:
            stage_name = f"{mode}_proposal"
            bundle = proposal_bundles[mode]
            payload = _cached_proposal_payload(
                cached_record or {},
                stage_name,
                mode,
                input_fingerprint=bundle["input_fingerprint"],
            )
            cached_errors = (
                _proposal_payload_errors(payload, case, allowed_types)
                if payload
                else ["missing"]
            )
            if payload and not cached_errors:
                proposal_results[mode] = (
                    payload,
                    {
                        "stage": stage_name,
                        "status": "reused",
                        "attempts": 0,
                        "duration_seconds": 0.0,
                        "errors": [],
                        "usage": _empty_usage(),
                        "raw": _cached_stage_raw(
                            cached_record or {}, stage_name
                        ),
                        "cache_source": "validated_previous_stage_raw",
                        "input_fingerprint": bundle[
                            "input_fingerprint"
                        ],
                        "prompt_packet": bundle["packet_stats"],
                    },
                )
            else:
                proposal_results[mode] = _call_json_stage(
                    factory,
                    bundle["messages"],
                    stage=stage_name,
                    max_tokens=max_tokens,
                    retries=retries,
                    thinking_mode="disabled",
                    validator=lambda value, allowed=allowed_types: _proposal_payload_errors(
                        value, case, allowed
                    ),
                    input_fingerprint=bundle["input_fingerprint"],
                    prompt_packet_stats=bundle["packet_stats"],
                )
            _emit_partial_proposal_checkpoint(
                checkpoint_callback,
                model=model,
                case=case,
                proposal_specs=proposal_specs,
                proposal_results=proposal_results,
            )
    else:
        proposal_futures = {}
        with ThreadPoolExecutor(max_workers=2) as proposal_executor:
            for mode, allowed_types in proposal_specs:
                stage_name = f"{mode}_proposal"
                bundle = proposal_bundles[mode]
                payload = _cached_proposal_payload(
                    cached_record or {},
                    stage_name,
                    mode,
                    input_fingerprint=bundle["input_fingerprint"],
                )
                cached_errors = (
                    _proposal_payload_errors(payload, case, allowed_types)
                    if payload
                    else ["missing"]
                )
                if payload and not cached_errors:
                    proposal_results[mode] = (
                        payload,
                        {
                            "stage": stage_name,
                            "status": "reused",
                            "attempts": 0,
                            "duration_seconds": 0.0,
                            "errors": [],
                            "usage": _empty_usage(),
                            "raw": _cached_stage_raw(
                                cached_record or {}, stage_name
                            ),
                            "cache_source": "validated_previous_stage_raw",
                            "input_fingerprint": bundle[
                                "input_fingerprint"
                            ],
                            "prompt_packet": bundle["packet_stats"],
                        },
                    )
                    continue
                if grouped_proposals:
                    future = proposal_executor.submit(
                        _call_grouped_proposal,
                        factory,
                        case,
                        mode,
                        allowed_types,
                        max_tokens=max_tokens,
                        retries=retries,
                        material_mode=material_mode,
                        parent_input_fingerprint=bundle[
                            "input_fingerprint"
                        ],
                    )
                else:
                    future = proposal_executor.submit(
                        _call_json_stage,
                        factory,
                        bundle["messages"],
                        stage=stage_name,
                        max_tokens=max_tokens,
                        retries=retries,
                        thinking_mode="disabled",
                        validator=lambda value, allowed=allowed_types: _proposal_payload_errors(
                            value, case, allowed
                        ),
                        input_fingerprint=bundle["input_fingerprint"],
                        prompt_packet_stats=bundle["packet_stats"],
                    )
                proposal_futures[future] = mode

            for future in as_completed(proposal_futures):
                proposal_results[proposal_futures[future]] = future.result()
                _emit_partial_proposal_checkpoint(
                    checkpoint_callback,
                    model=model,
                    case=case,
                    proposal_specs=proposal_specs,
                    proposal_results=proposal_results,
                )

    for mode, allowed_types in proposal_specs:
        payload, record = proposal_results[mode]
        usage = _merge_usage(usage, record.get("usage") or {})
        stage_records.append(record)
        candidates.extend(
            _normalize_proposals(
                payload,
                case,
                mode=mode,
                allowed_types=allowed_types,
            )
        )
    _emit_detective_checkpoint(
        checkpoint_callback,
        model=model,
        case=case,
        candidates=candidates,
        relations=[],
        stage_records=stage_records,
        usage=usage,
    )
    candidates = _deduplicate_candidates(candidates)
    for candidate in candidates:
        candidate["classic_check_hypothesis_ids"] = (
            _candidate_classic_check_ids(candidate, case)
        )
    if proposal_ready_callback:
        proposal_ready_callback(case, candidates)

    if candidates:
        adjudication_bundle = _adjudication_prompt_bundle(
            case,
            candidates,
            material_mode=material_mode,
        )
        adjudication_payload = _cached_stage_payload(
            cached_record or {},
            "adversarial_adjudication",
            input_fingerprint=adjudication_bundle["input_fingerprint"],
        )
        adjudication_errors = (
            _adjudication_payload_errors(adjudication_payload, candidates)
            if adjudication_payload
            else ["missing"]
        )
        if adjudication_payload and not adjudication_errors:
            adjudication_record = _reused_stage_record(
                "adversarial_adjudication",
                _cached_stage_raw(cached_record or {}, "adversarial_adjudication"),
                input_fingerprint=adjudication_bundle[
                    "input_fingerprint"
                ],
                prompt_packet_stats=adjudication_bundle[
                    "packet_stats"
                ],
            )
        else:
            if execution_mode == "legacy_sequential":
                adjudication_payload, adjudication_record = _call_json_stage(
                    factory,
                    adjudication_bundle["messages"],
                    stage="adversarial_adjudication",
                    max_tokens=max(max_tokens, 16384),
                    retries=retries,
                    thinking_mode="enabled",
                    validator=lambda value: _adjudication_payload_errors(
                        value, candidates
                    ),
                    input_fingerprint=adjudication_bundle[
                        "input_fingerprint"
                    ],
                    prompt_packet_stats=adjudication_bundle[
                        "packet_stats"
                    ],
                )
            else:
                adjudication_payload, adjudication_record = _call_sharded_adjudication(
                    factory,
                    case,
                    candidates,
                    max_tokens=max_tokens,
                    retries=retries,
                    material_mode=material_mode,
                    parent_input_fingerprint=adjudication_bundle[
                        "input_fingerprint"
                    ],
                )
        usage = _merge_usage(usage, adjudication_record.get("usage") or {})
        stage_records.append(adjudication_record)
        relations = _adjudicate_candidates(adjudication_payload, candidates)
    else:
        relations = []
        stage_records.append(
            {
                "stage": "adversarial_adjudication",
                "status": "not_needed",
                "attempts": 0,
                "usage": _empty_usage(),
                "errors": [],
            }
        )
    _emit_detective_checkpoint(
        checkpoint_callback,
        model=model,
        case=case,
        candidates=candidates,
        relations=relations,
        stage_records=stage_records,
        usage=usage,
    )

    if any(item.get("classic_check_hypothesis_ids") for item in candidates):
        classic_review_bundle = _classic_review_prompt_bundle(
            case,
            candidates,
            relations,
            material_mode=material_mode,
        )
        classic_review_payload = _cached_stage_payload(
            cached_record or {},
            "classic_post_review",
            input_fingerprint=classic_review_bundle[
                "input_fingerprint"
            ],
        )
        classic_review_errors = (
            _classic_review_payload_errors(classic_review_payload, candidates)
            if classic_review_payload
            else ["missing"]
        )
        if classic_review_payload and not classic_review_errors:
            classic_review_record = _reused_stage_record(
                "classic_post_review",
                _cached_stage_raw(cached_record or {}, "classic_post_review"),
                input_fingerprint=classic_review_bundle[
                    "input_fingerprint"
                ],
                prompt_packet_stats=classic_review_bundle[
                    "packet_stats"
                ],
            )
        else:
            classic_review_payload, classic_review_record = _call_json_stage(
                factory,
                classic_review_bundle["messages"],
                stage="classic_post_review",
                max_tokens=max(2048, min(max_tokens, 4096)),
                retries=retries,
                thinking_mode="disabled",
                validator=lambda value: _classic_review_payload_errors(
                    value, candidates
                ),
                input_fingerprint=classic_review_bundle[
                    "input_fingerprint"
                ],
                prompt_packet_stats=classic_review_bundle[
                    "packet_stats"
                ],
            )
        usage = _merge_usage(
            usage, classic_review_record.get("usage") or {}
        )
        stage_records.append(classic_review_record)
        relations = _apply_classic_reviews(
            relations, candidates, classic_review_payload
        )
    else:
        stage_records.append(
            {
                "stage": "classic_post_review",
                "status": "not_needed",
                "attempts": 0,
                "usage": _empty_usage(),
                "errors": [],
            }
        )

    counts = Counter(item.get("status") for item in relations)
    packet_records = [
        item.get("prompt_packet") or {}
        for item in stage_records
        if (item.get("prompt_packet") or {}).get("coverage_validation")
        == "passed"
    ]
    prompt_packet_audit = _aggregate_packet_stats(
        packet_records,
        stage="detective_pipeline",
        material_mode=material_mode,
    )
    stage_statuses = [item.get("status") for item in stage_records]
    overall_status = (
        "succeeded"
        if stage_statuses
        and all(
            item
            in {
                "succeeded",
                "succeeded_truncated_recovery",
                "reused",
                "not_needed",
            }
            for item in stage_statuses
        )
        else "partial"
    )
    audit = {
        "mode": "live_proposal_and_adversarial_adjudication",
        "execution_mode": execution_mode,
        "proposal_execution": (
            "relation_family_grouped_parallel"
            if grouped_proposals
            else "mode_parallel"
        ),
        "status": overall_status,
        "model": model,
        "selected_piece_count": len(case["piece_index"]),
        "total_piece_count": len(materials.get("pieces", []) or []),
        "candidate_relation_count": len(candidates),
        "classic_connection_hypothesis_count": len(
            case.get("classic_connection_hypotheses") or []
        ),
        "classic_post_check_hypothesis_count": len(
            case.get("classic_post_check_hypotheses") or []
        ),
        "native_candidate_count": sum(
            "native" in (item.get("discovery_routes") or [])
            for item in candidates
        ),
        "classic_assisted_candidate_count": sum(
            "classic_assisted" in (item.get("discovery_routes") or [])
            for item in candidates
        ),
        "classic_check_candidate_count": sum(
            bool(item.get("classic_check_hypothesis_ids"))
            for item in candidates
        ),
        "dual_route_candidate_count": sum(
            set(item.get("discovery_routes") or [])
            == {"native", "classic_assisted"}
            for item in candidates
        ),
        "classic_guided_candidate_count": sum(
            bool(item.get("connection_hypothesis_ids")) for item in candidates
        ),
        "rejected_classic_hypothesis_reference_count": sum(
            len(item.get("rejected_connection_hypothesis_ids") or [])
            for item in candidates
        ),
        "classic_guided_accepted_count": sum(
            item.get("status") == "accepted"
            and bool(
                (item.get("provenance") or {}).get(
                    "connection_hypothesis_ids"
                )
            )
            for item in relations
        ),
        "classic_discovered_accepted_count": sum(
            item.get("status") == "accepted"
            and bool(
                (item.get("provenance") or {}).get(
                    "classic_discovery_hypothesis_ids"
                )
            )
            for item in relations
        ),
        "classic_check_validated_count": sum(
            bool(
                (item.get("provenance") or {}).get(
                    "connection_hypothesis_ids"
                )
            )
            for item in relations
        ),
        "classic_check_rejected_count": sum(
            bool(
                (item.get("provenance") or {}).get(
                    "rejected_connection_hypothesis_ids"
                )
            )
            for item in relations
        ),
        "classic_check_unreviewed_count": sum(
            bool(
                (item.get("provenance") or {}).get(
                    "unreviewed_connection_hypothesis_ids"
                )
            )
            for item in relations
        ),
        "native_accepted_count": sum(
            item.get("status") == "accepted"
            and "native"
            in ((item.get("provenance") or {}).get("discovery_routes") or [])
            for item in relations
        ),
        "classic_assisted_accepted_count": sum(
            item.get("status") == "accepted"
            and "classic_assisted"
            in ((item.get("provenance") or {}).get("discovery_routes") or [])
            for item in relations
        ),
        "classic_truth_weight_bonus": False,
        "accepted_relation_count": counts.get("accepted", 0),
        "rejected_relation_count": counts.get("rejected", 0),
        "unresolved_relation_count": counts.get("unresolved", 0),
        "forced_relation_count": 0,
        "prompt_packet": prompt_packet_audit,
        "stages": stage_records,
        "usage": usage,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    detective = build_detective_output(
        question,
        materials,
        semantic_relations=relations,
        semantic_audit=audit,
    )
    record = {
        "pipeline_version": DETECTIVE_PIPELINE_VERSION,
        "execution_mode": execution_mode,
        "status": overall_status,
        "model": model,
        "case": case,
        "candidate_relations": candidates,
        "adjudicated_relations": relations,
        "audit": audit,
        "usage": usage,
    }
    if checkpoint_callback:
        checkpoint_callback(record)
    return detective, record


def _emit_detective_checkpoint(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    model: str,
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    stage_records: list[dict[str, Any]],
    usage: dict[str, Any],
) -> None:
    if not callback:
        return
    callback(
        {
            "pipeline_version": DETECTIVE_PIPELINE_VERSION,
            "status": "running",
            "model": model,
            "case": case,
            "candidate_relations": candidates,
            "adjudicated_relations": relations,
            "audit": {
                "status": "running",
                "model": model,
                "stages": stage_records,
                "usage": usage,
            },
            "usage": usage,
        }
    )


def _emit_partial_proposal_checkpoint(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    model: str,
    case: dict[str, Any],
    proposal_specs: tuple[tuple[str, set[str]], ...],
    proposal_results: dict[
        str, tuple[dict[str, Any], dict[str, Any]]
    ],
) -> None:
    if not callback:
        return
    candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    usage = _empty_usage()
    for mode, allowed_types in proposal_specs:
        if mode not in proposal_results:
            continue
        payload, record = proposal_results[mode]
        records.append(record)
        usage = _merge_usage(usage, record.get("usage") or {})
        candidates.extend(
            _normalize_proposals(
                payload,
                case,
                mode=mode,
                allowed_types=allowed_types,
            )
        )
    _emit_detective_checkpoint(
        callback,
        model=model,
        case=case,
        candidates=candidates,
        relations=[],
        stage_records=records,
        usage=usage,
    )


def _reused_stage_record(
    stage: str,
    raw: dict[str, Any],
    *,
    input_fingerprint: str = "",
    prompt_packet_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "stage": stage,
        "status": "reused",
        "attempts": 0,
        "duration_seconds": 0.0,
        "errors": [],
        "usage": _empty_usage(),
        "raw": raw,
        "cache_source": "validated_previous_stage_raw",
        "input_fingerprint": input_fingerprint,
    }
    if prompt_packet_stats:
        record["prompt_packet"] = prompt_packet_stats
    return record


def build_detective_case(
    question: str,
    materials: dict[str, Any],
) -> dict[str, Any]:
    selected = _select_pieces(materials.get("pieces", []) or [])
    mechanism_index = _compact_mechanism_index(
        materials.get("mechanism_index") or {}
    )
    selected_piece_ids = {
        item.get("piece_id") for item in selected if item.get("piece_id")
    }
    connection_hypotheses = [
        _compact_connection_hypothesis(item)
        for item in select_connection_hypotheses(
            materials.get("classic_connection_hypotheses") or {},
            selected_piece_ids,
            max_items=20,
        )
    ]
    post_check_hypotheses = [
        _compact_connection_hypothesis(item)
        for item in (
            materials.get("classic_connection_hypotheses") or {}
        ).get("hypotheses", [])
        or []
        if item.get("source_piece_id") in selected_piece_ids
    ]
    return {
        "pipeline_version": DETECTIVE_PIPELINE_VERSION,
        "question": compact_text(question),
        "question_id": materials["question_id"],
        "piece_index": [_compact_piece(item) for item in selected],
        "source_dependency_clusters": _source_clusters(selected),
        "source_lineage_rules": {
            "digital_people_share_base_model_prior": True,
            "shared_model_means_interpretation_correlated_not_evidence_shared": True,
            "equal_lineage_ids_mean_same_tracked_family": True,
        },
        "mechanism_index": mechanism_index,
        "classic_connection_hypotheses": connection_hypotheses,
        "classic_post_check_hypotheses": post_check_hypotheses,
        "selection_audit": {
            "total_piece_count": len(materials.get("pieces", []) or []),
            "selected_piece_count": len(selected),
            "rule": "总量不增加；每人仍保留关键链条，但同类材料优先选择稀缺机制，原始材料不被修改。",
            "adds_model_calls": False,
            "native_detective_remains_primary": True,
            "classic_assistance_is_additive": True,
            "classic_truth_weight_bonus": False,
            "classic_discovery_hypothesis_count": len(
                connection_hypotheses
            ),
            "classic_post_check_hypothesis_count": len(
                post_check_hypotheses
            ),
        },
    }


def _proposal_case_payload(
    case: dict[str, Any], mode: str, allowed_types: set[str]
) -> tuple[str, dict[str, Any]]:
    if mode == "content":
        task = (
            "只寻找内容关系：同一命题上的独立会合、真实冲突、表面冲突、条件互补、范围修正、"
            "因果接力、共同假设、定义分支、价值分支、盲区填补或不可比较。"
        )
        kinds = {
            "observed_claim",
            "retained_local",
            "evidence_claim",
            "assumption",
            "value_condition",
            "fracture",
            "blind_spot_candidate",
        }
    else:
        task = (
            "只寻找阴影关系：相同错误机制形成的相关错误、方向相反但未校准的扭曲、"
            "一人的材料怎样填补另一人的盲区、真正的集体盲区、诊断之间的挑战，以及由共同假设、定义或价值造成的结构。"
        )
        kinds = {
            "observed_claim",
            "retained_local",
            "distortion",
            "fracture",
            "blind_spot_candidate",
            "assumption",
            "value_condition",
        }
    relevant_hypotheses = [
        item
        for item in case.get("classic_connection_hypotheses", []) or []
        if set(item.get("allowed_relation_types") or []) & allowed_types
    ]
    preferred_piece_ids = {
        item.get("source_piece_id")
        for item in relevant_hypotheses
        if item.get("source_piece_id")
    }
    preferred_piece_ids.update(
        piece_id
        for item in relevant_hypotheses
        for piece_id in item.get("candidate_partner_piece_ids", []) or []
    )
    pieces = _proposal_piece_selection(
        case["piece_index"],
        kinds,
        mode=mode,
        preferred_piece_ids=preferred_piece_ids,
        classic_source_piece_ids={
            item.get("source_piece_id")
            for item in relevant_hypotheses
            if item.get("source_piece_id")
        },
    )
    visible_piece_ids = {item.get("piece_id") for item in pieces}
    relevant_hypotheses = [
        {
            **item,
            "candidate_partner_piece_ids": [
                piece_id
                for piece_id in item.get("candidate_partner_piece_ids", [])
                or []
                if piece_id in visible_piece_ids
            ],
        }
        for item in relevant_hypotheses
        if item.get("source_piece_id") in visible_piece_ids
    ]
    return task, {
        "question": case["question"],
        "question_id": case["question_id"],
        "pieces": pieces,
        "source_dependency_clusters": case[
            "source_dependency_clusters"
        ],
        "source_lineage_rules": case["source_lineage_rules"],
        "mechanism_index": case.get("mechanism_index") or {},
        "classic_connection_hypotheses": relevant_hypotheses,
    }


def _proposal_prompt_bundle(
    case: dict[str, Any],
    mode: str,
    allowed_types: set[str],
    *,
    material_mode: str,
    native_limit: int = MAX_NATIVE_PROPOSALS,
    classic_limit: int = MAX_CLASSIC_ASSISTED_PROPOSALS,
    lane_id: str = "full",
) -> dict[str, Any]:
    task, full_payload = _proposal_case_payload(case, mode, allowed_types)
    message_material, packet_stats = _prepare_detective_prompt_material(
        full_payload,
        material_mode=material_mode,
        stage=f"{mode}_proposal",
    )
    schema = {
        "relations": [
            {
                "relation_type": "allowed relation type",
                "piece_ids": ["piece_id_1", "piece_id_2"],
                "plain_language_explanation": "具体说明两块以上材料怎样连接",
                "shared_coordinate": {
                    "same_proposition": "它们共同回答的具体命题",
                    "definition": "兼容或不兼容的定义",
                    "scope": "兼容或不兼容的范围",
                    "time": "兼容或不兼容的时间",
                    "condition": "使关系成立的条件",
                },
                "source_independence": "independent|shared|unknown|not_applicable",
                "common_error_risk": "high|medium|low|unknown",
                "competing_explanation": "为什么这条连接也可能不成立",
                "falsification_test": "出现什么材料时应撤销或改写这条连接",
                "discovery_route": "native|classic_assisted",
                "connection_hypothesis_ids": [
                    "若检验了经典连接假设，填写其 hypothesis_id"
                ],
            }
        ]
    }
    messages = [
        {"role": "system", "content": DETECTIVE_SYSTEM + CLASSIC_TRACE_RULES},
        {
            "role": "user",
            "content": (
                f"任务：{task}\n"
                f"只允许 relation_type：{sorted(allowed_types)}；可以返回空列表。\n"
                "每条关系至少连接两个不同数字人的材料。collective 盲区候选可与具体数字人连接。\n"
                f"先不依赖经典，只凭 pieces 自由寻找最多 {native_limit} 条高价值原生连接，"
                "这一步是侦探主干。随后才把 classic_connection_hypotheses 当作查漏清单，"
                f"最多补充 {classic_limit} 条原生阶段遗漏的连接；经典不得挤掉原生候选，也不要求凑满。"
                "原生阶段已经发现、后来又通过经典假设检查的关系，discovery_route 仍填 native，同时保留 hypothesis_id。"
                "只有经典提示后才发现的关系，discovery_route 才填 classic_assisted。"
                "这些假设只是搜索路线：采用时必须引用 hypothesis_id、包含其 source_piece_id、遵守 allowed_relation_types，"
                "并逐项满足 proof_obligations；经典知识不得补写任何现实事实。\n"
                "连接假设的搭档必须来自 source_person_id 之外的数字人；优先检查 candidate_partner_piece_ids，"
                "但只有材料真的满足证明责任时才能连接。\n"
                "若 CASE_JSON.schema 是 detective-prompt-packet，先按 packet_rules 解读 tables 和 reference_indexes；"
                "它与完整材料逐项等价，不得把表格编码误当成信息缺失。\n"
                "CASE_JSON:\n"
                + _prompt_json(message_material)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]
    return {
        "messages": messages,
        "input_fingerprint": _stage_input_fingerprint(
            PROPOSAL_PROMPT_REVISION,
            {
                "mode": mode,
                "lane_id": lane_id,
                "allowed_types": sorted(allowed_types),
                "native_limit": native_limit,
                "classic_limit": classic_limit,
                "case": full_payload,
            },
        ),
        "packet_stats": packet_stats,
    }


def _proposal_messages(
    case: dict[str, Any],
    mode: str,
    allowed_types: set[str],
    *,
    material_mode: str = "legacy_full_case",
) -> list[dict[str, str]]:
    return _proposal_prompt_bundle(
        case,
        mode,
        allowed_types,
        material_mode=material_mode,
    )["messages"]


def _adjudication_case_payload(
    case: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    endpoint_ids = {
        piece_id for item in candidates for piece_id in item.get("piece_ids", [])
    }
    pieces = [
        item for item in case["piece_index"] if item["piece_id"] in endpoint_ids
    ]
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
        "question": case["question"],
        "pieces": pieces,
        "source_dependency_clusters": case[
            "source_dependency_clusters"
        ],
        "source_lineage_rules": case["source_lineage_rules"],
        "mechanism_index": case.get("mechanism_index") or {},
        "candidates": material_candidates,
    }


def _adjudication_prompt_bundle(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    material_mode: str,
) -> dict[str, Any]:
    full_payload = _adjudication_case_payload(case, candidates)
    message_material, packet_stats = _prepare_detective_prompt_material(
        full_payload,
        material_mode=material_mode,
        stage="adversarial_adjudication",
    )
    schema = {
        "decisions": [
            {
                "candidate_id": "candidate id from input",
                "decision": "accepted|rejected|unresolved",
                "plain_language_explanation": "大白话裁决理由",
                "competing_explanation": "该关系最强的竞争解释",
                "falsification_test": "何种材料会撤销或降级该裁决",
                "source_independence": "independent|shared|unknown|not_applicable",
                "common_error_risk": "high|medium|low|unknown",
                "confidence_profile": {
                    "semantic_alignment": "high|medium|low|unknown",
                    "scope_compatibility": "high|medium|low|unknown",
                    "source_independence": "high|medium|low|unknown",
                    "diagnostic_certainty": "high|medium|low|unknown",
                },
            }
        ]
    }
    messages = [
        {"role": "system", "content": ADJUDICATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                "逐条裁决下列候选，不得漏项。相同结论若只有共同基础模型先验或相同来源，不能接受为独立会合。\n"
                "本阶段输入已物理移除发现路线和全部经典知识。只依据 pieces、shared_coordinate、来源谱系、竞争解释与证伪条件，"
                "独立决定 accepted/rejected/unresolved。\n"
                "若 CASE_JSON.schema 是 detective-prompt-packet，先按 packet_rules 解读 tables 和 reference_indexes；"
                "它与完整材料逐项等价。\n"
                "CASE_JSON:\n"
                + _prompt_json(message_material)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]
    return {
        "messages": messages,
        "input_fingerprint": _stage_input_fingerprint(
            ADJUDICATION_PROMPT_REVISION,
            full_payload,
        ),
        "packet_stats": packet_stats,
    }


def _adjudication_messages(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    material_mode: str = "legacy_full_case",
) -> list[dict[str, str]]:
    return _adjudication_prompt_bundle(
        case,
        candidates,
        material_mode=material_mode,
    )["messages"]


def _classic_review_case_payload(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    relation_by_candidate_id = {
        compact_text((item.get("provenance") or {}).get("candidate_id")): item
        for item in relations
    }
    review_candidates = [
        item for item in candidates if item.get("classic_check_hypothesis_ids")
    ]
    endpoint_ids = {
        piece_id
        for item in review_candidates
        for piece_id in item.get("piece_ids", []) or []
    }
    classic_check_ids = {
        hypothesis_id
        for item in review_candidates
        for hypothesis_id in item.get("classic_check_hypothesis_ids") or []
    }
    tasks = []
    for candidate in review_candidates:
        relation = relation_by_candidate_id.get(candidate["candidate_id"]) or {}
        for hypothesis_id in candidate.get("classic_check_hypothesis_ids") or []:
            tasks.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "relation_type": candidate["relation_type"],
                    "piece_ids": candidate["piece_ids"],
                    "material_decision": {
                        "status": relation.get("status") or "unresolved",
                        "explanation": relation.get(
                            "plain_language_explanation"
                        )
                        or "",
                    },
                    "hypothesis_id": hypothesis_id,
                }
            )
    return {
        "question": case["question"],
        "pieces": [
            item
            for item in case["piece_index"]
            if item["piece_id"] in endpoint_ids
        ],
        "classic_connection_hypotheses": [
            item
            for item in case.get(
                "classic_post_check_hypotheses", []
            )
            or []
            if item.get("hypothesis_id") in classic_check_ids
        ],
        "review_tasks": tasks,
    }


def _classic_review_prompt_bundle(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    material_mode: str,
) -> dict[str, Any]:
    full_payload = _classic_review_case_payload(case, candidates, relations)
    message_material, packet_stats = _prepare_detective_prompt_material(
        full_payload,
        material_mode=material_mode,
        stage="classic_post_review",
    )
    schema = {
        "reviews": [
            {
                "candidate_id": "candidate id from input",
                "hypothesis_id": "hypothesis id from input",
                "decision": "validated|rejected",
                "reason": "具体说明证明责任怎样满足或哪里不满足",
            }
        ]
    }
    messages = [
        {"role": "system", "content": CLASSIC_REVIEW_SYSTEM + CLASSIC_TRACE_RULES},
        {
            "role": "user",
            "content": (
                "逐项复核，不得漏项。material_decision 已经冻结，不得重新裁决或建议修改。"
                "你的 decision 只针对经典工具的解释适配度。\n"
                "若 CASE_JSON.schema 是 detective-prompt-packet，先按 packet_rules 解读 tables 和 reference_indexes；"
                "它与完整材料逐项等价。\n"
                "CASE_JSON:\n"
                + _prompt_json(message_material)
                + "\nOUTPUT_SCHEMA:\n"
                + _prompt_json(schema)
            ),
        },
    ]
    return {
        "messages": messages,
        "input_fingerprint": _stage_input_fingerprint(
            CLASSIC_REVIEW_PROMPT_REVISION,
            full_payload,
        ),
        "packet_stats": packet_stats,
    }


def _classic_review_messages(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    material_mode: str = "legacy_full_case",
) -> list[dict[str, str]]:
    return _classic_review_prompt_bundle(
        case,
        candidates,
        relations,
        material_mode=material_mode,
    )["messages"]


def build_detective_prompt_packet(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Encode detective material once while retaining exact reconstruction."""

    shared = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "pieces",
            "candidates",
            "classic_connection_hypotheses",
        }
    }
    indexes: dict[str, dict[str, Any]] = {
        "coordinates": {},
        "source_lineage": {},
        "diagnostic_context": {},
        "classic_guard": {},
        "mechanism_ids": {},
        "parent_conclusion": {},
        "classic_contract": {},
        "person_id": {},
    }
    refs_by_fingerprint: dict[str, dict[str, str]] = {
        key: {} for key in indexes
    }

    def intern(index_name: str, value: Any) -> str:
        fingerprint = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = refs_by_fingerprint[index_name].get(fingerprint)
        if existing:
            return existing
        ref = f"{index_name}_{len(indexes[index_name]) + 1:03d}"
        refs_by_fingerprint[index_name][fingerprint] = ref
        indexes[index_name][ref] = value
        return ref

    def encode_table(
        items: list[dict[str, Any]], id_field: str
    ) -> dict[str, Any]:
        fields: list[str] = []
        for item in items:
            for key in item:
                if key != id_field and key not in fields:
                    fields.append(key)
        rows = []
        for item in items:
            presence = sum(
                1 << index
                for index, field in enumerate(fields)
                if field in item
            )
            rows.append(
                [
                    item.get(id_field),
                    presence,
                    *[item.get(field) for field in fields],
                ]
            )
        return {"id_field": id_field, "fields": fields, "rows": rows}

    tables: dict[str, dict[str, Any]] = {}
    if "pieces" in payload:
        encoded_pieces = []
        for piece in payload.get("pieces", []) or []:
            encoded = {
                key: value
                for key, value in piece.items()
                if key
                not in {
                    "piece_id",
                    "person_id",
                    "coordinates",
                    "source_lineage",
                    "diagnostic_context",
                    "classic_guard",
                    "mechanism_ids",
                    "parent_conclusion",
                }
            }
            if "person_id" in piece:
                encoded["person_id_ref"] = intern(
                    "person_id", piece.get("person_id")
                )
            for field in (
                "coordinates",
                "source_lineage",
                "diagnostic_context",
                "classic_guard",
                "mechanism_ids",
                "parent_conclusion",
            ):
                if field in piece:
                    encoded[f"{field}_ref"] = intern(field, piece[field])
            encoded_pieces.append(
                {"piece_id": piece.get("piece_id"), **encoded}
            )
        tables["pieces"] = encode_table(encoded_pieces, "piece_id")

    if "candidates" in payload:
        tables["candidates"] = encode_table(
            list(payload.get("candidates", []) or []), "candidate_id"
        )

    if "classic_connection_hypotheses" in payload:
        contract_fields = {
            "operator_id",
            "allowed_relation_types",
            "partner_piece_kinds",
            "partner_must_be_different_person",
            "constructive_task",
            "proof_obligations",
            "forbidden_inferences",
            "classic_as_world_evidence",
        }
        encoded_hypotheses = []
        for hypothesis in (
            payload.get("classic_connection_hypotheses", []) or []
        ):
            contract = {
                key: hypothesis[key]
                for key in hypothesis
                if key in contract_fields
            }
            encoded = {
                key: value
                for key, value in hypothesis.items()
                if key not in contract_fields | {"hypothesis_id"}
            }
            encoded["classic_contract_ref"] = intern(
                "classic_contract", contract
            )
            encoded_hypotheses.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    **encoded,
                }
            )
        tables["classic_connection_hypotheses"] = encode_table(
            encoded_hypotheses, "hypothesis_id"
        )

    return {
        "schema": DETECTIVE_PROMPT_PACKET_SCHEMA,
        "packet_rules": {
            "lossless_collective_material": True,
            "table_row_0_is_id_row_1_is_presence_mask": True,
            "remaining_row_cells_follow_fields_in_order": True,
            "decoding": (
                "tables 中每行依次为 [ID, presence_mask, ...fields 对应值]；"
                "presence_mask 的第 n 位说明 fields[n] 是否在原材料中存在。"
            ),
            "field_ref_suffix_resolves_in_reference_indexes": True,
            "classic_contract_ref_restores_hypothesis_contract": True,
            "classic_as_world_evidence": False,
        },
        "shared": shared,
        "tables": tables,
        "reference_indexes": {
            key: value for key, value in indexes.items() if value
        },
    }


def restore_detective_prompt_packet(
    packet: dict[str, Any],
) -> dict[str, Any]:
    if packet.get("schema") != DETECTIVE_PROMPT_PACKET_SCHEMA:
        raise ValueError("unknown detective prompt packet schema")
    output = dict(packet.get("shared") or {})
    tables = packet.get("tables") or {}
    indexes = packet.get("reference_indexes") or {}

    def decode_table(table: dict[str, Any]) -> list[dict[str, Any]]:
        id_field = str(table["id_field"])
        fields = list(table.get("fields") or [])
        decoded = []
        for row in table.get("rows") or []:
            if len(row) != len(fields) + 2:
                raise ValueError(f"malformed table row for {id_field}")
            presence = int(row[1])
            item = {id_field: row[0]}
            for index, field in enumerate(fields):
                if presence & (1 << index):
                    item[field] = row[index + 2]
            decoded.append(item)
        return decoded

    if "pieces" in tables:
        pieces = decode_table(tables["pieces"])
        for piece in pieces:
            for key in list(piece):
                if not key.endswith("_ref"):
                    continue
                field = key[:-4]
                piece[field] = indexes[field][piece.pop(key)]
        output["pieces"] = pieces

    if "candidates" in tables:
        output["candidates"] = decode_table(tables["candidates"])

    if "classic_connection_hypotheses" in tables:
        hypotheses = decode_table(tables["classic_connection_hypotheses"])
        for hypothesis in hypotheses:
            contract_ref = hypothesis.pop("classic_contract_ref")
            contract = dict(indexes["classic_contract"][contract_ref])
            hypothesis.update(contract)
        output["classic_connection_hypotheses"] = hypotheses
    return output


def validate_detective_prompt_packet(
    packet: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    try:
        restored = restore_detective_prompt_packet(packet)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"packet cannot be reconstructed: {exc}"]
    if restored != payload:
        expected_keys = set(payload)
        actual_keys = set(restored)
        if expected_keys != actual_keys:
            errors.append("top-level key coverage mismatch")
        for key in sorted(expected_keys & actual_keys):
            if restored[key] != payload[key]:
                errors.append(f"material mismatch: {key}")
    referenced: defaultdict[str, set[str]] = defaultdict(set)
    for table in (packet.get("tables") or {}).values():
        fields = list(table.get("fields") or [])
        for row in table.get("rows") or []:
            presence = int(row[1])
            for index, field in enumerate(fields):
                if field.endswith("_ref") and presence & (1 << index):
                    referenced[field[:-4]].add(str(row[index + 2]))
    for index_name, values in (
        packet.get("reference_indexes") or {}
    ).items():
        if set(values) != referenced.get(index_name, set()):
            errors.append(f"unused or missing reference entries: {index_name}")
    return errors[:20]


def _prepare_detective_prompt_material(
    payload: dict[str, Any],
    *,
    material_mode: str,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if material_mode not in {"legacy_full_case", "lossless_packet_v2"}:
        raise ValueError(
            "material_mode must be 'legacy_full_case' or 'lossless_packet_v2'"
        )
    packet = build_detective_prompt_packet(payload)
    errors = validate_detective_prompt_packet(packet, payload)
    if errors:
        raise RuntimeError(
            f"Detective prompt packet failed at {stage}: "
            + "; ".join(errors)
        )
    full_chars = len(_prompt_json(payload))
    packet_chars = len(_prompt_json(packet))
    use_packet = (
        material_mode == "lossless_packet_v2" and packet_chars < full_chars
    )
    material = packet if use_packet else payload
    sent_chars = len(_prompt_json(material))
    return material, {
        "schema": DETECTIVE_PROMPT_PACKET_SCHEMA,
        "stage": stage,
        "material_mode": material_mode,
        "encoding_selected": (
            "lossless_packet" if use_packet else "full_case_smaller"
        ),
        "full_case_characters": full_chars,
        "prompt_packet_characters": packet_chars,
        "sent_material_characters": sent_chars,
        "character_reduction_ratio": round(
            1 - sent_chars / max(1, full_chars), 4
        ),
        "coverage_validation": "passed",
        "task_relevant_material_removed": False,
    }


def _aggregate_packet_stats(
    stats: list[dict[str, Any]],
    *,
    stage: str,
    material_mode: str,
) -> dict[str, Any]:
    full_chars = sum(int(item.get("full_case_characters") or 0) for item in stats)
    packet_chars = sum(
        int(item.get("prompt_packet_characters") or 0) for item in stats
    )
    sent_chars = sum(
        int(item.get("sent_material_characters") or 0) for item in stats
    )
    return {
        "schema": DETECTIVE_PROMPT_PACKET_SCHEMA,
        "stage": stage,
        "material_mode": material_mode,
        "shard_count": len(stats),
        "full_case_characters": full_chars,
        "prompt_packet_characters": packet_chars,
        "sent_material_characters": sent_chars,
        "character_reduction_ratio": round(
            1 - sent_chars / max(1, full_chars), 4
        ),
        "coverage_validation": (
            "passed"
            if all(item.get("coverage_validation") == "passed" for item in stats)
            else "failed"
        ),
        "task_relevant_material_removed": any(
            bool(item.get("task_relevant_material_removed")) for item in stats
        ),
    }


def _stage_input_fingerprint(revision: str, payload: Any) -> str:
    serialized = json.dumps(
        {"revision": revision, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_proposals(
    payload: Any,
    case: dict[str, Any],
    *,
    mode: str,
    allowed_types: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    piece_by_id = {item["piece_id"]: item for item in case["piece_index"]}
    hypothesis_by_id = {
        item["hypothesis_id"]: item
        for item in case.get("classic_connection_hypotheses", []) or []
        if item.get("hypothesis_id")
    }
    output = []
    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        raw_relations = [payload] if payload.get("relation_type") else []
    for item in raw_relations[:30]:
        if not isinstance(item, dict):
            continue
        relation_type = compact_text(item.get("relation_type"))
        if relation_type not in allowed_types or relation_type not in RELATION_TYPES:
            continue
        piece_ids = _known_ids(item.get("piece_ids"), set(piece_by_id))
        people = {piece_by_id[piece_id]["person_id"] for piece_id in piece_ids}
        if len(piece_ids) < 2 or len(people) < 2:
            continue
        explanation = compact_text(item.get("plain_language_explanation"))
        if not explanation:
            continue
        declared_hypothesis_ids = list(
            dict.fromkeys(
                compact_text(value)
                for value in item.get("connection_hypothesis_ids", []) or []
                if compact_text(value)
            )
        )
        hypothesis_ids = [
            hypothesis_id
            for hypothesis_id in declared_hypothesis_ids
            if hypothesis_id in hypothesis_by_id
            and hypothesis_by_id[hypothesis_id].get("source_piece_id")
            in piece_ids
            and relation_type
            in (
                hypothesis_by_id[hypothesis_id].get("allowed_relation_types")
                or []
            )
        ]
        rejected_hypothesis_ids = [
            hypothesis_id
            for hypothesis_id in declared_hypothesis_ids
            if hypothesis_id not in hypothesis_ids
        ]
        declared_route = compact_text(item.get("discovery_route"))
        if hypothesis_ids:
            discovery_routes = ["classic_assisted"]
            if declared_route == "native":
                discovery_routes.insert(0, "native")
        else:
            discovery_routes = ["native"]
        candidate_id = stable_id(
            "candidate",
            case["question_id"],
            relation_type,
            "|".join(sorted(piece_ids)),
        )
        output.append(
            {
                "candidate_id": candidate_id,
                "question_id": case["question_id"],
                "proposal_mode": mode,
                "relation_type": relation_type,
                "piece_ids": piece_ids,
                "plain_language_explanation": explanation,
                "shared_coordinate": item.get("shared_coordinate") or {},
                "source_independence": compact_text(
                    item.get("source_independence")
                )
                or "unknown",
                "common_error_risk": compact_text(item.get("common_error_risk"))
                or "unknown",
                "competing_explanation": compact_text(
                    item.get("competing_explanation")
                ),
                "falsification_test": compact_text(item.get("falsification_test")),
                "connection_hypothesis_ids": hypothesis_ids,
                "rejected_connection_hypothesis_ids": rejected_hypothesis_ids,
                "discovery_routes": discovery_routes,
                "discovery_route": (
                    "dual" if len(discovery_routes) > 1 else discovery_routes[0]
                ),
            }
        )
    return _select_dual_track_proposals(output)


def _adjudicate_candidates(
    payload: Any, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    decisions = {}
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
        decisions = {
            compact_text(item.get("candidate_id")): item
            for item in payload["decisions"]
            if isinstance(item, dict) and compact_text(item.get("candidate_id"))
        }
    output = []
    for candidate in candidates:
        decision = decisions.get(candidate["candidate_id"], {})
        status = compact_text(decision.get("decision"))
        if status not in {"accepted", "rejected", "unresolved"}:
            status = "unresolved"
        explanation = compact_text(
            decision.get("plain_language_explanation")
            or candidate.get("plain_language_explanation")
        )
        competing = compact_text(
            decision.get("competing_explanation")
            or candidate.get("competing_explanation")
        )
        falsifier = compact_text(
            decision.get("falsification_test")
            or candidate.get("falsification_test")
        )
        if status == "accepted" and (not competing or not falsifier):
            status = "unresolved"
        output.append(
            {
                "relation_type": candidate["relation_type"],
                "piece_ids": candidate["piece_ids"],
                "status": status,
                "plain_language_explanation": explanation,
                "shared_coordinate": decision.get("shared_coordinate")
                or candidate.get("shared_coordinate")
                or {},
                "source_independence": compact_text(
                    decision.get("source_independence")
                    or candidate.get("source_independence")
                )
                or "unknown",
                "common_error_risk": compact_text(
                    decision.get("common_error_risk")
                    or candidate.get("common_error_risk")
                )
                or "unknown",
                "competing_explanation": competing
                or "现有材料不足以排除另一种关系解释。",
                "falsification_test": falsifier
                or "补充同定义、同范围、同时间且来源可追溯的材料后重新裁决。",
                "confidence_profile": decision.get("confidence_profile") or {},
                "provenance": {
                    "mode": "live_adversarial_adjudication",
                    "candidate_id": candidate["candidate_id"],
                    "proposal_mode": candidate["proposal_mode"],
                    "classic_discovery_hypothesis_ids": candidate.get(
                        "connection_hypothesis_ids"
                    )
                    or [],
                    "connection_hypothesis_ids": [],
                    "rejected_connection_hypothesis_ids": [],
                    "unreviewed_connection_hypothesis_ids": candidate.get(
                        "classic_check_hypothesis_ids"
                    )
                    or [],
                    "classic_check": {},
                    "discovery_routes": candidate.get("discovery_routes")
                    or ["native"],
                    "discovery_route": candidate.get("discovery_route")
                    or "native",
                    "classic_truth_weight_bonus": False,
                    "adjudication_transport_fallback": bool(
                        decision.get("transport_fallback")
                    ),
                },
            }
        )
    return output


def _apply_classic_reviews(
    relations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    payload: Any,
) -> list[dict[str, Any]]:
    reviews = {}
    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        reviews = {
            (
                compact_text(item.get("candidate_id")),
                compact_text(item.get("hypothesis_id")),
            ): item
            for item in payload["reviews"]
            if isinstance(item, dict)
        }
    checks_by_candidate = {
        item["candidate_id"]: item.get("classic_check_hypothesis_ids") or []
        for item in candidates
    }
    output = []
    for relation in relations:
        guarded = dict(relation)
        provenance = dict(relation.get("provenance") or {})
        candidate_id = compact_text(provenance.get("candidate_id"))
        offered = checks_by_candidate.get(candidate_id, [])
        validated: list[str] = []
        rejected: list[str] = []
        unreviewed: list[str] = []
        review_records = []
        for hypothesis_id in offered:
            review = reviews.get((candidate_id, hypothesis_id)) or {}
            decision = compact_text(review.get("decision"))
            reason = compact_text(review.get("reason"))
            if decision == "validated" and reason:
                validated.append(hypothesis_id)
            elif decision == "rejected" and reason:
                rejected.append(hypothesis_id)
            else:
                unreviewed.append(hypothesis_id)
                continue
            review_records.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "decision": decision,
                    "reason": reason,
                }
            )
        provenance["connection_hypothesis_ids"] = validated
        provenance["rejected_connection_hypothesis_ids"] = rejected
        provenance["unreviewed_connection_hypothesis_ids"] = unreviewed
        provenance["classic_check"] = (
            review_records[0] if review_records else {}
        )
        provenance["classic_review_cannot_modify_relation"] = True
        provenance["classic_truth_weight_bonus"] = False
        guarded["provenance"] = provenance
        output.append(guarded)
    return output


def _call_grouped_proposal(
    client_factory: Callable[[], DeepSeekClient],
    case: dict[str, Any],
    mode: str,
    allowed_types: set[str],
    *,
    max_tokens: int,
    retries: int,
    material_mode: str,
    parent_input_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Propose disjoint relation families in parallel, then restore one stage."""

    started = time.perf_counter()
    lane_specs = list(PROPOSAL_RELATION_LANES.get(mode) or [])
    lane_type_sets = [set(item[1]) for item in lane_specs]
    covered_types = set().union(*lane_type_sets) if lane_type_sets else set()
    overlaps = set()
    for index, left in enumerate(lane_type_sets):
        for right in lane_type_sets[index + 1 :]:
            overlaps.update(left & right)
    if covered_types != set(allowed_types) or overlaps:
        raise RuntimeError(
            f"Invalid grouped proposal coverage for {mode}: "
            f"missing={sorted(set(allowed_types) - covered_types)}, "
            f"extra={sorted(covered_types - set(allowed_types))}, "
            f"overlap={sorted(overlaps)}"
        )
    if sum(item[2] for item in lane_specs) != MAX_NATIVE_PROPOSALS:
        raise RuntimeError("Grouped proposal native capacity changed.")
    if sum(item[3] for item in lane_specs) != MAX_CLASSIC_ASSISTED_PROPOSALS:
        raise RuntimeError("Grouped proposal classic capacity changed.")

    bundles = []
    for lane_id, lane_types, native_limit, classic_limit in lane_specs:
        bundle = _proposal_prompt_bundle(
            case,
            mode,
            set(lane_types),
            material_mode=material_mode,
            native_limit=native_limit,
            classic_limit=classic_limit,
            lane_id=lane_id,
        )
        bundles.append(
            (
                lane_id,
                set(lane_types),
                native_limit,
                classic_limit,
                bundle,
            )
        )

    records: list[dict[str, Any] | None] = [None] * len(bundles)
    payloads: list[dict[str, Any] | None] = [None] * len(bundles)
    with ThreadPoolExecutor(max_workers=len(bundles)) as executor:
        futures = {}
        for index, (
            lane_id,
            lane_types,
            native_limit,
            classic_limit,
            bundle,
        ) in enumerate(bundles):
            capacity_ratio = (native_limit + classic_limit) / (
                MAX_NATIVE_PROPOSALS + MAX_CLASSIC_ASSISTED_PROPOSALS
            )
            lane_max_tokens = max(
                3072, min(max_tokens, int(max_tokens * capacity_ratio))
            )
            future = executor.submit(
                _call_json_stage,
                client_factory,
                bundle["messages"],
                stage=f"{mode}_{lane_id}_proposal",
                max_tokens=lane_max_tokens,
                retries=retries,
                thinking_mode="disabled",
                validator=lambda value, allowed=lane_types: _proposal_payload_errors(
                    value, case, allowed
                ),
                input_fingerprint=bundle["input_fingerprint"],
                prompt_packet_stats=bundle["packet_stats"],
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
    successful_statuses = {
        "succeeded",
        "succeeded_truncated_recovery",
    }
    shards_succeeded = (
        len(complete_records) == len(bundles)
        and all(
            record.get("status") in successful_statuses
            for record in complete_records
        )
    )
    raw_relations: list[dict[str, Any]] = []
    for index, payload_item in enumerate(payloads):
        if not isinstance(payload_item, dict):
            continue
        _lane_id, _lane_types, native_limit, classic_limit, _bundle = (
            bundles[index]
        )
        raw_relations.extend(
            _cap_grouped_proposal_relations(
                [
                    relation
                    for relation in payload_item.get("relations", []) or []
                    if isinstance(relation, dict)
                ],
                native_limit=native_limit,
                classic_limit=classic_limit,
            )
        )
    payload = {
        "relations": _cap_grouped_proposal_relations(raw_relations)
    }
    validation_errors = (
        _proposal_payload_errors(payload, case, allowed_types)
        if shards_succeeded
        else ["one or more proposal groups failed"]
    )
    errors.extend(validation_errors)
    status = "succeeded" if not validation_errors else "failed"
    aggregate_packet_stats = _aggregate_packet_stats(
        [bundle[4]["packet_stats"] for bundle in bundles],
        stage=f"{mode}_proposal",
        material_mode=material_mode,
    )
    aggregate_packet_stats["relation_type_coverage"] = {
        lane_id: sorted(lane_types)
        for lane_id, lane_types, *_rest in lane_specs
    }
    aggregate_packet_stats["capacity_preserved"] = {
        "native": MAX_NATIVE_PROPOSALS,
        "classic_assisted": MAX_CLASSIC_ASSISTED_PROPOSALS,
    }
    synthetic_raw = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            }
        ]
    }
    return (payload if status == "succeeded" else {}), {
        "stage": f"{mode}_proposal",
        "status": status,
        "attempts": sum(
            int(record.get("attempts") or 0) for record in complete_records
        ),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
        "usage": usage,
        "raw": synthetic_raw,
        "thinking_mode": "disabled",
        "retryable": all(
            record.get("retryable", True) for record in complete_records
        ),
        "shard_count": len(bundles),
        "shards": complete_records,
        "input_fingerprint": parent_input_fingerprint,
        "prompt_packet": aggregate_packet_stats,
    }


def _cap_grouped_proposal_relations(
    relations: list[dict[str, Any]],
    *,
    native_limit: int = MAX_NATIVE_PROPOSALS,
    classic_limit: int = MAX_CLASSIC_ASSISTED_PROPOSALS,
) -> list[dict[str, Any]]:
    native: list[dict[str, Any]] = []
    classic: list[dict[str, Any]] = []
    for relation in relations:
        route = compact_text(relation.get("discovery_route"))
        if route == "classic_assisted":
            classic.append(relation)
        else:
            native.append(relation)
    return [
        *native[:native_limit],
        *classic[:classic_limit],
    ]


def _call_sharded_adjudication(
    client_factory: Callable[[], DeepSeekClient],
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    max_tokens: int,
    retries: int,
    material_mode: str = "legacy_full_case",
    parent_input_fingerprint: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adjudicate independent certificates in parallel under one rubric."""

    started = time.perf_counter()
    shard_count = min(8, max(1, (len(candidates) + 2) // 3))
    shard_size = max(1, (len(candidates) + shard_count - 1) // shard_count)
    shards = [
        candidates[start : start + shard_size]
        for start in range(0, len(candidates), shard_size)
    ]
    shard_bundles = [
        _adjudication_prompt_bundle(
            case,
            shard,
            material_mode=material_mode,
        )
        for shard in shards
    ]
    aggregate_packet_stats = _aggregate_packet_stats(
        [bundle["packet_stats"] for bundle in shard_bundles],
        stage="adversarial_adjudication",
        material_mode=material_mode,
    )
    parent_input_fingerprint = parent_input_fingerprint or (
        _adjudication_prompt_bundle(
            case,
            candidates,
            material_mode=material_mode,
        )["input_fingerprint"]
    )
    records: list[dict[str, Any] | None] = [None] * len(shards)
    payloads: list[dict[str, Any] | None] = [None] * len(shards)
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(
                _call_json_stage,
                client_factory,
                shard_bundles[index]["messages"],
                stage=f"adversarial_adjudication_shard_{index + 1:02d}",
                max_tokens=max(max_tokens, 8192),
                retries=0,
                thinking_mode="enabled",
                validator=lambda value, expected=shard: _adjudication_payload_errors(
                    value, expected
                ),
                input_fingerprint=shard_bundles[index][
                    "input_fingerprint"
                ],
                prompt_packet_stats=shard_bundles[index][
                    "packet_stats"
                ],
            ): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            index = futures[future]
            payloads[index], records[index] = future.result()

    failed_shard_indices = [
        index
        for index, record in enumerate(records)
        if not _adjudication_stage_succeeded(record)
    ]
    recovery_records: dict[int, list[dict[str, Any]]] = {
        index: [] for index in failed_shard_indices
    }
    recovery_payloads: dict[int, dict[str, dict[str, Any]]] = {
        index: {} for index in failed_shard_indices
    }
    retryable_failed_shard_indices = [
        index
        for index in failed_shard_indices
        if (records[index] or {}).get("retryable", True)
    ]
    if retryable_failed_shard_indices and retries > 0:
        recovery_jobs = [
            (index, candidate)
            for index in retryable_failed_shard_indices
            for candidate in shards[index]
        ]
        with ThreadPoolExecutor(max_workers=min(8, len(recovery_jobs))) as executor:
            futures = {}
            for index, candidate in recovery_jobs:
                bundle = _adjudication_prompt_bundle(
                    case,
                    [candidate],
                    material_mode=material_mode,
                )
                future = executor.submit(
                    _call_json_stage,
                    client_factory,
                    bundle["messages"],
                    stage=(
                        f"adversarial_adjudication_recovery_"
                        f"{index + 1:02d}_{candidate['candidate_id']}"
                    ),
                    max_tokens=max(max_tokens, 8192),
                    retries=max(0, retries - 1),
                    thinking_mode="enabled",
                    validator=lambda value, expected=[candidate]: (
                        _adjudication_payload_errors(value, expected)
                    ),
                    input_fingerprint=bundle["input_fingerprint"],
                    prompt_packet_stats=bundle["packet_stats"],
                )
                futures[future] = (index, candidate)
            for future in as_completed(futures):
                index, candidate = futures[future]
                payload, record = future.result()
                recovery_records[index].append(record)
                decisions = payload.get("decisions", []) if payload else []
                decision = next(
                    (
                        item
                        for item in decisions
                        if compact_text(item.get("candidate_id"))
                        == candidate["candidate_id"]
                    ),
                    None,
                )
                if decision is not None:
                    recovery_payloads[index][candidate["candidate_id"]] = decision

    transport_unresolved_ids: list[str] = []
    for index in failed_shard_indices:
        initial_record = records[index] or {}
        decisions = recovery_payloads.get(index, {})
        for candidate in shards[index]:
            if candidate["candidate_id"] not in decisions:
                transport_unresolved_ids.append(candidate["candidate_id"])
                decisions[candidate["candidate_id"]] = (
                    _transport_unresolved_decision(candidate)
                )
        recovered_payload = {
            "decisions": [
                decisions[candidate["candidate_id"]]
                for candidate in shards[index]
            ]
        }
        payloads[index] = recovered_payload
        component_records = [
            initial_record,
            *recovery_records.get(index, []),
        ]
        recovered_usage = _empty_usage()
        recovered_errors: list[str] = []
        for component in component_records:
            recovered_usage = _merge_usage(
                recovered_usage, component.get("usage") or {}
            )
            recovered_errors.extend(component.get("errors") or [])
        records[index] = {
            "stage": f"adversarial_adjudication_shard_{index + 1:02d}",
            "status": (
                "succeeded_with_unresolved_transport"
                if any(
                    candidate["candidate_id"] in transport_unresolved_ids
                    for candidate in shards[index]
                )
                else "succeeded_targeted_recovery"
            ),
            "attempts": sum(
                int(component.get("attempts") or 0)
                for component in component_records
            ),
            "duration_seconds": round(
                max(
                    [
                        float(component.get("duration_seconds") or 0.0)
                        for component in component_records
                    ]
                    or [0.0]
                ),
                3,
            ),
            "errors": recovered_errors,
            "usage": recovered_usage,
            "raw": _synthetic_json_raw(recovered_payload),
            "thinking_mode": "enabled",
            "retryable": True,
            "targeted_recovery": True,
            "initial_record": initial_record,
            "recovery_shards": recovery_records.get(index, []),
            "transport_unresolved_candidate_ids": [
                candidate["candidate_id"]
                for candidate in shards[index]
                if candidate["candidate_id"] in transport_unresolved_ids
            ],
        }

    complete_records = [item for item in records if item is not None]
    usage = _empty_usage()
    errors: list[str] = []
    for record in complete_records:
        usage = _merge_usage(usage, record.get("usage") or {})
        errors.extend(record.get("errors") or [])
    if any(
        not _adjudication_stage_succeeded(record)
        for record in complete_records
    ):
        return {}, {
            "stage": "adversarial_adjudication",
            "status": "failed",
            "attempts": sum(
                int(record.get("attempts") or 0)
                for record in complete_records
            ),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "errors": errors,
            "usage": usage,
            "raw": {},
            "thinking_mode": "enabled",
            "retryable": all(
                record.get("retryable", True) for record in complete_records
            ),
            "shard_count": len(shards),
            "shards": complete_records,
            "input_fingerprint": parent_input_fingerprint,
            "prompt_packet": aggregate_packet_stats,
        }

    decisions_by_id = {
        compact_text(item.get("candidate_id")): item
        for payload in payloads
        if isinstance(payload, dict)
        for item in payload.get("decisions", []) or []
        if isinstance(item, dict) and compact_text(item.get("candidate_id"))
    }
    payload = {
        "decisions": [
            decisions_by_id[candidate["candidate_id"]]
            for candidate in candidates
            if candidate["candidate_id"] in decisions_by_id
        ]
    }
    validation_errors = _adjudication_payload_errors(payload, candidates)
    if validation_errors:
        errors.extend(validation_errors)
        status = "failed"
    else:
        status = "succeeded"
    synthetic_raw = _synthetic_json_raw(payload)
    return (payload if status == "succeeded" else {}), {
        "stage": "adversarial_adjudication",
        "status": status,
        "attempts": sum(
            int(record.get("attempts") or 0) for record in complete_records
        ),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
        "usage": usage,
        "raw": synthetic_raw,
        "thinking_mode": "enabled",
        "retryable": all(
            record.get("retryable", True) for record in complete_records
        ),
        "shard_count": len(shards),
        "shards": complete_records,
        "targeted_recovery_shard_count": len(failed_shard_indices),
        "targeted_recovery_attempted_shard_count": len(
            retryable_failed_shard_indices
        ),
        "transport_unresolved_candidate_ids": transport_unresolved_ids,
        "input_fingerprint": parent_input_fingerprint,
        "prompt_packet": aggregate_packet_stats,
    }


def _adjudication_stage_succeeded(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("status")
        in {
            "succeeded",
            "succeeded_truncated_recovery",
            "succeeded_targeted_recovery",
            "succeeded_with_unresolved_transport",
        }
    )


def _transport_unresolved_decision(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "decision": "unresolved",
        "plain_language_explanation": (
            "本轮模型连接未能完整返回这条关系的裁决，因此它保持未决，"
            "不得进入主轮廓。"
        ),
        "shared_coordinate": candidate.get("shared_coordinate") or {},
        "source_independence": "unknown",
        "common_error_risk": "transport_incomplete",
        "competing_explanation": candidate.get("competing_explanation")
        or "关系也可能来自定义、范围或来源依赖差异。",
        "falsification_test": candidate.get("falsification_test")
        or "在同一范围和可追溯来源下重新裁决这条关系。",
        "confidence_profile": {
            "semantic_alignment": "unknown",
            "scope_compatibility": "unknown",
            "source_independence": "unknown",
            "diagnostic_certainty": "unknown",
        },
        "transport_fallback": True,
    }


def _synthetic_json_raw(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            }
        ]
    }


def _call_json_stage(
    client_factory: Callable[[], DeepSeekClient],
    messages: list[dict[str, str]],
    *,
    stage: str,
    max_tokens: int,
    retries: int,
    thinking_mode: str,
    validator: Callable[[dict[str, Any]], list[str]] | None = None,
    input_fingerprint: str = "",
    prompt_packet_stats: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = _empty_usage()
    errors: list[str] = []
    last_raw: dict[str, Any] = {}
    started = time.perf_counter()
    attempts = 0
    retryable = True
    current_messages = list(messages)
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response = client_factory().chat(
                current_messages,
                temperature=0.15,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                thinking={"type": thinking_mode},
                reasoning_effort=(
                    "high" if thinking_mode == "enabled" else None
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
                recovered = (
                    _recover_complete_relation_array(response.content)
                    if stage.endswith("_proposal")
                    else []
                )
                if recovered:
                    parsed = {"relations": recovered}
                    validation_errors = validator(parsed) if validator else []
                    if not validation_errors:
                        return parsed, {
                            "stage": stage,
                            "status": "succeeded_truncated_recovery",
                            "attempts": attempts,
                            "duration_seconds": round(
                                time.perf_counter() - started, 3
                            ),
                            "errors": [
                                "输出末尾被截断；只保留通过 JSON 解码的完整关系对象。"
                            ],
                            "usage": usage,
                            "raw": last_raw,
                            "thinking_mode": thinking_mode,
                            "recovered_relation_count": len(recovered),
                            "input_fingerprint": input_fingerprint,
                            "prompt_packet": prompt_packet_stats or {},
                        }
                errors.append(
                    "generation stopped at max_tokens before a complete stage output"
                )
                break
            parsed = parse_json_content(response.content)
            if isinstance(parsed, dict) and "raw_text" not in parsed:
                validation_errors = validator(parsed) if validator else []
                if not validation_errors:
                    return parsed, {
                        "stage": stage,
                        "status": "succeeded",
                        "attempts": attempts,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "errors": errors,
                        "usage": usage,
                        "raw": last_raw,
                        "thinking_mode": thinking_mode,
                        "input_fingerprint": input_fingerprint,
                        "prompt_packet": prompt_packet_stats or {},
                    }
                errors.extend(validation_errors)
                current_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": "上一版未通过关系结构校验，请完全重写并修复："
                        + "；".join(validation_errors[:8]),
                    }
                ]
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
                continue
            errors.append("response was not a JSON object")
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
        except Exception as exc:  # noqa: BLE001 - preserve partial analysis.
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
        "retryable": retryable,
        "input_fingerprint": input_fingerprint,
        "prompt_packet": prompt_packet_stats or {},
    }


def _proposal_payload_errors(
    payload: dict[str, Any],
    case: dict[str, Any],
    allowed_types: set[str],
) -> list[str]:
    raw = payload.get("relations")
    if isinstance(raw, list):
        if not raw:
            return []
    elif payload.get("relation_type"):
        raw = [payload]
    else:
        return ["输出必须是 relations 列表或一个完整的单关系对象"]
    hypothesis_errors = _connection_hypothesis_reference_errors(payload, case)
    if hypothesis_errors:
        return hypothesis_errors
    normalized = _normalize_proposals(
        payload,
        case,
        mode="validation",
        allowed_types=allowed_types,
    )
    if not normalized:
        return ["所有关系都使用了非法类型、未知 piece_id、同一人端点或缺少具体解释"]
    return []


def _connection_hypothesis_reference_errors(
    payload: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    hypothesis_by_id = {
        item["hypothesis_id"]: item
        for item in case.get("classic_connection_hypotheses", []) or []
        if item.get("hypothesis_id")
    }
    piece_by_id = {
        item.get("piece_id"): item for item in case.get("piece_index", []) or []
    }
    raw = payload.get("relations")
    relations = raw if isinstance(raw, list) else [payload]
    errors: list[str] = []
    for item in relations:
        if not isinstance(item, dict):
            continue
        raw_ids = item.get("connection_hypothesis_ids")
        if raw_ids is None:
            continue
        if not isinstance(raw_ids, list):
            errors.append("connection_hypothesis_ids 必须是列表")
            continue
        piece_ids = {
            compact_text(value) for value in item.get("piece_ids", []) or []
        }
        endpoint_people = {
            piece_by_id[piece_id].get("person_id")
            for piece_id in piece_ids
            if piece_id in piece_by_id
        }
        relation_type = compact_text(item.get("relation_type"))
        for hypothesis_id in raw_ids:
            hypothesis = hypothesis_by_id.get(compact_text(hypothesis_id))
            if not hypothesis:
                errors.append(
                    f"关系引用未知 connection hypothesis: {compact_text(hypothesis_id)}"
                )
                continue
            if hypothesis.get("source_piece_id") not in piece_ids:
                errors.append(
                    f"关系未包含连接假设的 source_piece_id: {hypothesis['hypothesis_id']}"
                )
            if relation_type not in (
                hypothesis.get("allowed_relation_types") or []
            ):
                errors.append(
                    f"关系类型超出连接假设授权: {hypothesis['hypothesis_id']}"
                )
            if len(endpoint_people) < 2:
                errors.append(
                    f"连接假设必须连接不同数字人: {hypothesis['hypothesis_id']}"
                )
    return errors[:8]


def _adjudication_payload_errors(
    payload: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[str]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return ["decisions 必须是列表"]
    expected = {item["candidate_id"] for item in candidates}
    actual = {
        compact_text(item.get("candidate_id"))
        for item in decisions
        if isinstance(item, dict)
    }
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    errors = []
    if unknown:
        errors.append("裁决引用未知 candidate_id: " + ",".join(unknown[:3]))
    if missing:
        errors.append("裁决漏掉 candidate_id: " + ",".join(missing[:3]))
    return errors[:8]


def _classic_review_payload_errors(
    payload: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[str]:
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        return ["reviews 必须是列表"]
    expected = {
        (item["candidate_id"], hypothesis_id)
        for item in candidates
        for hypothesis_id in item.get("classic_check_hypothesis_ids") or []
    }
    actual = {
        (
            compact_text(item.get("candidate_id")),
            compact_text(item.get("hypothesis_id")),
        )
        for item in reviews
        if isinstance(item, dict)
    }
    errors = []
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        errors.append(
            "经典复核引用未知任务: "
            + ",".join(f"{left}/{right}" for left, right in unknown[:3])
        )
    if missing:
        errors.append(
            "经典复核漏掉任务: "
            + ",".join(f"{left}/{right}" for left, right in missing[:3])
        )
    for item in reviews:
        if not isinstance(item, dict):
            errors.append("经典复核项必须是对象")
            continue
        if compact_text(item.get("decision")) not in {
            "validated",
            "rejected",
        }:
            errors.append("经典复核决定必须是 validated 或 rejected")
        if not compact_text(item.get("reason")):
            errors.append("经典复核缺少具体理由")
    return errors[:8]


def _cached_stage_payload(
    cached_record: dict[str, Any],
    stage_name: str,
    *,
    input_fingerprint: str = "",
) -> dict[str, Any]:
    audit = cached_record.get("audit") or {}
    for item in audit.get("stages", []) or []:
        if item.get("stage") != stage_name:
            continue
        if not _cached_stage_record_is_reusable(item):
            return {}
        if input_fingerprint and item.get("input_fingerprint") != input_fingerprint:
            return {}
        message = ((item.get("raw") or {}).get("choices") or [{}])[0].get(
            "message"
        ) or {}
        finish_reason = compact_text(
            (((item.get("raw") or {}).get("choices") or [{}])[0] or {}).get(
                "finish_reason"
            )
        )
        if finish_reason == "length":
            recovered = _recover_complete_relation_array(
                str(message.get("content") or "")
            )
            if recovered:
                return {"relations": recovered}
        parsed = parse_json_content(str(message.get("content") or ""))
        if isinstance(parsed, dict) and "raw_text" not in parsed:
            return parsed
    return {}


def _cached_proposal_payload(
    cached_record: dict[str, Any],
    stage_name: str,
    mode: str,
    *,
    input_fingerprint: str = "",
) -> dict[str, Any]:
    if cached_record.get("pipeline_version") not in (
        PROPOSAL_CACHE_COMPATIBLE_VERSIONS
    ):
        return {}
    if not _cached_stage_is_reusable(cached_record, stage_name):
        return {}
    if input_fingerprint and not _cached_stage_has_fingerprint(
        cached_record, stage_name, input_fingerprint
    ):
        return {}
    normalized = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "relation_type",
                "piece_ids",
                "plain_language_explanation",
                "shared_coordinate",
                "source_independence",
                "common_error_risk",
                "competing_explanation",
                "falsification_test",
                "connection_hypothesis_ids",
            }
        }
        for item in cached_record.get("candidate_relations", []) or []
        if item.get("proposal_mode") == mode
    ]
    if normalized:
        cached_raw = _cached_stage_payload(
            cached_record,
            stage_name,
            input_fingerprint=input_fingerprint,
        )
        if len(cached_raw.get("relations", []) or []) > len(normalized):
            return cached_raw
        return {"relations": normalized}
    return _cached_stage_payload(
        cached_record,
        stage_name,
        input_fingerprint=input_fingerprint,
    )


def _cached_stage_has_fingerprint(
    cached_record: dict[str, Any],
    stage_name: str,
    input_fingerprint: str,
) -> bool:
    return any(
        item.get("stage") == stage_name
        and item.get("input_fingerprint") == input_fingerprint
        for item in ((cached_record.get("audit") or {}).get("stages", []) or [])
        if isinstance(item, dict)
    )


def _cached_stage_is_reusable(
    cached_record: dict[str, Any],
    stage_name: str,
) -> bool:
    return any(
        item.get("stage") == stage_name
        and _cached_stage_record_is_reusable(item)
        for item in ((cached_record.get("audit") or {}).get("stages", []) or [])
        if isinstance(item, dict)
    )


def _cached_stage_record_is_reusable(item: dict[str, Any]) -> bool:
    return item.get("status") in {
        "succeeded",
        "succeeded_truncated_recovery",
        "succeeded_targeted_recovery",
        "succeeded_with_unresolved_transport",
        "reused",
    }


def _recover_complete_relation_array(content: str) -> list[dict[str, Any]]:
    """Recover only complete JSON objects from a truncated relations array."""

    key_position = content.find('"relations"')
    array_start = content.find("[", key_position)
    if key_position < 0 or array_start < 0:
        return []
    decoder = json.JSONDecoder()
    position = array_start + 1
    complete: list[dict[str, Any]] = []
    while position < len(content) and len(complete) < 40:
        while position < len(content) and (
            content[position].isspace() or content[position] == ","
        ):
            position += 1
        if position >= len(content) or content[position] == "]":
            break
        try:
            value, end = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            break
        complete.append(value)
        position = end
    native = [
        item
        for item in complete
        if compact_text(item.get("discovery_route")) == "native"
        or not item.get("connection_hypothesis_ids")
    ][:MAX_NATIVE_PROPOSALS]
    selected_ids = {id(item) for item in native}
    classic_additions = [
        item
        for item in complete
        if id(item) not in selected_ids
        and item.get("connection_hypothesis_ids")
    ][:MAX_CLASSIC_ASSISTED_PROPOSALS]
    return [*native, *classic_additions]


def _cached_stage_raw(
    cached_record: dict[str, Any], stage_name: str
) -> dict[str, Any]:
    for item in (cached_record.get("audit") or {}).get("stages", []) or []:
        if item.get("stage") == stage_name and isinstance(item.get("raw"), dict):
            return item["raw"]
    return {}


def _select_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_person: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for piece in pieces:
        by_person[str(piece.get("person_id"))].append(piece)
    mechanism_counts = Counter(
        mechanism_id
        for piece in pieces
        for mechanism_id in piece.get("mechanism_ids", []) or []
    )
    priority = {
        "observed_claim": 0,
        "retained_local": 1,
        "distortion": 2,
        "fracture": 3,
        "assumption": 4,
        "value_condition": 5,
        "blind_spot_candidate": 6,
        "evidence_claim": 7,
    }
    per_kind_limit = {
        "observed_claim": 1,
        "retained_local": 2,
        "distortion": 2,
        "fracture": 2,
        "assumption": 2,
        "value_condition": 1,
        "blind_spot_candidate": 2,
        "evidence_claim": 2,
    }
    output = []
    for person_id, values in sorted(by_person.items()):
        counts: Counter[str] = Counter()
        selected = []
        for item in sorted(
            values,
            key=lambda value: (
                priority.get(value.get("piece_kind"), 99),
                0 if value.get("conclusion_leverage") == "high" else 1,
                -mechanism_rarity(value, mechanism_counts),
                value.get("piece_id", ""),
            ),
        ):
            kind = str(item.get("piece_kind"))
            if counts[kind] >= per_kind_limit.get(kind, 0):
                continue
            selected.append(item)
            counts[kind] += 1
            if len(selected) >= (12 if person_id == "collective" else 10):
                break
        output.extend(selected)
    return output


def _compact_piece(piece: dict[str, Any]) -> dict[str, Any]:
    coordinates = piece.get("coordinates") or {}
    knowledge = piece.get("knowledge_trace") or {}
    diagnostic_context = {
        key: _limit(value, 72)
        for key, value in (piece.get("diagnostic_context") or {}).items()
        if key
        in {
            "from_claim",
            "to_conclusion",
            "dimension",
        }
    }
    result = {
        "piece_id": piece.get("piece_id"),
        "person_id": piece.get("person_id"),
        "piece_kind": piece.get("piece_kind"),
        "text": _limit(piece.get("text"), 135),
        "coordinates": {
            "direction": coordinates.get("answer_direction"),
            "scope": _limit(coordinates.get("scope"), 60),
            "time": _limit(coordinates.get("time_horizon"), 45),
        },
        "source_lineage": _compact_source_lineage(piece),
        "mechanism_ids": piece.get("mechanism_ids") or [],
        "conclusion_leverage": piece.get("conclusion_leverage"),
    }
    parent = _limit(piece.get("parent_conclusion"), 90)
    if (
        piece.get("piece_kind") == "retained_local"
        and parent
        and parent != result["text"]
    ):
        result["parent_conclusion"] = parent
    if diagnostic_context:
        result["diagnostic_context"] = diagnostic_context
    if knowledge:
        result["classic_guard"] = {
            "lens_ids": knowledge.get("lens_ids") or [],
            "differential_statuses": knowledge.get(
                "differential_statuses"
            )
            or [],
            "inversion_authorization": knowledge.get(
                "inversion_authorization"
            ),
        }
    return result


def _proposal_piece_selection(
    pieces: list[dict[str, Any]],
    allowed_kinds: set[str],
    *,
    mode: str,
    preferred_piece_ids: set[str] | None = None,
    classic_source_piece_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    preferred_piece_ids = preferred_piece_ids or set()
    classic_source_piece_ids = classic_source_piece_ids or set()
    by_person: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pieces:
        if item.get("piece_kind") in allowed_kinds:
            by_person[str(item.get("person_id"))].append(item)
    mechanism_counts = Counter(
        mechanism_id
        for piece in pieces
        for mechanism_id in piece.get("mechanism_ids", []) or []
    )
    if mode == "content":
        limits = {
            "observed_claim": 1,
            "retained_local": 2,
            "fracture": 1,
            "assumption": 1,
            "value_condition": 1,
            "blind_spot_candidate": 1,
            "evidence_claim": 1,
        }
    else:
        limits = {
            "observed_claim": 1,
            "retained_local": 1,
            "distortion": 2,
            "fracture": 2,
            "blind_spot_candidate": 1,
            "assumption": 1,
            "value_condition": 1,
        }
    priority = {kind: index for index, kind in enumerate(limits)}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for person_id, values in sorted(by_person.items()):
        counts: Counter[str] = Counter()
        person_limit = 4 if person_id != "collective" else 6
        for item in sorted(
            values,
            key=lambda value: (
                priority.get(str(value.get("piece_kind")), 99),
                0 if value.get("conclusion_leverage") == "high" else 1,
                -mechanism_rarity(value, mechanism_counts),
                value.get("piece_id", ""),
            ),
        ):
            kind = str(item.get("piece_kind"))
            if counts[kind] >= limits.get(kind, 0):
                continue
            selected.append(item)
            selected_ids.add(str(item.get("piece_id")))
            counts[kind] += 1
            if sum(counts.values()) >= person_limit:
                break

    # Classic material is a bounded addition after the native sample is fixed.
    # It can expose a missed route, but it never displaces a piece the detective
    # would have inspected without the classic library.
    extras_by_person: Counter[str] = Counter()
    extras = sorted(
        (
            item
            for item in pieces
            if item.get("piece_kind") in allowed_kinds
            and item.get("piece_id") in preferred_piece_ids
            and str(item.get("piece_id")) not in selected_ids
        ),
        key=lambda value: (
            0 if value.get("piece_id") in classic_source_piece_ids else 1,
            0 if value.get("conclusion_leverage") == "high" else 1,
            -mechanism_rarity(value, mechanism_counts),
            priority.get(str(value.get("piece_kind")), 99),
            str(value.get("person_id")),
            str(value.get("piece_id")),
        ),
    )
    for item in extras:
        person_id = str(item.get("person_id"))
        if extras_by_person[person_id] >= 1:
            continue
        selected.append(item)
        selected_ids.add(str(item.get("piece_id")))
        extras_by_person[person_id] += 1
        if sum(extras_by_person.values()) >= MAX_CLASSIC_MATERIAL_ADDITIONS:
            break
    return selected


def _compact_mechanism_index(index: dict[str, Any]) -> dict[str, Any]:
    slots = [
        {
            "slot_id": item.get("slot_id"),
            "label": item.get("label"),
            "description": _limit(item.get("description"), 100),
            "priority": item.get("priority"),
            "material_status": item.get("material_status"),
            "piece_count": item.get("piece_count"),
            "person_count": item.get("person_count"),
        }
        for item in index.get("slots", []) or []
    ]
    return {
        "generation_mode": index.get("generation_mode"),
        "adds_model_calls": False,
        "slots": slots,
        "instruction": (
            "优先检查高影响但连接稀少的机制；空白可以保持空白，不得为了覆盖率硬连。"
        ),
    }


def _compact_connection_hypothesis(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "hypothesis_id": item.get("hypothesis_id"),
        "operator_id": item.get("operator_id"),
        "source_piece_id": item.get("source_piece_id"),
        "applicable_source_piece_ids": item.get(
            "applicable_source_piece_ids"
        )
        or [item.get("source_piece_id")],
        "source_person_id": item.get("source_person_id"),
        "source_excerpt": _limit(item.get("source_excerpt"), 140),
        "mechanism_ids": item.get("mechanism_ids") or [],
        "authorization": item.get("authorization"),
        "allowed_relation_types": item.get("allowed_relation_types") or [],
        "partner_piece_kinds": item.get("partner_piece_kinds") or [],
        "partner_must_be_different_person": True,
        "candidate_partner_piece_ids": item.get(
            "candidate_partner_piece_ids"
        )
        or [],
        "constructive_task": _limit(item.get("constructive_task"), 150),
        "search_priority_not_truth_weight": int(item.get("priority") or 0),
        "proof_obligations": [
            _limit(value, 100)
            for value in (item.get("proof_obligations") or [])[:3]
        ],
        "forbidden_inferences": [
            _limit(value, 100)
            for value in (item.get("forbidden_inferences") or [])[:3]
        ],
        "differential_statuses": item.get("differential_statuses") or [],
        "classic_as_world_evidence": False,
    }


def _source_clusters(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: defaultdict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"person_ids": set(), "piece_ids": set()}
    )
    for piece in pieces:
        for family in piece.get("source_families", []) or []:
            if family == "shared-base-model-prior":
                continue
            clusters[family]["person_ids"].add(str(piece.get("person_id")))
            clusters[family]["piece_ids"].add(str(piece.get("piece_id")))
    return [
        {
            "source_family_id": _lineage_token("source", family),
            "family_type": (
                "shared_model"
                if family == "shared-base-model-prior"
                else "external_evidence"
            ),
            "person_ids": sorted(values["person_ids"]),
            "independence_rule": "same family counts as one dependent source bundle",
        }
        for family, values in sorted(clusters.items())
        if len(values["person_ids"]) >= 2
    ]


def _compact_source_lineage(piece: dict[str, Any]) -> dict[str, Any]:
    profile = piece.get("independence_profile") or {}
    evidence = piece.get("evidence_source_families") or profile.get(
        "evidence_families"
    ) or []
    publishers = piece.get("publisher_families") or profile.get(
        "publisher_families"
    ) or []
    output = {
        "evidence_family_ids": [
            _lineage_token("evidence", family) for family in evidence[:10]
        ],
        "publisher_family_ids": [
            _lineage_token("publisher", family) for family in publishers[:10]
        ],
        "verification_statuses": profile.get("verification_statuses") or [],
        "lineage_status": profile.get("lineage_status"),
    }
    if len(evidence) > 10:
        output["evidence_family_count"] = len(evidence)
    if len(publishers) > 10:
        output["publisher_family_count"] = len(publishers)
    return {
        key: value
        for key, value in output.items()
        if value not in (None, "", [], {})
    }


def _lineage_token(kind: str, family: Any) -> str:
    return stable_id(kind, compact_text(family), size=8)


def _known_ids(values: Any, allowed: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(item) for item in values if str(item) in allowed))


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = {}
    for item in candidates:
        key = (item["relation_type"], tuple(sorted(item["piece_ids"])))
        if key not in output:
            output[key] = item
            continue
        output[key]["connection_hypothesis_ids"] = list(
            dict.fromkeys(
                [
                    *(output[key].get("connection_hypothesis_ids") or []),
                    *(item.get("connection_hypothesis_ids") or []),
                ]
            )
        )
        output[key]["rejected_connection_hypothesis_ids"] = list(
            dict.fromkeys(
                [
                    *(
                        output[key].get(
                            "rejected_connection_hypothesis_ids"
                        )
                        or []
                    ),
                    *(
                        item.get("rejected_connection_hypothesis_ids")
                        or []
                    ),
                ]
            )
        )
        output[key]["discovery_routes"] = list(
            dict.fromkeys(
                [
                    *(output[key].get("discovery_routes") or ["native"]),
                    *(item.get("discovery_routes") or ["native"]),
                ]
            )
        )
    for item in output.values():
        routes = item.get("discovery_routes") or ["native"]
        item["discovery_route"] = "dual" if len(routes) > 1 else routes[0]
    return sorted(output.values(), key=lambda item: item["candidate_id"])


def _candidate_classic_check_ids(
    candidate: dict[str, Any], case: dict[str, Any]
) -> list[str]:
    """Offer a bounded post-decision classic review without changing status."""

    piece_by_id = {
        item.get("piece_id"): item for item in case.get("piece_index", []) or []
    }
    piece_ids = set(candidate.get("piece_ids") or [])
    discovery_ids = set(candidate.get("connection_hypothesis_ids") or [])
    relation_type = candidate.get("relation_type")
    ranked: list[tuple[int, int, int, int, str]] = []
    for hypothesis in (
        case.get("classic_post_check_hypotheses")
        or case.get("classic_connection_hypotheses", [])
        or []
    ):
        hypothesis_id = compact_text(hypothesis.get("hypothesis_id"))
        source_piece_id = hypothesis.get("source_piece_id")
        applicable_source_piece_ids = set(
            hypothesis.get("applicable_source_piece_ids")
            or [source_piece_id]
        )
        matched_source_piece_ids = piece_ids.intersection(
            applicable_source_piece_ids
        )
        if (
            not hypothesis_id
            or not matched_source_piece_ids
            or relation_type not in (hypothesis.get("allowed_relation_types") or [])
        ):
            continue
        source_person_id = hypothesis.get("source_person_id")
        partner_ids = [
            piece_id
            for piece_id in piece_ids
            if piece_id not in matched_source_piece_ids
            and piece_by_id.get(piece_id, {}).get("person_id")
            != source_person_id
        ]
        allowed_partner_kinds = set(
            hypothesis.get("partner_piece_kinds") or []
        )
        compatible_partner_ids = [
            piece_id
            for piece_id in partner_ids
            if piece_by_id.get(piece_id, {}).get("piece_kind")
            in allowed_partner_kinds
        ]
        if not compatible_partner_ids:
            continue
        suggested_partner_ids = set(
            hypothesis.get("candidate_partner_piece_ids") or []
        )
        ranked.append(
            (
                0 if hypothesis_id in discovery_ids else 1,
                0
                if suggested_partner_ids.intersection(compatible_partner_ids)
                else 1,
                0 if hypothesis.get("authorization") == "provisional" else 1,
                -int(
                    hypothesis.get("search_priority_not_truth_weight") or 0
                ),
                hypothesis_id,
            )
        )
    return [
        hypothesis_id
        for *_, hypothesis_id in sorted(ranked)[
            :MAX_CLASSIC_CHECKS_PER_CANDIDATE
        ]
    ]


def _select_dual_track_proposals(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the native budget intact and append bounded classic-only finds."""

    native = [
        item
        for item in candidates
        if "native" in (item.get("discovery_routes") or [])
    ][:MAX_NATIVE_PROPOSALS]
    selected_ids = {id(item) for item in native}
    classic_additions = [
        item
        for item in candidates
        if id(item) not in selected_ids
        and "classic_assisted" in (item.get("discovery_routes") or [])
    ][:MAX_CLASSIC_ASSISTED_PROPOSALS]
    return [*native, *classic_additions]


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
