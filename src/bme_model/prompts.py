from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .meta_model.lens_registry import (
    load_calibration_cases,
    load_lens_conflicts,
    load_lenses,
    load_micro_lenses,
)
from .meta_model.forensics import DETECTOR_IDS
from .meta_model.shadow_ontology import load_capture_protocols, load_shadow_mechanisms
from .model import DigitalPerson


DIGITAL_PERSON_SYSTEM = """你是盲人摸象模型中的 AI 数字人。

你不是中立分析器，而是一个拥有有限理性、信息过滤器、价值权重和认知偏差的观察者。
你必须只按照你的认知身份证思考。不要试图覆盖所有视角，不要假装中立。
你必须区分“模型先验”和“外部证据”：模型先验是你搜索前自然会想起和相信的东西，不等同于外部事实；外部证据也必须经过你的信息过滤器采信、忽略或排斥。
你只能处理当前 payload 中的 question。不得带入其他问题的对象、案例、结论、时间窗口或专用措辞。如果材料的 question_context_id 与当前 context_id 不一致，必须拒绝使用并在输出中登记。

输出必须是 JSON，不要输出 Markdown。
"""


META_MODEL_SYSTEM = """你是盲人摸象模型中的元模型。

你的任务不是给出标准答案，而是研究数字人群思考本身。你的第一职责是逐个完成“认知链透视”：从问题理解、搜索方式、信息过滤、证据取舍、核心假设、推理过程、价值影响、结论到改判条件，完整审视每一个数字人。

9 张经典透镜与“局部整体越界”共同组成 10 个平行检测器。每个检测器都必须检查每个数字人的完整思考链；不得突出其中某一个，也不得因为没有发现问题而跳过。结论只能是：检出、暂未检出、证据不足。

经典、透镜、微透镜和机制名称是你的内部工具，不是用户语言。对用户必须用大白话一针见血地说明：问题从哪里开始，怎样影响后续思考，最终把结论带偏、带窄或带漏到哪里，以及哪些局部仍可保留。没有具体链条证据，不得贴标签；说得更长不等于看得更深。

输出必须是 JSON，不要输出 Markdown。
"""


def build_digital_person_context_packet(
    retrieval_context: dict[str, Any],
) -> dict[str, Any]:
    """Remove transport duplication while retaining every cognition-relevant item."""
    if not retrieval_context:
        return {}
    ledger = retrieval_context.get("evidence_ledger") or {}
    strategy = retrieval_context.get("search_strategy") or {}
    decisions = ledger.get("source_decisions") or []

    source_catalog: dict[str, dict[str, Any]] = {}
    compact_decisions: list[dict[str, Any]] = []
    source_fields = (
        "id",
        "query",
        "title",
        "url",
        "snippet",
        "source_name",
        "source_type",
        "evidence_type",
        "retrieval_layer",
        "verification_status",
        "retrieval_note",
    )
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        result = decision.get("result") or {}
        source_id = str(result.get("id") or "").strip()
        if not source_id:
            continue
        source_catalog[source_id] = {
            field: result.get(field) for field in source_fields
        }
        compact_decisions.append(
            {
                "source_id": source_id,
                "decision": decision.get("decision"),
                "trust_score": decision.get("trust_score"),
                "reasons": decision.get("reasons") or [],
                "shadow_hint": decision.get("shadow_hint"),
            }
        )

    for result in retrieval_context.get("candidate_sources") or []:
        if not isinstance(result, dict):
            continue
        source_id = str(result.get("id") or "").strip()
        if source_id and source_id not in source_catalog:
            source_catalog[source_id] = {
                field: result.get(field) for field in source_fields
            }

    search_requests = []
    for request in strategy.get("search_tool_requests") or []:
        if not isinstance(request, dict):
            continue
        search_requests.append(_compact_search_request(request))

    stack = retrieval_context.get("retrieval_stack") or {}
    strategy_core = {
        key: value
        for key, value in strategy.items()
        if key != "search_tool_requests"
    }
    return {
        "schema": "bme.digital-person-context-packet.v1",
        "coverage_validation": "must_pass_before_model_call",
        "question_context_id": retrieval_context.get("question_context_id"),
        "question_frame": retrieval_context.get("question_frame") or {},
        "search_strategy": strategy_core,
        "search_requests": search_requests,
        "retrieval_stack": {
            "name": stack.get("name"),
            "evidence_mode": stack.get("evidence_mode"),
            "agentic_tool_calls": stack.get("agentic_tool_calls"),
            "model_prior_enabled": stack.get("model_prior_enabled"),
            "external_provider": stack.get("external_provider"),
            "external_error_count": len(stack.get("external_errors") or []),
        },
        "source_catalog": [source_catalog[key] for key in sorted(source_catalog)],
        "source_decisions": compact_decisions,
        "decision_sets": {
            "accepted_source_ids": _decision_source_ids(
                ledger.get("accepted_sources") or []
            ),
            "rejected_source_ids": _decision_source_ids(
                ledger.get("rejected_sources") or []
            ),
            "ignored_source_ids": _decision_source_ids(
                ledger.get("ignored_sources") or []
            ),
        },
        "information_shadow_hints": ledger.get("information_shadow_hints") or [],
        "accepted_evidence_entries": ledger.get("entries") or [],
        "source_layer_rules": retrieval_context.get("source_layer_rules") or {},
        "instruction": retrieval_context.get("instruction"),
    }


def validate_digital_person_context_packet(
    packet: dict[str, Any], retrieval_context: dict[str, Any]
) -> list[str]:
    if not retrieval_context:
        return [] if not packet else ["empty retrieval context produced a packet"]
    errors: list[str] = []
    ledger = retrieval_context.get("evidence_ledger") or {}
    strategy = retrieval_context.get("search_strategy") or {}
    original_decisions = [
        item
        for item in ledger.get("source_decisions") or []
        if isinstance(item, dict) and (item.get("result") or {}).get("id")
    ]
    original_sources = {
        str((item.get("result") or {}).get("id")): item.get("result") or {}
        for item in original_decisions
    }
    for result in retrieval_context.get("candidate_sources") or []:
        if isinstance(result, dict) and result.get("id"):
            original_sources.setdefault(str(result["id"]), result)
    packet_sources = {
        str(item.get("id")): item
        for item in packet.get("source_catalog") or []
        if isinstance(item, dict) and item.get("id")
    }
    if set(packet_sources) != set(original_sources):
        errors.append("source catalog IDs do not exactly cover retrieval sources")
    source_fields = (
        "id",
        "query",
        "title",
        "url",
        "snippet",
        "source_name",
        "source_type",
        "evidence_type",
        "retrieval_layer",
        "verification_status",
        "retrieval_note",
    )
    for source_id in set(packet_sources) & set(original_sources):
        if any(
            packet_sources[source_id].get(field)
            != original_sources[source_id].get(field)
            for field in source_fields
        ):
            errors.append(f"source material changed for {source_id}")

    packet_decisions = {
        str(item.get("source_id")): item
        for item in packet.get("source_decisions") or []
        if isinstance(item, dict) and item.get("source_id")
    }
    expected_decisions = {
        str((item.get("result") or {}).get("id")): item
        for item in original_decisions
    }
    if set(packet_decisions) != set(expected_decisions):
        errors.append("source decisions do not exactly cover the evidence ledger")
    for source_id in set(packet_decisions) & set(expected_decisions):
        original = expected_decisions[source_id]
        expected = {
            "source_id": source_id,
            "decision": original.get("decision"),
            "trust_score": original.get("trust_score"),
            "reasons": original.get("reasons") or [],
            "shadow_hint": original.get("shadow_hint"),
        }
        if packet_decisions[source_id] != expected:
            errors.append(f"source decision changed for {source_id}")

    expected_sets = {
        "accepted_source_ids": _decision_source_ids(
            ledger.get("accepted_sources") or []
        ),
        "rejected_source_ids": _decision_source_ids(
            ledger.get("rejected_sources") or []
        ),
        "ignored_source_ids": _decision_source_ids(
            ledger.get("ignored_sources") or []
        ),
    }
    if packet.get("decision_sets") != expected_sets:
        errors.append("accepted/rejected/ignored source sets changed")
    if packet.get("accepted_evidence_entries") != (ledger.get("entries") or []):
        errors.append("accepted evidence entries changed")
    if packet.get("information_shadow_hints") != (
        ledger.get("information_shadow_hints") or []
    ):
        errors.append("information shadow hints changed")

    expected_strategy = {
        key: value
        for key, value in strategy.items()
        if key != "search_tool_requests"
    }
    if packet.get("search_strategy") != expected_strategy:
        errors.append("search strategy changed")
    original_requests = strategy.get("search_tool_requests") or []
    compact_requests = packet.get("search_requests") or []
    if len(original_requests) != len(compact_requests):
        errors.append("search request count changed")
    else:
        for index, (original, compact) in enumerate(
            zip(original_requests, compact_requests), start=1
        ):
            if not isinstance(original, dict) or compact != _compact_search_request(original):
                errors.append(f"search request {index} changed")

    if packet.get("question_context_id") != retrieval_context.get(
        "question_context_id"
    ):
        errors.append("question context ID changed")
    if packet.get("question_frame") != (retrieval_context.get("question_frame") or {}):
        errors.append("question frame changed")
    stack = retrieval_context.get("retrieval_stack") or {}
    expected_stack = {
        "name": stack.get("name"),
        "evidence_mode": stack.get("evidence_mode"),
        "agentic_tool_calls": stack.get("agentic_tool_calls"),
        "model_prior_enabled": stack.get("model_prior_enabled"),
        "external_provider": stack.get("external_provider"),
        "external_error_count": len(stack.get("external_errors") or []),
    }
    if packet.get("retrieval_stack") != expected_stack:
        errors.append("retrieval stack summary changed")
    if packet.get("source_layer_rules") != (
        retrieval_context.get("source_layer_rules") or {}
    ):
        errors.append("source layer rules changed")
    if packet.get("instruction") != retrieval_context.get("instruction"):
        errors.append("digital-person instruction changed")
    return errors


def _compact_search_request(request: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize both executed and fallback search-request records.

    Fallback requests legitimately omit provider execution fields. Normalizing
    those absent list fields to empty lists is lossless, but the validator must
    use the same representation as the packet builder.
    """
    return {
        "tool_call_id": request.get("tool_call_id"),
        "query": request.get("query"),
        "why_this_person_searches_it": request.get("why_this_person_searches_it"),
        "evidence_sought": request.get("evidence_sought"),
        "requested_limit": request.get("requested_limit"),
        "providers_attempted": request.get("providers_attempted") or [],
        "providers_succeeded": request.get("providers_succeeded") or [],
        "result_count": request.get("result_count"),
        "status": request.get("status"),
        "provider_failure_count": len(request.get("provider_errors") or []),
    }


def _decision_source_ids(decisions: list[Any]) -> list[str]:
    return [
        str((item.get("result") or {}).get("id"))
        for item in decisions
        if isinstance(item, dict) and (item.get("result") or {}).get("id")
    ]


def render_digital_person_messages(
    question: str,
    person: DigitalPerson,
    retrieval_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    context = dict(retrieval_context or {})
    context_mode = str(context.pop("_prompt_context_mode", "legacy_full"))
    if context_mode == "lossless_packet_v2":
        prompt_context = build_digital_person_context_packet(context)
        packet_errors = validate_digital_person_context_packet(
            prompt_context, context
        )
        if packet_errors:
            raise ValueError(
                "Digital-person context packet failed coverage validation: "
                + "; ".join(packet_errors)
            )
        prompt_context["coverage_validation"] = "passed"
    elif context_mode == "legacy_full":
        prompt_context = context
    else:
        raise ValueError(f"unknown digital-person prompt context mode: {context_mode}")
    payload = {
        "question": question,
        "cognitive_identity": asdict(person),
        "retrieval_context_mode": context_mode,
        "retrieval_context": prompt_context,
        "required_output": {
            "person_id": person.id,
            "question": question,
            "search_strategy": {
                "preferred_queries": [],
                "trusted_sources": [],
                "distrusted_sources": [],
                "ignored_evidence_types": [],
            },
            "model_prior_before_search": [],
            "source_layer_usage": {
                "model_prior": [],
                "external_search": [],
                "mock_search": [],
            },
            "evidence_ledger": [],
            "core_assumptions": [],
            "reasoning_path": [],
            "value_judgements": [],
            "conclusion": "",
            "answer_position": {
                "directness": "direct|reframed|uncertain",
                "direction": "yes|no|conditional|uncertain|not_applicable",
                "scope": "这项判断实际覆盖到哪里",
                "exact_answer": "只用一句话直接回应当前 question",
            },
            "confidence": 0.0,
            "what_i_underweighted": [],
            "what_would_change_my_mind": [],
        },
    }
    return [
        {"role": "system", "content": DIGITAL_PERSON_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def render_meta_model_messages(
    question: str,
    person_outputs: list[dict[str, Any]],
    diagnostic_report: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "digital_person_outputs": person_outputs,
        "diagnostic_lens_protocol": {
            "instruction": "把经典知识库当作内部诊断工具，不当作用户可见的术语书单。每项判断必须钉到该数字人的搜索、来源决定、证据、假设、推理步骤、价值权重、结论或改判条件。",
            "cognitive_chain_scan_protocol": {
                "chain_stages": [
                    "problem_framing",
                    "information_filter",
                    "retrieval",
                    "evidence_selection",
                    "assumption",
                    "reasoning",
                    "value_evaluation",
                    "conclusion",
                    "self_correction",
                ],
                "detectors": list(DETECTOR_IDS),
                "equal_status_rule": "10 个检测器平行工作；局部整体越界只是其中之一。",
                "decision_states": ["detected", "not_detected", "insufficient_evidence"],
                "proof_burden": [
                    "指出具体发生位置和可观察材料。",
                    "解释阴影怎样生成并传到结论。",
                    "提出至少一个竞争解释，防止误诊。",
                    "给出能推翻本诊断的条件。",
                    "说明受到影响的部分与仍可保留的部分。",
                ],
                "scope_rule": "本阶段只完成逐个数字人的认知链透视，不建立跨数字人的错、对、没看见连接。",
            },
            "available_lenses": _render_lens_cards_for_prompt(),
            "available_micro_lenses": _render_micro_lenses_for_prompt(),
            "lens_conflict_rules": _render_lens_conflicts_for_prompt(),
            "calibration_cases": _render_calibration_cases_for_prompt(),
            "shadow_mechanisms": _render_shadow_mechanisms_for_prompt(),
            "capture_protocols": _render_capture_protocols_for_prompt(),
            "required_modules": [
                "cognitive_chain_scan",
                "ten_detector_panel",
                "differential_diagnosis",
                "plain_language_compressor",
                "shadow_inverter",
                "self_auditor",
            ],
        },
        "first_pass_diagnostic_report": diagnostic_report or {},
        "required_output": {
            "question": question,
            "cognitive_terrain": {
                "stable_anchors": [],
                "constraint_bands": [],
                "negative_spaces": [],
                "major_disagreements": [],
                "irreducible_conflicts": [],
            },
            "shadow_registry": [],
            "cognitive_chain_conclusions": [
                {
                    "person_id": "",
                    "headline": "",
                    "key_problems": [
                        {
                            "problem": "",
                            "where_it_appeared": "",
                            "how_it_affected_thinking": "",
                            "what_can_still_be_kept": "",
                        }
                    ],
                    "unresolved_checks": [],
                }
            ],
            "chain_detector_coverage": [
                {
                    "person_id": "",
                    "checked_detector_ids": [],
                    "insufficient_detector_ids": [],
                }
            ],
            "bias_attribution": [],
            "collective_blind_spots": [],
            "candidate_truth_contour": "",
            "next_probe": {
                "new_evidence_needed": [],
                "new_digital_persons_needed": [],
                "stress_tests": [],
            },
            "meta_reflection": {
                "possible_meta_biases": [],
                "underweighted_dimensions": [],
                "confidence": 0.0,
            },
        },
    }
    return [
        {"role": "system", "content": META_MODEL_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _render_lens_cards_for_prompt() -> list[dict[str, Any]]:
    lens_cards = []
    for lens in load_lenses().values():
        lens_cards.append(
            {
                "id": lens.id,
                "name": lens.name,
                "source_works": lens.source_works,
                "core_idea": lens.core_idea,
                "diagnostic_questions": lens.diagnostic_questions,
                "shadow_types": lens.shadow_types,
                "source_stages": lens.source_stages,
                "mechanism_chain": lens.mechanism_chain,
                "observable_markers": lens.observable_markers,
                "attribution_rules": lens.attribution_rules,
                "inversion_rules": lens.inversion_rules,
                "puzzle_uses": lens.puzzle_uses,
                "boundary_conditions": lens.boundary_conditions,
                "anti_misuse_rules": lens.anti_misuse_rules,
                "model_training_notes": lens.model_training_notes,
                "self_reflection": lens.self_reflection,
            }
        )
    return lens_cards


def _render_micro_lenses_for_prompt() -> list[dict[str, Any]]:
    micro_lenses = []
    for lens in load_micro_lenses().values():
        micro_lenses.append(
            {
                "id": lens.id,
                "parent_lens": lens.parent_lens,
                "name": lens.name,
                "source_work": lens.source_work,
                "mechanism": lens.mechanism,
                "trigger_signals": lens.trigger_signals,
                "diagnostic_questions": lens.diagnostic_questions,
                "differentiates_from": lens.differentiates_from,
                "inversion_hint": lens.inversion_hint,
                "misuse_warning": lens.misuse_warning,
            }
        )
    return micro_lenses


def _render_lens_conflicts_for_prompt() -> list[dict[str, Any]]:
    return [
        {
            "id": rule.id,
            "competing_lenses": rule.competing_lenses,
            "confusion_pattern": rule.confusion_pattern,
            "decision_questions": rule.decision_questions,
            "prefer_first_when": rule.prefer_first_when,
            "prefer_second_when": rule.prefer_second_when,
            "coexist_when": rule.coexist_when,
            "misdiagnosis_risk": rule.misdiagnosis_risk,
        }
        for rule in load_lens_conflicts()
    ]


def _render_calibration_cases_for_prompt() -> list[dict[str, Any]]:
    return [
        {
            "id": case.id,
            "title": case.title,
            "scenario": case.scenario,
            "expected_primary_lenses": case.expected_primary_lenses,
            "expected_confusions": case.expected_confusions,
            "calibration_questions": case.calibration_questions,
            "failure_mode": case.failure_mode,
        }
        for case in load_calibration_cases()
    ]


def _render_shadow_mechanisms_for_prompt() -> list[dict[str, Any]]:
    return [
        {
            "id": mechanism.id,
            "name": mechanism.name,
            "shadow_family": mechanism.shadow_family,
            "core_mechanism": mechanism.core_mechanism,
            "generation_chain": mechanism.generation_chain,
            "chain_stages": mechanism.chain_stages,
            "visible_signals": mechanism.visible_signals,
            "capture_questions": mechanism.capture_questions,
            "false_positive_risks": mechanism.false_positive_risks,
            "false_negative_risks": mechanism.false_negative_risks,
            "linked_lenses": mechanism.linked_lenses,
            "puzzle_material": mechanism.puzzle_material,
        }
        for mechanism in load_shadow_mechanisms().values()
    ]


def _render_capture_protocols_for_prompt() -> list[dict[str, Any]]:
    return [
        {
            "stage": stage.stage,
            "touch_target": stage.touch_target,
            "what_to_inspect": stage.what_to_inspect,
            "shadow_questions": stage.shadow_questions,
            "evidence_needed": stage.evidence_needed,
            "failure_if_skipped": stage.failure_if_skipped,
        }
        for stage in load_capture_protocols()
    ]
