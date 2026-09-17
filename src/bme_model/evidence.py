from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .model import DigitalPerson, EvidenceLedger
from .search import SearchResult

Decision = Literal["accept", "reject", "ignore"]


@dataclass(frozen=True)
class SourceDecision:
    result: SearchResult
    decision: Decision
    trust_score: float
    reasons: list[str]
    shadow_hint: str | None = None


def decide_source(
    person: DigitalPerson,
    result: SearchResult,
    semantic_decision: dict[str, Any] | None = None,
) -> SourceDecision:
    if result.retrieval_layer == "external_search" and semantic_decision is not None:
        return _semantic_source_decision(result, semantic_decision)

    trust_score = 0.5
    reasons: list[str] = []
    shadow_hint: str | None = None
    retrieval_layer = getattr(result, "retrieval_layer", "external_search")

    if retrieval_layer == "model_prior":
        trust_score += 0.05
        reasons.append("模型先验：用于显影搜索前预设，不等同于外部事实。")
        shadow_hint = "先验阴影：这个数字人进入搜索前已经带着稳定的注意力偏向。"
    elif retrieval_layer == "mock_search":
        reasons.append("模拟搜索：用于开发测试，不等同于真实外部证据。")

    if result.source_type in person.information_filter.trusted_sources:
        trust_score += 0.35
        reasons.append(f"来源类型匹配信任列表：{result.source_type}")
    if result.evidence_type in person.information_filter.preferred_evidence:
        trust_score += 0.25
        reasons.append(f"证据类型匹配偏好：{result.evidence_type}")
    if result.source_type in person.information_filter.distrusted_sources:
        trust_score -= 0.45
        reasons.append(f"来源类型命中低信任列表：{result.source_type}")
        shadow_hint = "信息阴影：该数字人可能系统性排斥这一来源类型。"
    if result.evidence_type in person.information_filter.ignored_evidence:
        trust_score -= 0.35
        reasons.append(f"证据类型命中忽略列表：{result.evidence_type}")
        shadow_hint = "信息阴影：该数字人可能系统性忽略这一证据类型。"

    trust_score = max(0.0, min(1.0, trust_score))

    if trust_score >= 0.68:
        decision: Decision = "accept"
    elif trust_score <= 0.28:
        decision = "reject"
    else:
        decision = "ignore"
        if shadow_hint is None:
            shadow_hint = "信息阴影：该来源处于低注意力区，可能不会进入该数字人的推理。"

    if not reasons:
        reasons.append("未明显匹配信任、排斥、偏好或忽略规则。")

    return SourceDecision(
        result=result,
        decision=decision,
        trust_score=round(trust_score, 3),
        reasons=reasons,
        shadow_hint=shadow_hint,
    )


def build_evidence_ledger(
    question: str,
    person: DigitalPerson,
    search_strategy: dict[str, object],
    candidates: list[SearchResult],
    semantic_source_decisions: list[dict[str, Any]] | None = None,
) -> EvidenceLedger:
    semantic_by_id = {
        str(item.get("result_id")): item
        for item in (semantic_source_decisions or [])
        if isinstance(item, dict) and item.get("result_id")
    }
    decisions = [
        decide_source(person, result, semantic_by_id.get(result.id))
        for result in candidates
    ]
    accepted = [decision for decision in decisions if decision.decision == "accept"]
    rejected = [decision for decision in decisions if decision.decision == "reject"]
    ignored = [decision for decision in decisions if decision.decision == "ignore"]

    entries = [
        {
            "entry_id": decision.result.id,
            "source_id": decision.result.id,
            "query": decision.result.query,
            "claim": decision.result.snippet,
            "source": decision.result.url,
            "source_name": decision.result.source_name,
            "source_title": decision.result.title,
            "source_type": decision.result.source_type,
            "evidence_type": decision.result.evidence_type,
            "retrieval_layer": decision.result.retrieval_layer,
            "verification_status": decision.result.verification_status,
            "evidence_use_role": _evidence_use_role(decision.result),
            "trust_reason": "; ".join(decision.reasons),
            "used_for": _used_for(person, decision.result),
            "confidence": decision.trust_score,
        }
        for decision in accepted
    ]

    shadow_hints = [
        hint
        for hint in (decision.shadow_hint for decision in decisions)
        if hint is not None
    ]

    return EvidenceLedger(
        person_id=person.id,
        question=question,
        search_strategy=search_strategy,
        source_decisions=[_decision_to_dict(decision) for decision in decisions],
        accepted_sources=[_decision_to_dict(decision) for decision in accepted],
        rejected_sources=[_decision_to_dict(decision) for decision in rejected],
        ignored_sources=[_decision_to_dict(decision) for decision in ignored],
        information_shadow_hints=shadow_hints,
        entries=entries,
    )


def _decision_to_dict(decision: SourceDecision) -> dict[str, object]:
    return {
        "decision": decision.decision,
        "trust_score": decision.trust_score,
        "reasons": decision.reasons,
        "shadow_hint": decision.shadow_hint,
        "result": asdict(decision.result),
    }


def _used_for(person: DigitalPerson, result: SearchResult) -> str:
    if result.retrieval_layer == "model_prior":
        return "显影该数字人搜索前会自然带入的先验，不作为外部事实"
    if result.evidence_type in person.information_filter.preferred_evidence:
        return "支撑该数字人偏好的核心证据通道"
    if result.source_type in person.information_filter.trusted_sources:
        return "支撑该数字人信任的信息来源通道"
    return "补充背景证据"


def _evidence_use_role(result: SearchResult) -> str:
    if result.retrieval_layer in {"model_prior", "mock_search"}:
        return "shadow_only"
    if result.verification_status in {
        "page_verified",
        "primary_verified",
        "cross_verified",
    }:
        return "directional_eligible"
    return "candidate_observation"


def _semantic_source_decision(
    result: SearchResult,
    payload: dict[str, Any],
) -> SourceDecision:
    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject", "ignore"}:
        raise ValueError(f"Invalid semantic source decision for {result.id}: {decision!r}")
    try:
        trust_score = float(payload.get("trust_score"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing semantic trust score for {result.id}") from exc
    reason = str(payload.get("reason") or "").strip()
    filter_basis = str(payload.get("filter_basis") or "").strip()
    if not reason or not filter_basis:
        raise ValueError(f"Incomplete semantic source decision for {result.id}")
    shadow_hint = None
    if decision == "reject":
        shadow_hint = "信息阴影：该数字人的过滤器明确排斥了这条外部材料。"
    elif decision == "ignore":
        shadow_hint = "信息阴影：该数字人看见了这条外部材料，但没有让它进入推理。"
    return SourceDecision(
        result=result,
        decision=decision,  # type: ignore[arg-type]
        trust_score=round(max(0.0, min(1.0, trust_score)), 3),
        reasons=[f"数字人语义筛选：{reason}", f"过滤依据：{filter_basis}"],
        shadow_hint=shadow_hint,
    )
