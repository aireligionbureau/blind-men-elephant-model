from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..evidence_normalization import normalize_output_evidence_entries
from .classic_connectors import build_connection_hypotheses
from .mechanisms import build_mechanism_index
from .knowledge_trace import (
    build_classic_trace_bundle,
    piece_knowledge_trace,
    traces_for_verdict,
)
from .schemas import (
    PUZZLE_SCHEMA_VERSION,
    compact_text,
    normalize_claim,
    question_id,
    stable_id,
    string_list,
    unique_strings,
    validate_puzzle_materials,
)


def materialize_puzzle_pieces(
    question: str,
    diagnosis: dict[str, Any],
    person_outputs: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]] | None = None,
    question_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert sealed per-person chains and verdicts into traceable pieces.

    The materializer does not decide what is true and does not connect people.
    It only creates the standardized material the detective layer may inspect.
    """

    qid = question_id(question)
    outputs = _index_outputs(person_outputs)
    ledgers = {
        str(item.get("person_id", "")): item
        for item in (evidence_ledgers or [])
        if item.get("person_id")
    }
    scans = {
        str(item.get("person_id", "")): item
        for item in diagnosis.get("cognitive_chain_scans", []) or []
        if item.get("person_id")
    }
    source_catalog = _build_source_catalog(ledgers)
    source_family_by_id = {
        source_id: item["source_family"]
        for source_id, item in source_catalog.items()
    }
    classic_bundle = build_classic_trace_bundle(question, diagnosis)
    diagnostic_bridges: list[dict[str, Any]] = []
    source_resolution_counts: Counter[str] = Counter()

    pieces: list[dict[str, Any]] = []
    for person_id, record in sorted(outputs.items()):
        output = record.get("output", record)
        person = record.get("person", {}) or {}
        scan = scans.get(person_id, {})
        conclusion = compact_text(output.get("conclusion"))
        answer_position = output.get("answer_position") or {}
        evidence_entries = normalize_output_evidence_entries(
            output.get("evidence_ledger")
        )
        resolved_evidence = [
            _resolve_evidence_entry_source(
                item,
                ledger=ledgers.get(person_id, {}),
                source_catalog=source_catalog,
            )
            for item in evidence_entries
        ]
        source_resolution_counts.update(
            mode for _, mode in resolved_evidence
        )
        output_source_ids = unique_strings(
            source_id for source_id, _ in resolved_evidence
        )

        if conclusion:
            pieces.append(
                _piece(
                    qid=qid,
                    person_id=person_id,
                    person_name=person.get("name") or record.get("person_name"),
                    piece_kind="observed_claim",
                    text=conclusion,
                    status="unassessed_observation",
                    origin="person_output.conclusion",
                    anchor_ids=["C1"],
                    source_ids=output_source_ids,
                    source_family_by_id=source_family_by_id,
                    source_catalog_by_id=source_catalog,
                    question=question,
                    answer_position=answer_position,
                    time_horizon=person.get("time_horizon"),
                    conclusion_leverage="high",
                    parent_text=conclusion,
                )
            )

        for index, assumption in enumerate(string_list(output.get("core_assumptions")), start=1):
            pieces.append(
                _piece(
                    qid=qid,
                    person_id=person_id,
                    person_name=person.get("name"),
                    piece_kind="assumption",
                    text=assumption,
                    status="declared_assumption",
                    origin="person_output.core_assumptions",
                    anchor_ids=[f"A{index}"],
                    source_ids=[],
                    source_family_by_id=source_family_by_id,
                    question=question,
                    answer_position=answer_position,
                    time_horizon=person.get("time_horizon"),
                    conclusion_leverage="medium",
                    parent_text=conclusion,
                )
            )

        for index, value in enumerate(string_list(output.get("value_judgements")), start=1):
            pieces.append(
                _piece(
                    qid=qid,
                    person_id=person_id,
                    person_name=person.get("name"),
                    piece_kind="value_condition",
                    text=value,
                    status="declared_value_condition",
                    origin="person_output.value_judgements",
                    anchor_ids=[f"V{index}"],
                    source_ids=[],
                    source_family_by_id=source_family_by_id,
                    question=question,
                    answer_position=answer_position,
                    time_horizon=person.get("time_horizon"),
                    conclusion_leverage="medium",
                    parent_text=conclusion,
                )
            )

        for index, evidence in enumerate(evidence_entries[:12], start=1):
            claim = compact_text(evidence.get("claim"))
            source_id = (
                resolved_evidence[index - 1][0]
                if index <= len(resolved_evidence)
                else ""
            )
            if not claim:
                continue
            pieces.append(
                _piece(
                    qid=qid,
                    person_id=person_id,
                    person_name=person.get("name"),
                    piece_kind="evidence_claim",
                    text=claim,
                    status="source_claim_unverified",
                    origin="person_output.evidence_ledger",
                    anchor_ids=[f"PE{index}"],
                    source_ids=[source_id] if source_id else [],
                    source_family_by_id=source_family_by_id,
                    source_catalog_by_id=source_catalog,
                    question=question,
                    answer_position=answer_position,
                    time_horizon=person.get("time_horizon"),
                    conclusion_leverage="low",
                    parent_text=conclusion,
                )
            )

        semantic = scan.get("semantic_verdict") or {}
        for verdict_index, verdict in enumerate(semantic.get("verdicts", []) or [], start=1):
            diagnostic_id = stable_id(
                "diagnosis",
                qid,
                person_id,
                verdict_index,
                verdict.get("issue"),
            )
            bridge_id = stable_id("diagnostic_bridge", qid, diagnostic_id)
            source_detector_ids = unique_strings(
                verdict.get("source_detector_ids") or []
            )
            linked_traces = traces_for_verdict(
                classic_bundle,
                person_id,
                source_detector_ids,
            )
            bridge_piece_ids: list[str] = []
            preserved = compact_text(verdict.get("preserved_fragment"))
            if preserved:
                retained_piece = _piece(
                        qid=qid,
                        person_id=person_id,
                        person_name=person.get("name"),
                        piece_kind="retained_local",
                        text=preserved,
                        status="candidate_retained_not_truth",
                        origin="forensic.semantic_verdict.preserved_fragment",
                        anchor_ids=verdict.get("evidence_refs") or [],
                        source_ids=[],
                        source_family_by_id=source_family_by_id,
                        question=question,
                        answer_position=answer_position,
                        time_horizon=person.get("time_horizon"),
                        conclusion_leverage=verdict.get("conclusion_leverage", "medium"),
                        parent_text=conclusion,
                        diagnostic_ids=[diagnostic_id],
                        knowledge_trace=_bridge_knowledge_trace(
                            piece_knowledge_trace(
                                linked_traces,
                                classic_bundle,
                                transformation="preserved_fragment_candidate",
                            ),
                            bridge_id,
                        ),
                    )
                pieces.append(retained_piece)
                bridge_piece_ids.append(retained_piece["piece_id"])

            issue = compact_text(verdict.get("issue"))
            if issue:
                distortion_piece = _piece(
                        qid=qid,
                        person_id=person_id,
                        person_name=person.get("name"),
                        piece_kind="distortion",
                        text=issue,
                        status="diagnosed_distortion",
                        origin="forensic.semantic_verdict.issue",
                        anchor_ids=verdict.get("evidence_refs") or [],
                        source_ids=[],
                        source_family_by_id=source_family_by_id,
                        question=question,
                        answer_position=answer_position,
                        time_horizon=person.get("time_horizon"),
                        conclusion_leverage=verdict.get("conclusion_leverage", "medium"),
                        parent_text=conclusion,
                        diagnostic_ids=[diagnostic_id],
                        knowledge_trace=_bridge_knowledge_trace(
                            piece_knowledge_trace(
                                linked_traces,
                                classic_bundle,
                                transformation="diagnosed_distortion",
                            ),
                            bridge_id,
                        ),
                        shadow_operators=_shadow_operators(
                            " ".join(
                                [
                                    issue,
                                    compact_text(verdict.get("missing_bridge")),
                                    compact_text(verdict.get("impact_on_conclusion")),
                                ]
                            )
                        ),
                        extra={
                            "from_claim": compact_text(verdict.get("from_claim")),
                            "to_conclusion": compact_text(verdict.get("to_conclusion")),
                            "impact_on_conclusion": compact_text(verdict.get("impact_on_conclusion")),
                            "competing_explanation": compact_text(verdict.get("competing_explanation")),
                            "falsification_test": compact_text(verdict.get("falsification_test")),
                        },
                    )
                pieces.append(distortion_piece)
                bridge_piece_ids.append(distortion_piece["piece_id"])

            missing_bridge = compact_text(verdict.get("missing_bridge"))
            if missing_bridge:
                fracture_piece = _piece(
                        qid=qid,
                        person_id=person_id,
                        person_name=person.get("name"),
                        piece_kind="fracture",
                        text=missing_bridge,
                        status="diagnosed_inference_fracture",
                        origin="forensic.semantic_verdict.missing_bridge",
                        anchor_ids=verdict.get("evidence_refs") or [],
                        source_ids=[],
                        source_family_by_id=source_family_by_id,
                        question=question,
                        answer_position=answer_position,
                        time_horizon=person.get("time_horizon"),
                        conclusion_leverage=verdict.get("conclusion_leverage", "medium"),
                        parent_text=conclusion,
                        diagnostic_ids=[diagnostic_id],
                        knowledge_trace=_bridge_knowledge_trace(
                            piece_knowledge_trace(
                                linked_traces,
                                classic_bundle,
                                transformation="missing_bridge_boundary",
                            ),
                            bridge_id,
                        ),
                        shadow_operators=["missing_reasoning_bridge"],
                        extra={
                            "from_claim": compact_text(verdict.get("from_claim")),
                            "to_conclusion": compact_text(verdict.get("to_conclusion")),
                        },
                    )
                pieces.append(fracture_piece)
                bridge_piece_ids.append(fracture_piece["piece_id"])

            diagnostic_bridges.append(
                _diagnostic_bridge(
                    bridge_id=bridge_id,
                    diagnostic_id=diagnostic_id,
                    person_id=person_id,
                    verdict_index=verdict_index,
                    verdict=verdict,
                    semantic=semantic,
                    source_detector_ids=source_detector_ids,
                    linked_traces=linked_traces,
                    piece_ids=bridge_piece_ids,
                )
            )

        unresolved = unique_strings(
            list(semantic.get("unresolved", []) or [])
            + list(output.get("what_i_underweighted", []) or [])
        )
        for index, item in enumerate(unresolved[:8], start=1):
            pieces.append(
                _piece(
                    qid=qid,
                    person_id=person_id,
                    person_name=person.get("name"),
                    piece_kind="blind_spot_candidate",
                    text=item,
                    status="candidate_unverified",
                    origin="forensic.unresolved_or_underweighted",
                    anchor_ids=[f"U{index}"],
                    source_ids=[],
                    source_family_by_id=source_family_by_id,
                    question=question,
                    answer_position=answer_position,
                    time_horizon=person.get("time_horizon"),
                    conclusion_leverage="unknown",
                    parent_text=conclusion,
                    relevance_status="candidate_unverified",
                )
            )

    pieces.extend(
        _collective_blind_spot_pieces(
            qid,
            question,
            diagnosis,
            source_family_by_id,
        )
    )
    pieces = _deduplicate_pieces(pieces)
    mechanism_index = build_mechanism_index(question, question_frame, pieces)
    classic_connectors = build_connection_hypotheses(
        question,
        pieces,
        classic_bundle["inversion_certificates"],
    )
    counts = Counter(piece["piece_kind"] for piece in pieces)
    payload = {
        "schema_version": PUZZLE_SCHEMA_VERSION,
        "question": compact_text(question),
        "question_id": qid,
        "person_count": len(outputs),
        "pieces": pieces,
        "source_catalog": list(source_catalog.values()),
        "mechanism_index": mechanism_index,
        "classic_connection_hypotheses": classic_connectors,
        "classic_diagnostic_traces": classic_bundle[
            "classic_diagnostic_traces"
        ],
        "inversion_certificates": classic_bundle["inversion_certificates"],
        "diagnostic_bridges": diagnostic_bridges,
        "material_counts": dict(sorted(counts.items())),
        "materialization_audit": {
            "rule": "未检出阴影不等于真相；盲区候选不等于已证明空白；不执行未经校准的数值反演。",
            "piece_count": len(pieces),
            "traceable_piece_count": sum(
                1 for piece in pieces if piece.get("provenance", {}).get("origin")
            ),
            "candidate_retained_count": counts.get("retained_local", 0),
            "distortion_count": counts.get("distortion", 0),
            "fracture_count": counts.get("fracture", 0),
            "blind_spot_candidate_count": counts.get("blind_spot_candidate", 0),
            "numeric_inversion_performed": False,
            "classic_connection_hypothesis_count": classic_connectors[
                "hypothesis_count"
            ],
            "classic_connection_operator_count": classic_connectors[
                "operators_loaded"
            ],
            "classic_connector_adds_model_calls": False,
            "source_resolution_counts": dict(source_resolution_counts),
            "source_fallback_match_count": source_resolution_counts.get(
                "fallback_unique_source_match", 0
            ),
            "source_fallback_unresolved_count": source_resolution_counts.get(
                "fallback_unresolved", 0
            ),
            **_source_lineage_audit(pieces, source_catalog),
            **_classic_trace_audit(
                pieces,
                classic_bundle["classic_diagnostic_traces"],
                classic_bundle["inversion_certificates"],
                diagnostic_bridges,
            ),
        },
    }
    errors = validate_puzzle_materials(payload, question)
    if errors:
        raise ValueError("Invalid puzzle materials: " + "; ".join(errors))
    return payload


def _bridge_knowledge_trace(
    knowledge_trace: dict[str, Any], bridge_id: str
) -> dict[str, Any]:
    return {
        **knowledge_trace,
        "diagnostic_bridge_ids": [bridge_id],
    }


def _diagnostic_bridge(
    *,
    bridge_id: str,
    diagnostic_id: str,
    person_id: str,
    verdict_index: int,
    verdict: dict[str, Any],
    semantic: dict[str, Any],
    source_detector_ids: list[str],
    linked_traces: list[dict[str, Any]],
    piece_ids: list[str],
) -> dict[str, Any]:
    trace_ids = [item["trace_id"] for item in linked_traces]
    certificate_ids = unique_strings(
        item.get("inversion_certificate_id") for item in linked_traces
    )
    evidence_validation = verdict.get("evidence_validation") or {}
    semantic_validation = semantic.get("validation") or {}
    verdict_valid = evidence_validation.get("valid") is not False
    batch_valid = semantic_validation.get("valid") is not False
    return {
        "bridge_id": bridge_id,
        "diagnostic_id": diagnostic_id,
        "person_id": person_id,
        "semantic_verdict_index": verdict_index,
        "semantic_issue": compact_text(verdict.get("issue")),
        "source_detector_ids": source_detector_ids,
        "evidence_refs": unique_strings(verdict.get("evidence_refs") or []),
        "semantic_validation": {
            "verdict_valid": verdict_valid,
            "batch_valid": batch_valid,
            "specificity_score": evidence_validation.get("specificity_score"),
            "invalid_refs": unique_strings(
                evidence_validation.get("invalid_refs") or []
            ),
        },
        "classic_trace_ids": trace_ids,
        "inversion_certificate_ids": certificate_ids,
        "output_piece_ids": unique_strings(piece_ids),
        "classic_link_status": "linked" if trace_ids else "missing_classic_trace",
        "trace_statuses": unique_strings(
            item.get("trace_status") for item in linked_traces
        ),
        "epistemic_limits": [
            "语义诊断只说明思考链哪里可能带偏，不证明数字人的方向性结论为假。",
            "经典透镜只提供检测与反演规则，不提供关于现实问题的新事实。",
            "输出拼图仍须经过跨人关系裁决，才能约束真相轮廓。",
        ],
        "epistemic_role": "diagnostic_method_not_world_evidence",
        "classic_as_world_evidence": False,
    }


def _classic_trace_audit(
    pieces: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    certificates: list[dict[str, Any]],
    bridges: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_pieces = [
        item for item in pieces if item.get("diagnostic_ids")
    ]
    linked_pieces = [
        item
        for item in diagnostic_pieces
        if (item.get("knowledge_trace") or {}).get("classic_trace_ids")
    ]
    used_trace_ids = {
        trace_id
        for item in linked_pieces
        for trace_id in (item.get("knowledge_trace") or {}).get(
            "classic_trace_ids", []
        )
    }
    authorization_counts = Counter(
        compact_text(
            (item.get("knowledge_trace") or {}).get(
                "inversion_authorization"
            )
        )
        or "missing"
        for item in diagnostic_pieces
    )
    return {
        "classic_trace_count": len(traces),
        "inversion_certificate_count": len(certificates),
        "diagnostic_bridge_count": len(bridges),
        "diagnostic_piece_count": len(diagnostic_pieces),
        "classic_linked_piece_count": len(linked_pieces),
        "classic_trace_link_rate": round(
            len(linked_pieces) / len(diagnostic_pieces), 4
        )
        if diagnostic_pieces
        else 1.0,
        "inversion_authorization_counts": dict(authorization_counts),
        "unused_classic_trace_ids": [
            item["trace_id"]
            for item in traces
            if item["trace_id"] not in used_trace_ids
        ],
        "classic_used_as_world_evidence_count": sum(
            bool(
                (item.get("knowledge_trace") or {}).get(
                    "classic_as_world_evidence"
                )
            )
            for item in diagnostic_pieces
        ),
    }


def _piece(
    *,
    qid: str,
    person_id: str,
    person_name: Any,
    piece_kind: str,
    text: Any,
    status: str,
    origin: str,
    anchor_ids: list[Any],
    source_ids: list[Any],
    source_family_by_id: dict[str, str],
    source_catalog_by_id: dict[str, dict[str, Any]] | None = None,
    question: str,
    answer_position: dict[str, Any],
    time_horizon: Any,
    conclusion_leverage: Any,
    parent_text: Any,
    diagnostic_ids: list[str] | None = None,
    knowledge_trace: dict[str, Any] | None = None,
    shadow_operators: list[str] | None = None,
    relevance_status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = compact_text(text)
    clean_source_ids = unique_strings(source_ids)
    piece_id = stable_id("piece", qid, person_id, piece_kind, body, origin)
    coordinates = {
        "epistemic_mode": _epistemic_mode(question, body, piece_kind),
        "answer_direction": compact_text(answer_position.get("direction")) or "not_applicable",
        "scope": compact_text(answer_position.get("scope")),
        "time_horizon": compact_text(time_horizon),
        "definitions": _definitions(body),
        "condition_markers": _condition_markers(body),
    }
    source_families = unique_strings(
        source_family_by_id.get(source_id, f"unknown:{source_id}")
        for source_id in clean_source_ids
    )
    if person_id != "collective":
        source_families = unique_strings(
            ["shared-base-model-prior", *source_families]
        )
    source_catalog_by_id = source_catalog_by_id or {}
    source_records = [
        source_catalog_by_id[source_id]
        for source_id in clean_source_ids
        if source_id in source_catalog_by_id
    ]
    evidence_source_families = [
        family
        for family in source_families
        if family != "shared-base-model-prior"
    ]
    model_source_families = [
        family
        for family in source_families
        if family == "shared-base-model-prior"
    ]
    publisher_families = unique_strings(
        item.get("publisher_family") for item in source_records
    )
    origin_families = unique_strings(
        item.get("origin_family") for item in source_records
    )
    verification_statuses = unique_strings(
        item.get("verification_status") for item in source_records
    )
    unknown_source_ids = [
        source_id
        for source_id in clean_source_ids
        if source_id not in source_catalog_by_id
    ]
    if evidence_source_families and not unknown_source_ids:
        lineage_status = "external_lineage_linked"
    elif clean_source_ids and unknown_source_ids:
        lineage_status = "partial_lineage"
    elif model_source_families:
        lineage_status = "model_prior_only"
    else:
        lineage_status = "reasoning_only"
    payload = {
        "piece_id": piece_id,
        "question_id": qid,
        "person_id": person_id,
        "person_name": compact_text(person_name),
        "piece_kind": piece_kind,
        "text": body,
        "normalized_claim": normalize_claim(body),
        "content_status": status,
        "coordinates": coordinates,
        "shadow_operators": unique_strings(shadow_operators or []),
        "anchor_ids": unique_strings(anchor_ids),
        "source_ids": clean_source_ids,
        "source_families": source_families,
        "evidence_source_families": evidence_source_families,
        "model_source_families": model_source_families,
        "publisher_families": publisher_families,
        "origin_families": origin_families,
        "independence_profile": {
            "model_families": model_source_families,
            "evidence_families": evidence_source_families,
            "publisher_families": publisher_families,
            "origin_families": origin_families,
            "verification_statuses": verification_statuses,
            "unknown_source_ids": unknown_source_ids,
            "lineage_status": lineage_status,
        },
        "diagnostic_ids": unique_strings(diagnostic_ids or []),
        "knowledge_trace": knowledge_trace or {},
        "conclusion_leverage": compact_text(conclusion_leverage) or "unknown",
        "relevance_status": relevance_status or "not_applicable",
        "parent_conclusion": compact_text(parent_text),
        "provenance": {
            "origin": origin,
            "person_id": person_id,
            "anchor_ids": unique_strings(anchor_ids),
            "diagnostic_ids": unique_strings(diagnostic_ids or []),
            "classic_trace_ids": unique_strings(
                (knowledge_trace or {}).get("classic_trace_ids") or []
            ),
            "inversion_certificate_ids": unique_strings(
                (knowledge_trace or {}).get("inversion_certificate_ids") or []
            ),
            "diagnostic_bridge_ids": unique_strings(
                (knowledge_trace or {}).get("diagnostic_bridge_ids") or []
            ),
        },
    }
    if extra:
        payload["diagnostic_context"] = extra
    return payload


def _build_source_catalog(ledgers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    people_by_source: defaultdict[str, set[str]] = defaultdict(set)
    decisions_by_source: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for person_id, ledger in ledgers.items():
        for decision in ledger.get("source_decisions", []) or []:
            result = decision.get("result") or {}
            source_id = compact_text(result.get("id"))
            if not source_id:
                continue
            layer = compact_text(result.get("retrieval_layer")) or "unknown"
            url = compact_text(result.get("url"))
            source_name = compact_text(result.get("source_name"))
            family = _source_family(source_id, url, layer)
            title = compact_text(result.get("title"))
            snippet = compact_text(result.get("snippet"))
            catalog[source_id] = {
                "source_id": source_id,
                "title": title,
                "snippet": snippet,
                "url": url,
                "domain": _domain(url),
                "source_name": source_name,
                "retrieval_layer": layer,
                "verification_status": compact_text(result.get("verification_status")),
                "source_type": compact_text(result.get("source_type")),
                "source_family": family,
                "canonical_url_family": family,
                "publisher_family": _publisher_family(source_name, url, layer),
                "origin_family": _origin_family(
                    title,
                    snippet,
                    url,
                    layer,
                ),
            }
            people_by_source[source_id].add(person_id)
            decisions_by_source[source_id][compact_text(decision.get("decision")) or "unknown"] += 1

    for source_id, item in catalog.items():
        item["person_ids"] = sorted(people_by_source[source_id])
        item["decision_counts"] = dict(decisions_by_source[source_id])
    return catalog


def _source_family(source_id: str, url: str, retrieval_layer: str) -> str:
    if retrieval_layer == "model_prior" or url.startswith("model-prior://"):
        return "shared-base-model-prior"
    if url.startswith("http://") or url.startswith("https://"):
        split = urlsplit(url)
        normalized = urlunsplit(
            (
                split.scheme.lower(),
                split.netloc.lower(),
                split.path.rstrip("/"),
                "",
                "",
            )
        )
        return f"url:{normalized}"
    return f"source:{source_id}"


def _publisher_family(source_name: str, url: str, retrieval_layer: str) -> str:
    if retrieval_layer == "model_prior" or url.startswith("model-prior://"):
        return ""
    cleaned = re.sub(r"\s+", " ", compact_text(source_name)).strip().casefold()
    generic_names = {
        "",
        "digital-person-prior",
        "google news",
        "google-news-rss",
        "bing rss",
        "bing-rss",
    }
    if cleaned not in generic_names:
        return f"publisher:{cleaned}"
    domain = _domain(url)
    return f"publisher:{domain}" if domain else ""


def _origin_family(
    title: str,
    snippet: str,
    url: str,
    retrieval_layer: str,
) -> str:
    """Conservatively group mirrors and syndicated copies without fetching pages."""

    if retrieval_layer == "model_prior" or url.startswith("model-prior://"):
        return "shared-base-model-prior"
    title_key = _lineage_text_key(_strip_publisher_suffix(title))
    snippet_key = _lineage_text_key(snippet)[:240]
    if len(title_key) >= 12:
        seed = f"title:{title_key}"
    elif len(snippet_key) >= 32:
        seed = f"snippet:{snippet_key}"
    else:
        return _source_family("", url, retrieval_layer)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    return f"origin:{digest}"


def _strip_publisher_suffix(title: str) -> str:
    parts = re.split(r"\s+(?:[-|–—])\s+", compact_text(title), maxsplit=1)
    return parts[0] if parts else compact_text(title)


def _lineage_text_key(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _evidence_entry_source_id(entry: dict[str, Any]) -> str:
    """Read both current and legacy evidence-ledger identifiers."""

    return compact_text(
        entry.get("source_id")
        or entry.get("entry_id")
        or entry.get("id")
    )


def _resolve_evidence_entry_source(
    entry: dict[str, Any],
    *,
    ledger: dict[str, Any],
    source_catalog: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    explicit = _evidence_entry_source_id(entry)
    if explicit:
        return explicit, (
            "explicit_id" if explicit in source_catalog else "explicit_unknown_id"
        )

    source_label = compact_text(
        entry.get("source") or entry.get("url") or entry.get("publisher")
    )
    if not source_label:
        return "", "fallback_unresolved"
    accepted_ids = []
    for decision in ledger.get("source_decisions", []) or []:
        status = compact_text(decision.get("decision")).lower()
        result = decision.get("result") or {}
        source_id = compact_text(result.get("id"))
        if status in {"accept", "accepted", "use", "采信"} and source_id in source_catalog:
            accepted_ids.append(source_id)
    candidates = []
    for source_id in unique_strings(accepted_ids):
        record = source_catalog[source_id]
        score = _source_match_score(entry, source_label, record)
        if score > 0:
            candidates.append((score, source_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates or candidates[0][0] < 0.88:
        return "", "fallback_unresolved"
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.06:
        return "", "fallback_unresolved"
    return candidates[0][1], "fallback_unique_source_match"


def _source_match_score(
    entry: dict[str, Any], source_label: str, record: dict[str, Any]
) -> float:
    source_url = compact_text(entry.get("url"))
    record_url = compact_text(record.get("url"))
    if source_label.startswith(("http://", "https://")):
        source_url = source_label
    if source_url and record_url and _source_family("", source_url, "") == _source_family("", record_url, ""):
        return 1.0

    source_key = _source_match_key(source_label)
    if len(source_key) < 3 or source_key in {
        "模型先验",
        "公司财报",
        "市场数据",
        "新闻报道",
        "研究报告",
    }:
        return 0.0
    title_key = _source_match_key(record.get("title"))
    domain_key = _source_match_key(record.get("domain"))
    base = 0.0
    if source_key == domain_key or source_key in domain_key:
        base = 0.96
    elif source_key in title_key:
        base = 0.93
    elif source_key in _source_match_key(record_url):
        base = 0.9
    if not base:
        return 0.0
    claim = compact_text(entry.get("claim"))
    reference = " ".join(
        [compact_text(record.get("title")), compact_text(record.get("snippet"))]
    )
    return min(1.0, base + 0.06 * _text_overlap(claim, reference))


def _source_match_key(value: Any) -> str:
    text = compact_text(value).lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def _text_overlap(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        normalized = _source_match_key(value)
        if len(normalized) < 3:
            return {normalized} if normalized else set()
        return {
            normalized[index : index + 3]
            for index in range(len(normalized) - 2)
        }

    left_grams = grams(left)
    right_grams = grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _source_lineage_audit(
    pieces: list[dict[str, Any]],
    source_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    referenced = [piece for piece in pieces if piece.get("source_ids")]
    fully_linked = [
        piece
        for piece in referenced
        if all(
            source_id in source_catalog
            for source_id in piece.get("source_ids", []) or []
        )
    ]
    external = [
        piece
        for piece in pieces
        if piece.get("evidence_source_families")
    ]
    denominator = len(referenced)
    return {
        "source_referenced_piece_count": denominator,
        "source_lineage_complete_piece_count": len(fully_linked),
        "source_lineage_complete_rate": (
            round(len(fully_linked) / denominator, 4) if denominator else 1.0
        ),
            "external_lineage_piece_count": len(external),
    }


def _domain(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    return urlsplit(url).netloc.lower()


def _collective_blind_spot_pieces(
    qid: str,
    question: str,
    diagnosis: dict[str, Any],
    source_family_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    items = diagnosis.get("blind_spot_registry") or diagnosis.get("collective_blind_spots") or []
    pieces: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        text = compact_text(item.get("description") or item.get("dimension"))
        if not text:
            continue
        pieces.append(
            _piece(
                qid=qid,
                person_id="collective",
                person_name="数字人群",
                piece_kind="blind_spot_candidate",
                text=text,
                status="candidate_unverified",
                origin="forensic.collective_blind_spot_candidate",
                anchor_ids=[f"CB{index}"],
                source_ids=[],
                source_family_by_id=source_family_by_id,
                question=question,
                answer_position={},
                time_horizon="",
                conclusion_leverage="unknown",
                parent_text="",
                relevance_status=compact_text(item.get("relevance_status")) or "candidate_unverified",
                extra={
                    "dimension": compact_text(item.get("dimension")),
                    "recommended_probe": compact_text(item.get("recommended_probe")),
                    "source": compact_text(item.get("source")),
                },
            )
        )
    return pieces


def _index_outputs(person_outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in person_outputs:
        output = record.get("output", record)
        person = record.get("person", {}) or {}
        person_id = compact_text(
            record.get("person_id") or person.get("id") or output.get("person_id")
        )
        if person_id:
            indexed[person_id] = record
    return indexed


def _deduplicate_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for piece in pieces:
        by_id[piece["piece_id"]] = piece
    return [by_id[key] for key in sorted(by_id)]


def _epistemic_mode(question: str, text: str, piece_kind: str) -> str:
    combined = f"{question} {text}"
    if piece_kind == "value_condition" or any(token in combined for token in ["应该", "正义", "权利", "道德", "伦理"]):
        return "normative"
    if any(token in combined for token in ["定义", "概念", "所谓", "指的是"]):
        return "conceptual"
    if any(token in combined for token in ["会不会", "未来", "预测", "将会", "可能发生", "很快"]):
        return "forecast"
    if any(token in combined for token in ["导致", "因为", "机制", "因果", "驱动", "源于"]):
        return "causal"
    return "empirical"


def _definitions(text: str) -> list[str]:
    matches = re.findall(r"[“\"‘']([^”\"’']{1,36})[”\"’']", text)
    return unique_strings(matches)[:8]


def _condition_markers(text: str) -> list[str]:
    markers = []
    for token in ["如果", "若", "除非", "取决于", "在此条件下", "前提是", "只要"]:
        if token in text:
            markers.append(token)
    return markers


def _shadow_operators(text: str) -> list[str]:
    rules = [
        ("scope_overreach", ["越界", "扩大", "普遍", "整体", "所有", "不可能", "必然"]),
        ("missing_reasoning_bridge", ["缺少", "缺失", "没有证明", "跳到", "桥梁"]),
        ("definition_shift", ["定义", "概念", "偷换", "口径"]),
        ("time_conflation", ["时间", "短期", "长期", "未来", "当前"]),
        ("evidence_gap", ["证据不足", "未经核验", "没有证据", "样本"]),
        ("omitted_variable", ["忽略", "遗漏", "变量", "没有考虑"]),
        ("confidence_inflation", ["置信度", "确定性", "过度自信", "断言"]),
        ("causal_overreach", ["因果", "相关", "机制"]),
        ("value_fact_conflation", ["价值", "偏好", "应该", "事实"]),
    ]
    operators = [name for name, tokens in rules if any(token in text for token in tokens)]
    return operators or ["diagnosed_distortion"]
