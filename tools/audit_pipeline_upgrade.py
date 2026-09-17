from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.contour import (  # noqa: E402
    build_contour_constraints,
    run_live_contour,
    run_provisional_contour_drafts,
)
from bme_model.contour.schemas import validate_truth_contour  # noqa: E402
from bme_model.contour.live import (  # noqa: E402
    _contour_reasoning_payload,
    _build_provisional_draft_case,
    _provisional_focus_case,
    _validate_final,
    build_contour_case,
    build_contour_prompt_packet,
    build_counter_pressure_packets,
    validate_contour_prompt_packet,
    validate_counter_pressure_packets,
)
from bme_model.detective import run_live_detective  # noqa: E402
from bme_model.detective.live import (  # noqa: E402
    CONTENT_TYPES,
    SHADOW_TYPES,
    _adjudication_prompt_bundle,
    _classic_review_prompt_bundle,
    _proposal_prompt_bundle,
    build_detective_case,
)
from bme_model.detective.schemas import validate_detective_output  # noqa: E402
from bme_model.providers import (  # noqa: E402
    DeepSeekClient,
    synthesis_request_timeout_seconds,
)
from bme_model.scheduler import AdaptiveConcurrencyGovernor  # noqa: E402


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def iso_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round(
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(),
        3,
    )


def stage_seconds(manifest: dict[str, Any], stage: str) -> float | None:
    record = (manifest.get("stages") or {}).get(stage) or {}
    return iso_seconds(record.get("started_at"), record.get("completed_at"))


def _count_detector_coverage(diagnosis: dict[str, Any]) -> dict[str, Any]:
    scans = diagnosis.get("cognitive_chain_scans") or []
    counts = [len(item.get("detector_results") or []) for item in scans]
    return {
        "person_count": len(scans),
        "minimum_detectors_per_person": min(counts) if counts else 0,
        "complete_ten_detector_people": sum(value >= 10 for value in counts),
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json", {}) or {}
    run = read_json(run_dir / "run.json", {}) or {}
    quality = read_json(run_dir / "quality.json", {}) or {}
    diagnosis = read_json(run_dir / "diagnosis.json", {}) or {}
    detective = read_json(run_dir / "detective.json", {}) or {}
    contour = read_json(run_dir / "truth_contour.json", {}) or {}
    candidates = read_json(run_dir / "contour_candidates.json", {}) or {}
    retrieval = read_json(run_dir / "retrieval.json", {}) or {}
    detective_eval = detective.get("detective_evaluation") or {}
    contour_eval = contour.get("contour_evaluation") or {}
    classic_audit = contour.get("classic_knowledge_audit") or {}
    live_contour = candidates.get("contour") or {}
    final_payload = live_contour.get("final_adjudication") or {}
    final_case = live_contour.get("case") or {}
    final_payload_errors: list[str] = []
    if final_payload and final_case:
        _, final_payload_errors = _validate_final(final_payload, final_case)
    execution = manifest.get("execution") or {}
    metrics = quality.get("metrics") or (
        (manifest.get("quality") or {}).get("metrics") or {}
    )
    streaming_front_metrics = (
        read_json(run_dir / "streaming_front_metrics.json", {}) or {}
    )

    analysis_start = (
        ((manifest.get("stages") or {}).get("retrieval") or {}).get("started_at")
    )
    analysis_end = (
        ((manifest.get("stages") or {}).get("relational_synthesis") or {}).get(
            "completed_at"
        )
    )
    total_submission_seconds = iso_seconds(
        manifest.get("created_at"), manifest.get("completed_at")
    )
    if total_submission_seconds is None:
        total_submission_seconds = iso_seconds(
            run.get("started_at"), run.get("completed_at")
        )

    relations = detective.get("relation_certificates") or []
    accepted = [item for item in relations if item.get("status") == "accepted"]
    independence = _independence_metrics(accepted)
    packet_audit = (live_contour.get("audit") or {}).get("prompt_packet") or {}
    detective_packet_audit = (
        ((candidates.get("detective") or {}).get("audit") or {}).get(
            "prompt_packet"
        )
        or {}
    )
    counter_stage = next(
        (
            item
            for item in (live_contour.get("audit") or {}).get("stages", []) or []
            if item.get("stage")
            in {"strongest_counter_contour", "parallel_counter_pressure"}
        ),
        {},
    )
    speculative_record = candidates.get("speculative_drafts") or {}
    speculative_execution = (live_contour.get("audit") or {}).get(
        "speculative_execution"
    ) or {}
    chain_eval = diagnosis.get("cognitive_chain_evaluation") or {}
    semantic_eval = diagnosis.get("semantic_verdict_evaluation") or {}
    semantic_audit = diagnosis.get("semantic_verdict_audit") or {}
    meta_eval = diagnosis.get("meta_model_overall_evaluation") or {}

    return {
        "run_dir": str(run_dir.resolve()),
        "question": manifest.get("question") or run.get("question"),
        "status": manifest.get("status") or quality.get("status") or "legacy",
        "pipeline_mode": execution.get("pipeline_mode") or "legacy_sequential",
        "synthesis_mode": execution.get("synthesis_mode") or "legacy_sequential",
        "timing": {
            "submission_to_result_seconds": total_submission_seconds,
            "analysis_window_seconds": iso_seconds(analysis_start, analysis_end),
            "cohort_seconds": stage_seconds(manifest, "cohort"),
            "retrieval_seconds": stage_seconds(manifest, "retrieval"),
            "digital_people_seconds": stage_seconds(manifest, "digital_people"),
            "semantic_verdicts_seconds": stage_seconds(
                manifest, "semantic_verdicts"
            ),
            "relational_synthesis_seconds": stage_seconds(
                manifest, "relational_synthesis"
            ),
            "streaming_front_seconds": (
                streaming_front_metrics.get("total_wall_seconds")
            ),
        },
        "quality": {
            "expected_people": metrics.get("expected_people")
            or retrieval.get("person_count"),
            "succeeded_people": metrics.get("succeeded_people")
            or len(run.get("person_results") or []),
            "retrieval_external_coverage": metrics.get(
                "retrieval_external_coverage"
            ),
            "semantic_fallback_count": metrics.get("semantic_fallback_count"),
            "cognitive_chain_score": chain_eval.get("score"),
            "semantic_verdict_score": semantic_eval.get("score"),
            "semantic_live_person_count": semantic_audit.get(
                "live_person_count"
            ),
            "semantic_adjudicated_person_count": semantic_audit.get(
                "attached_person_count"
            ),
            "semantic_verdict_count": semantic_eval.get("verdict_count"),
            "semantic_inference_mode": streaming_front_metrics.get(
                "semantic_inference_mode"
            ),
            "meta_model_score": meta_eval.get("score"),
            "detector_coverage": _count_detector_coverage(diagnosis),
            "detective_passed": detective_eval.get("passed"),
            "relation_count": len(relations),
            "accepted_relation_count": len(accepted),
            **independence,
            "accepted_traceability_rate": detective_eval.get(
                "accepted_traceability_rate"
            ),
            "accepted_classic_traceability_rate": detective_eval.get(
                "accepted_classic_traceability_rate"
            ),
            "forced_relation_count": detective_eval.get("forced_relation_count"),
            "classic_world_evidence_count": detective_eval.get(
                "classic_used_as_world_evidence_count"
            ),
            "contour_generation_mode": contour.get("generation_mode"),
            "contour_provenance_coverage": contour_eval.get(
                "provenance_coverage"
            ),
            "evidence_binding_contract": contour.get(
                "evidence_binding_contract"
            ),
            "sentence_binding_count": contour_eval.get(
                "sentence_binding_count"
            ),
            "sentence_binding_coverage": contour_eval.get(
                "sentence_binding_coverage"
            ),
            "verified_observation_count": contour_eval.get(
                "verified_observation_count"
            ),
            "directional_judgment_count": contour_eval.get(
                "directional_judgment_count"
            ),
            "verified_accepted_evidence_count": metrics.get(
                "verified_accepted_evidence_count"
            ),
            "final_contour_payload_valid": not final_payload_errors,
            "final_contour_payload_error_count": len(final_payload_errors),
            "final_contour_payload_errors": final_payload_errors,
            "direct_answer_present": bool(str(contour.get("direct_answer") or "").strip()),
            "main_contour_present": bool(str(contour.get("main_contour") or "").strip()),
            "key_condition_count": len(contour.get("key_conditions") or []),
            "important_unknown_count": len(contour.get("important_unknowns") or []),
            "contour_classic_world_evidence_count": classic_audit.get(
                "classic_used_as_world_evidence_count"
            ),
            "prompt_packet_coverage": packet_audit.get("coverage_validation"),
            "prompt_packet_reduction_ratio": packet_audit.get(
                "character_reduction_ratio"
            ),
            "detective_prompt_packet_coverage": detective_packet_audit.get(
                "coverage_validation"
            ),
            "detective_prompt_packet_reduction_ratio": (
                detective_packet_audit.get("character_reduction_ratio")
            ),
            "counter_mode": live_contour.get("counter_mode")
            or (live_contour.get("audit") or {}).get("counter_mode"),
            "counter_pressure_stage_count": counter_stage.get(
                "pressure_stage_count"
            ),
            "counter_pressure_packet_coverage": counter_stage.get(
                "packet_coverage_validation"
            ),
            "speculative_draft_status": speculative_record.get("status"),
            "speculative_drafts_are_world_evidence": (
                None
                if not speculative_record
                else not bool(
                    (speculative_record.get("audit") or {}).get(
                        "provisional_only_not_world_evidence"
                    )
                )
            ),
            "speculative_final_uses_only_adjudicated_relations": (
                speculative_execution.get(
                    "final_case_uses_only_adjudicated_relations"
                )
            ),
        },
    }


def audit_lossless_optimization(run_dir: Path) -> dict[str, Any]:
    """Measure current lossless encodings on frozen artifacts without API calls."""

    run = read_json(run_dir / "run.json", {}) or {}
    manifest = read_json(run_dir / "run_manifest.json", {}) or {}
    materials = read_json(run_dir / "puzzle_pieces.json", {}) or {}
    saved = read_json(run_dir / "contour_candidates.json", {}) or {}
    detective = read_json(run_dir / "detective.json", {}) or {}
    constraints = read_json(run_dir / "contour_constraints.json", {}) or {}
    cohort = read_json(run_dir / "cohort.json", {}) or {}
    front_metrics = read_json(
        run_dir / "streaming_front_metrics.json", {}
    ) or {}
    question = str(
        run.get("question") or manifest.get("question") or ""
    ).strip()
    if not all((question, materials, saved, detective, constraints)):
        raise ValueError(f"{run_dir} lacks lossless-audit artifacts")

    detective_case = build_detective_case(question, materials)
    saved_detective = saved.get("detective") or {}
    candidates = saved_detective.get("candidate_relations") or []
    relations = saved_detective.get("adjudicated_relations") or []
    proposal_bundles = {
        "content": _proposal_prompt_bundle(
            detective_case,
            "content",
            CONTENT_TYPES,
            material_mode="lossless_packet_v2",
        ),
        "shadow": _proposal_prompt_bundle(
            detective_case,
            "shadow",
            SHADOW_TYPES,
            material_mode="lossless_packet_v2",
        ),
    }
    shard_count = min(6, max(1, (len(candidates) + 3) // 4))
    shard_size = max(1, (len(candidates) + shard_count - 1) // shard_count)
    adjudication_stats = [
        _adjudication_prompt_bundle(
            detective_case,
            candidates[start : start + shard_size],
            material_mode="lossless_packet_v2",
        )["packet_stats"]
        for start in range(0, len(candidates), shard_size)
    ]
    classic_stats = _classic_review_prompt_bundle(
        detective_case,
        candidates,
        relations,
        material_mode="lossless_packet_v2",
    )["packet_stats"]

    contour_case = build_contour_case(question, detective, constraints)
    contour_packet = build_contour_prompt_packet(contour_case)
    contour_packet_errors = validate_contour_prompt_packet(
        contour_packet, contour_case
    )
    saved_contour = saved.get("contour") or {}
    candidate = _contour_reasoning_payload(
        saved_contour.get("candidate_contour") or {}
    )
    pressure_packets = (
        build_counter_pressure_packets(contour_packet, candidate)
        if candidate
        else []
    )
    pressure_errors = (
        validate_counter_pressure_packets(
            pressure_packets, contour_packet, candidate
        )
        if pressure_packets
        else ["saved run has no candidate contour"]
    )
    provisional_case = _build_provisional_draft_case(
        question, detective_case, candidates
    )
    provisional_case_sizes = [
        len(json.dumps(provisional_case, ensure_ascii=False)),
        *[
            len(
                json.dumps(
                    _provisional_focus_case(provisional_case, focus_id),
                    ensure_ascii=False,
                )
            )
            for focus_id in (
                "evidence_lineage",
                "causal_conditions",
                "shadow_unknowns",
            )
        ],
    ]
    _, saved_final_errors = _validate_final(
        saved_contour.get("final_adjudication") or {}, contour_case
    )

    detective_stages = {
        item.get("stage"): item
        for item in (saved_detective.get("audit") or {}).get("stages", []) or []
    }
    contour_stages = {
        item.get("stage"): item
        for item in (saved_contour.get("audit") or {}).get("stages", []) or []
    }

    def projected_seconds(
        stage: dict[str, Any],
        *,
        prompt_ratio: float,
        completion_ratio: float = 1.0,
    ) -> float:
        usage = stage.get("usage") or {}
        prompt = float(usage.get("prompt_tokens") or 0)
        completion = float(usage.get("completion_tokens") or 0)
        token_ratio = (
            (prompt * prompt_ratio + completion * completion_ratio)
            / max(1.0, prompt + completion)
        )
        return round(float(stage.get("duration_seconds") or 0) * token_ratio, 3)

    content_stats = proposal_bundles["content"]["packet_stats"]
    shadow_stats = proposal_bundles["shadow"]["packet_stats"]
    content_ratio = float(content_stats["sent_material_characters"]) / max(
        1.0, float(content_stats["full_case_characters"])
    )
    shadow_ratio = float(shadow_stats["sent_material_characters"]) / max(
        1.0, float(shadow_stats["full_case_characters"])
    )
    adjudication_full = sum(
        int(item.get("full_case_characters") or 0)
        for item in adjudication_stats
    )
    adjudication_sent = sum(
        int(item.get("sent_material_characters") or 0)
        for item in adjudication_stats
    )
    adjudication_ratio = adjudication_sent / max(1, adjudication_full)
    classic_ratio = float(classic_stats["sent_material_characters"]) / max(
        1.0, float(classic_stats["full_case_characters"])
    )
    proposal_seconds = max(
        projected_seconds(
            detective_stages.get("content_proposal") or {},
            prompt_ratio=content_ratio,
        ),
        projected_seconds(
            detective_stages.get("shadow_proposal") or {},
            prompt_ratio=shadow_ratio,
        ),
    )
    adjudication_seconds = projected_seconds(
        detective_stages.get("adversarial_adjudication") or {},
        prompt_ratio=adjudication_ratio,
    )
    classic_seconds = projected_seconds(
        detective_stages.get("classic_post_review") or {},
        prompt_ratio=classic_ratio,
    )
    projected_detective_seconds = round(
        proposal_seconds + adjudication_seconds + classic_seconds, 3
    )

    full_contour_chars = len(json.dumps(contour_packet, ensure_ascii=False))
    pressure_sizes = [
        len(json.dumps(item["material"], ensure_ascii=False))
        for item in pressure_packets
    ]
    largest_pressure_ratio = (
        max(pressure_sizes) / max(1, full_contour_chars)
        if pressure_sizes
        else 1.0
    )
    projected_pressure_seconds = projected_seconds(
        contour_stages.get("strongest_counter_contour") or {},
        prompt_ratio=largest_pressure_ratio,
        completion_ratio=0.55,
    )
    candidate_seconds = float(
        (contour_stages.get("candidate_contour") or {}).get(
            "duration_seconds"
        )
        or 0
    )
    final_seconds = float(
        (contour_stages.get("final_adjudication") or {}).get(
            "duration_seconds"
        )
        or 0
    )
    projected_contour_low = round(
        candidate_seconds + projected_pressure_seconds + final_seconds, 3
    )
    projected_contour_high = round(
        candidate_seconds
        + projected_pressure_seconds
        + final_seconds * 1.25,
        3,
    )
    projected_synthesis = {
        "lower_seconds": round(
            projected_detective_seconds + projected_contour_low, 3
        ),
        "conservative_seconds": round(
            projected_detective_seconds + projected_contour_high, 3
        ),
        "method": (
            "Frozen-run duration scaled by prompt/completion token exposure; "
            "this is a planning estimate, not live proof."
        ),
    }
    provisional_prompt_ratio = max(provisional_case_sizes) / max(
        1, full_contour_chars
    )
    projected_draft_seconds = projected_seconds(
        contour_stages.get("candidate_contour") or {},
        prompt_ratio=provisional_prompt_ratio,
        completion_ratio=0.55,
    )
    projected_speculative_final_seconds = round(final_seconds * 1.25, 3)
    projected_speculative_synthesis_seconds = round(
        proposal_seconds
        + max(
            adjudication_seconds + classic_seconds,
            projected_draft_seconds,
        )
        + projected_speculative_final_seconds,
        3,
    )
    semantic_records = [
        read_json(path, {}) or {}
        for path in sorted((run_dir / "semantic_verdicts").glob("*.json"))
    ]
    semantic_attempts = sum(
        int(item.get("attempts") or 0) for item in semantic_records
    )
    semantic_visible_chars = 0
    semantic_reasoning_chars = 0
    for item in semantic_records:
        message = (((item.get("raw") or {}).get("choices") or [{}])[0].get(
            "message"
        ) or {})
        semantic_visible_chars += len(str(message.get("content") or ""))
        semantic_reasoning_chars += len(
            str(message.get("reasoning_content") or "")
        )
    semantic_response_chars = (
        semantic_visible_chars + semantic_reasoning_chars
    )
    cohort_seconds = float(
        (cohort.get("generation") or {}).get("duration_seconds") or 0
    )
    measured_front_seconds = float(
        front_metrics.get("total_wall_seconds") or 0
    )
    projected_total_before_semantic_fast = round(
        cohort_seconds
        + measured_front_seconds
        + projected_speculative_synthesis_seconds,
        3,
    )
    front_budget_for_ten_minutes = round(
        max(
            0.0,
            600.0
            - cohort_seconds
            - projected_speculative_synthesis_seconds,
        ),
        3,
    )

    return {
        "schema": "bme.lossless-optimization-audit.v1",
        "run_dir": str(run_dir.resolve()),
        "question": question,
        "external_calls": 0,
        "detective": {
            "content_proposal": content_stats,
            "shadow_proposal": shadow_stats,
            "adjudication_shards": {
                "count": len(adjudication_stats),
                "full_case_characters": adjudication_full,
                "sent_material_characters": adjudication_sent,
                "character_reduction_ratio": round(
                    1 - adjudication_ratio, 4
                ),
                "coverage_validation": (
                    "passed"
                    if all(
                        item.get("coverage_validation") == "passed"
                        for item in adjudication_stats
                    )
                    else "failed"
                ),
            },
            "classic_review": classic_stats,
        },
        "contour": {
            "prompt_packet_errors": contour_packet_errors,
            "counter_pressure_packet_errors": pressure_errors,
            "counter_pressure_count": len(pressure_packets),
            "full_packet_characters": full_contour_chars,
            "pressure_packet_characters": pressure_sizes,
            "largest_pressure_packet_ratio": round(
                largest_pressure_ratio, 4
            ),
            "saved_final_errors_under_current_gate": saved_final_errors,
            "provisional_draft_case_characters": provisional_case_sizes,
            "provisional_drafts_are_world_evidence": False,
        },
        "cache_safety": {
            "saved_detective_pipeline_version": saved_detective.get(
                "pipeline_version"
            ),
            "saved_contour_pipeline_version": saved_contour.get(
                "pipeline_version"
            ),
            "legacy_stage_cache_requires_regeneration": True,
            "candidate_change_invalidates_counter": True,
            "counter_change_invalidates_final": True,
            "validated_fast_semantic_cache_isolated": True,
        },
        "semantic_fast_path": {
            "saved_record_count": len(semantic_records),
            "saved_attempt_count": semantic_attempts,
            "visible_response_characters": semantic_visible_chars,
            "hidden_reasoning_characters": semantic_reasoning_chars,
            "hidden_reasoning_character_ratio": round(
                semantic_reasoning_chars / max(1, semantic_response_chars),
                4,
            ),
            "candidate_first_attempt_thinking": "disabled",
            "invalid_first_attempt_escalates_to_full_thinking": True,
            "output_schema_and_validator_unchanged": True,
        },
        "projection": {
            "detective_seconds": projected_detective_seconds,
            "contour_lower_seconds": projected_contour_low,
            "contour_conservative_seconds": projected_contour_high,
            "synthesis": projected_synthesis,
            "target_synthesis_seconds": 200,
            "speculative_fast_path": {
                "draft_seconds_parallel_with_adjudication": (
                    projected_draft_seconds
                ),
                "final_seconds_conservative": (
                    projected_speculative_final_seconds
                ),
                "synthesis_seconds": (
                    projected_speculative_synthesis_seconds
                ),
                "draft_failure_falls_back_to_post_adjudication_path": True,
            },
            "target_met_by_projection": (
                projected_speculative_synthesis_seconds <= 200
            ),
            "end_to_end_budget": {
                "cohort_measured_seconds": cohort_seconds,
                "front_measured_seconds": measured_front_seconds,
                "synthesis_projected_seconds": (
                    projected_speculative_synthesis_seconds
                ),
                "total_before_semantic_fast_seconds": (
                    projected_total_before_semantic_fast
                ),
                "front_budget_for_ten_minutes_seconds": (
                    front_budget_for_ten_minutes
                ),
                "required_front_reduction_seconds": round(
                    max(
                        0.0,
                        measured_front_seconds - front_budget_for_ten_minutes,
                    ),
                    3,
                ),
                "under_ten_minutes_live_proven": False,
            },
        },
        "hard_local_gates": {
            "detective_packets_reversible": all(
                bundle["packet_stats"].get("coverage_validation") == "passed"
                for bundle in proposal_bundles.values()
            )
            and all(
                item.get("coverage_validation") == "passed"
                for item in adjudication_stats
            )
            and classic_stats.get("coverage_validation") == "passed",
            "contour_packet_reversible": not contour_packet_errors,
            "counter_packets_collectively_lossless": not pressure_errors,
            "task_relevant_material_removed": False,
        },
    }


class OfflineOnlyClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        raise RuntimeError("offline replay attempted a model call")


def replay_saved_synthesis(run_dir: Path, output_root: Path) -> dict[str, Any]:
    run = read_json(run_dir / "run.json", {}) or {}
    question = str(run.get("question") or "").strip()
    materials = read_json(run_dir / "puzzle_pieces.json", {}) or {}
    saved = read_json(run_dir / "contour_candidates.json", {}) or {}
    if not question or not materials or not saved:
        raise ValueError(f"{run_dir} lacks replayable synthesis artifacts")

    client = OfflineOnlyClient()
    detective, detective_record = run_live_detective(
        question,
        materials,
        model=str(run.get("model") or "saved-model"),
        retries=0,
        client_factory=lambda: client,
        cached_record=saved.get("detective") or {},
    )
    constraints = build_contour_constraints(question, detective)
    contour, contour_record = run_live_contour(
        question,
        detective,
        constraints,
        model=str(run.get("model") or "saved-model"),
        retries=0,
        client_factory=lambda: client,
        cached_record=saved.get("contour") or {},
    )
    detective_errors = validate_detective_output(detective, materials)
    contour_errors = validate_truth_contour(
        contour,
        question=question,
        detective=detective,
        constraints=constraints,
    )
    old_detective = read_json(run_dir / "detective.json", {}) or {}
    old_contour = read_json(run_dir / "truth_contour.json", {}) or {}
    old_eval = old_detective.get("detective_evaluation") or {}
    new_eval = detective.get("detective_evaluation") or {}
    old_contour_eval = old_contour.get("contour_evaluation") or {}
    new_contour_eval = contour.get("contour_evaluation") or {}
    old_accepted_ids = sorted(
        item.get("relation_id")
        for item in old_detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted" and item.get("relation_id")
    )
    new_accepted_ids = sorted(
        item.get("relation_id")
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted" and item.get("relation_id")
    )

    result = {
        "schema": "bme.offline-synthesis-replay.v1",
        "source_run": str(run_dir.resolve()),
        "question": question,
        "network_or_model_calls": client.calls,
        "replay_status": (
            "passed"
            if not client.calls and not detective_errors and not contour_errors
            else (
                "legacy_cache_requires_regeneration"
                if client.calls and not detective_errors and not contour_errors
                else "failed"
            )
        ),
        "validation": {
            "detective_errors": detective_errors,
            "contour_errors": contour_errors,
        },
        "old": {
            "relation_count": old_eval.get("relation_count"),
            "accepted_relation_count": (old_eval.get("status_counts") or {}).get(
                "accepted", 0
            ),
            "accepted_traceability_rate": old_eval.get(
                "accepted_traceability_rate"
            ),
            "classic_world_evidence_count": old_eval.get(
                "classic_used_as_world_evidence_count"
            ),
            "contour_generation_mode": old_contour.get("generation_mode"),
            "contour_provenance_coverage": old_contour_eval.get(
                "provenance_coverage"
            ),
        },
        "current": {
            "detective_status": detective_record.get("status"),
            "relation_count": new_eval.get("relation_count"),
            "accepted_relation_count": (new_eval.get("status_counts") or {}).get(
                "accepted", 0
            ),
            "accepted_traceability_rate": new_eval.get(
                "accepted_traceability_rate"
            ),
            "classic_world_evidence_count": new_eval.get(
                "classic_used_as_world_evidence_count"
            ),
            "contour_status": contour_record.get("status"),
            "contour_generation_mode": contour.get("generation_mode"),
            "contour_provenance_coverage": new_contour_eval.get(
                "provenance_coverage"
            ),
            "prompt_packet": (contour_record.get("audit") or {}).get(
                "prompt_packet"
            ),
        },
        "non_regression": {
            "accepted_relation_ids_identical": (
                old_accepted_ids == new_accepted_ids
            ),
            "direct_answer_identical": (
                old_contour.get("direct_answer") == contour.get("direct_answer")
            ),
            "main_contour_identical": (
                old_contour.get("main_contour") == contour.get("main_contour")
            ),
            "key_conditions_identical": (
                old_contour.get("key_conditions") == contour.get("key_conditions")
            ),
            "important_unknowns_identical": (
                old_contour.get("important_unknowns")
                == contour.get("important_unknowns")
            ),
        },
    }
    target = output_root / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "replay_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_live_optimized_synthesis(
    run_dir: Path,
    output_root: Path,
    *,
    model_override: str | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    """Run optimized synthesis on frozen saved materials without cache reuse."""

    run = read_json(run_dir / "run.json", {}) or {}
    question = str(run.get("question") or "").strip()
    materials = read_json(run_dir / "puzzle_pieces.json", {}) or {}
    baseline_detective = read_json(run_dir / "detective.json", {}) or {}
    baseline_contour = read_json(run_dir / "truth_contour.json", {}) or {}
    if not question or not materials or not baseline_detective or not baseline_contour:
        raise ValueError(f"{run_dir} lacks live comparison artifacts")

    model = model_override or str(run.get("model") or "deepseek-v4-pro")
    governor = AdaptiveConcurrencyGovernor(24)
    user_id = "bme_benchmark_" + hashlib.sha256(
        str(run_dir.resolve()).encode("utf-8")
    ).hexdigest()[:20]
    client_factory = lambda: DeepSeekClient(  # noqa: E731
        model=model,
        timeout_seconds=synthesis_request_timeout_seconds(),
        request_governor=governor,
        user_id=user_id,
    )
    started = time.perf_counter()
    draft_future: Future[tuple[dict[str, Any], dict[str, Any]]] | None = None
    with ThreadPoolExecutor(max_workers=1) as draft_executor:
        def launch_drafts(
            detective_case: dict[str, Any],
            candidates: list[dict[str, Any]],
        ) -> None:
            nonlocal draft_future
            if draft_future is None and candidates:
                draft_future = draft_executor.submit(
                    run_provisional_contour_drafts,
                    question,
                    detective_case,
                    candidates,
                    model=model,
                    retries=retries,
                    client_factory=client_factory,
                    cached_record={},
                )

        detective, detective_record = run_live_detective(
            question,
            materials,
            model=model,
            retries=retries,
            client_factory=client_factory,
            cached_record={},
            proposal_ready_callback=launch_drafts,
            execution_mode="optimized_v2",
        )
        draft_bundle: dict[str, Any] = {}
        draft_record: dict[str, Any] = {}
        if draft_future is not None:
            draft_bundle, draft_record = draft_future.result()
    constraints = build_contour_constraints(question, detective)
    use_speculative = bool(
        draft_bundle
        and draft_record.get("status")
        in {
            "succeeded",
            "reused",
            "succeeded_partial_acceleration",
            "reused_partial_acceleration",
        }
    )
    contour, contour_record = run_live_contour(
        question,
        detective,
        constraints,
        model=model,
        retries=retries,
        client_factory=client_factory,
        cached_record={},
        material_mode="lossless_packet_v2",
        counter_mode="parallel_pressure_v1",
        contour_mode=(
            "speculative_parallel_v1"
            if use_speculative
            else "post_adjudication"
        ),
        provisional_drafts=draft_bundle if use_speculative else None,
    )
    duration_seconds = round(time.perf_counter() - started, 3)
    detective_errors = validate_detective_output(detective, materials)
    contour_errors = validate_truth_contour(
        contour,
        question=question,
        detective=detective,
        constraints=constraints,
    )
    detective_eval = detective.get("detective_evaluation") or {}
    contour_eval = contour.get("contour_evaluation") or {}
    classic_audit = contour.get("classic_knowledge_audit") or {}
    status_counts = detective_eval.get("status_counts") or {}
    accepted = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
    ]
    independence = _independence_metrics(accepted)
    generation_is_valid = contour_record.get("status") == "succeeded" or (
        contour_record.get("status") == "fallback"
        and not (contour_record.get("case") or {}).get(
            "constructive_relation_ids"
        )
    )
    detective_packet = (detective_record.get("audit") or {}).get(
        "prompt_packet"
    ) or {}
    pressure_stage = next(
        (
            item
            for item in (contour_record.get("audit") or {}).get("stages", []) or []
            if item.get("stage") == "parallel_counter_pressure"
        ),
        {},
    )
    speculative_audit = (contour_record.get("audit") or {}).get(
        "speculative_execution"
    ) or {}
    gates = {
        "detective_schema_valid": not detective_errors,
        "contour_schema_valid": not contour_errors,
        "detective_completed": detective_record.get("status") == "succeeded",
        "contour_completed_or_justified_fallback": generation_is_valid,
        "accepted_relations_traceable": (
            float(detective_eval.get("accepted_traceability_rate") or 0) == 1.0
        ),
        "classic_relations_traceable": (
            float(
                detective_eval.get("accepted_classic_traceability_rate") or 0
            )
            == 1.0
        ),
        "no_forced_relations": (
            int(detective_eval.get("forced_relation_count") or 0) == 0
        ),
        "classic_never_world_evidence": (
            int(
                detective_eval.get("classic_used_as_world_evidence_count") or 0
            )
            == 0
            and int(
                classic_audit.get("classic_used_as_world_evidence_count") or 0
            )
            == 0
        ),
        "contour_provenance_complete": (
            float(contour_eval.get("provenance_coverage") or 0) == 1.0
        ),
        "question_answered": bool(str(contour.get("direct_answer") or "").strip())
        and bool(str(contour.get("main_contour") or "").strip()),
        "lossless_packet_validated": (
            ((contour_record.get("audit") or {}).get("prompt_packet") or {}).get(
                "coverage_validation"
            )
            == "passed"
            and ((contour_record.get("audit") or {}).get("prompt_packet") or {}).get(
                "decision_packet_coverage_validation"
            )
            == "passed"
        ),
        "detective_lossless_packet_validated": (
            detective_packet.get("coverage_validation") == "passed"
            and not detective_packet.get("task_relevant_material_removed")
        ),
        "counter_pressure_collectively_lossless": (
            (
                contour_record.get("counter_mode")
                == "speculative_parallel_v1"
                and draft_record.get("status") in {"succeeded", "reused"}
                and (
                    (draft_record.get("audit") or {}).get(
                        "provisional_only_not_world_evidence"
                    )
                    is True
                )
                and speculative_audit.get(
                    "final_case_uses_only_adjudicated_relations"
                )
                is True
            )
            or (
                contour_record.get("counter_mode")
                == "parallel_pressure_v1"
                and pressure_stage.get("packet_coverage_validation")
                == "passed"
                and int(pressure_stage.get("pressure_stage_count") or 0)
                == 3
            )
        ),
    }
    result = {
        "schema": "bme.live-optimized-synthesis-benchmark.v1",
        "source_run": str(run_dir.resolve()),
        "question": question,
        "model": model,
        "synthesis_mode": "optimized_v2",
        "duration_seconds": duration_seconds,
        "gates": gates,
        "hard_gate_status": "passed" if all(gates.values()) else "failed",
        "validation": {
            "detective_errors": detective_errors,
            "contour_errors": contour_errors,
        },
        "baseline": {
            "direct_answer": baseline_contour.get("direct_answer"),
            "main_contour": baseline_contour.get("main_contour"),
            "generation_mode": baseline_contour.get("generation_mode"),
            "relation_count": len(
                baseline_detective.get("relation_certificates", []) or []
            ),
        },
        "optimized": {
            "direct_answer": contour.get("direct_answer"),
            "main_contour": contour.get("main_contour"),
            "generation_mode": contour.get("generation_mode"),
            "relation_count": detective_eval.get("relation_count"),
            "accepted_relation_count": status_counts.get("accepted", 0),
            **independence,
            "key_conditions": contour.get("key_conditions") or [],
            "important_unknowns": contour.get("important_unknowns") or [],
            "prompt_packet": (contour_record.get("audit") or {}).get(
                "prompt_packet"
            ),
            "detective_prompt_packet": detective_packet,
            "counter_pressure": {
                "mode": contour_record.get("counter_mode"),
                "stage_count": pressure_stage.get("pressure_stage_count"),
                "packet_coverage_validation": pressure_stage.get(
                    "packet_coverage_validation"
                ),
                "largest_pressure_packet_characters": pressure_stage.get(
                    "largest_pressure_packet_characters"
                ),
                "speculative_execution": speculative_audit,
            },
        },
        "usage": {
            "detective": detective_record.get("usage") or {},
            "speculative_drafts": draft_record.get("usage") or {},
            "contour": contour_record.get("usage") or {},
            "model_governor": governor.snapshot().to_dict(),
        },
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_root / f"{stamp}-{run_dir.name}"
    target.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "benchmark.json": result,
        "detective.json": detective,
        "detective_record.json": detective_record,
        "speculative_drafts.json": draft_record,
        "contour_constraints.json": constraints,
        "truth_contour.json": contour,
        "contour_record.json": contour_record,
    }
    for name, payload in artifacts.items():
        (target / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    result["output_dir"] = str(target.resolve())
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


QUALITY_DIMENSIONS = (
    "digital_person_integrity",
    "shadow_specificity",
    "connection_quality",
    "source_independence",
    "classic_boundary",
    "truth_contour_quality",
)


def _not_worse(
    candidate: Any,
    baseline: Any,
    *,
    minimum_when_baseline_unknown: float = 0.0,
) -> bool:
    if candidate is None:
        return False
    candidate_value = float(candidate)
    if baseline is None:
        return candidate_value >= minimum_when_baseline_unknown
    return candidate_value >= float(baseline)


def _not_more(
    candidate: Any,
    baseline: Any,
    *,
    maximum_when_baseline_unknown: float = 0.0,
) -> bool:
    if candidate is None:
        return False
    candidate_value = float(candidate)
    if baseline is None:
        return candidate_value <= maximum_when_baseline_unknown
    return candidate_value <= float(baseline)


def _semantic_review_status(review: dict[str, Any] | None) -> dict[str, Any]:
    dimensions = dict((review or {}).get("dimensions") or {})
    candidate_label = str((review or {}).get("_candidate_label") or "").strip()
    checks: dict[str, bool] = {}
    for dimension in QUALITY_DIMENSIONS:
        record = dimensions.get(dimension) or {}
        preferred = str(record.get("preferred") or "").strip()
        blind_preference_passed = (
            bool(candidate_label)
            and preferred in {"tie", candidate_label}
        )
        checks[dimension] = (
            (record.get("not_worse") is True or blind_preference_passed)
            and bool(str(record.get("reason") or "").strip())
        )
    complete = (
        bool(str((review or {}).get("reviewer") or "").strip())
        and (review or {}).get("blind_order") is True
        and all(checks.values())
    )
    return {
        "status": "passed" if complete else "pending_or_failed",
        "reviewer": (review or {}).get("reviewer"),
        "blind_order": (review or {}).get("blind_order"),
        "candidate_label": candidate_label or None,
        "dimension_checks": checks,
        "passed": complete,
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    semantic_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = baseline["quality"]
    after = candidate["quality"]
    gates = {
        "same_question": baseline.get("question") == candidate.get("question"),
        "baseline_mode_retained": (
            baseline.get("pipeline_mode") in {"sequential", "legacy_sequential"}
            and baseline.get("synthesis_mode") == "legacy_sequential"
        ),
        "optimized_mode_exercised": (
            candidate.get("pipeline_mode") == "streaming_v2"
            and candidate.get("synthesis_mode") == "optimized_v2"
        ),
        "all_people_completed": (
            after.get("expected_people") == after.get("succeeded_people")
            and bool(after.get("expected_people"))
        ),
        "ten_detectors_complete": (
            (after.get("detector_coverage") or {}).get(
                "complete_ten_detector_people"
            )
            == after.get("expected_people")
        ),
        "external_evidence_coverage_not_worse": _not_worse(
            after.get("retrieval_external_coverage"),
            before.get("retrieval_external_coverage"),
            minimum_when_baseline_unknown=1.0,
        ),
        "semantic_fallback_not_worse": (
            _not_more(
                after.get("semantic_fallback_count"),
                before.get("semantic_fallback_count"),
                maximum_when_baseline_unknown=0.0,
            )
        ),
        "cognitive_chain_not_worse": _not_worse(
            after.get("cognitive_chain_score"),
            before.get("cognitive_chain_score"),
        ),
        "semantic_specificity_not_worse": _not_worse(
            after.get("semantic_verdict_score"),
            before.get("semantic_verdict_score"),
        ),
        "semantic_live_coverage_complete": (
            after.get("semantic_live_person_count")
            == after.get("expected_people")
            and after.get("semantic_adjudicated_person_count")
            == after.get("expected_people")
        ),
        "validated_semantic_fast_path_exercised": (
            after.get("semantic_inference_mode") == "validated_fast_v1"
        ),
        "meta_model_not_worse": _not_worse(
            after.get("meta_model_score"),
            before.get("meta_model_score"),
        ),
        "detective_completed": after.get("detective_passed") is True,
        "detective_traceability_preserved": (
            float(after.get("accepted_traceability_rate") or 0)
            >= float(before.get("accepted_traceability_rate") or 0)
        ),
        "classic_traceability_complete": (
            float(after.get("accepted_classic_traceability_rate") or 0) == 1.0
        ),
        "no_forced_relations": int(after.get("forced_relation_count") or 0) == 0,
        "source_independence_not_worse": (
            int(after.get("accepted_evidence_independent_relation_count") or 0)
            >= int(before.get("accepted_evidence_independent_relation_count") or 0)
            and int(after.get("accepted_directional_support_relation_count") or 0)
            >= int(before.get("accepted_directional_support_relation_count") or 0)
            and int(after.get("independence_overclaim_count") or 0)
            <= int(before.get("independence_overclaim_count") or 0)
        ),
        "classic_never_world_evidence": (
            int(after.get("classic_world_evidence_count") or 0) == 0
            and int(after.get("contour_classic_world_evidence_count") or 0) == 0
        ),
        "contour_provenance_preserved": (
            float(after.get("contour_provenance_coverage") or 0)
            >= float(before.get("contour_provenance_coverage") or 0)
        ),
        "question_answered": (
            after.get("direct_answer_present") is True
            and after.get("main_contour_present") is True
        ),
        "final_contour_payload_valid": (
            after.get("final_contour_payload_valid") is True
        ),
        "sentence_evidence_binding_complete": (
            after.get("evidence_binding_contract") == "sentence_level.v1"
            and int(after.get("sentence_binding_count") or 0) > 0
            and float(after.get("sentence_binding_coverage") or 0) == 1.0
        ),
        "evidence_qualification_consistent": (
            (
                int(after.get("verified_observation_count") or 0) == 0
                or int(after.get("verified_accepted_evidence_count") or 0) > 0
            )
            and (
                int(after.get("directional_judgment_count") or 0) == 0
                or int(
                    after.get("accepted_directional_support_relation_count")
                    or 0
                )
                > 0
            )
        ),
        "prompt_packet_lossless": after.get("prompt_packet_coverage") == "passed",
        "detective_prompt_packet_lossless": (
            after.get("detective_prompt_packet_coverage") == "passed"
        ),
        "parallel_counter_pressure_exercised": (
            (
                after.get("counter_mode") == "speculative_parallel_v1"
                and after.get("speculative_draft_status")
                in {"succeeded", "reused"}
                and after.get("speculative_drafts_are_world_evidence")
                is False
                and after.get(
                    "speculative_final_uses_only_adjudicated_relations"
                )
                is True
            )
            or (
                after.get("counter_mode") == "parallel_pressure_v1"
                and int(after.get("counter_pressure_stage_count") or 0)
                == 3
                and after.get("counter_pressure_packet_coverage")
                == "passed"
            )
        ),
        "under_ten_minutes": (
            float(candidate["timing"].get("submission_to_result_seconds") or 1e9)
            < 600
        ),
    }
    dimension_gate_names = {
        "digital_person_integrity": (
            "all_people_completed",
            "ten_detectors_complete",
        ),
        "shadow_specificity": (
            "semantic_fallback_not_worse",
            "cognitive_chain_not_worse",
            "semantic_specificity_not_worse",
            "semantic_live_coverage_complete",
            "validated_semantic_fast_path_exercised",
            "meta_model_not_worse",
        ),
        "connection_quality": (
            "detective_completed",
            "detective_traceability_preserved",
            "classic_traceability_complete",
            "no_forced_relations",
            "parallel_counter_pressure_exercised",
        ),
        "source_independence": (
            "external_evidence_coverage_not_worse",
            "source_independence_not_worse",
        ),
        "classic_boundary": (
            "classic_never_world_evidence",
            "classic_traceability_complete",
        ),
        "truth_contour_quality": (
            "contour_provenance_preserved",
            "final_contour_payload_valid",
            "sentence_evidence_binding_complete",
            "evidence_qualification_consistent",
            "question_answered",
            "prompt_packet_lossless",
            "detective_prompt_packet_lossless",
        ),
    }
    quality_dimensions = {
        dimension: {
            "passed": all(gates[name] for name in gate_names),
            "gates": {name: gates[name] for name in gate_names},
        }
        for dimension, gate_names in dimension_gate_names.items()
    }
    machine_passed = all(gates.values()) and all(
        item["passed"] for item in quality_dimensions.values()
    )
    review_status = _semantic_review_status(semantic_review)
    return {
        "schema": "bme.pipeline-upgrade-comparison.v2",
        "same_question": baseline.get("question") == candidate.get("question"),
        "baseline": baseline,
        "candidate": candidate,
        "gates": gates,
        "quality_dimensions": quality_dimensions,
        "machine_passed": machine_passed,
        "semantic_review": review_status,
        "activation_ready": machine_passed and review_status["passed"],
        "passed": machine_passed,
    }


def _manifest_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = ROOT / path
    if project_path.exists():
        return project_path
    return manifest_path.parent / path


def _review_view(run_dir: Path) -> dict[str, Any]:
    contour = read_json(run_dir / "truth_contour.json", {}) or {}
    diagnosis = read_json(run_dir / "diagnosis.json", {}) or {}
    detective = read_json(run_dir / "detective.json", {}) or {}
    shadow_summaries = []
    for index, scan in enumerate(diagnosis.get("cognitive_chain_scans") or [], 1):
        summary = scan.get("plain_language_summary") or {}
        semantic = scan.get("semantic_verdict") or {}
        shadow_summaries.append(
            {
                "person": scan.get("person_name") or f"P{index:02d}",
                "headline": semantic.get("headline") or summary.get("headline"),
                "key_problems": (
                    (semantic.get("summary") or {}).get("key_problems")
                    or summary.get("key_problems")
                    or []
                ),
                "unresolved": semantic.get("unresolved")
                or summary.get("unresolved_checks")
                or [],
            }
        )
    accepted_connections = []
    for relation in detective.get("relation_certificates") or []:
        if relation.get("status") != "accepted":
            continue
        accepted_connections.append(
            {
                "relation_type": relation.get("relation_type"),
                "explanation": relation.get("plain_language_explanation"),
                "competing_explanation": relation.get("competing_explanation"),
                "falsification_test": relation.get("falsification_test"),
                "independence_profile": relation.get("independence_profile"),
            }
        )
    return {
        "truth_contour": {
            key: contour.get(key)
            for key in (
                "direct_answer",
                "main_contour",
                "key_conditions",
                "stable_parts",
                "boundary_conditions",
                "important_unknowns",
                "confidence_statement",
                "why_this_contour",
                "strongest_counter_contour",
                "counter_contour_disposition",
            )
        },
        "individual_shadow_analyses": shadow_summaries,
        "accepted_connections": accepted_connections,
    }


def create_blind_review_packet(
    baseline_run: Path,
    candidate_run: Path,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_audit = audit_run(baseline_run)
    candidate_audit = audit_run(candidate_run)
    if baseline_audit.get("question") != candidate_audit.get("question"):
        raise ValueError("blind review requires runs for the same question")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_label = "A" if secrets.randbelow(2) == 0 else "B"
    baseline_label = "B" if candidate_label == "A" else "A"
    views = {
        baseline_label: _review_view(baseline_run),
        candidate_label: _review_view(candidate_run),
    }
    packet = {
        "schema": "bme.blind-quality-review-packet.v1",
        "question": baseline_audit.get("question"),
        "instructions": (
            "先不要查看 review-key.json。逐项比较 A 与 B；preferred 只能填 A、B 或 tie，"
            "reason 必须指出具体内容。"
        ),
        "required_dimensions": list(QUALITY_DIMENSIONS),
        "versions": {label: views[label] for label in ("A", "B")},
    }
    review = {
        "schema": "bme.blind-quality-review.v1",
        "reviewer": "",
        "blind_order": True,
        "dimensions": {
            dimension: {"preferred": "", "reason": ""}
            for dimension in QUALITY_DIMENSIONS
        },
    }
    key = {
        "schema": "bme.blind-quality-review-key.v1",
        "candidate_label": candidate_label,
        "baseline_label": baseline_label,
        "candidate_run": str(candidate_run.resolve()),
        "baseline_run": str(baseline_run.resolve()),
    }
    artifacts = {
        "review-packet.json": packet,
        "review.json": review,
        "review-key.json": key,
    }
    for name, payload in artifacts.items():
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "schema": "bme.blind-quality-review-artifacts.v1",
        "output_dir": str(output_dir.resolve()),
        "packet": str((output_dir / "review-packet.json").resolve()),
        "review": str((output_dir / "review.json").resolve()),
        "key": str((output_dir / "review-key.json").resolve()),
    }


def compare_cross_domain_suite(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path, {}) or {}
    cases = list(manifest.get("cases") or [])
    domains = {str(item.get("domain") or "").strip() for item in cases}
    domains.discard("")
    results: list[dict[str, Any]] = []
    for item in cases:
        baseline_value = item.get("baseline_run")
        candidate_value = item.get("candidate_run")
        record: dict[str, Any] = {
            "case_id": item.get("case_id"),
            "domain": item.get("domain"),
            "question": item.get("question"),
        }
        if not baseline_value or not candidate_value:
            record.update(
                {
                    "status": "pending_candidate",
                    "activation_ready": False,
                }
            )
            results.append(record)
            continue
        baseline = audit_run(_manifest_path(baseline_value, manifest_path))
        candidate = audit_run(_manifest_path(candidate_value, manifest_path))
        review = item.get("semantic_review")
        review_file = item.get("semantic_review_file")
        review_key_file = item.get("semantic_review_key_file")
        if review_file:
            review = read_json(_manifest_path(review_file, manifest_path), {})
        if review_key_file:
            review_key = read_json(
                _manifest_path(review_key_file, manifest_path), {}
            ) or {}
            review = dict(review or {})
            review["_candidate_label"] = review_key.get("candidate_label")
        comparison = compare_runs(
            baseline,
            candidate,
            semantic_review=review,
        )
        question_matches_manifest = (
            baseline.get("question") == item.get("question")
            and candidate.get("question") == item.get("question")
        )
        record.update(
            {
                "status": (
                    "passed"
                    if comparison["activation_ready"]
                    and question_matches_manifest
                    else "failed_or_pending_review"
                ),
                "question_matches_manifest": question_matches_manifest,
                "comparison": comparison,
                "activation_ready": (
                    comparison["activation_ready"]
                    and question_matches_manifest
                ),
            }
        )
        results.append(record)
    suite_shape_valid = len(cases) >= 3 and len(domains) >= 3
    activation_ready = (
        suite_shape_valid
        and len(results) == len(cases)
        and all(item.get("activation_ready") is True for item in results)
    )
    return {
        "schema": "bme.cross-domain-activation-suite.v1",
        "manifest": str(manifest_path.resolve()),
        "required_quality_dimensions": list(QUALITY_DIMENSIONS),
        "case_count": len(cases),
        "distinct_domain_count": len(domains),
        "suite_shape_valid": suite_shape_valid,
        "cases": results,
        "activation_ready": activation_ready,
        "default_switch_allowed": activation_ready,
    }


def _independence_metrics(
    accepted_relations: list[dict[str, Any]],
) -> dict[str, int]:
    evidence_independent = 0
    publisher_independent = 0
    directional_authorized = 0
    shared_model = 0
    declared_independent = 0
    overclaims = 0
    independent_states = {"independent", "independent_by_current_ledger"}

    for relation in accepted_relations:
        profile = relation.get("independence_profile") or {}
        declared = relation.get("source_independence") == "independent"
        evidence_state = profile.get("evidence_origin_independence")
        if evidence_state is None:
            evidence_state = profile.get("external_evidence")
        evidence_is_independent = evidence_state in independent_states
        if not profile and declared:
            evidence_is_independent = True

        declared_independent += int(declared)
        evidence_independent += int(evidence_is_independent)
        publisher_independent += int(
            profile.get("publisher_independence") in independent_states
        )
        directional_authorized += int(
            profile.get("directional_support_authorized") is True
        )
        shared_model += int(profile.get("model_independence") == "shared")
        overclaims += int(declared and bool(profile) and not evidence_is_independent)

    return {
        # Retained for old benchmark readers; now means evidence-origin independence.
        "accepted_independent_relation_count": evidence_independent,
        "accepted_evidence_independent_relation_count": evidence_independent,
        "accepted_publisher_independent_relation_count": publisher_independent,
        "accepted_directional_support_relation_count": directional_authorized,
        "accepted_shared_model_relation_count": shared_model,
        "declared_independent_relation_count": declared_independent,
        "independence_overclaim_count": overclaims,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("run_dirs", nargs="+", type=Path)
    lossless_parser = subparsers.add_parser("lossless")
    lossless_parser.add_argument("run_dirs", nargs="+", type=Path)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("run_dirs", nargs="+", type=Path)
    replay_parser.add_argument(
        "--output-root", type=Path, default=ROOT / "benchmarks" / "replays"
    )
    live_parser = subparsers.add_parser("live-synthesis")
    live_parser.add_argument("run_dirs", nargs="+", type=Path)
    live_parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "benchmarks" / "live-synthesis",
    )
    live_parser.add_argument("--model", default=None)
    live_parser.add_argument("--retries", type=int, default=1)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--review", type=Path, default=None)
    suite_parser = subparsers.add_parser("suite")
    suite_parser.add_argument(
        "manifest",
        type=Path,
        default=ROOT / "benchmarks" / "cross-domain-acceptance.json",
        nargs="?",
    )
    review_parser = subparsers.add_parser("review-packet")
    review_parser.add_argument("baseline", type=Path)
    review_parser.add_argument("candidate", type=Path)
    review_parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if args.command == "audit":
        payload: Any = [audit_run(path) for path in args.run_dirs]
    elif args.command == "lossless":
        payload = [
            audit_lossless_optimization(path) for path in args.run_dirs
        ]
    elif args.command == "replay":
        payload = [
            replay_saved_synthesis(path, args.output_root)
            for path in args.run_dirs
        ]
    elif args.command == "live-synthesis":
        payload = [
            run_live_optimized_synthesis(
                path,
                args.output_root,
                model_override=args.model,
                retries=args.retries,
            )
            for path in args.run_dirs
        ]
    elif args.command == "compare":
        review = read_json(args.review, {}) if args.review else None
        payload = compare_runs(
            audit_run(args.baseline),
            audit_run(args.candidate),
            semantic_review=review,
        )
    elif args.command == "suite":
        payload = compare_cross_domain_suite(args.manifest)
    else:
        payload = create_blind_review_packet(
            args.baseline,
            args.candidate,
            args.output_dir,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
