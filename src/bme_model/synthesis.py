from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .contour import (
    build_contour_constraints,
    build_truth_contour,
    run_live_contour,
    run_provisional_contour_drafts,
)
from .contour.schemas import validate_truth_contour
from .detective import (
    build_detective_output,
    materialize_puzzle_pieces,
    run_live_detective,
)
from .detective.schemas import validate_detective_output
from .providers import DeepSeekClient
from .runtime import atomic_write_json


SYNTHESIS_ARTIFACTS = {
    "puzzle_materials": "puzzle_pieces.json",
    "detective": "detective.json",
    "contour_constraints": "contour_constraints.json",
    "contour_candidates": "contour_candidates.json",
    "truth_contour": "truth_contour.json",
}


def build_deterministic_synthesis(
    question: str,
    diagnosis: dict[str, Any],
    person_outputs: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]],
    *,
    question_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materials = materialize_puzzle_pieces(
        question,
        diagnosis,
        person_outputs,
        evidence_ledgers,
        question_frame,
    )
    detective = build_detective_output(question, materials)
    constraints = build_contour_constraints(question, detective)
    contour = build_truth_contour(question, detective, constraints)
    return {
        "puzzle_materials": materials,
        "detective": detective,
        "contour_constraints": constraints,
        "contour_candidates": {},
        "truth_contour": contour,
        "relational_synthesis_usage": {
            "mode": "deterministic",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        },
    }


def run_live_synthesis(
    question: str,
    diagnosis: dict[str, Any],
    person_outputs: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]],
    *,
    question_frame: dict[str, Any] | None = None,
    model: str,
    max_tokens: int = 8192,
    retries: int = 2,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    resume_record: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    synthesis_mode: str = "optimized_v2",
) -> dict[str, Any]:
    if synthesis_mode not in {"legacy_sequential", "optimized_v2"}:
        raise ValueError(
            "synthesis_mode must be 'legacy_sequential' or 'optimized_v2'"
        )
    checkpoint_record = {
        "detective": dict((resume_record or {}).get("detective") or {}),
        "speculative_drafts": dict(
            (resume_record or {}).get("speculative_drafts") or {}
        ),
        "contour": dict((resume_record or {}).get("contour") or {}),
    }
    checkpoint_lock = threading.Lock()

    def save_component(name: str, record: dict[str, Any]) -> None:
        with checkpoint_lock:
            checkpoint_record[name] = record
            snapshot = copy.deepcopy(checkpoint_record)
            if checkpoint_callback:
                checkpoint_callback(snapshot)

    def save_detective(record: dict[str, Any]) -> None:
        save_component("detective", record)

    def save_contour(record: dict[str, Any]) -> None:
        save_component("contour", record)

    def save_speculative_drafts(record: dict[str, Any]) -> None:
        save_component("speculative_drafts", record)

    materials = materialize_puzzle_pieces(
        question,
        diagnosis,
        person_outputs,
        evidence_ledgers,
        question_frame,
    )
    draft_executor = (
        ThreadPoolExecutor(max_workers=1)
        if synthesis_mode == "optimized_v2"
        else None
    )
    draft_future: Future[tuple[dict[str, Any], dict[str, Any]]] | None = None

    def launch_speculative_drafts(
        detective_case: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> None:
        nonlocal draft_future
        if not draft_executor or draft_future is not None or not candidates:
            return
        def run_and_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
            bundle, record = run_provisional_contour_drafts(
                question,
                detective_case,
                candidates,
                model=model,
                max_tokens=min(max_tokens, 4096),
                retries=retries,
                client_factory=client_factory,
                cached_record=(resume_record or {}).get("speculative_drafts")
                or {},
            )
            save_speculative_drafts(record)
            return bundle, record

        draft_future = draft_executor.submit(run_and_checkpoint)

    try:
        detective, detective_record = run_live_detective(
            question,
            materials,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            client_factory=client_factory,
            cached_record=(resume_record or {}).get("detective") or {},
            checkpoint_callback=save_detective,
            proposal_ready_callback=launch_speculative_drafts,
            execution_mode=synthesis_mode,
        )
        draft_bundle: dict[str, Any] = {}
        draft_record: dict[str, Any] = {}
        if draft_future is not None:
            draft_bundle, draft_record = draft_future.result()
    finally:
        if draft_executor:
            draft_executor.shutdown(wait=True)
    constraints = build_contour_constraints(question, detective)
    use_speculative_fast_path = bool(
        synthesis_mode == "optimized_v2"
        and draft_bundle
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
        max_tokens=max_tokens,
        retries=retries,
        client_factory=client_factory,
        cached_record=(resume_record or {}).get("contour") or {},
        checkpoint_callback=save_contour,
        material_mode=(
            "legacy_full_case"
            if synthesis_mode == "legacy_sequential"
            else "lossless_packet_v2"
        ),
        counter_mode=(
            "single_counter"
            if synthesis_mode == "legacy_sequential"
            else "parallel_pressure_v1"
        ),
        contour_mode=(
            "speculative_parallel_v1"
            if use_speculative_fast_path
            else "post_adjudication"
        ),
        provisional_drafts=(draft_bundle if use_speculative_fast_path else None),
    )
    usage = combine_synthesis_usage(
        detective_record.get("usage") or {},
        contour_record.get("usage") or {},
        model=model,
        draft_usage=draft_record.get("usage") or {},
    )
    usage["calls"] = sum(
        int(item.get("attempts") or 0)
        for item in detective_record.get("audit", {}).get("stages", [])
    ) + sum(
        int(item.get("attempts") or 0)
        for item in contour_record.get("audit", {}).get("stages", [])
    ) + sum(
        int(item.get("attempts") or 0)
        for item in draft_record.get("audit", {}).get("stages", [])
    )
    usage["synthesis_mode"] = synthesis_mode
    result = {
        "puzzle_materials": materials,
        "detective": detective,
        "contour_constraints": constraints,
        "contour_candidates": {
            "detective": detective_record,
            "speculative_drafts": draft_record,
            "contour": contour_record,
        },
        "truth_contour": contour,
        "relational_synthesis_usage": usage,
    }
    with checkpoint_lock:
        checkpoint_record["detective"] = detective_record
        checkpoint_record["speculative_drafts"] = draft_record
        checkpoint_record["contour"] = contour_record
        final_checkpoint = copy.deepcopy(checkpoint_record)
        if checkpoint_callback:
            checkpoint_callback(final_checkpoint)
    return result


def restore_or_build_synthesis(
    run_path: Path,
    question: str,
    diagnosis: dict[str, Any],
    person_outputs: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]],
    *,
    question_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild deterministically while preserving a still-valid live contour."""

    base = build_deterministic_synthesis(
        question,
        diagnosis,
        person_outputs,
        evidence_ledgers,
        question_frame=question_frame,
    )
    existing_detective = _read_optional(run_path / "detective.json")
    if existing_detective and not validate_detective_output(
        existing_detective, base["puzzle_materials"]
    ):
        semantic_relations = [
            item
            for item in existing_detective.get("relation_certificates", []) or []
            if (item.get("provenance") or {}).get("mode")
            in {"live_adversarial_adjudication", "semantic_adjudication"}
        ]
        base["detective"] = build_detective_output(
            question,
            base["puzzle_materials"],
            semantic_relations=semantic_relations,
            semantic_audit=(existing_detective.get("detective_self_audit") or {}).get(
                "semantic_stage"
            ),
        )
        base["contour_constraints"] = build_contour_constraints(
            question, base["detective"]
        )

    existing_candidates = _read_optional(run_path / "contour_candidates.json")
    contour_record = existing_candidates.get("contour") or {}
    final_adjudication = contour_record.get("final_adjudication") or {}
    regenerated_contour = {}
    if final_adjudication:
        regenerated_contour = build_truth_contour(
            question,
            base["detective"],
            base["contour_constraints"],
            adjudicated_payload=final_adjudication,
            generation_audit=contour_record.get("audit") or {},
        )
    existing_contour = _read_optional(run_path / "truth_contour.json")
    contour_to_restore = regenerated_contour or existing_contour
    if contour_to_restore and not validate_truth_contour(
        contour_to_restore,
        question=question,
        detective=base["detective"],
        constraints=base["contour_constraints"],
    ):
        base["truth_contour"] = contour_to_restore
        base["contour_candidates"] = existing_candidates
        existing_usage = _read_existing_usage(run_path)
        if existing_usage:
            base["relational_synthesis_usage"] = existing_usage
    return base


def combine_synthesis_usage(
    detective_usage: dict[str, Any],
    contour_usage: dict[str, Any],
    *,
    model: str,
    draft_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    draft_usage = draft_usage or {}
    prompt = (
        int(detective_usage.get("prompt_tokens") or 0)
        + int(contour_usage.get("prompt_tokens") or 0)
        + int(draft_usage.get("prompt_tokens") or 0)
    )
    completion = (
        int(detective_usage.get("completion_tokens") or 0)
        + int(contour_usage.get("completion_tokens") or 0)
        + int(draft_usage.get("completion_tokens") or 0)
    )
    total = (
        int(detective_usage.get("total_tokens") or 0)
        + int(contour_usage.get("total_tokens") or 0)
        + int(draft_usage.get("total_tokens") or 0)
    )
    return {
        "mode": "live",
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
        "calls": 0,
        "detective_usage": detective_usage,
        "speculative_draft_usage": draft_usage,
        "contour_usage": contour_usage,
    }


def write_synthesis_artifacts(run_path: Path, synthesis: dict[str, Any]) -> None:
    for key, filename in SYNTHESIS_ARTIFACTS.items():
        payload = synthesis.get(key)
        if payload is None:
            continue
        atomic_write_json(run_path / filename, payload)


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_existing_usage(run_path: Path) -> dict[str, Any]:
    run_payload = _read_optional(run_path / "run.json")
    return run_payload.get("relational_synthesis_usage") or {}
