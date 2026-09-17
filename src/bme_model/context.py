from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


class ContextIsolationError(RuntimeError):
    """Raised when artifacts from different question runs are mixed."""


def validate_run_isolation(
    question: str,
    cohort_id: str,
    personas: list[Any],
    *,
    question_frame: dict[str, Any] | None = None,
    cohort_generation: dict[str, Any] | None = None,
    retrieval_records: list[dict[str, Any]] | None = None,
    evidence_ledgers: list[dict[str, Any]] | None = None,
    person_results: list[dict[str, Any]] | None = None,
) -> None:
    if not question.strip():
        raise ContextIsolationError("current question is empty")
    if not cohort_id.strip():
        raise ContextIsolationError("cohort_id is empty")

    persona_ids = [_person_id(person) for person in personas]
    if any(not person_id for person_id in persona_ids):
        raise ContextIsolationError("a persona has no id")
    if len(set(persona_ids)) != len(persona_ids):
        raise ContextIsolationError("duplicate persona ids in current cohort")
    expected_ids = set(persona_ids)

    frame = question_frame or {}
    if not frame:
        raise ContextIsolationError("question frame is missing; legacy run is not isolation-safe")
    frame_question = str(frame.get("exact_question", "")).strip()
    frame_context_id = str(frame.get("context_id", "")).strip()
    if not frame_question or frame_question != question:
        raise ContextIsolationError("question frame belongs to a different question")
    if not frame_context_id or frame_context_id != cohort_id:
        raise ContextIsolationError("question frame belongs to a different cohort")

    generation = cohort_generation or {}
    if generation.get("context_isolation") != "only_current_question":
        raise ContextIsolationError("cohort lacks the only-current-question isolation contract")

    if retrieval_records is None:
        raise ContextIsolationError("retrieval records are missing")
    _validate_retrieval_records(
        retrieval_records,
        question=question,
        cohort_id=cohort_id,
        expected_ids=expected_ids,
    )
    if evidence_ledgers is None:
        raise ContextIsolationError("evidence ledgers are missing")
    _validate_ledgers(
        evidence_ledgers,
        question=question,
        expected_ids=expected_ids,
    )
    if person_results is not None:
        _validate_person_results(person_results, expected_ids=expected_ids)


def validate_artifact_questions(question: str, artifacts: dict[str, dict[str, Any]]) -> None:
    mismatches = {
        name: str(payload.get("question", "")).strip()
        for name, payload in artifacts.items()
        if str(payload.get("question", "")).strip()
        and str(payload.get("question", "")).strip() != question
    }
    if mismatches:
        detail = ", ".join(f"{name}={value!r}" for name, value in sorted(mismatches.items()))
        raise ContextIsolationError(f"run artifacts contain a different question: {detail}")


def _validate_retrieval_records(
    records: list[dict[str, Any]],
    *,
    question: str,
    cohort_id: str,
    expected_ids: set[str],
) -> None:
    seen: set[str] = set()
    for record in records:
        person_id = str(record.get("person_id", "")).strip()
        if person_id not in expected_ids:
            raise ContextIsolationError(f"retrieval record has foreign person_id: {person_id!r}")
        if person_id in seen:
            raise ContextIsolationError(f"duplicate retrieval record for person_id: {person_id!r}")
        seen.add(person_id)
        context_id = str(record.get("question_context_id", "")).strip()
        if context_id != cohort_id:
            raise ContextIsolationError(f"retrieval record {person_id!r} has a foreign context id")
        frame = record.get("question_frame") or {}
        if not isinstance(frame, dict):
            raise ContextIsolationError(f"retrieval record {person_id!r} has no question frame")
        frame_question = str(frame.get("exact_question", "")).strip()
        frame_context_id = str(frame.get("context_id", "")).strip()
        if frame_question != question:
            raise ContextIsolationError(f"retrieval record {person_id!r} has a foreign question")
        if frame_context_id != cohort_id:
            raise ContextIsolationError(f"retrieval record {person_id!r} has a foreign frame context")
    missing = expected_ids - seen
    if missing:
        raise ContextIsolationError(f"retrieval records are missing current personas: {sorted(missing)}")


def _validate_ledgers(
    ledgers: list[dict[str, Any]],
    *,
    question: str,
    expected_ids: set[str],
) -> None:
    seen: set[str] = set()
    for ledger in ledgers:
        person_id = str(ledger.get("person_id", "")).strip()
        if person_id not in expected_ids:
            raise ContextIsolationError(f"evidence ledger has foreign person_id: {person_id!r}")
        if person_id in seen:
            raise ContextIsolationError(f"duplicate evidence ledger for person_id: {person_id!r}")
        seen.add(person_id)
        ledger_question = str(ledger.get("question", "")).strip()
        if ledger_question != question:
            raise ContextIsolationError(f"evidence ledger {person_id!r} belongs to a different question")
    missing = expected_ids - seen
    if missing:
        raise ContextIsolationError(f"evidence ledgers are missing current personas: {sorted(missing)}")


def _validate_person_results(results: list[dict[str, Any]], *, expected_ids: set[str]) -> None:
    seen: set[str] = set()
    for result in results:
        person_id = _person_id(result)
        if person_id not in expected_ids:
            raise ContextIsolationError(f"person output has foreign person_id: {person_id!r}")
        if person_id in seen:
            raise ContextIsolationError(f"duplicate person output for person_id: {person_id!r}")
        seen.add(person_id)
    missing = expected_ids - seen
    if missing:
        raise ContextIsolationError(f"person outputs are missing current personas: {sorted(missing)}")


def _person_id(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        return str(getattr(value, "id", "")).strip()
    return str(
        value.get("person_id")
        or value.get("id")
        or (value.get("person") or {}).get("id")
        or (value.get("output") or {}).get("person_id")
        or ""
    ).strip()
