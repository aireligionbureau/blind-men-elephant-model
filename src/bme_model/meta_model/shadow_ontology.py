from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowMechanism:
    id: str
    name: str
    shadow_family: str
    core_mechanism: str
    generation_chain: list[str]
    chain_stages: list[str]
    visible_signals: list[str]
    capture_questions: list[str]
    false_positive_risks: list[str]
    false_negative_risks: list[str]
    linked_lenses: list[str]
    puzzle_material: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShadowMechanism":
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            shadow_family=str(payload["shadow_family"]),
            core_mechanism=str(payload["core_mechanism"]),
            generation_chain=list(payload["generation_chain"]),
            chain_stages=list(payload["chain_stages"]),
            visible_signals=list(payload["visible_signals"]),
            capture_questions=list(payload["capture_questions"]),
            false_positive_risks=list(payload["false_positive_risks"]),
            false_negative_risks=list(payload["false_negative_risks"]),
            linked_lenses=list(payload["linked_lenses"]),
            puzzle_material=str(payload["puzzle_material"]),
        )


@dataclass(frozen=True)
class CaptureProtocolStage:
    stage: str
    touch_target: str
    what_to_inspect: list[str]
    shadow_questions: list[str]
    evidence_needed: list[str]
    failure_if_skipped: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptureProtocolStage":
        return cls(
            stage=str(payload["stage"]),
            touch_target=str(payload["touch_target"]),
            what_to_inspect=list(payload["what_to_inspect"]),
            shadow_questions=list(payload["shadow_questions"]),
            evidence_needed=list(payload["evidence_needed"]),
            failure_if_skipped=str(payload["failure_if_skipped"]),
        )


def load_shadow_mechanisms(path: str | Path | None = None) -> dict[str, ShadowMechanism]:
    file_path = Path(path) if path is not None else _default_shadow_mechanism_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return {
            mechanism.id: mechanism
            for mechanism in (ShadowMechanism.from_dict(item) for item in json.load(handle))
        }


def load_capture_protocols(path: str | Path | None = None) -> list[CaptureProtocolStage]:
    file_path = Path(path) if path is not None else _default_capture_protocol_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return [CaptureProtocolStage.from_dict(item) for item in json.load(handle)]


def _default_shadow_mechanism_path() -> Path:
    return _project_root() / "knowledge" / "shadow_mechanisms.json"


def _default_capture_protocol_path() -> Path:
    return _project_root() / "knowledge" / "capture_protocols.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
