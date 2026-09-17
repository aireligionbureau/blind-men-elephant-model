from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationCase:
    id: str
    title: str
    scenario: str
    expected_primary_lenses: list[str]
    expected_confusions: list[str]
    calibration_questions: list[str]
    failure_mode: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationCase":
        return cls(
            id=str(payload["id"]),
            title=str(payload["title"]),
            scenario=str(payload["scenario"]),
            expected_primary_lenses=list(payload["expected_primary_lenses"]),
            expected_confusions=list(payload["expected_confusions"]),
            calibration_questions=list(payload["calibration_questions"]),
            failure_mode=str(payload["failure_mode"]),
        )


@dataclass(frozen=True)
class MicroLens:
    id: str
    parent_lens: str
    name: str
    source_work: str
    mechanism: str
    trigger_signals: list[str]
    diagnostic_questions: list[str]
    differentiates_from: list[str]
    inversion_hint: str
    misuse_warning: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MicroLens":
        return cls(
            id=str(payload["id"]),
            parent_lens=str(payload["parent_lens"]),
            name=str(payload["name"]),
            source_work=str(payload["source_work"]),
            mechanism=str(payload["mechanism"]),
            trigger_signals=list(payload["trigger_signals"]),
            diagnostic_questions=list(payload["diagnostic_questions"]),
            differentiates_from=list(payload["differentiates_from"]),
            inversion_hint=str(payload["inversion_hint"]),
            misuse_warning=str(payload["misuse_warning"]),
        )


@dataclass(frozen=True)
class LensConflictRule:
    id: str
    competing_lenses: list[str]
    confusion_pattern: str
    decision_questions: list[str]
    prefer_first_when: list[str]
    prefer_second_when: list[str]
    coexist_when: list[str]
    misdiagnosis_risk: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LensConflictRule":
        return cls(
            id=str(payload["id"]),
            competing_lenses=list(payload["competing_lenses"]),
            confusion_pattern=str(payload["confusion_pattern"]),
            decision_questions=list(payload["decision_questions"]),
            prefer_first_when=list(payload["prefer_first_when"]),
            prefer_second_when=list(payload["prefer_second_when"]),
            coexist_when=list(payload["coexist_when"]),
            misdiagnosis_risk=str(payload["misdiagnosis_risk"]),
        )


@dataclass(frozen=True)
class LensCard:
    id: str
    name: str
    source_works: list[str]
    category: str
    core_idea: str
    detect_signals: list[str]
    diagnostic_questions: list[str]
    shadow_types: list[str]
    source_stages: list[str]
    mechanism_chain: list[str]
    observable_markers: list[str]
    attribution_rules: list[str]
    inversion_rules: list[str]
    puzzle_uses: list[str]
    boundary_conditions: list[str]
    anti_misuse_rules: list[str]
    model_training_notes: list[str]
    self_reflection: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LensCard":
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            source_works=list(payload["source_works"]),
            category=str(payload["category"]),
            core_idea=str(payload["core_idea"]),
            detect_signals=list(payload["detect_signals"]),
            diagnostic_questions=list(payload["diagnostic_questions"]),
            shadow_types=list(payload["shadow_types"]),
            source_stages=list(payload["source_stages"]),
            mechanism_chain=list(payload.get("mechanism_chain", [])),
            observable_markers=list(payload.get("observable_markers", payload.get("detect_signals", []))),
            attribution_rules=list(payload["attribution_rules"]),
            inversion_rules=list(payload["inversion_rules"]),
            puzzle_uses=list(payload.get("puzzle_uses", ["unknown"])),
            boundary_conditions=list(payload.get("boundary_conditions", [])),
            anti_misuse_rules=list(payload.get("anti_misuse_rules", [])),
            model_training_notes=list(payload.get("model_training_notes", [])),
            self_reflection=list(payload["self_reflection"]),
        )


def load_lenses(path: str | Path | None = None) -> dict[str, LensCard]:
    lens_dir = Path(path) if path is not None else _default_lens_dir()
    lenses: dict[str, LensCard] = {}
    for file_path in sorted(lens_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as handle:
            lens = LensCard.from_dict(json.load(handle))
        lenses[lens.id] = lens
    return lenses


def load_micro_lenses(path: str | Path | None = None) -> dict[str, MicroLens]:
    file_path = Path(path) if path is not None else _default_micro_lens_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return {
            lens.id: lens
            for lens in (MicroLens.from_dict(item) for item in json.load(handle))
        }


def load_lens_conflicts(path: str | Path | None = None) -> list[LensConflictRule]:
    file_path = Path(path) if path is not None else _default_lens_conflict_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return [LensConflictRule.from_dict(item) for item in json.load(handle)]


def load_calibration_cases(path: str | Path | None = None) -> list[CalibrationCase]:
    file_path = Path(path) if path is not None else _default_calibration_case_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return [CalibrationCase.from_dict(item) for item in json.load(handle)]


def _default_lens_dir() -> Path:
    return _project_root() / "knowledge" / "lenses"


def _default_micro_lens_path() -> Path:
    return _project_root() / "knowledge" / "micro_lenses.json"


def _default_lens_conflict_path() -> Path:
    return _project_root() / "knowledge" / "lens_conflicts.json"


def _default_calibration_case_path() -> Path:
    return _project_root() / "knowledge" / "calibration_cases.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
