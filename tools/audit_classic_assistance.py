from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.detective.live import (  # noqa: E402
    CONTENT_TYPES,
    MAX_CLASSIC_MATERIAL_ADDITIONS,
    SHADOW_TYPES,
    _adjudication_messages,
    _adjudicate_candidates,
    _candidate_classic_check_ids,
    _classic_review_messages,
    _proposal_messages,
    _proposal_piece_selection,
    build_detective_case,
)
from bme_model.json_utils import parse_json_content  # noqa: E402


MODE_CONFIG = {
    "content": (
        CONTENT_TYPES,
        {
            "observed_claim",
            "retained_local",
            "evidence_claim",
            "assumption",
            "value_condition",
            "fracture",
            "blind_spot_candidate",
        },
    ),
    "shadow": (
        SHADOW_TYPES,
        {
            "observed_claim",
            "retained_local",
            "distortion",
            "fracture",
            "blind_spot_candidate",
            "assumption",
            "value_condition",
        },
    ),
}


def audit(run_dir: Path) -> dict:
    materials = json.loads(
        (run_dir / "puzzle_pieces.json").read_text(encoding="utf-8")
    )
    case = build_detective_case(materials["question"], materials)
    person_by_piece = {
        item["piece_id"]: item.get("person_id") for item in case["piece_index"]
    }
    modes = []
    violations = []
    for mode, (allowed_types, allowed_kinds) in MODE_CONFIG.items():
        native = _proposal_piece_selection(
            case["piece_index"], allowed_kinds, mode=mode
        )
        message = _proposal_messages(case, mode, allowed_types)[1]["content"]
        case_json = message.split("CASE_JSON:\n", 1)[1].split(
            "\nOUTPUT_SCHEMA:\n", 1
        )[0]
        prompt_case = json.loads(case_json)
        prompted = prompt_case["pieces"]
        native_ids = {item["piece_id"] for item in native}
        prompted_ids = {item["piece_id"] for item in prompted}
        extra_ids = prompted_ids - native_ids
        extra_people = Counter(person_by_piece.get(item) for item in extra_ids)
        native_preserved = native_ids.issubset(prompted_ids)
        bounded_additions = (
            len(extra_ids) <= MAX_CLASSIC_MATERIAL_ADDITIONS
            and all(count <= 1 for count in extra_people.values())
        )
        if not native_preserved:
            violations.append(f"{mode}: classic material displaced native material")
        if not bounded_additions:
            violations.append(f"{mode}: classic material additions exceeded limits")

        hypotheses = prompt_case["classic_connection_hypotheses"]
        cross_person_partner_errors = 0
        for hypothesis in hypotheses:
            source_person = hypothesis.get("source_person_id")
            for partner_id in hypothesis.get("candidate_partner_piece_ids", []) or []:
                if person_by_piece.get(partner_id) == source_person:
                    cross_person_partner_errors += 1
        if cross_person_partner_errors:
            violations.append(
                f"{mode}: classic hypotheses contain same-person partners"
            )
        modes.append(
            {
                "mode": mode,
                "prompt_characters": len(message),
                "native_material_count": len(native_ids),
                "prompted_material_count": len(prompted_ids),
                "classic_material_addition_count": len(extra_ids),
                "native_material_preserved": native_preserved,
                "classic_additions_bounded": bounded_additions,
                "visible_classic_hypothesis_count": len(hypotheses),
                "hypotheses_with_visible_partner_count": sum(
                    bool(item.get("candidate_partner_piece_ids"))
                    for item in hypotheses
                ),
                "cross_person_partner_error_count": cross_person_partner_errors,
            }
        )
    candidate_audit = {
        "candidate_count": 0,
        "candidate_with_classic_post_check_count": 0,
        "native_candidate_with_classic_post_check_count": 0,
    }
    candidate_path = run_dir / "contour_candidates.json"
    if candidate_path.exists():
        candidate_payload = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        detective_record = candidate_payload.get("detective") or {}
        candidates = detective_record.get("candidate_relations") or []
        offered = [
            (item, _candidate_classic_check_ids(item, case))
            for item in candidates
        ]
        candidates_with_checks = [
            {**item, "classic_check_hypothesis_ids": checks}
            for item, checks in offered
        ]
        adjudication_message = _adjudication_messages(
            case, candidates_with_checks
        )[1]["content"]
        classic_review_message = _classic_review_messages(
            case,
            candidates_with_checks,
            detective_record.get("adjudicated_relations") or [],
        )[1]["content"]
        material_stage = next(
            (
                item
                for item in (detective_record.get("audit") or {}).get(
                    "stages", []
                )
                if item.get("stage") == "adversarial_adjudication"
            ),
            {},
        )
        material_content = str(
            (
                ((material_stage.get("raw") or {}).get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        )
        material_payload = parse_json_content(material_content)
        frozen_relations = _adjudicate_candidates(
            material_payload if isinstance(material_payload, dict) else {},
            candidates,
        )
        final_relations = detective_record.get("adjudicated_relations") or []
        final_by_candidate = {
            (item.get("provenance") or {}).get("candidate_id"): item
            for item in final_relations
        }
        immutable_fields = {
            "relation_type",
            "piece_ids",
            "status",
            "plain_language_explanation",
            "shared_coordinate",
            "source_independence",
            "common_error_risk",
            "competing_explanation",
            "falsification_test",
            "confidence_profile",
        }
        forbidden_mutations = []
        for frozen in frozen_relations:
            candidate_id = (frozen.get("provenance") or {}).get("candidate_id")
            final = final_by_candidate.get(candidate_id) or {}
            changed = sorted(
                field
                for field in immutable_fields
                if frozen.get(field) != final.get(field)
            )
            if changed:
                forbidden_mutations.append(
                    {"candidate_id": candidate_id, "changed_fields": changed}
                )
        candidate_audit = {
            "candidate_count": len(candidates),
            "candidate_with_classic_post_check_count": sum(
                bool(checks) for _, checks in offered
            ),
            "native_candidate_with_classic_post_check_count": sum(
                bool(checks)
                and "native" in (item.get("discovery_routes") or [])
                for item, checks in offered
            ),
            "offered_checks": [
                {
                    "candidate_id": item.get("candidate_id"),
                    "relation_type": item.get("relation_type"),
                    "discovery_routes": item.get("discovery_routes") or [],
                    "hypothesis_ids": checks,
                }
                for item, checks in offered
                if checks
            ],
            "material_adjudication_prompt_characters": len(
                adjudication_message
            ),
            "classic_review_prompt_characters": len(classic_review_message),
            "classic_review_isolation_passed": not forbidden_mutations,
            "forbidden_relation_mutations": forbidden_mutations,
        }
        if forbidden_mutations:
            violations.append(
                "classic review modified frozen material adjudication"
            )
    return {
        "run_dir": str(run_dir),
        "case_piece_count": len(case["piece_index"]),
        "classic_hypothesis_count": len(
            case.get("classic_connection_hypotheses") or []
        ),
        "classic_post_check_hypothesis_count": len(
            case.get("classic_post_check_hypotheses") or []
        ),
        "selected_classic_operator_counts": dict(
            Counter(
                item.get("operator_id")
                for item in case.get("classic_connection_hypotheses") or []
            )
        ),
        "selected_shadow_crosscheck_sources": [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "source_person_id": item.get("source_person_id"),
                "source_piece_id": item.get("source_piece_id"),
            }
            for item in case.get("classic_connection_hypotheses") or []
            if item.get("operator_id") == "shadow_mechanism_crosscheck"
        ],
        "native_detective_remains_primary": not violations,
        "classic_truth_weight_bonus": False,
        "post_decision_classic_check": candidate_audit,
        "modes": modes,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit that classic connection help is additive to detective work."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
