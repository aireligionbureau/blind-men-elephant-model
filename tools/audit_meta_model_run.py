from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_run(run_dir: Path, expected_people: int) -> dict[str, Any]:
    diagnosis = _read_json(run_dir / "diagnosis.json")
    scans = diagnosis.get("cognitive_chain_scans", [])
    verdicts = [
        verdict
        for scan in scans
        for verdict in scan.get("semantic_verdict", {}).get("verdicts", [])
    ]
    substantive_drops = [
        {"person": scan.get("person_name"), "errors": dropped.get("errors", [])}
        for scan in scans
        for dropped in scan.get("semantic_verdict", {}).get("validation", {}).get("dropped_verdicts", [])
        if any(
            "重复" not in reason and "优先级低于" not in reason
            for reason in dropped.get("errors", [])
        )
    ]
    checks = {
        "expected_person_count": len(scans) == expected_people,
        "ten_detectors_per_person": all(len(scan.get("detector_results", [])) == 10 for scan in scans),
        "complete_chain_per_person": all(
            float(scan.get("chain_coverage", {}).get("score", 0)) == 100 for scan in scans
        ),
        "all_live_no_fallback": all(
            scan.get("semantic_verdict", {}).get("execution_status") == "succeeded" for scan in scans
        ),
        "all_semantic_outputs_valid": all(
            scan.get("semantic_verdict", {}).get("validation", {}).get("valid") for scan in scans
        ),
        "at_most_two_root_problems": all(
            len(scan.get("semantic_verdict", {}).get("verdicts", [])) <= 2 for scan in scans
        ),
        "no_substantive_drop_hidden": not substantive_drops,
        "specificity_floor": all(
            float(item.get("evidence_validation", {}).get("specificity_score", 0)) >= 0.9
            for item in verdicts
        ),
        "canonical_falsification": all(
            "有可核验证据" in item.get("falsification_test", "")
            and "仍支持原结论" in item.get("falsification_test", "")
            and "这条批评应撤销或降级" in item.get("falsification_test", "")
            for item in verdicts
        ),
        "classic_foundation_100": float(diagnosis.get("readiness_evaluation", {}).get("score", 0)) == 100,
        "cognitive_chain_100": float(diagnosis.get("cognitive_chain_evaluation", {}).get("score", 0)) == 100,
        "semantic_verdict_100": float(diagnosis.get("semantic_verdict_evaluation", {}).get("score", 0)) == 100,
        "meta_model_overall_100": float(diagnosis.get("meta_model_overall_evaluation", {}).get("score", 0)) == 100,
    }
    return {
        "passed": all(checks.values()),
        "run_dir": str(run_dir),
        "person_count": len(scans),
        "verdict_count": len(verdicts),
        "checks": checks,
        "substantive_drops": substantive_drops,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a saved meta-model run against the full per-person standard.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-people", type=int, default=24)
    args = parser.parse_args()
    result = audit_run(args.run_dir, args.expected_people)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
