from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bme_model.pipeline import run_resilient_batch  # noqa: E402
from bme_model.runtime import atomic_write_json  # noqa: E402
from bme_model.search import choose_regional_provider  # noqa: E402
from tools.audit_pipeline_upgrade import (  # noqa: E402
    audit_run,
    compare_cross_domain_suite,
    compare_runs,
    create_blind_review_packet,
    read_json,
)


PIPELINE_MODE = "streaming_v2"
SYNTHESIS_MODE = "optimized_v2"
RUNS_DIR = ROOT / "runs-candidate"


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = ROOT / path
    return project_path if project_path.exists() else manifest_path.parent / path


def load_case(manifest_path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(manifest_path, {}) or {}
    matches = [
        item
        for item in manifest.get("cases") or []
        if item.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"case_id {case_id!r} must identify exactly one case")
    return manifest, matches[0]


def candidate_preflight(manifest_path: Path, case_id: str) -> dict[str, Any]:
    manifest, case = load_case(manifest_path, case_id)
    del manifest
    baseline_value = case.get("baseline_run")
    if not baseline_value:
        raise ValueError(f"case {case_id!r} has no baseline_run")
    baseline_run = _manifest_path(baseline_value, manifest_path)
    baseline = audit_run(baseline_run)
    expected_question = str(case.get("question") or "").strip()
    if baseline.get("question") != expected_question:
        raise ValueError("baseline question does not match acceptance manifest")
    provider = choose_regional_provider("global")
    return {
        "schema": "bme.activation-candidate-preflight.v1",
        "external_calls": 0,
        "case_id": case_id,
        "question": expected_question,
        "baseline_run": str(baseline_run.resolve()),
        "existing_candidate_run": case.get("candidate_run"),
        "deepseek_key_available": bool(os.getenv("DEEPSEEK_API_KEY")),
        "search_provider": provider.__class__.__name__,
        "mock_search_fallback": provider.__class__.__name__ == "MockSearchProvider",
        "pipeline_mode": PIPELINE_MODE,
        "synthesis_mode": SYNTHESIS_MODE,
        "person_count": 24,
        "runs_namespace": RUNS_DIR.name,
        "will_send_external_data": True,
        "ready": (
            bool(os.getenv("DEEPSEEK_API_KEY"))
            and provider.__class__.__name__ != "MockSearchProvider"
            and not case.get("candidate_run")
        ),
    }


def run_candidate(
    manifest_path: Path,
    case_id: str,
    *,
    replace_candidate: bool = False,
) -> dict[str, Any]:
    manifest, case = load_case(manifest_path, case_id)
    preflight = candidate_preflight(manifest_path, case_id)
    if not preflight["deepseek_key_available"]:
        raise RuntimeError("DEEPSEEK_API_KEY is not available in this process")
    if preflight["mock_search_fallback"]:
        raise RuntimeError("candidate benchmark cannot use mock search")
    if case.get("candidate_run") and not replace_candidate:
        raise RuntimeError(
            "candidate_run already exists; use --replace-candidate only after reviewing it"
        )

    started = time.perf_counter()
    result = run_resilient_batch(
        str(case["question"]),
        profile_name="standard",
        provider_name="three-layer",
        results_per_query=5,
        concurrency=24,
        retrieval_concurrency=12,
        semantic_concurrency=24,
        model_concurrency=24,
        pipeline_mode=PIPELINE_MODE,
        synthesis_mode=SYNTHESIS_MODE,
        retries=1,
        recovery_passes=0,
        time_budget_seconds=570,
        output_dir=RUNS_DIR,
    )
    wall_seconds = round(time.perf_counter() - started, 3)
    candidate_run = Path(result["run_dir"]).resolve()
    baseline_run = _manifest_path(case["baseline_run"], manifest_path)
    comparison = compare_runs(audit_run(baseline_run), audit_run(candidate_run))

    review_dir = (
        ROOT
        / "benchmarks"
        / "reviews"
        / f"{case_id}-{candidate_run.name}"
    )
    review_artifacts = create_blind_review_packet(
        baseline_run,
        candidate_run,
        review_dir,
    )
    case["candidate_run"] = _project_path(candidate_run)
    case["semantic_review_file"] = _project_path(review_dir / "review.json")
    case["semantic_review_key_file"] = _project_path(
        review_dir / "review-key.json"
    )
    case["semantic_review"] = None
    atomic_write_json(manifest_path, manifest)
    suite = compare_cross_domain_suite(manifest_path)

    benchmark = {
        "schema": "bme.activation-candidate-run.v1",
        "case_id": case_id,
        "question": case["question"],
        "wall_seconds_observed_by_runner": wall_seconds,
        "candidate_run": str(candidate_run),
        "comparison": comparison,
        "review_artifacts": review_artifacts,
        "suite_activation_ready": suite["activation_ready"],
        "default_switch_allowed": suite["default_switch_allowed"],
    }
    atomic_write_json(candidate_run / "activation_benchmark.json", benchmark)
    return benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or execute one isolated optimized activation candidate."
    )
    parser.add_argument("case_id")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "benchmarks" / "cross-domain-acceptance.json",
    )
    parser.add_argument(
        "--authorize-external-calls",
        action="store_true",
        help="Actually send the case to DeepSeek and live search providers.",
    )
    parser.add_argument("--replace-candidate", action="store_true")
    args = parser.parse_args()

    if not args.authorize_external_calls:
        payload = candidate_preflight(args.manifest, args.case_id)
    else:
        payload = run_candidate(
            args.manifest,
            args.case_id,
            replace_candidate=args.replace_candidate,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
