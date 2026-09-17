from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.config import DEFAULT_MODEL  # noqa: E402
from bme_model.runner import run_filtered_retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real three-layer retrieval smoke test.")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    result = run_filtered_retrieval(
        args.question,
        person_count=1,
        provider_name="three-layer",
        results_per_query=3,
        model=args.model,
        tool_retries=1,
    )
    ledger = result["evidence_ledgers"][0]
    external = [
        item
        for item in ledger["source_decisions"]
        if item["result"].get("retrieval_layer") == "external_search"
    ]
    counts = {
        decision: sum(item["decision"] == decision for item in external)
        for decision in ["accept", "reject", "ignore"]
    }
    summary = {
        "question": result["question"],
        "evidence_mode": result["evidence_mode"],
        "search_tool_usage": result["search_tool_usage"],
        "external_decisions": counts,
        "selection_summary": result["search_tool_traces"][0].get("selection_summary"),
        "decision_samples": [
            {
                "title": item["result"]["title"],
                "decision": item["decision"],
                "reasons": item["reasons"],
            }
            for item in external[:3]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if external and counts["accept"] + counts["reject"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
