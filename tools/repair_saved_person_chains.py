from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.batch import repair_saved_run_person_outputs  # noqa: E402
from bme_model.config import DEFAULT_MODEL  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair incomplete saved digital-person chains.")
    parser.add_argument("run_dir")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    result = repair_saved_run_person_outputs(
        args.run_dir,
        model=args.model,
        concurrency=args.concurrency,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )
    summary = result.get("person_chain_repair", {"repaired_count": 0})
    summary.update(
        {
            "run_dir": result["run_dir"],
            "question": result["question"],
            "report": str(Path(result["run_dir"]) / "report.html"),
            "cognitive_chain_evaluation": result["meta_model"].get("cognitive_chain_evaluation", {}),
            "semantic_verdict_evaluation": result["meta_model"].get("semantic_verdict_evaluation", {}),
            "meta_model_overall_evaluation": result["meta_model"].get("meta_model_overall_evaluation", {}),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
