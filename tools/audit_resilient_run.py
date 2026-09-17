from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.pipeline import audit_resilient_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit checkpoints, quality gates, and artifact integrity for one run."
    )
    parser.add_argument("run_dir")
    args = parser.parse_args()
    result = audit_resilient_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
