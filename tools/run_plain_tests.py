from __future__ import annotations

import importlib
import inspect
import sys
import traceback
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

MODULES = [
    f"tests.{path.stem}"
    for path in sorted((ROOT / "tests").glob("test_*.py"))
]


def main() -> int:
    failures: list[tuple[str, str, str]] = []
    total = 0
    skipped = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                function()
            except Exception:  # noqa: BLE001 - this is a dependency-free test harness.
                failures.append((module_name, name, traceback.format_exc()))

        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TestResult()
        suite.run(result)
        total += result.testsRun
        skipped += len(result.skipped)
        failures.extend(
            (module_name, test.id(), details)
            for test, details in [*result.failures, *result.errors]
        )

    suffix = f", {skipped} skipped" if skipped else ""
    print(f"{total - len(failures)}/{total} passed{suffix}")
    for module_name, name, details in failures:
        print(f"FAIL {module_name}.{name}\n{details}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
