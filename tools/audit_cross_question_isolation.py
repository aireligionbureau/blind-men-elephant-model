from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bme_model.runner import run_filtered_retrieval  # noqa: E402
from bme_model.search import MockSearchProvider, choose_regional_provider  # noqa: E402


CASES = [
    ("ALPHA_ONLY", "ALPHA_ONLY：城市是否应该实行四天工作制？"),
    ("BETA_ONLY", "BETA_ONLY：深海采矿是否应该暂停？"),
    ("GAMMA_ONLY", "GAMMA_ONLY：中学是否应该推迟到九点上课？"),
]

RUNTIME_FILES = [
    ROOT / "src/bme_model/model.py",
    ROOT / "src/bme_model/cohort.py",
    ROOT / "src/bme_model/context.py",
    ROOT / "src/bme_model/search.py",
    ROOT / "src/bme_model/prompts.py",
    ROOT / "src/bme_model/report.py",
    ROOT / "src/bme_model/meta_model/diagnostics.py",
]

KNOWN_STALE_LITERALS = [
    "美股AI泡沫",
    "美股 AI 泡沫",
    "互联网泡沫",
    "泡沫速度",
    "破裂速度",
    "AI会不会产生意识",
    "主观体验",
    "fast_burst",
    "no_fast_burst",
]


def main() -> int:
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for marker, question in CASES:
        result = run_filtered_retrieval(
            question,
            person_count=4,
            provider_name="hybrid-mock",
            results_per_query=2,
        )
        artifacts[marker] = result
        serialized = json.dumps(result, ensure_ascii=False)
        foreign_markers = [other for other, _ in CASES if other != marker and other in serialized]
        checks.append(
            _check(
                f"{marker}: no foreign question marker",
                not foreign_markers,
                {"foreign_markers": foreign_markers},
            )
        )
        checks.append(
            _check(
                f"{marker}: one context id from cohort through ledgers",
                _single_context(result),
                {"cohort_id": result.get("cohort_id")},
            )
        )
        checks.append(
            _check(
                f"{marker}: mock is explicit test evidence",
                result.get("evidence_mode") == "test_fixture"
                and result.get("cohort_generation", {}).get("context_isolation") == "only_current_question",
                {
                    "evidence_mode": result.get("evidence_mode"),
                    "cohort_mode": result.get("cohort_generation", {}).get("mode"),
                },
            )
        )

    cross_cohort_ids = {payload.get("cohort_id") for payload in artifacts.values()}
    checks.append(
        _check(
            "different questions receive different cohorts",
            len(cross_cohort_ids) == len(CASES),
            {"cohort_ids": sorted(str(value) for value in cross_cohort_ids)},
        )
    )

    stale_hits: list[dict[str, str]] = []
    for path in RUNTIME_FILES:
        text = path.read_text(encoding="utf-8")
        for literal in KNOWN_STALE_LITERALS:
            if literal in text:
                stale_hits.append({"file": str(path.relative_to(ROOT)), "literal": literal})
    checks.append(
        _check(
            "runtime has no known old-question literals",
            not stale_hits,
            {"hits": stale_hits},
        )
    )

    checks.append(
        _check(
            "global live search without credentials uses real sources",
            _global_live_uses_real_sources_without_credentials(),
            {},
        )
    )

    passed = all(item["passed"] for item in checks)
    print(json.dumps({"passed": passed, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _single_context(result: dict[str, Any]) -> bool:
    context_id = result.get("cohort_id")
    if not context_id or result.get("question_frame", {}).get("context_id") != context_id:
        return False
    for record in result.get("retrieval_records", []):
        if record.get("question_context_id") != context_id:
            return False
        if record.get("question_frame", {}).get("context_id") != context_id:
            return False
    return all(ledger.get("question") == result.get("question") for ledger in result.get("evidence_ledgers", []))


def _global_live_uses_real_sources_without_credentials() -> bool:
    keys = ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "BME_SEARCH_ALLOW_MOCK_FALLBACK"]
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        provider = choose_regional_provider("global")
        return (
            not isinstance(provider, MockSearchProvider)
            and provider.name == "federated-html"
            and bool(provider.providers)
            and all(not isinstance(source, MockSearchProvider) for source in provider.providers)
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


if __name__ == "__main__":
    raise SystemExit(main())
