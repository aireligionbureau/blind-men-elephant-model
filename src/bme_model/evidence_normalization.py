from __future__ import annotations

import re
from typing import Any


_SOURCE_ID_SUFFIX = re.compile(
    r"\s*[（(]\s*ID\s*[:：]\s*([A-Za-z0-9._:-]+)\s*[）)]\s*[。.!！]?\s*$",
    re.IGNORECASE,
)


def normalize_output_evidence_entries(value: Any) -> list[dict[str, Any]]:
    """Normalize model-authored evidence ledgers without discarding lineage."""
    if isinstance(value, dict):
        value = value.get("entries", [])
    if not isinstance(value, list):
        return []

    entries: list[dict[str, Any]] = []
    for raw in value:
        if isinstance(raw, dict):
            entries.append(dict(raw))
            continue
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        match = _SOURCE_ID_SUFFIX.search(text)
        claim = _SOURCE_ID_SUFFIX.sub("", text).strip() if match else text
        entry: dict[str, Any] = {"claim": claim}
        if match:
            entry["source_id"] = match.group(1)
        entries.append(entry)
    return entries
