from __future__ import annotations

import json
import re
from typing import Any


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_content(content: str) -> Any:
    """Parse model JSON, including common fenced/trailing-text variants."""
    text = content.strip().lstrip("\ufeff")
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    return {"raw_text": content}


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(text))

    first_object = text.find("{")
    last_object = text.rfind("}")
    if 0 <= first_object < last_object:
        candidates.append(text[first_object : last_object + 1])

    first_array = text.find("[")
    last_array = text.rfind("]")
    if 0 <= first_array < last_array:
        candidates.append(text[first_array : last_array + 1])

    return list(dict.fromkeys(candidate for candidate in candidates if candidate))
