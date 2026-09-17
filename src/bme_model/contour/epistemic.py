from __future__ import annotations

import re
from typing import Any

from ..detective.schemas import compact_text


STRUCTURAL_UNCERTAINTY_MARKERS = (
    "不能确定",
    "尚不能",
    "无法确定",
    "无法可靠确定",
    "无法可靠判断",
    "无法给出可靠判断",
    "无法给出确定答案",
    "不能给出确定答案",
    "无法给出确定结论",
    "不能给出确定结论",
    "尚无定论",
    "不足以判断",
    "取决于",
    "要先定义",
    "必须先定义",
    "首先取决于",
    "不能用单一",
    "没有单一",
    "没有统一",
    "定义不同",
    "条件不同",
    "不能得出方向",
    "不能推出方向",
    "如果",
    "若",
    "条件下",
    "情况下",
    "定义下",
    "才会",
)


def has_structural_uncertainty(value: Any) -> bool:
    text = compact_text(value)
    if any(marker in text for marker in STRUCTURAL_UNCERTAINTY_MARKERS):
        return True
    return bool(
        re.search(
            r"(?:无法|不能|尚不能|不足以).{0,24}"
            r"(?:确定|判断|支持|得出|推出|回答|结论)",
            text,
        )
    )
