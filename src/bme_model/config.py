from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RunProfile:
    name: str
    person_count: int
    description: str
    target_input_tokens: int
    target_output_tokens: int


RUN_PROFILES = {
    "standard": RunProfile(
        name="standard",
        person_count=24,
        description="标准版：24 个数字人，适合大多数复杂问题的一次真相轮廓生成。",
        target_input_tokens=1_400_000,
        target_output_tokens=200_000,
    ),
    "deep": RunProfile(
        name="deep",
        person_count=50,
        description="深度版：50 个数字人，适合公共政策、战略决策和高风险议题。",
        target_input_tokens=5_000_000,
        target_output_tokens=600_000,
    ),
}


DEFAULT_MODEL = (
    os.getenv("BME_LLM_MODEL")
    or os.getenv("DEEPSEEK_MODEL")
    or "deepseek-v4-pro"
)
DEFAULT_COHORT_MODEL = (
    os.getenv("BME_COHORT_MODEL")
    or os.getenv("BME_LLM_MODEL")
    or os.getenv("DEEPSEEK_COHORT_MODEL")
    or "deepseek-v4-flash"
)
DEFAULT_BASE_URL = (
    os.getenv("BME_LLM_BASE_URL")
    or os.getenv("DEEPSEEK_BASE_URL")
    or "https://api.deepseek.com"
)
