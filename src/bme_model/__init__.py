"""Blind Men Elephant Model.

This package contains a minimal protocol-first prototype for the
盲人摸象模型: a cognitive-shadow inversion framework for complex problems.
"""

from .model import (
    DigitalPerson,
    EvidenceLedger,
    InformationFilter,
    Shadow,
    TruthContour,
    build_run_plan,
)
from .batch import (
    enrich_saved_run_with_semantic_verdicts,
    resume_live_batch,
    run_live_batch,
)
from .puzzle import build_shadow_puzzle
from .runner import build_profile_plan
from .runner import run_filtered_retrieval
from .runner import run_shadow_diagnosis

__all__ = [
    "DigitalPerson",
    "EvidenceLedger",
    "InformationFilter",
    "Shadow",
    "TruthContour",
    "build_profile_plan",
    "build_run_plan",
    "build_shadow_puzzle",
    "enrich_saved_run_with_semantic_verdicts",
    "run_live_batch",
    "resume_live_batch",
    "run_filtered_retrieval",
    "run_shadow_diagnosis",
]
