"""Deterministic grader for the public easy task."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .base_grader import compute_component_details, finalize_details, weighted_overall


EASY_WEIGHTS = {
    "beta_score": 0.30,
    "tremor_score": 0.18,
    "tracking_score": 0.16,
    "force_score": 0.14,
    "safety_score": 0.14,
    "smoothness_score": 0.04,
    "efficiency_score": 0.04,
}


def grade_easy(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grade the easy clinical titration task."""
    details = compute_component_details(task, trajectory)
    raw_overall = weighted_overall(details, EASY_WEIGHTS)
    return finalize_details(task, trajectory, details, raw_overall)
