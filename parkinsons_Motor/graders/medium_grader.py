"""Deterministic grader for the public medium task."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .base_grader import compute_component_details, finalize_details, weighted_overall


MEDIUM_WEIGHTS = {
    "force_score": 0.16,
    "beta_score": 0.06,
    "tremor_score": 0.14,
    "tracking_score": 0.14,
    "safety_score": 0.22,
    "smoothness_score": 0.04,
    "terminal_stability_score": 0.12,
    "efficiency_score": 0.06,
    "recovery_score": 0.06,
}


def grade_medium(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grade the medium tremor-rescue task."""
    details = compute_component_details(task, trajectory)
    raw_overall = weighted_overall(details, MEDIUM_WEIGHTS)
    return finalize_details(task, trajectory, details, raw_overall)
