"""Deterministic grader for the public hard task."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .base_grader import compute_component_details, finalize_details, weighted_overall


HARD_WEIGHTS = {
    "force_score": 0.14,
    "beta_score": 0.22,        # Suppressing pathological beta is the primary DBS goal
    "tremor_score": 0.14,      # Tremor control is co-primary
    "tracking_score": 0.16,
    "safety_score": 0.18,      # Safety matters but can't be gamed by low-amp coasting
    "smoothness_score": 0.04,
    "efficiency_score": 0.04,  # Gated by therapeutic engagement; secondary in hard sessions
    "terminal_stability_score": 0.08,  # 150-step episode must finish stable
}


def grade_hard(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grade the long-horizon full-session task."""
    details = compute_component_details(task, trajectory)
    raw_overall = weighted_overall(details, HARD_WEIGHTS)
    return finalize_details(task, trajectory, details, raw_overall)
