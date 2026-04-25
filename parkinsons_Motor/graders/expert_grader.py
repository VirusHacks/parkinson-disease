"""Deterministic graders for the extended expert tasks."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .base_grader import compute_component_details, finalize_details, weighted_overall


EXPERT_WEIGHT_MAP: Dict[str, Dict[str, float]] = {
    "fragile_patient": {
        "force_score": 0.18,
        "beta_score": 0.10,
        "tremor_score": 0.10,
        "tracking_score": 0.18,
        "safety_score": 0.28,
        "smoothness_score": 0.06,
        "efficiency_score": 0.05,
        "terminal_stability_score": 0.05,
    },
    "refractory_patient": {
        "force_score": 0.18,
        "beta_score": 0.08,
        "tremor_score": 0.12,
        "tracking_score": 0.14,
        "safety_score": 0.18,
        "smoothness_score": 0.05,
        "efficiency_score": 0.10,
        "terminal_stability_score": 0.07,
        "recovery_score": 0.08,
    },
    "personalization_generalization": {
        "force_score": 0.18,
        "beta_score": 0.08,
        "tremor_score": 0.12,
        "tracking_score": 0.18,
        "safety_score": 0.18,
        "smoothness_score": 0.04,
        "efficiency_score": 0.08,
        "terminal_stability_score": 0.05,
        "recovery_score": 0.09,
    },
}


def grade_expert(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grade the expert/extended tasks using the configured task-specific weights."""
    details = compute_component_details(task, trajectory)
    weights = EXPERT_WEIGHT_MAP.get(task.task_id)
    if weights is None:
        raise ValueError(f"No expert grader weights configured for task_id={task.task_id!r}")
    raw_overall = weighted_overall(details, weights)
    return finalize_details(task, trajectory, details, raw_overall)
