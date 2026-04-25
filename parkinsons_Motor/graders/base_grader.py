"""Shared grader logic for the Parkinson's Motor benchmark."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .components import (
    beta_score,
    clamp,
    efficiency_score,
    force_score,
    recovery_score,
    safety_score,
    smoothness_score,
    terminal_stability_score,
    tracking_score,
    tremor_score,
)
from .rules import hard_failure_penalty


def compute_component_details(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the shared component scores used by all task-specific graders."""
    details = {
        "force_score": force_score(trajectory, task),
        "beta_score": beta_score(trajectory, task),
        "tremor_score": tremor_score(trajectory, task),
        "tracking_score": tracking_score(trajectory, task),
        "safety_score": safety_score(trajectory, task),
        "smoothness_score": smoothness_score(trajectory),
        "terminal_stability_score": terminal_stability_score(trajectory, task),
        "recovery_score": recovery_score(trajectory, task),
    }
    therapeutic_engagement = clamp(
        0.40 * details["force_score"]
        + 0.30 * details["beta_score"]
        + 0.30 * details["tremor_score"]
    )
    details["therapeutic_engagement"] = therapeutic_engagement
    details["efficiency_score"] = efficiency_score(trajectory, task) * therapeutic_engagement
    return details


def weighted_overall(details: Dict[str, float], weights: Dict[str, float]) -> float:
    """Combine component scores using a task-specific weight map."""
    return clamp(sum(weights.get(key, 0.0) * details[key] for key in weights))


def finalize_details(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    details: Dict[str, float],
    raw_overall: float,
) -> Dict[str, float]:
    """Attach common gates, penalties, and the final overall score."""
    penalty = hard_failure_penalty(task, trajectory, details)
    details["pre_penalty_score"] = clamp(raw_overall)
    details["hard_failure_penalty"] = penalty
    details["passes_safety_gate"] = 1.0 if details["safety_score"] >= 0.45 else 0.0
    details["passes_symptom_gate"] = 1.0 if min(details["beta_score"], details["tremor_score"]) >= 0.35 else 0.0
    details["passes_motor_gate"] = 1.0 if min(details["force_score"], details["tracking_score"]) >= 0.30 else 0.0
    details["overall_score"] = clamp(raw_overall - penalty)
    return details
