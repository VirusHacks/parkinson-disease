"""Deterministic benchmark grader for the Parkinson's Motor environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def compute_score_details(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return all score components plus the final deterministic score."""
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
    details["efficiency_score"] = efficiency_score(trajectory, task) * therapeutic_engagement

    if task.task_id == "beta_suppression":
        # Efficiency has real bite here: constant max-amp policy scores low on
        # efficiency_score (gated by therapeutic_engagement), preventing trivial wins.
        overall = (
            0.30 * details["beta_score"]
            + 0.18 * details["tremor_score"]
            + 0.16 * details["tracking_score"]
            + 0.14 * details["force_score"]
            + 0.14 * details["safety_score"]
            + 0.04 * details["smoothness_score"]
            + 0.04 * details["efficiency_score"]
        )
    elif task.task_id == "tremor_correction":
        # recovery_score raised: the core question is "did you actually rescue?"
        # terminal_stability kept high: can't just spike-and-crash.
        overall = (
            0.16 * details["force_score"]
            + 0.06 * details["beta_score"]
            + 0.14 * details["tremor_score"]
            + 0.14 * details["tracking_score"]
            + 0.22 * details["safety_score"]
            + 0.04 * details["smoothness_score"]
            + 0.12 * details["terminal_stability_score"]
            + 0.06 * details["efficiency_score"]
            + 0.06 * details["recovery_score"]
        )
    elif task.task_id == "fragile_patient":
        overall = (
            0.18 * details["force_score"]
            + 0.10 * details["beta_score"]
            + 0.10 * details["tremor_score"]
            + 0.18 * details["tracking_score"]
            + 0.28 * details["safety_score"]
            + 0.06 * details["smoothness_score"]
            + 0.05 * details["efficiency_score"]
            + 0.05 * details["terminal_stability_score"]
        )
    elif task.task_id == "refractory_patient":
        overall = (
            0.18 * details["force_score"]
            + 0.08 * details["beta_score"]
            + 0.12 * details["tremor_score"]
            + 0.14 * details["tracking_score"]
            + 0.18 * details["safety_score"]
            + 0.05 * details["smoothness_score"]
            + 0.10 * details["efficiency_score"]
            + 0.07 * details["terminal_stability_score"]
            + 0.08 * details["recovery_score"]
        )
    elif task.task_id == "personalization_generalization":
        overall = (
            0.18 * details["force_score"]
            + 0.08 * details["beta_score"]
            + 0.12 * details["tremor_score"]
            + 0.18 * details["tracking_score"]
            + 0.18 * details["safety_score"]
            + 0.04 * details["smoothness_score"]
            + 0.08 * details["efficiency_score"]
            + 0.05 * details["terminal_stability_score"]
            + 0.09 * details["recovery_score"]
        )
    else:
        overall = (
            0.14 * details["force_score"]
            + 0.08 * details["beta_score"]
            + 0.06 * details["tremor_score"]
            + 0.16 * details["tracking_score"]
            + 0.36 * details["safety_score"]
            + 0.05 * details["smoothness_score"]
            + 0.10 * details["efficiency_score"]
            + 0.05 * details["terminal_stability_score"]
        )

    details["overall_score"] = clamp(overall - hard_failure_penalty(task, trajectory, details))
    return details


def compute_score(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """Deterministic benchmark-facing score in [0, 1]."""
    return compute_score_details(task, trajectory, info)["overall_score"]


def is_success(task: DBSTask, score: float) -> bool:
    """Return True if the score meets the task threshold."""
    return score >= task.success_threshold
