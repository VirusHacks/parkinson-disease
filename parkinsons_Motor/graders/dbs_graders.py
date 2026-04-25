"""Deterministic benchmark grader dispatcher for the Parkinson's Motor environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .easy_grader import grade_easy
from .expert_grader import grade_expert
from .hard_grader import grade_hard
from .medium_grader import grade_medium
from .scenario_graders import grade_scenario


GRADER_REGISTRY = {
    # Public.
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    # Expert (existing).
    "fragile_patient": grade_expert,
    "refractory_patient": grade_expert,
    "personalization_generalization": grade_expert,
    # Scenario (new).
    "exercise_bout": grade_scenario,
    "medication_interaction": grade_scenario,
    "nocturnal_transition": grade_scenario,
    "surgical_followup": grade_scenario,
}


def compute_score_details(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return all score components plus the final deterministic score."""
    grader = GRADER_REGISTRY.get(task.task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task_id={task.task_id!r}")
    return grader(task, trajectory)


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
