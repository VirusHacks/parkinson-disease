"""Dedicated graders for the new expert/scenario tasks.

Each grader inherits the shared component pipeline (`compute_component_details`
+ `finalize_details`) and applies a task-specific weight map. Penalty rules
specific to these scenarios live in `rules.py` so the deterministic-grader
contract remains: components → weighted sum → hard-failure penalty → clamp.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask

from .base_grader import compute_component_details, finalize_details, weighted_overall


# ---------------------------------------------------------------------------
# Weight maps
# ---------------------------------------------------------------------------

EXERCISE_BOUT_WEIGHTS: Dict[str, float] = {
    # Tracking + force are the dominant outcomes during an exercise bout.
    "force_score": 0.22,
    "tracking_score": 0.22,
    "beta_score": 0.08,
    "tremor_score": 0.10,
    "safety_score": 0.20,
    "smoothness_score": 0.04,
    "efficiency_score": 0.06,
    "terminal_stability_score": 0.08,
}

MEDICATION_INTERACTION_WEIGHTS: Dict[str, float] = {
    # Recovery + safety dominate: must compensate the off-med crisis without
    # over-treating the dyskinesia rebound.
    "force_score": 0.16,
    "tracking_score": 0.14,
    "beta_score": 0.10,
    "tremor_score": 0.12,
    "safety_score": 0.22,
    "smoothness_score": 0.04,
    "efficiency_score": 0.06,
    "terminal_stability_score": 0.06,
    "recovery_score": 0.10,
}

NOCTURNAL_TRANSITION_WEIGHTS: Dict[str, float] = {
    # Long horizon, time-varying setpoints: terminal stability and efficiency
    # matter (battery + side-effect budget over a full night).
    "force_score": 0.12,
    "tracking_score": 0.10,
    "beta_score": 0.12,
    "tremor_score": 0.14,
    "safety_score": 0.22,
    "smoothness_score": 0.06,
    "efficiency_score": 0.12,
    "terminal_stability_score": 0.12,
}

SURGICAL_FOLLOWUP_WEIGHTS: Dict[str, float] = {
    # Safety > everything during the microlesion window; recovery score tracks
    # the agent's ability to ramp up effective control as the cap relaxes.
    "force_score": 0.14,
    "tracking_score": 0.14,
    "beta_score": 0.10,
    "tremor_score": 0.10,
    "safety_score": 0.30,
    "smoothness_score": 0.06,
    "efficiency_score": 0.06,
    "terminal_stability_score": 0.04,
    "recovery_score": 0.06,
}


WEIGHT_MAP: Dict[str, Dict[str, float]] = {
    "exercise_bout": EXERCISE_BOUT_WEIGHTS,
    "medication_interaction": MEDICATION_INTERACTION_WEIGHTS,
    "nocturnal_transition": NOCTURNAL_TRANSITION_WEIGHTS,
    "surgical_followup": SURGICAL_FOLLOWUP_WEIGHTS,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def grade_scenario(task: DBSTask, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Grade a scenario task using its registered weight map."""
    weights = WEIGHT_MAP.get(task.task_id)
    if weights is None:
        raise ValueError(
            f"No scenario grader weights configured for task_id={task.task_id!r}"
        )
    details = compute_component_details(task, trajectory)
    raw_overall = weighted_overall(details, weights)
    return finalize_details(task, trajectory, details, raw_overall)
