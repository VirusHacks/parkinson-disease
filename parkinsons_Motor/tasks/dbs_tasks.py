"""
DBS task definitions for the Parkinson's Motor environment.

The upgraded tasks are clinically distinct scenarios rather than simple
length-based slices of the same trajectory. Each task specifies:
  - the episode window
  - the patient profile pool
  - allowed stimulation envelope
  - clinical success thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class DBSTask:
    """Immutable task specification."""

    task_id: str
    description: str
    difficulty: str

    # Episode extent on the calibrated timeline
    start_step: int
    n_steps: int

    # Allowed stimulation envelope
    max_dbs_amplitude: float
    max_dbs_pulse_width: float
    max_side_effect_load: float

    # Clinical performance targets
    target_force_preserved: float
    target_beta_arv: float
    target_tremor_arv: float
    target_tracking_error: float
    success_threshold: float

    # Scenario configuration
    patient_profile_ids: Tuple[str, ...] = field(default_factory=tuple)
    target_output_range: Tuple[float, float] = (-0.5, 0.5)


TASK_BETA_SUPPRESSION = DBSTask(
    task_id="beta_suppression",
    description=(
        "Fragile early-phase stabilization. The patient is highly sensitive to "
        "side effects, so the controller must suppress rising beta quickly using "
        "gentle stimulation and clean motor tracking."
    ),
    difficulty="easy",
    start_step=0,
    n_steps=24,
    max_dbs_amplitude=1.0,
    max_dbs_pulse_width=0.14,
    max_side_effect_load=0.30,
    target_force_preserved=0.86,
    target_beta_arv=0.22,
    target_tremor_arv=0.14,
    target_tracking_error=0.22,
    success_threshold=0.54,
    patient_profile_ids=("balanced", "responsive"),
    target_output_range=(-0.35, 0.35),
)

TASK_TREMOR_CORRECTION = DBSTask(
    task_id="tremor_correction",
    description=(
        "Acute tremor rescue. The episode begins during tremor escalation, and "
        "the agent must actively reverse symptom growth, recover functional force, "
        "and keep tracking accuracy and side effects within bounds."
    ),
    difficulty="medium",
    start_step=18,
    n_steps=48,
    max_dbs_amplitude=1.8,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.46,
    target_force_preserved=0.64,
    target_beta_arv=0.28,
    target_tremor_arv=0.34,
    target_tracking_error=0.28,
    success_threshold=0.32,
    patient_profile_ids=("balanced", "responsive"),
    target_output_range=(-0.55, 0.55),
)

TASK_FULL_EPISODE = DBSTask(
    task_id="full_episode",
    description=(
        "Sustained closed-loop control over the full clinical episode. The "
        "controller must manage symptom progression, cumulative side effects, "
        "tracking quality, recovery quality, and stimulation smoothness over a "
        "long horizon."
    ),
    difficulty="hard",
    start_step=0,
    n_steps=100,
    max_dbs_amplitude=2.4,
    max_dbs_pulse_width=0.20,
    max_side_effect_load=0.60,
    target_force_preserved=0.58,
    target_beta_arv=0.30,
    target_tremor_arv=0.38,
    target_tracking_error=0.30,
    success_threshold=0.60,
    patient_profile_ids=("balanced", "responsive", "refractory"),
    target_output_range=(-0.70, 0.70),
)


TASK_REGISTRY: Dict[str, DBSTask] = {
    task.task_id: task
    for task in (
        TASK_BETA_SUPPRESSION,
        TASK_TREMOR_CORRECTION,
        TASK_FULL_EPISODE,
    )
}

_DEFAULT_TASK_ID = TASK_FULL_EPISODE.task_id


def get_task(task_id: str | None = None) -> DBSTask:
    """Return the DBSTask for the given id, defaulting to the hard task."""
    if task_id is None:
        return TASK_REGISTRY[_DEFAULT_TASK_ID]
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Available: {list(TASK_REGISTRY)}"
        )
    return task
