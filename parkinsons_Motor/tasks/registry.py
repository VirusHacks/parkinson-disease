"""Task registry helpers."""

from __future__ import annotations

from typing import Dict

from .base import DBSTask
from .easy import TASK_EASY
from .exercise_bout import TASK_EXERCISE_BOUT
from .hard import TASK_HARD
from .medication_interaction import TASK_MEDICATION_INTERACTION
from .medium import TASK_MEDIUM
from .nocturnal_transition import TASK_NOCTURNAL_TRANSITION
from .scenarios import (
    TASK_FRAGILE_PATIENT,
    TASK_PERSONALIZATION_GENERALIZATION,
    TASK_REFRACTORY_PATIENT,
)
from .surgical_followup import TASK_SURGICAL_FOLLOWUP


TASK_REGISTRY: Dict[str, DBSTask] = {
    # Public tasks + legacy aliases.
    "easy": TASK_EASY,
    "beta_suppression": TASK_EASY,
    "calm_start": TASK_EASY,
    "medium": TASK_MEDIUM,
    "tremor_correction": TASK_MEDIUM,
    "rescue_phase": TASK_MEDIUM,
    "hard": TASK_HARD,
    "full_episode": TASK_HARD,
    # Expert tasks.
    "fragile_patient": TASK_FRAGILE_PATIENT,
    "refractory_patient": TASK_REFRACTORY_PATIENT,
    "personalization_generalization": TASK_PERSONALIZATION_GENERALIZATION,
    "exercise_bout": TASK_EXERCISE_BOUT,
    "medication_interaction": TASK_MEDICATION_INTERACTION,
    "nocturnal_transition": TASK_NOCTURNAL_TRANSITION,
    "surgical_followup": TASK_SURGICAL_FOLLOWUP,
}

DEFAULT_TASK_ID = TASK_HARD.task_id


def get_task(task_id: str | None = None) -> DBSTask:
    """Return the DBSTask for the given id, defaulting to the hard task."""
    if task_id is None:
        return TASK_REGISTRY[DEFAULT_TASK_ID]
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Available: {list(TASK_REGISTRY)}"
        )
    return task
