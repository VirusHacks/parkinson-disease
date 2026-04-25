"""Task package for the Parkinson's Motor environment."""

from .base import DBSTask
from .easy import TASK_EASY, get_easy_task
from .exercise_bout import TASK_EXERCISE_BOUT, get_exercise_bout_task
from .hard import TASK_HARD, get_hard_task
from .medication_interaction import (
    TASK_MEDICATION_INTERACTION,
    get_medication_interaction_task,
)
from .medium import TASK_MEDIUM, get_medium_task
from .nocturnal_transition import (
    TASK_NOCTURNAL_TRANSITION,
    get_nocturnal_transition_task,
)
from .registry import TASK_REGISTRY, get_task
from .scenarios import (
    TASK_BETA_SUPPRESSION,
    TASK_FRAGILE_PATIENT,
    TASK_FULL_EPISODE,
    TASK_PERSONALIZATION_GENERALIZATION,
    TASK_REFRACTORY_PATIENT,
    TASK_TREMOR_CORRECTION,
)
from .surgical_followup import TASK_SURGICAL_FOLLOWUP, get_surgical_followup_task

__all__ = [
    "DBSTask",
    "TASK_EASY",
    "TASK_MEDIUM",
    "TASK_HARD",
    "TASK_BETA_SUPPRESSION",
    "TASK_TREMOR_CORRECTION",
    "TASK_FULL_EPISODE",
    "TASK_FRAGILE_PATIENT",
    "TASK_REFRACTORY_PATIENT",
    "TASK_PERSONALIZATION_GENERALIZATION",
    "TASK_EXERCISE_BOUT",
    "TASK_MEDICATION_INTERACTION",
    "TASK_NOCTURNAL_TRANSITION",
    "TASK_SURGICAL_FOLLOWUP",
    "get_easy_task",
    "get_medium_task",
    "get_hard_task",
    "get_exercise_bout_task",
    "get_medication_interaction_task",
    "get_nocturnal_transition_task",
    "get_surgical_followup_task",
    "TASK_REGISTRY",
    "get_task",
]
