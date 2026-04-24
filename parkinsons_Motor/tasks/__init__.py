"""Task package for the Parkinson's Motor environment."""

from .base import DBSTask
from .registry import TASK_REGISTRY, get_task
from .scenarios import (
    TASK_BETA_SUPPRESSION,
    TASK_FRAGILE_PATIENT,
    TASK_FULL_EPISODE,
    TASK_PERSONALIZATION_GENERALIZATION,
    TASK_REFRACTORY_PATIENT,
    TASK_TREMOR_CORRECTION,
)

__all__ = [
    "DBSTask",
    "TASK_BETA_SUPPRESSION",
    "TASK_TREMOR_CORRECTION",
    "TASK_FULL_EPISODE",
    "TASK_FRAGILE_PATIENT",
    "TASK_REFRACTORY_PATIENT",
    "TASK_PERSONALIZATION_GENERALIZATION",
    "TASK_REGISTRY",
    "get_task",
]
