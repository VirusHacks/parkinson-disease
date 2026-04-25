"""Backward-compatible re-export for the task suite."""

from .base import DBSTask
from .easy import TASK_EASY, get_easy_task
from .hard import TASK_HARD, get_hard_task
from .medium import TASK_MEDIUM, get_medium_task
from .registry import TASK_REGISTRY, get_task
from .scenarios import (
    TASK_BETA_SUPPRESSION,
    TASK_FULL_EPISODE,
    TASK_TREMOR_CORRECTION,
)

__all__ = [
    "DBSTask",
    "TASK_EASY",
    "TASK_MEDIUM",
    "TASK_HARD",
    "TASK_BETA_SUPPRESSION",
    "TASK_TREMOR_CORRECTION",
    "TASK_FULL_EPISODE",
    "get_easy_task",
    "get_medium_task",
    "get_hard_task",
    "TASK_REGISTRY",
    "get_task",
]
