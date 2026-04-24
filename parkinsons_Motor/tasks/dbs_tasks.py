"""Backward-compatible re-export for the task suite."""

from .base import DBSTask
from .registry import TASK_REGISTRY, get_task
from .scenarios import (
    TASK_BETA_SUPPRESSION,
    TASK_FULL_EPISODE,
    TASK_TREMOR_CORRECTION,
)

__all__ = [
    "DBSTask",
    "TASK_BETA_SUPPRESSION",
    "TASK_TREMOR_CORRECTION",
    "TASK_FULL_EPISODE",
    "TASK_REGISTRY",
    "get_task",
]
