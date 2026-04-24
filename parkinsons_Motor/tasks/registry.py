"""Task registry helpers."""

from __future__ import annotations

from typing import Dict

from .base import DBSTask
from .scenarios import (
    TASK_BETA_SUPPRESSION,
    TASK_FRAGILE_PATIENT,
    TASK_FULL_EPISODE,
    TASK_PERSONALIZATION_GENERALIZATION,
    TASK_REFRACTORY_PATIENT,
    TASK_TREMOR_CORRECTION,
)


TASK_REGISTRY: Dict[str, DBSTask] = {
    task.task_id: task
    for task in (
        TASK_BETA_SUPPRESSION,
        TASK_TREMOR_CORRECTION,
        TASK_FULL_EPISODE,
        TASK_FRAGILE_PATIENT,
        TASK_REFRACTORY_PATIENT,
        TASK_PERSONALIZATION_GENERALIZATION,
    )
}

DEFAULT_TASK_ID = TASK_FULL_EPISODE.task_id


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
