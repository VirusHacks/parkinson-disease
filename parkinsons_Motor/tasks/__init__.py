"""
Tasks for the Parkinson's Motor (DBS) Environment.

Three tasks of increasing clinical difficulty:
  - Task 1 (Easy):   beta_suppression  — suppress a static beta oscillation spike
  - Task 2 (Medium): tremor_correction  — prevent force decay during active tremor
  - Task 3 (Hard):   full_episode       — optimise full 100-step DBS trajectory
"""

from .dbs_tasks import TASK_REGISTRY, DBSTask, get_task

__all__ = ["TASK_REGISTRY", "DBSTask", "get_task"]
