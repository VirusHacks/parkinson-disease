"""Task datamodels for the Parkinson's Motor environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DBSTask:
    """Immutable task specification."""

    task_id: str
    description: str
    difficulty: str
    start_step: int
    n_steps: int
    max_dbs_amplitude: float
    max_dbs_pulse_width: float
    max_side_effect_load: float
    target_force_preserved: float
    target_beta_arv: float
    target_tremor_arv: float
    target_tracking_error: float
    success_threshold: float
    patient_profile_ids: Tuple[str, ...] = field(default_factory=tuple)
    target_output_range: Tuple[float, float] = (-0.5, 0.5)
