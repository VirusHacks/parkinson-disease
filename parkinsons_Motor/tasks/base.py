"""Task datamodels for the Parkinson's Motor environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DBSTask:
    """Immutable task specification.

    Fields below define both the clinical scenario (targets, budgets, profiles)
    and the runtime behaviour (episode length, target output range, optional
    stochastic event mix). Event configuration is opt-in: tasks that omit
    `event_profile` run a deterministic deterioration curve, while expert tasks
    can request specific event distributions to model real clinical disturbances.
    """

    task_id: str
    name: str
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
    # Optional stochastic event profile id. None → deterministic episode (legacy
    # behaviour). See parkinsons_Motor.core.events for available profiles.
    event_profile: str | None = None
    # Optional per-step biomarker sensor noise (std). 0.0 → clean signals.
    sensor_noise_std: float = 0.0
    # Optional time-varying setpoint schedule id (e.g. nocturnal taper).
    schedule_id: str | None = None
