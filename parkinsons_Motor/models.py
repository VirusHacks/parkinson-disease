# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Parkinsons Motor environment.

Action:
  - motor intent command
  - DBS amplitude
  - DBS pulse width

Observation:
  - clinically plausible sensed signals
  - short-horizon trend summaries
  - task/constraint diagnostics relevant to learning
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ParkinsonsMotorAction(Action):
    """Agent action at each step."""

    motor_command: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Intended normalized motor command in [-1, 1].",
    )
    dbs_amplitude: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Requested DBS amplitude in mA before task-level clipping.",
    )
    dbs_pulse_width: float = Field(
        default=0.06,
        ge=0.06,
        le=0.20,
        description="Requested DBS pulse width in ms before task-level clipping.",
    )
    task_id: str = Field(
        default="",
        description="Task ID to load on reset. Empty keeps the current task.",
    )


class ParkinsonsMotorObservation(Observation):
    """Clinically oriented observation returned by the environment."""

    # Sensed neural and motor state
    beta_arv: float = Field(default=0.0, ge=0.0, le=1.0)
    tremor_arv: float = Field(default=0.0, ge=0.0, le=1.0)
    semg_arv: float = Field(default=0.0, ge=0.0, le=1.0)
    force_amplitude: float = Field(default=0.0, ge=0.0)
    force_preserved: float = Field(default=0.0, ge=0.0, le=1.0)

    # Derived but clinically interpretable summaries
    disease_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    beta_suppression: float = Field(default=0.0, ge=0.0, le=1.0)
    beta_trend: float = Field(default=0.0, ge=-1.0, le=1.0)
    tremor_trend: float = Field(default=0.0, ge=-1.0, le=1.0)
    side_effect_rate: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Task performance
    target_output: float = Field(default=0.0, ge=-1.0, le=1.0)
    effective_motor_output: float = Field(default=0.0, ge=-1.0, le=1.0)
    task_error: float = Field(default=0.0, ge=0.0, le=2.0)
    tracking_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # DBS device state
    dbs_amplitude_ma: float = Field(default=0.0, ge=0.0)
    dbs_pulse_width_ms: float = Field(default=0.06, ge=0.06, le=0.20)
    dbs_entrainment: float = Field(default=0.0, ge=0.0, le=1.0)
    recent_dbs_avg_ma: float = Field(default=0.0, ge=0.0)
    recent_dbs_avg_pw_ms: float = Field(default=0.06, ge=0.06, le=0.20)
    side_effect_load: float = Field(default=0.0, ge=0.0, le=1.0)
    action_smoothness_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    dbs_constraint_violation: float = Field(default=0.0, ge=0.0, le=1.0)

    # Timing and task info
    sim_time_s: float = Field(default=0.0, ge=0.0)
    task_id: str = Field(default="full_episode")
    grader_score: float = Field(default=-1.0)
    episode_success: bool = Field(default=False)
