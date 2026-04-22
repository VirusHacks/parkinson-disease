# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Parkinsons Motor Environment.

Action  : motor command (intended force) + optional DBS parameters.
Observation : full brain-state derived from park-sen closed-loop simulation.

All normalized fields are in [0, 1] unless noted.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ParkinsonsMotorAction(Action):
    """
    Agent action at each step.

    motor_command    : intended normalised force/torque in [-1, 1].
    dbs_amplitude    : DBS stimulation amplitude in mA [0, 5]. 0 = off.
    dbs_pulse_width  : DBS pulse width in ms [0.06, 0.20].
    task_id          : task to run on reset (ignored during step).  One of
                       'beta_suppression' | 'tremor_correction' | 'full_episode'.
                       Leave empty to keep the current task.
    """
    motor_command: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Intended motor output normalised to [-1, 1]",
    )
    dbs_amplitude: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="DBS stimulation amplitude in mA (0 = off)",
    )
    dbs_pulse_width: float = Field(
        default=0.06, ge=0.06, le=0.20,
        description="DBS pulse width in ms",
    )
    task_id: str = Field(
        default="",
        description="Task ID to load on reset. Empty = keep current task.",
    )


class ParkinsonsMotorObservation(Observation):
    """
    Brain-state observation grounded in park-sen simulation data.

    Neural state (normalized [0,1]):
      beta_arv          : STN beta-band amplitude (ARV). High = strong PD oscillation.
                          Pre-DBS baseline ~0.78. DBS suppresses toward 0.
      tremor_arv        : Tremor amplitude envelope. Grows from ~0 to ~0.69 over episode.
      semg_arv          : Surface EMG envelope. Mirrors beta activity.

    Motor output:
      force_amplitude   : Simulated muscle force (mN, raw — not normalized).
                          Starts ~55000 mN, decays as tremor grows.
      effective_motor_output : Agent's command after Parkinsonian distortion [-1, 1].
      task_error        : |target - effective_output| [0, 2].

    DBS state:
      dbs_amplitude_ma  : DBS amplitude actually delivered this step (mA).
      dbs_pulse_width_ms: DBS pulse width (ms).
      dbs_entrainment   : Fraction of cortical collaterals entrained [0,1].
      side_effect_load  : Cumulative DBS side-effect proxy [0,1].

    Controller internals (for transparency / debugging):
      scheduler_class   : 0=tremor controller active, 1=beta controller active.
      beta_ctrl_error   : Beta controller tracking error (raw, positive=undershoot).
    """
    # ── neural state ──────────────────────────────────────────────────────────
    beta_arv: float = Field(default=0.0, ge=0.0, le=1.0,
        description="STN beta oscillation (0=suppressed, 1=peak Parkinson's)")
    tremor_arv: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Tremor amplitude envelope (0=none, 1=max observed)")
    semg_arv: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Surface EMG envelope (normalized)")

    # ── disease state ─────────────────────────────────────────────────────────
    disease_severity: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Current Parkinson severity (= tremor_arv). 0=early, 1=severe")
    beta_suppression: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Fraction of beta oscillation suppressed by DBS so far")

    # ── motor output ──────────────────────────────────────────────────────────
    force_amplitude: float = Field(default=0.0, ge=0.0,
        description="Simulated muscle force (mN, raw). Healthy baseline ~59752 mN")
    force_preserved: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Fraction of healthy force preserved (force / 59752 mN). "
                    "Pre-DBS Parkinson: ~0.80. End of episode without DBS: ~0.04")
    effective_motor_output: float = Field(default=0.0, ge=-1.0, le=1.0,
        description="Agent command after Parkinsonian distortion")
    task_error: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── DBS ───────────────────────────────────────────────────────────────────
    dbs_amplitude_ma: float = Field(default=0.0, ge=0.0,
        description="DBS amplitude delivered (mA). Ground-truth optimal: ~0.49-0.63 mA")
    dbs_pulse_width_ms: float = Field(default=0.06)
    dbs_entrainment: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Fraction of cortical collaterals entrained by DBS")
    side_effect_load: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── controller internals ──────────────────────────────────────────────────
    scheduler_class: int = Field(default=1,
        description="Active sub-controller: 0=tremor, 1=beta (active 99% of time)")
    beta_ctrl_error: float = Field(default=0.0,
        description="Beta controller tracking error. +ve=undershoot, -ve=suppressed")

    # ── simulation time ───────────────────────────────────────────────────────
    sim_time_s: float = Field(default=0.0, ge=0.0)

    # ── task & grader ─────────────────────────────────────────────────────────
    task_id: str = Field(default="full_episode",
        description="Active task: 'beta_suppression' | 'tremor_correction' | 'full_episode'")
    grader_score: float = Field(default=-1.0,
        description="Final grader score in [0, 1]. -1.0 means episode not yet finished.")
    episode_success: bool = Field(default=False,
        description="True if grader_score >= task success_threshold at episode end.")
