# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parkinson's Motor Environment — CSV-grounded implementation.

At each step the environment replays the park-sen closed-loop DBS simulation
timeline (100 steps, t=10.02–12.00s, 20ms intervals).

State at each step comes directly from the calibrated park-sen CSV data:
  - beta_arv, tremor_arv, semg_arv   (normalized neural signals)
  - force_amplitude                   (simulated muscle force, mN)
  - dbs_amplitude_ma, dbs_pulse_width_ms  (delivered DBS)
  - scheduler_class, beta_ctrl_error  (closed-loop controller state)
  - side_effect_load                  (DBS side-effect proxy)

The agent's motor_command is distorted by the current neural state:
    effective = command
                * (1 - BETA_WEIGHT  * beta_arv)
                * (1 - SYNC_WEIGHT  * tremor_arv)
                + tremor_noise(tremor_arv)

If the agent applies DBS (dbs_amplitude > 0), entrainment is looked up from
the 12×15 parameter sweep table and applied to the NEXT step's brain state
(one-step lag, matching clinical reality).

Reward = -|task_error| - DBS_PENALTY * dbs_amplitude
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from ..brain_calibrator import (
        calibrate, get_window_idx, query_dbs_effect, CalibratedBrainState,
    )
except ImportError:
    from models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from brain_calibrator import (
        calibrate, get_window_idx, query_dbs_effect, CalibratedBrainState,
    )

# ── physics weights ───────────────────────────────────────────────────────────
BETA_MOTOR_WEIGHT   = 0.55   # beta oscillation suppresses motor output
TREMOR_MOTOR_WEIGHT = 0.30   # tremor reduces effective output
TREMOR_NOISE_SCALE  = 0.12   # amplitude of tremor-driven noise on output

# ── reward weights ─────────────────────────────────────────────────────────────
# Reward per step (in [0, 1] range):
#   FORCE_WEIGHT  * force_preserved          ← primary: preserve motor function
#   TASK_WEIGHT   * (1 - task_error)         ← secondary: hit motor target
#   ENTRAINMENT_BONUS * dbs_entrainment      ← bonus when DBS suppresses beta
#   - DBS_PENALTY * dbs_amplitude            ← small penalty for side effects
#
# force_preserved from real simulation data: starts ~0.93, decays to ~0.04
# without DBS. Ground-truth park-sen controller (0.49–0.63 mA) achieves ~0.20
# at episode end. Full entrainment (2 mA) can push this higher.
# DBS_PENALTY is intentionally tiny so agent isn't discouraged from using DBS.
FORCE_WEIGHT      = 0.50
TASK_WEIGHT       = 0.30
ENTRAINMENT_BONUS = 0.15   # extra reward per unit of DBS entrainment
DBS_PENALTY       = 0.005  # per mA — tiny side-effect cost

N_STEPS = 100               # one full episode = 100 CSV windows


class ParkinsonsMotorEnvironment(Environment):
    """
    Parkinson's brain-state RL environment.

    The brain state at every step is sourced directly from the park-sen
    closed-loop DBS simulation — real neural dynamics, not approximations.
    The agent must learn to issue motor commands that achieve a target despite
    Parkinsonian beta oscillations and tremor, optionally using DBS.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._brain: CalibratedBrainState = calibrate()
        self._step_idx: int = 0
        self._dbs_entrainment: float = 0.0   # DBS effect carried from previous step
        self._target_output: float = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _brain_window(self):
        return get_window_idx(self._brain, self._step_idx)

    def _apply_distortion(self, command: float, beta: float, tremor: float) -> float:
        noise = (random.random() * 2.0 - 1.0) * TREMOR_NOISE_SCALE * tremor
        effective = (
            command
            * (1.0 - BETA_MOTOR_WEIGHT  * beta)
            * (1.0 - TREMOR_MOTOR_WEIGHT * tremor)
            + noise
        )
        return float(max(-1.0, min(1.0, effective)))

    def _make_obs(self, w, effective: float, task_error: float,
                  dbs_amp: float, dbs_pw: float, reward: float, done: bool,
                  meta: dict) -> ParkinsonsMotorObservation:
        sup = self._dbs_entrainment
        beta   = float(max(0.0, min(1.0, w.beta_arv   * (1.0 - sup))))
        tremor = float(max(0.0, min(1.0, w.tremor_arv * (1.0 - 0.6 * sup))))
        semg   = float(max(0.0, min(1.0, w.semg_arv   * (1.0 - sup))))
        # force_preserved already grounded in real data; DBS entrainment
        # raises it proportionally (more suppression → more force preserved)
        force_p = float(min(1.0, w.force_preserved * (1.0 + 0.4 * sup)))

        return ParkinsonsMotorObservation(
            beta_arv=beta,
            tremor_arv=tremor,
            semg_arv=semg,
            disease_severity=float(max(0.0, min(1.0, w.disease_severity * (1.0 - 0.5 * sup)))),
            beta_suppression=float(max(0.0, min(1.0, w.beta_suppression + sup * (1.0 - w.beta_suppression)))),
            force_amplitude=w.force_amplitude,
            force_preserved=force_p,
            effective_motor_output=effective,
            task_error=task_error,
            dbs_amplitude_ma=dbs_amp,
            dbs_pulse_width_ms=dbs_pw,
            dbs_entrainment=self._dbs_entrainment,
            side_effect_load=w.side_effect_load,
            scheduler_class=w.scheduler_class,
            beta_ctrl_error=w.beta_ctrl_error,
            sim_time_s=w.t_s,
            done=done,
            reward=reward,
            metadata=meta,
        )

    # ── Environment interface ─────────────────────────────────────────────────

    def reset(self) -> ParkinsonsMotorObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._step_idx = 0
        self._dbs_entrainment = 0.0
        self._target_output = round(random.uniform(-0.8, 0.8), 2)

        w = self._brain_window()
        return self._make_obs(
            w, effective=0.0, task_error=abs(self._target_output),
            dbs_amp=0.0, dbs_pw=0.06, reward=0.0, done=False,
            meta={"target_output": self._target_output, "step": 0},
        )

    def step(self, action: ParkinsonsMotorAction) -> ParkinsonsMotorObservation:  # type: ignore[override]
        self._state.step_count += 1
        w = self._brain_window()

        # Apply Parkinsonian motor distortion using current brain state
        # (already suppressed by last step's DBS via self._dbs_entrainment)
        effective = self._apply_distortion(
            action.motor_command,
            w.beta_arv   * (1.0 - self._dbs_entrainment),
            w.tremor_arv * (1.0 - 0.6 * self._dbs_entrainment),
        )
        task_error = abs(self._target_output - effective)

        # force_preserved from real CSV data (what ground-truth DBS achieved).
        # Agent's additional entrainment adds a bonus on top.
        force_p = w.force_preserved
        reward = float(
            FORCE_WEIGHT * force_p
            + TASK_WEIGHT * (1.0 - task_error)
            + ENTRAINMENT_BONUS * self._dbs_entrainment
            - DBS_PENALTY * action.dbs_amplitude
        )

        # Advance timeline
        self._step_idx += 1
        done = self._step_idx >= N_STEPS

        # Update DBS entrainment for next step
        self._dbs_entrainment = query_dbs_effect(
            self._brain, action.dbs_amplitude, action.dbs_pulse_width
        )

        next_w = self._brain_window()
        return self._make_obs(
            next_w, effective=effective, task_error=task_error,
            dbs_amp=action.dbs_amplitude, dbs_pw=action.dbs_pulse_width,
            reward=reward, done=done,
            meta={
                "target_output": self._target_output,
                "step": self._state.step_count,
                "ground_truth_dbs_ma": w.dbs_amplitude_ma,
                "ground_truth_force": w.force_amplitude,
            },
        )

    @property
    def state(self) -> State:
        return self._state
