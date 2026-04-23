"""
Parkinson's Motor Environment — Task-aware, grader-backed implementation.

At each step the environment replays the Fleming et al. (2023) closed-loop DBS
simulation timeline (100 steps, t=10.02–12.00 s, 20 ms intervals).

The episode window and success criteria are determined by the active task:
  beta_suppression  — 20 steps, easy, focus on beta suppression
  tremor_correction — 50 steps, medium, dynamic tremor management
  full_episode      — 100 steps, hard, full clinical optimisation

Brain state at each step comes from the calibrated fleming-model-based-brain data.
The agent supplies DBS parameters and a motor command; the environment
distorts the command via the Parkinsonian brain state and computes reward.
At episode end the registered grader scores the full trajectory [0.0–1.0].
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from ..brain_calibrator import (
        calibrate, get_window_idx, query_dbs_effect, CalibratedBrainState,
    )
    from ..tasks import get_task, DBSTask
    from ..graders import compute_score, is_success
except ImportError:
    from models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from brain_calibrator import (
        calibrate, get_window_idx, query_dbs_effect, CalibratedBrainState,
    )
    from tasks import get_task, DBSTask
    from graders import compute_score, is_success

# ── physics weights ────────────────────────────────────────────────────────────
BETA_MOTOR_WEIGHT   = 0.55   # beta oscillation suppresses motor output
TREMOR_MOTOR_WEIGHT = 0.30   # tremor reduces effective output
TREMOR_NOISE_SCALE  = 0.12   # amplitude of tremor-driven noise on output

# ── dense per-step reward weights ─────────────────────────────────────────────
# Dense signal so the agent learns continuously during the episode.
# The FINAL score (0–1) is computed by the task's grader at episode end.
FORCE_WEIGHT      = 0.50
TASK_WEIGHT       = 0.30
ENTRAINMENT_BONUS = 0.15
DBS_PENALTY       = 0.005  # per mA — tiny side-effect cost discourage brute force


class ParkinsonsMotorEnvironment(Environment):
    """
    Parkinson's brain-state RL environment with task-based grading.

    Each episode replays a task-specific slice of the Fleming et al. (2023)
    closed-loop DBS simulation.  The agent must tune DBS parameters and issue
    motor commands; at episode end the deterministic grader returns a score
    in [0.0, 1.0] representing clinical performance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._brain: CalibratedBrainState = calibrate()
        self._task: DBSTask = get_task()          # default: full_episode (hard)
        self._step_idx: int = 0                   # index into calibrated timeline
        self._local_step: int = 0                 # step index within this episode
        self._dbs_entrainment: float = 0.0
        self._target_output: float = 0.0
        self._trajectory: List[Dict[str, Any]] = []  # accumulates per-step dicts

    # ── helpers ───────────────────────────────────────────────────────────────

    def _abs_step(self) -> int:
        """Absolute index in the 100-step calibrated timeline."""
        return self._task.start_step + self._local_step

    def _brain_window(self):
        return get_window_idx(self._brain, self._abs_step())

    def _apply_distortion(self, command: float, beta: float, tremor: float) -> float:
        noise = (random.random() * 2.0 - 1.0) * TREMOR_NOISE_SCALE * tremor
        effective = (
            command
            * (1.0 - BETA_MOTOR_WEIGHT  * beta)
            * (1.0 - TREMOR_MOTOR_WEIGHT * tremor)
            + noise
        )
        return float(max(-1.0, min(1.0, effective)))

    def _record_step(self, w, dbs_amp: float, effective: float) -> None:
        """Append the step's key clinical metrics to the episode trajectory."""
        sup = self._dbs_entrainment
        self._trajectory.append({
            "force_preserved":  float(min(1.0, w.force_preserved * (1.0 + 0.4 * sup))),
            "beta_arv":         float(max(0.0, min(1.0, w.beta_arv * (1.0 - sup)))),
            "side_effect_load": float(max(0.0, min(1.0, w.side_effect_load))),
            "dbs_amplitude_ma": dbs_amp,
            "tremor_arv":       float(max(0.0, min(1.0, w.tremor_arv * (1.0 - 0.6 * sup)))),
            "effective_motor":  effective,
        })

    def _make_obs(self, w, effective: float, task_error: float,
                  dbs_amp: float, dbs_pw: float, reward: float, done: bool,
                  grader_score: Optional[float], meta: dict) -> ParkinsonsMotorObservation:
        sup = self._dbs_entrainment
        beta   = float(max(0.0, min(1.0, w.beta_arv   * (1.0 - sup))))
        tremor = float(max(0.0, min(1.0, w.tremor_arv * (1.0 - 0.6 * sup))))
        semg   = float(max(0.0, min(1.0, w.semg_arv   * (1.0 - sup))))
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
            side_effect_load=float(max(0.0, min(1.0, w.side_effect_load))),
            scheduler_class=w.scheduler_class,
            beta_ctrl_error=w.beta_ctrl_error,
            sim_time_s=w.t_s,
            task_id=self._task.task_id,
            grader_score=grader_score if grader_score is not None else -1.0,
            episode_success=is_success(self._task, grader_score) if grader_score is not None else False,
            done=done,
            reward=reward,
            metadata=meta,
        )

    # ── Environment interface ──────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> ParkinsonsMotorObservation:
        """
        Reset the environment.

        Args:
            task_id: One of 'beta_suppression', 'tremor_correction',
                     'full_episode'.  Defaults to 'full_episode'.
        """
        self._task = get_task(task_id)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._local_step = 0
        self._dbs_entrainment = 0.0
        self._trajectory = []
        # Random motor target in a realistic range (±0.6) to vary task demands
        self._target_output = round(random.uniform(-0.6, 0.6), 2)

        if len(self._brain.timeline) == 0:
            raise RuntimeError(f"Calibration failed remotely! Windows=0.")
        w = self._brain_window()
        return self._make_obs(
            w, effective=0.0, task_error=abs(self._target_output),
            dbs_amp=0.0, dbs_pw=0.06, reward=0.0, done=False,
            grader_score=None,
            meta={
                "task_id": self._task.task_id,
                "task_difficulty": self._task.difficulty,
                "target_output": self._target_output,
                "step": 0,
                "episode_steps": self._task.n_steps,
            },
        )

    def step(self, action: ParkinsonsMotorAction) -> ParkinsonsMotorObservation:  # type: ignore[override]
        self._state.step_count += 1
        w = self._brain_window()

        # ── Apply Parkinsonian motor distortion ───────────────────────────────
        effective = self._apply_distortion(
            action.motor_command,
            w.beta_arv   * (1.0 - self._dbs_entrainment),
            w.tremor_arv * (1.0 - 0.6 * self._dbs_entrainment),
        )
        task_error = abs(self._target_output - effective)

        # ── Dense per-step reward ─────────────────────────────────────────────
        force_p = w.force_preserved
        reward = float(
            FORCE_WEIGHT * float(min(1.0, force_p * (1.0 + 0.4 * self._dbs_entrainment)))
            + TASK_WEIGHT * (1.0 - task_error)
            + ENTRAINMENT_BONUS * self._dbs_entrainment
            - DBS_PENALTY * action.dbs_amplitude
        )

        # ── Record for grader ─────────────────────────────────────────────────
        self._record_step(w, action.dbs_amplitude, effective)

        # ── Advance timeline ──────────────────────────────────────────────────
        self._local_step += 1
        done = self._local_step >= self._task.n_steps

        # ── Update DBS entrainment for next step ──────────────────────────────
        self._dbs_entrainment = query_dbs_effect(
            self._brain, action.dbs_amplitude, action.dbs_pulse_width
        )

        # ── Compute grader score at episode end ───────────────────────────────
        grader_score: Optional[float] = None
        if done:
            grader_score = compute_score(self._task, self._trajectory)

        next_w = self._brain_window()
        return self._make_obs(
            next_w, effective=effective, task_error=task_error,
            dbs_amp=action.dbs_amplitude, dbs_pw=action.dbs_pulse_width,
            reward=reward, done=done, grader_score=grader_score,
            meta={
                "task_id": self._task.task_id,
                "target_output": self._target_output,
                "step": self._state.step_count,
                "local_step": self._local_step,
                "episode_steps": self._task.n_steps,
                "ground_truth_dbs_ma": w.dbs_amplitude_ma,
                "ground_truth_force": w.force_amplitude,
            },
        )

    @property
    def state(self) -> State:
        return self._state
