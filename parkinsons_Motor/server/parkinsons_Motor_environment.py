"""
Parkinson's Motor Environment — upgraded causal control loop.

This environment keeps the Fleming et al. (2023) trajectory as a physiological
anchor, but the online state is no longer a direct replay. Each step now uses:
  - task-specific stimulation constraints
  - seeded patient profiles
  - action-coupled side-effect accumulation
  - short-term stimulation wash-in / wash-out
  - latent state carryover across the episode
  - benchmark-aligned diagnostics for grading
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..brain_calibrator import CalibratedBrainState, calibrate, get_window_idx, query_dbs_effect
    from ..graders import compute_score_details, is_success
    from ..models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from ..patient_profiles import PatientProfile, get_profile
    from ..tasks import DBSTask, get_task
except ImportError:
    from brain_calibrator import CalibratedBrainState, calibrate, get_window_idx, query_dbs_effect
    from graders import compute_score_details, is_success
    from models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from patient_profiles import PatientProfile, get_profile
    from tasks import DBSTask, get_task


FORCE_WEIGHT = 0.28
TRACKING_WEIGHT = 0.20
BETA_WEIGHT = 0.16
TREMOR_WEIGHT = 0.12
SAFETY_WEIGHT = 0.12
SMOOTHNESS_WEIGHT = 0.07
EFFICIENCY_WEIGHT = 0.05


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class ParkinsonsMotorEnvironment(Environment):
    """Task-aware Parkinson's control environment with patient variation."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: int = 7):
        self._base_seed = seed
        self._episode_index = 0
        self._rng = random.Random(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._brain: CalibratedBrainState = calibrate()
        self._task: DBSTask = get_task()
        self._profile: PatientProfile = get_profile("balanced")

        self._local_step = 0
        self._target_output = 0.0
        self._trajectory: List[Dict[str, Any]] = []

        self._beta_state = 0.0
        self._tremor_state = 0.0
        self._semg_state = 0.0
        self._force_state = 0.0
        self._side_effect_state = 0.0
        self._fatigue_state = 0.0
        self._entrainment_state = 0.0

        self._prev_beta = 0.0
        self._prev_tremor = 0.0
        self._prev_side_effect = 0.0
        self._prev_amp = 0.0
        self._prev_pw = 0.06
        self._recent_amp: deque[float] = deque(maxlen=5)
        self._recent_pw: deque[float] = deque(maxlen=5)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _abs_step(self) -> int:
        return self._task.start_step + self._local_step

    def _brain_window(self, idx: Optional[int] = None):
        step = self._abs_step() if idx is None else idx
        return get_window_idx(self._brain, step)

    def _sample_profile(self, explicit_profile_id: Optional[str]) -> PatientProfile:
        if explicit_profile_id:
            return get_profile(explicit_profile_id)
        options = list(self._task.patient_profile_ids)
        return get_profile(self._rng.choice(options))

    def _init_latent_state(self) -> None:
        base = self._brain_window(self._task.start_step)
        self._beta_state = _clamp(base.beta_arv * self._profile.beta_scale)
        self._tremor_state = _clamp(base.tremor_arv * self._profile.tremor_scale)
        self._semg_state = _clamp(base.semg_arv * self._profile.semg_scale)
        self._force_state = _clamp(base.force_preserved * self._profile.force_scale)
        self._side_effect_state = 0.02 * self._profile.side_effect_sensitivity
        self._fatigue_state = 0.03 * self._profile.progression_scale
        self._entrainment_state = 0.0

        self._prev_beta = self._beta_state
        self._prev_tremor = self._tremor_state
        self._prev_side_effect = self._side_effect_state
        self._prev_amp = 0.0
        self._prev_pw = 0.06
        self._recent_amp.clear()
        self._recent_pw.clear()
        self._recent_amp.append(0.0)
        self._recent_pw.append(0.06)

    def _clip_action(self, action: ParkinsonsMotorAction) -> Tuple[float, float, float]:
        amp_cap = self._task.max_dbs_amplitude * self._profile.max_amp_scale
        pw_cap = self._task.max_dbs_pulse_width

        amp = _clamp(action.dbs_amplitude, 0.0, amp_cap)
        pw = _clamp(action.dbs_pulse_width, 0.06, pw_cap)

        amp_violation = max(0.0, action.dbs_amplitude - amp_cap) / max(amp_cap, 1e-6)
        pw_violation = max(0.0, action.dbs_pulse_width - pw_cap) / max(pw_cap - 0.06, 1e-6)
        violation = _clamp(0.75 * amp_violation + 0.25 * pw_violation)
        return amp, pw, violation

    def _smoothness_cost(self, amp: float, pw: float) -> float:
        amp_span = max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6)
        pw_span = max(self._task.max_dbs_pulse_width - 0.06, 1e-6)
        amp_delta = abs(amp - self._prev_amp) / amp_span
        pw_delta = abs(pw - self._prev_pw) / pw_span
        return _clamp(0.70 * amp_delta + 0.30 * pw_delta)

    def _mean_recent(self, values: deque[float], fallback: float) -> float:
        return sum(values) / len(values) if values else fallback

    def _apply_motor_distortion(self, command: float) -> float:
        noise = (
            self._rng.uniform(-1.0, 1.0)
            * 0.10
            * self._profile.motor_noise_scale
            * (0.35 + self._tremor_state)
        )
        effective = (
            command
            * (1.0 - 0.52 * self._beta_state)
            * (1.0 - 0.30 * self._tremor_state)
            * (1.0 - 0.10 * self._side_effect_state)
            + noise
        )
        return _clamp(effective, -1.0, 1.0)

    def _update_side_effects(self, amp: float, pw: float, smoothness: float, violation: float) -> None:
        amp_cap = max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6)
        amp_norm = amp / amp_cap
        pw_norm = (pw - 0.06) / max(self._task.max_dbs_pulse_width - 0.06, 1e-6)
        stimulation_burden = (
            0.58 * (amp_norm ** 1.70)
            + 0.22 * (pw_norm ** 1.30)
            + 0.20 * (amp_norm * pw_norm)
        )
        burden = (
            0.24 * stimulation_burden * self._profile.side_effect_sensitivity
            + 0.08 * self._entrainment_state
            + 0.10 * smoothness
            + 0.35 * violation
        )
        recovery = self._profile.recovery_rate * (1.10 - 0.55 * amp_norm)
        self._side_effect_state = _clamp(self._side_effect_state * (1.0 - recovery) + burden)

    def _update_latent_state(
        self,
        current_base,
        next_base,
        effective_motor: float,
        tracking_accuracy: float,
        smoothness: float,
    ) -> None:
        base_beta = _clamp(next_base.beta_arv * self._profile.beta_scale)
        base_tremor = _clamp(next_base.tremor_arv * self._profile.tremor_scale)
        base_semg = _clamp(next_base.semg_arv * self._profile.semg_scale)
        base_force = _clamp(next_base.force_preserved * self._profile.force_scale)

        beta_progression = (next_base.beta_arv - current_base.beta_arv) * self._profile.progression_scale
        tremor_progression = (next_base.tremor_arv - current_base.tremor_arv) * self._profile.progression_scale
        disease_pressure = 0.05 * self._profile.progression_scale * (0.35 + current_base.disease_severity)
        under_treated_pressure = 0.08 * (1.0 - self._entrainment_state) * (0.30 + current_base.disease_severity)

        target_beta = _clamp(
            base_beta
            + beta_progression
            + disease_pressure
            + under_treated_pressure
            + 0.06 * self._fatigue_state
            + 0.04 * self._side_effect_state
            - 0.82 * self._entrainment_state * self._profile.beta_responsiveness
        )
        self._beta_state = _clamp(0.45 * self._beta_state + 0.55 * target_beta)

        target_tremor = _clamp(
            base_tremor
            + tremor_progression
            + 0.24 * self._beta_state
            + 0.08 * under_treated_pressure
            + 0.06 * self._fatigue_state
            - 0.50 * self._entrainment_state * self._profile.tremor_responsiveness
        )
        self._tremor_state = _clamp(0.40 * self._tremor_state + 0.60 * target_tremor)

        self._semg_state = _clamp(
            0.55 * base_semg
            + 0.25 * self._beta_state
            + 0.20 * self._tremor_state
        )

        desired_force = _clamp(
            0.88 * base_force
            + 0.30 * tracking_accuracy
            + 0.20 * self._entrainment_state
            - 0.30 * self._beta_state
            - 0.34 * self._tremor_state
            - 0.20 * self._side_effect_state
            - 0.08 * smoothness
            - 0.10 * self._fatigue_state
        )
        self._force_state = _clamp(0.42 * self._force_state + 0.58 * desired_force)

        fatigue_gain = (
            self._profile.fatigue_rate
            * (0.22 + 0.25 * self._entrainment_state + 0.55 * self._tremor_state + 0.20 * under_treated_pressure)
        )
        self._fatigue_state = _clamp(self._fatigue_state * 0.95 + fatigue_gain)

    def _build_reward(
        self,
        tracking_accuracy: float,
        smoothness: float,
        violation: float,
        amp: float,
        pw: float,
    ) -> float:
        amp_eff = 1.0 - (amp / max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6))
        pw_eff = 1.0 - ((pw - 0.06) / max(self._task.max_dbs_pulse_width - 0.06, 1e-6))
        efficiency = _clamp(0.65 * amp_eff + 0.35 * pw_eff)
        safety = _clamp(1.0 - (self._side_effect_state / max(self._task.max_side_effect_load, 1e-6)))
        reward = (
            FORCE_WEIGHT * self._force_state
            + TRACKING_WEIGHT * tracking_accuracy
            + BETA_WEIGHT * (1.0 - self._beta_state)
            + TREMOR_WEIGHT * (1.0 - self._tremor_state)
            + SAFETY_WEIGHT * safety
            + SMOOTHNESS_WEIGHT * (1.0 - smoothness)
            + EFFICIENCY_WEIGHT * efficiency
            - 0.08 * violation
        )
        return float(_clamp(reward))

    def _record_step(
        self,
        amp: float,
        pw: float,
        effective_motor: float,
        task_error: float,
        tracking_accuracy: float,
        smoothness: float,
        violation: float,
    ) -> None:
        self._trajectory.append(
            {
                "force_preserved": self._force_state,
                "beta_arv": self._beta_state,
                "tremor_arv": self._tremor_state,
                "task_error": task_error,
                "tracking_accuracy": tracking_accuracy,
                "side_effect_load": self._side_effect_state,
                "smoothness_cost": smoothness,
                "constraint_violation": violation,
                "dbs_amplitude_ma": amp,
                "dbs_pulse_width_ms": pw,
                "effective_motor_output": effective_motor,
                "target_output": self._target_output,
            }
        )

    def _make_obs(
        self,
        *,
        sim_time_s: float,
        reward: float,
        done: bool,
        grader_score: Optional[float],
        effective_motor: float,
        task_error: float,
        tracking_accuracy: float,
        amp: float,
        pw: float,
        smoothness: float,
        violation: float,
        metadata: Dict[str, Any],
    ) -> ParkinsonsMotorObservation:
        beta_trend = _clamp(self._beta_state - self._prev_beta, -1.0, 1.0)
        tremor_trend = _clamp(self._tremor_state - self._prev_tremor, -1.0, 1.0)
        side_effect_rate = _clamp(self._side_effect_state - self._prev_side_effect, -1.0, 1.0)
        recent_amp = self._mean_recent(self._recent_amp, amp)
        recent_pw = self._mean_recent(self._recent_pw, pw)

        force_amplitude = self._force_state * self._brain.healthy_force_mn

        return ParkinsonsMotorObservation(
            beta_arv=self._beta_state,
            tremor_arv=self._tremor_state,
            semg_arv=self._semg_state,
            force_amplitude=force_amplitude,
            force_preserved=self._force_state,
            disease_severity=_clamp(0.55 * self._tremor_state + 0.45 * self._beta_state),
            beta_suppression=_clamp(1.0 - self._beta_state),
            beta_trend=beta_trend,
            tremor_trend=tremor_trend,
            side_effect_rate=side_effect_rate,
            target_output=self._target_output,
            effective_motor_output=effective_motor,
            task_error=task_error,
            tracking_accuracy=tracking_accuracy,
            dbs_amplitude_ma=amp,
            dbs_pulse_width_ms=pw,
            dbs_entrainment=self._entrainment_state,
            recent_dbs_avg_ma=recent_amp,
            recent_dbs_avg_pw_ms=recent_pw,
            side_effect_load=self._side_effect_state,
            action_smoothness_cost=smoothness,
            dbs_constraint_violation=violation,
            sim_time_s=sim_time_s,
            task_id=self._task.task_id,
            grader_score=grader_score if grader_score is not None else -1.0,
            episode_success=is_success(self._task, grader_score) if grader_score is not None else False,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        patient_profile_id: Optional[str] = None,
    ) -> ParkinsonsMotorObservation:
        """Reset the environment."""
        self._task = get_task(task_id)
        episode_seed = self._base_seed + self._episode_index if seed is None else seed
        self._episode_index += 1
        self._rng = random.Random(episode_seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._local_step = 0
        self._trajectory = []
        self._profile = self._sample_profile(patient_profile_id)
        lo, hi = self._task.target_output_range
        self._target_output = round(self._rng.uniform(lo, hi), 2)
        self._init_latent_state()

        base = self._brain_window(self._task.start_step)
        tracking_accuracy = _clamp(1.0 - abs(self._target_output) / 2.0)
        return self._make_obs(
            sim_time_s=base.t_s,
            reward=0.0,
            done=False,
            grader_score=None,
            effective_motor=0.0,
            task_error=abs(self._target_output),
            tracking_accuracy=tracking_accuracy,
            amp=0.0,
            pw=0.06,
            smoothness=0.0,
            violation=0.0,
            metadata={
                "task_id": self._task.task_id,
                "task_difficulty": self._task.difficulty,
                "episode_steps": self._task.n_steps,
                "target_output": self._target_output,
                "seed": episode_seed,
                "patient_profile_id": self._profile.profile_id,
                "patient_profile_description": self._profile.description,
                "ground_truth_dbs_ma": base.dbs_amplitude_ma,
                "ground_truth_force": base.force_amplitude,
                "ground_truth_scheduler_class": base.scheduler_class,
                "ground_truth_beta_ctrl_error": base.beta_ctrl_error,
            },
        )

    def step(self, action: ParkinsonsMotorAction) -> ParkinsonsMotorObservation:  # type: ignore[override]
        self._state.step_count += 1
        current_idx = self._abs_step()
        current_base = self._brain_window(current_idx)
        next_idx = min(current_idx + 1, self._task.start_step + self._task.n_steps - 1)
        next_base = self._brain_window(next_idx)

        amp, pw, violation = self._clip_action(action)
        smoothness = self._smoothness_cost(amp, pw)
        target_entrainment = _clamp(
            query_dbs_effect(self._brain, amp, pw) * self._profile.entrainment_scale
        )
        self._entrainment_state = _clamp(0.35 * self._entrainment_state + 0.65 * target_entrainment)

        effective_motor = self._apply_motor_distortion(action.motor_command)
        task_error = abs(self._target_output - effective_motor)
        tracking_accuracy = _clamp(1.0 - task_error / 2.0)

        self._update_side_effects(amp, pw, smoothness, violation)
        self._update_latent_state(current_base, next_base, effective_motor, tracking_accuracy, smoothness)

        reward = self._build_reward(tracking_accuracy, smoothness, violation, amp, pw)
        self._record_step(amp, pw, effective_motor, task_error, tracking_accuracy, smoothness, violation)

        self._recent_amp.append(amp)
        self._recent_pw.append(pw)
        self._local_step += 1
        done = self._local_step >= self._task.n_steps

        score_details: Optional[Dict[str, float]] = None
        grader_score: Optional[float] = None
        if done:
            score_details = compute_score_details(self._task, self._trajectory)
            grader_score = score_details["overall_score"]

        metadata = {
            "task_id": self._task.task_id,
            "target_output": self._target_output,
            "step": self._state.step_count,
            "local_step": self._local_step,
            "episode_steps": self._task.n_steps,
            "patient_profile_id": self._profile.profile_id,
            "patient_profile_description": self._profile.description,
            "ground_truth_dbs_ma": next_base.dbs_amplitude_ma,
            "ground_truth_force": next_base.force_amplitude,
            "ground_truth_scheduler_class": next_base.scheduler_class,
            "ground_truth_beta_ctrl_error": next_base.beta_ctrl_error,
            "constraint_violation": violation,
            "score_details": score_details or {},
        }

        obs = self._make_obs(
            sim_time_s=next_base.t_s,
            reward=reward,
            done=done,
            grader_score=grader_score,
            effective_motor=effective_motor,
            task_error=task_error,
            tracking_accuracy=tracking_accuracy,
            amp=amp,
            pw=pw,
            smoothness=smoothness,
            violation=violation,
            metadata=metadata,
        )

        self._prev_beta = self._beta_state
        self._prev_tremor = self._tremor_state
        self._prev_side_effect = self._side_effect_state
        self._prev_amp = amp
        self._prev_pw = pw
        return obs

    @property
    def state(self) -> State:
        return self._state
