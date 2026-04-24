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
    from ..core.calibration import CalibratedBrainState, calibrate, get_window_idx, query_dbs_effect
    from ..core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from ..core.patient_profiles import PatientProfile, get_profile
    from ..graders import compute_score_details, is_success
    from ..tasks import DBSTask, get_task
except ImportError:
    from core.calibration import CalibratedBrainState, calibrate, get_window_idx, query_dbs_effect
    from core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from core.patient_profiles import PatientProfile, get_profile
    from graders import compute_score_details, is_success
    from tasks import DBSTask, get_task


# Task-specific reward weight sets — mirror grader weights so training signal
# aligns with evaluation. Keys match task_id; "default" is the fallback.
_REWARD_WEIGHTS: dict = {
    "beta_suppression": dict(
        force=0.16, tracking=0.12, beta=0.30, tremor=0.18, safety=0.14,
        smoothness=0.05, efficiency=0.05,
    ),
    "tremor_correction": dict(
        force=0.16, tracking=0.16, beta=0.06, tremor=0.14, safety=0.22,
        smoothness=0.04, efficiency=0.08,
    ),
    "full_episode": dict(
        force=0.14, tracking=0.16, beta=0.08, tremor=0.06, safety=0.36,
        smoothness=0.05, efficiency=0.10,
    ),
    "fragile_patient": dict(
        force=0.18, tracking=0.18, beta=0.10, tremor=0.10, safety=0.28,
        smoothness=0.06, efficiency=0.05,
    ),
    "refractory_patient": dict(
        force=0.18, tracking=0.14, beta=0.08, tremor=0.12, safety=0.18,
        smoothness=0.05, efficiency=0.10,
    ),
    "personalization_generalization": dict(
        force=0.18, tracking=0.18, beta=0.08, tremor=0.12, safety=0.18,
        smoothness=0.04, efficiency=0.08,
    ),
    "default": dict(
        force=0.20, tracking=0.18, beta=0.14, tremor=0.12, safety=0.18,
        smoothness=0.06, efficiency=0.07,
    ),
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _freq_beta_factor(freq_hz: float) -> float:
    """Beta-suppression efficiency vs stimulation frequency.

    Derived from the DBS frequency-response literature (Kühn et al. 2008;
    Tinkhauser et al. 2017). Peak beta suppression at ~130 Hz; falls off
    symmetrically at lower and higher frequencies.
    """
    # Normalized position in [0,1] over [60, 185] Hz range
    f = (freq_hz - 60.0) / 125.0
    # Gaussian centred at 0.56 (=130 Hz), std≈0.26 → at 60 Hz factor≈0.68, at 185 Hz≈0.78
    peak = 0.56
    factor = 0.68 + 0.32 * max(0.0, 1.0 - ((f - peak) / 0.26) ** 2)
    return _clamp(factor, 0.65, 1.02)


def _freq_side_effect_factor(freq_hz: float) -> float:
    """Side-effect multiplier vs frequency.

    Higher frequencies drive more rapid axonal fatigue and increased
    charge delivery per second, raising dyskinesia risk.
    """
    f = (freq_hz - 60.0) / 125.0  # [0, 1]
    return _clamp(0.82 + 0.36 * f)  # 0.82 at 60 Hz → 1.18 at 185 Hz


def _battery_drain(amp: float, pw: float, freq: float) -> float:
    """Instantaneous battery drain proxy (normalized [0,1]).

    Charge per second ∝ amplitude × pulse_width × frequency.
    Normalized to max possible charge: 5 mA × 0.20 ms × 185 Hz.
    """
    charge_per_s = amp * (pw * 1e-3) * freq  # mA × s × Hz = mC/s
    max_charge = 5.0 * 0.20e-3 * 185.0
    return _clamp(charge_per_s / max_charge)


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
        self._adaptation_state = 0.0

        self._prev_beta = 0.0
        self._prev_tremor = 0.0
        self._prev_side_effect = 0.0
        self._prev_amp = 0.0
        self._prev_pw = 0.06
        self._prev_freq = 130.0
        self._recent_amp: deque[float] = deque(maxlen=5)
        self._recent_pw: deque[float] = deque(maxlen=5)

        # Per-episode trajectory noise (seeded at reset, prevents memorization)
        self._ep_beta_noise: float = 1.0
        self._ep_tremor_noise: float = 1.0
        self._ep_force_noise: float = 1.0
        self._ep_semg_noise: float = 1.0

        # Extended physiological state
        self._gamma_state: float = 0.0       # over-stimulation biomarker
        self._stim_washout: float = 0.0      # wash-in/wash-out accumulator
        self._medication_phase: float = 0.5  # L-DOPA cycle position
        self._med_phase_offset: float = 0.0  # per-episode phase offset
        self._battery_drain: float = 0.0     # cumulative battery drain proxy
        self._initial_force_state: float = 0.0
        self._initial_tremor_state: float = 0.0
        self._initial_tracking_accuracy: float = 0.0

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
        self._beta_state = _clamp(base.beta_arv * self._profile.beta_scale * self._ep_beta_noise)
        self._tremor_state = _clamp(base.tremor_arv * self._profile.tremor_scale * self._ep_tremor_noise)
        self._semg_state = _clamp(base.semg_arv * self._profile.semg_scale * self._ep_semg_noise)
        self._force_state = _clamp(base.force_preserved * self._profile.force_scale * self._ep_force_noise)
        self._side_effect_state = 0.02 * self._profile.side_effect_sensitivity
        self._fatigue_state = 0.03 * self._profile.progression_scale
        self._entrainment_state = 0.0
        self._adaptation_state = 0.0

        # Extended physiological state — reset per episode
        self._gamma_state = 0.0
        self._stim_washout = 0.0
        self._battery_drain = 0.0
        # L-DOPA phase: random offset per episode, advances ~1 full cycle over episode
        self._med_phase_offset = self._rng.uniform(0.0, 1.0)
        self._medication_phase = self._med_phase_offset

        self._prev_beta = self._beta_state
        self._prev_tremor = self._tremor_state
        self._prev_side_effect = self._side_effect_state
        self._prev_amp = 0.0
        self._prev_pw = 0.06
        self._recent_amp.clear()
        self._recent_pw.clear()
        self._recent_amp.append(0.0)
        self._recent_pw.append(0.06)
        self._initial_force_state = self._force_state
        self._initial_tremor_state = self._tremor_state

    def _long_horizon_shaping(self, tracking_accuracy: float) -> float:
        progress = (self._local_step + 1) / max(self._task.n_steps, 1)
        target_force = max(self._task.target_force_preserved, 1e-6)
        target_tremor = max(self._task.target_tremor_arv, 1e-6)
        target_error = max(self._task.target_tracking_error, 1e-6)

        terminal_proxy = _clamp(
            0.45 * min(self._force_state / target_force, 1.0)
            + 0.30 * (1.0 - min(self._tremor_state / target_tremor, 1.0))
            + 0.25 * (1.0 - min((2.0 * (1.0 - tracking_accuracy)) / target_error, 1.0))
        )

        if self._task.task_id == "tremor_correction":
            recovery_force = _clamp((self._force_state - self._initial_force_state + 0.12) / 0.28)
            recovery_tremor = _clamp((self._initial_tremor_state - self._tremor_state + 0.08) / 0.24)
            recovery_tracking = _clamp(
                (tracking_accuracy - self._initial_tracking_accuracy + 0.06) / 0.20
            )
            recovery_proxy = _clamp(
                0.40 * recovery_force + 0.40 * recovery_tremor + 0.20 * recovery_tracking
            )
            recovery_weight = min(progress / 0.5, 1.0)
            terminal_weight = _clamp((progress - 0.65) / 0.35)
            return 0.03 * recovery_proxy * recovery_weight + 0.02 * terminal_proxy * terminal_weight

        if self._task.task_id == "full_episode":
            terminal_weight = _clamp((progress - 0.75) / 0.25)
            return 0.03 * terminal_proxy * terminal_weight

        return 0.0

    def _clip_action(self, action: ParkinsonsMotorAction) -> Tuple[float, float, float, float]:
        amp_cap = self._task.max_dbs_amplitude * self._profile.max_amp_scale
        pw_cap = self._task.max_dbs_pulse_width

        amp = _clamp(action.dbs_amplitude, 0.0, amp_cap)
        pw = _clamp(action.dbs_pulse_width, 0.06, pw_cap)
        freq = _clamp(action.dbs_frequency, 60.0, 185.0)

        amp_violation = max(0.0, action.dbs_amplitude - amp_cap) / max(amp_cap, 1e-6)
        pw_violation = max(0.0, action.dbs_pulse_width - pw_cap) / max(pw_cap - 0.06, 1e-6)
        violation = _clamp(0.75 * amp_violation + 0.25 * pw_violation)
        return amp, pw, freq, violation

    def _smoothness_cost(self, amp: float, pw: float, freq: float) -> float:
        amp_span = max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6)
        pw_span = max(self._task.max_dbs_pulse_width - 0.06, 1e-6)
        freq_span = 125.0  # 185 - 60
        amp_delta = abs(amp - self._prev_amp) / amp_span
        pw_delta = abs(pw - self._prev_pw) / pw_span
        freq_delta = abs(freq - self._prev_freq) / freq_span
        return _clamp(0.60 * amp_delta + 0.25 * pw_delta + 0.15 * freq_delta)

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

    def _update_side_effects(self, amp: float, pw: float, freq: float, smoothness: float, violation: float) -> None:
        amp_cap = max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6)
        amp_norm = amp / amp_cap
        pw_norm = (pw - 0.06) / max(self._task.max_dbs_pulse_width - 0.06, 1e-6)
        stimulation_burden = (
            0.58 * (amp_norm ** 1.70)
            + 0.22 * (pw_norm ** 1.30)
            + 0.20 * (amp_norm * pw_norm)
        )
        # Frequency modulates side-effect accumulation rate (high frequency → more axonal fatigue)
        freq_se = _freq_side_effect_factor(freq)
        burden = (
            0.04 * stimulation_burden * self._profile.side_effect_sensitivity * freq_se
            + 0.02 * self._entrainment_state
            + 0.04 * smoothness
            + 0.20 * violation
        )
        recovery = self._profile.recovery_rate * (1.10 - 0.55 * amp_norm)
        self._side_effect_state = _clamp(self._side_effect_state * (1.0 - recovery) + burden)

        # Gamma ARV: over-stimulation biomarker — rises when side effects accumulate fast
        gamma_target = _clamp(
            0.60 * self._side_effect_state
            + 0.25 * (amp_norm ** 1.5)
            + 0.15 * max(0.0, (freq - 140.0) / 45.0)  # high-freq contribution
        )
        self._gamma_state = _clamp(0.70 * self._gamma_state + 0.30 * gamma_target)

        # Stim washout: wash-in when stimulating, wash-out when reducing
        washout_target = _clamp(0.80 * amp_norm + 0.20 * pw_norm)
        washout_rate = 0.30 if washout_target > self._stim_washout else 0.15
        self._stim_washout = _clamp(self._stim_washout + washout_rate * (washout_target - self._stim_washout))

        # Battery drain accumulation
        self._battery_drain = _battery_drain(amp, pw, freq)

    def _update_adaptation_state(self, amp: float, pw: float) -> None:
        """Model diminishing returns under prolonged aggressive stimulation."""
        amp_cap = max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6)
        amp_norm = amp / amp_cap
        pw_norm = (pw - 0.06) / max(self._task.max_dbs_pulse_width - 0.06, 1e-6)
        high_drive = _clamp(0.75 * amp_norm + 0.25 * pw_norm)
        adaptation_gain = 0.12 * max(high_drive - 0.58, 0.0) ** 1.35
        if self._profile.profile_id == "refractory":
            adaptation_gain *= 1.25
        elif self._profile.profile_id == "responsive":
            adaptation_gain *= 0.85

        recovery = 0.07 + 0.08 * max(0.42 - high_drive, 0.0)
        self._adaptation_state = _clamp(
            self._adaptation_state * (1.0 - recovery) + adaptation_gain
        )

    def _update_latent_state(
        self,
        current_base,
        next_base,
        effective_motor: float,
        tracking_accuracy: float,
        smoothness: float,
    ) -> None:
        base_beta = _clamp(next_base.beta_arv * self._profile.beta_scale * self._ep_beta_noise)
        base_tremor = _clamp(next_base.tremor_arv * self._profile.tremor_scale * self._ep_tremor_noise)
        base_semg = _clamp(next_base.semg_arv * self._profile.semg_scale * self._ep_semg_noise)
        base_force = _clamp(next_base.force_preserved * self._profile.force_scale * self._ep_force_noise)

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

        # effective_motor reflects actual motor output quality; better output → more force
        motor_quality = _clamp(abs(effective_motor) if abs(effective_motor) > 0.05 else tracking_accuracy)
        desired_force = _clamp(
            0.82 * base_force
            + 0.18 * motor_quality
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
        w = _REWARD_WEIGHTS.get(self._task.task_id, _REWARD_WEIGHTS["default"])
        amp_eff = 1.0 - (amp / max(self._task.max_dbs_amplitude * self._profile.max_amp_scale, 1e-6))
        pw_eff = 1.0 - ((pw - 0.06) / max(self._task.max_dbs_pulse_width - 0.06, 1e-6))
        efficiency = _clamp(0.65 * amp_eff + 0.35 * pw_eff)
        safety = _clamp(1.0 - (self._side_effect_state / max(self._task.max_side_effect_load, 1e-6)))
        reward = (
            w["force"] * self._force_state
            + w["tracking"] * tracking_accuracy
            + w["beta"] * (1.0 - self._beta_state)
            + w["tremor"] * (1.0 - self._tremor_state)
            + w["safety"] * safety
            + w["smoothness"] * (1.0 - smoothness)
            + w["efficiency"] * efficiency
            + self._long_horizon_shaping(tracking_accuracy)
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
                "adaptation_state": self._adaptation_state,
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
            gamma_arv=self._gamma_state,
            medication_phase=self._medication_phase,
            battery_drain_rate=self._battery_drain,
            stim_washout=self._stim_washout,
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

        # Sample per-episode noise to prevent trajectory memorization.
        # Gaussian with std=0.08 → ~68% of episodes within ±8% of base trajectory.
        self._ep_beta_noise = _clamp(1.0 + self._rng.gauss(0.0, 0.08), 0.75, 1.30)
        self._ep_tremor_noise = _clamp(1.0 + self._rng.gauss(0.0, 0.08), 0.75, 1.30)
        self._ep_force_noise = _clamp(1.0 + self._rng.gauss(0.0, 0.05), 0.82, 1.18)
        self._ep_semg_noise = _clamp(1.0 + self._rng.gauss(0.0, 0.06), 0.80, 1.20)

        self._init_latent_state()

        base = self._brain_window(self._task.start_step)
        tracking_accuracy = _clamp(1.0 - abs(self._target_output) / 2.0)
        self._initial_tracking_accuracy = tracking_accuracy
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

        amp, pw, freq, violation = self._clip_action(action)
        smoothness = self._smoothness_cost(amp, pw, freq)
        self._update_adaptation_state(amp, pw)
        adaptation_penalty = (
            0.45 * self._adaptation_state
            if self._task.task_id != "beta_suppression"
            else 0.25 * self._adaptation_state
        )
        # Frequency modulates beta-suppression efficiency (peak at ~130 Hz)
        freq_beta = _freq_beta_factor(freq)
        target_entrainment = _clamp(
            query_dbs_effect(self._brain, amp, pw)
            * self._profile.entrainment_scale
            * freq_beta
            * (1.0 - adaptation_penalty)
        )
        self._entrainment_state = _clamp(0.35 * self._entrainment_state + 0.65 * target_entrainment)

        # motor_command tracks target_output; disease state distorts the signal.
        motor_intent = _clamp(action.motor_command, -1.0, 1.0)
        effective_motor = self._apply_motor_distortion(motor_intent)
        task_error = abs(self._target_output - effective_motor)
        tracking_accuracy = _clamp(1.0 - task_error / 2.0)

        self._update_side_effects(amp, pw, freq, smoothness, violation)
        self._update_latent_state(current_base, next_base, effective_motor, tracking_accuracy, smoothness)

        # Advance medication phase (L-DOPA 4–6 hr cycle → ~0.5–1 full cycle per 100-step episode)
        self._medication_phase = _clamp(
            (self._med_phase_offset + self._local_step / max(self._task.n_steps, 1)) % 1.0
        )

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
            "adaptation_state": self._adaptation_state,
            "ground_truth_dbs_ma": next_base.dbs_amplitude_ma,
            "ground_truth_force": next_base.force_amplitude,
            "ground_truth_scheduler_class": next_base.scheduler_class,
            "ground_truth_beta_ctrl_error": next_base.beta_ctrl_error,
            "constraint_violation": violation,
            "dbs_frequency_hz": freq,
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
        self._prev_freq = freq
        return obs

    @property
    def state(self) -> State:
        return self._state
