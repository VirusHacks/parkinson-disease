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

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ParkinsonsMotorAction(Action):
    """
    Agent action at each 20ms timestep.

    The agent controls three DBS parameters (the standard clinical triad) plus a motor command:

      1. dbs_amplitude [0, 5 mA] — stimulation current. Drives cortical entrainment.
      2. dbs_pulse_width [60, 200 μs] — pulse duration. Controls spatial spread (more neurons recruited).
      3. dbs_frequency [60, 185 Hz] — stimulation frequency. Beta suppression peaks at ~130 Hz;
         very high frequency increases side effects; very low frequency favors tremor over beta.
      4. motor_command [-1, 1] — set to target_output (visible in obs) each step. The environment
         degrades this proportional to beta_arv and tremor_arv — better DBS → less degradation.

    Clinical basis: the three-parameter DBS action space matches the tunable parameters on
    commercial implantable pulse generators (Medtronic Activa, Abbott Infinity, Boston Scientific
    Vercise). Frequency was fixed at 130 Hz in early systems but is now actively tuned in
    adaptive DBS (aDBS) research (Little et al. 2016; Tinkhauser et al. 2017).

    Strategy: set motor_command = target_output, start at (1.0–1.5 mA, 0.10 ms, 130 Hz),
    increase amplitude when beta_trend > 0, reduce when side_effect_rate > 0.
    """

    motor_command: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Voluntary motor intent in [-1, 1]. Should match target_output in the observation. "
            "Parkinsonian distortion degrades this proportionally to beta_arv and tremor_arv. "
            "Better DBS suppression → less distortion → lower task_error → higher tracking_accuracy."
        ),
    )
    dbs_amplitude: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description=(
            "DBS stimulation current in mA [0, 5]. Clipped to task ceiling at runtime. "
            "Primary driver of cortical axon entrainment (via 12×15 lookup table). "
            "Higher amplitude → stronger beta and tremor suppression but faster side_effect_load "
            "accumulation. Sustained max amplitude causes adaptation (diminishing returns)."
        ),
    )
    dbs_pulse_width: float = Field(
        default=0.06,
        ge=0.06,
        le=0.20,
        description=(
            "DBS pulse duration in ms [0.06, 0.20] (equivalent to 60–200 μs). "
            "Controls spatial spread: wider pulses recruit more axons beyond the target nucleus. "
            "Jointly determines entrainment with amplitude via the parameter sweep table. "
            "Wider pulses increase side effects faster than narrow pulses at the same amplitude."
        ),
    )
    dbs_frequency: float = Field(
        default=130.0,
        ge=60.0,
        le=185.0,
        description=(
            "DBS pulse train frequency in Hz [60, 185]. "
            "Beta oscillation suppression peaks at ~130 Hz (standard clinical default). "
            "Frequencies below 80 Hz preferentially affect tremor over beta. "
            "Frequencies above 160 Hz increase side-effect load without proportional entrainment gain. "
            "Clinical range per FDA-approved IPG specs: 2–185 Hz; therapeutic range: 60–185 Hz."
        ),
    )
    task_id: str = Field(
        default="",
        description="Task ID to load on reset. Empty string keeps the current task.",
    )


class ParkinsonsMotorObservation(Observation):
    """
    Clinically oriented observation at each timestep.

    Key fields for DBS tuning:
      - beta_arv, tremor_arv: primary disease signals to suppress (targets in task spec)
      - side_effect_load: cumulative safety budget — keep below task max or score drops sharply
      - beta_trend, tremor_trend: direction of change (negative = improving)
      - dbs_entrainment: fraction of cortical axons entrained by current DBS settings

    Key fields for motor tracking:
      - target_output: the motor goal this episode (set motor_command to match this)
      - task_error: |target_output - effective_motor_output| — minimize this
      - tracking_accuracy: 1 - task_error/2, normalized to [0,1]

    Derived convenience fields (not independent — computed from primary signals):
      - disease_severity = 0.55*tremor_arv + 0.45*beta_arv
      - beta_suppression = 1 - beta_arv
    """

    # Primary neural signals
    beta_arv: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="STN beta-band oscillation amplitude (0=suppressed/healthy, 1=peak Parkinson's). Primary DBS target.",
    )
    tremor_arv: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Tremor envelope (0=no tremor, 1=maximal). Grows from baseline and is suppressed by DBS entrainment.",
    )
    semg_arv: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Surface EMG envelope. Reflects combined beta and tremor state.",
    )
    force_amplitude: float = Field(
        default=0.0, ge=0.0,
        description="Raw muscle force in mN. Healthy baseline ≈ 59,752 mN.",
    )
    force_preserved: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of healthy force preserved (0=total loss, 1=fully healthy). Primary graded outcome.",
    )

    # Derived summaries (computed from primary signals — not independent)
    disease_severity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Composite disease index = 0.55*tremor_arv + 0.45*beta_arv. Convenience aggregate.",
    )
    beta_suppression: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="DBS suppression fraction = 1 - beta_arv. Convenience inverse of beta_arv.",
    )
    beta_trend: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="5-step delta of beta_arv. Negative = improving (beta falling). Use to anticipate trajectory.",
    )
    tremor_trend: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="5-step delta of tremor_arv. Negative = improving. Use to decide when to modulate amplitude.",
    )
    side_effect_rate: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="5-step delta of side_effect_load. Positive = load growing (reduce amplitude). Negative = recovering.",
    )

    # Motor tracking
    target_output: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Task-defined motor target. Set motor_command to this value every step.",
    )
    effective_motor_output: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Actual motor output after Parkinsonian distortion. Approaches target_output as DBS improves.",
    )
    task_error: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="|target_output - effective_motor_output|. Minimize by setting motor_command=target_output and tuning DBS.",
    )
    tracking_accuracy: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1 - task_error/2. Normalized tracking quality in [0,1]. Higher is better.",
    )

    # DBS device state
    dbs_amplitude_ma: float = Field(
        default=0.0, ge=0.0,
        description="DBS amplitude delivered last step in mA (after task clipping).",
    )
    dbs_pulse_width_ms: float = Field(
        default=0.06, ge=0.06, le=0.20,
        description="DBS pulse width delivered last step in ms.",
    )
    dbs_entrainment: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of cortical axons entrained (0=no effect, 1=full entrainment). One-step lag from action.",
    )
    recent_dbs_avg_ma: float = Field(
        default=0.0, ge=0.0,
        description="5-step rolling mean of delivered amplitude. Use to track accumulated stimulation burden.",
    )
    recent_dbs_avg_pw_ms: float = Field(
        default=0.06, ge=0.06, le=0.20,
        description="5-step rolling mean of delivered pulse width.",
    )
    side_effect_load: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Cumulative DBS side-effect proxy. Must stay below task max_side_effect_load or safety score collapses. "
            "Decreases slowly during low-amplitude periods (recovery_rate ≈ 0.07/step)."
        ),
    )
    action_smoothness_cost: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Normalized cost for abrupt amplitude/pulse-width changes. Smooth adjustments score better.",
    )
    dbs_constraint_violation: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction by which action exceeded task amplitude or pulse-width cap. Avoid — incurs hard penalty.",
    )

    # Extended physiological signals (real closed-loop DBS devices observe these)
    gamma_arv: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "High-gamma band (60–90 Hz) LFP power, normalized. Elevated gamma indicates "
            "over-stimulation — the brain's response to excessive DBS drive. "
            "Clinically used as a real-time over-stimulation biomarker (Kühn et al. 2008). "
            "Correlated with side_effect_load but measured independently from a different frequency band."
        ),
    )
    medication_phase: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Normalized L-DOPA medication cycle phase [0, 1]. Parkinson's patients on oral "
            "levodopa experience 4–6 hour on/off cycles (Nutt & Holford 1996). "
            "0.0 = medication trough (worst motor state, DBS must compensate more). "
            "1.0 = medication peak (best baseline, less DBS needed). "
            "Agents should modulate DBS amplitude inversely with medication_phase."
        ),
    )
    battery_drain_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Normalized instantaneous DBS battery consumption rate [0, 1]. "
            "Proportional to amplitude × pulse_width × frequency (charge per second delivered). "
            "Implanted pulse generators (IPGs) have 5–10 year battery life — high drain "
            "requires surgical replacement. Efficiency-conscious agents minimize this. "
            "Formula: (amp/5.0)^1.2 × (pw_norm)^0.8 × (freq_norm)^0.6, normalized to [0,1]."
        ),
    )
    stim_washout: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "DBS wash-in/wash-out state [0, 1]. Reflects how much stimulation effect has "
            "accumulated in the neural circuit. Ramps up over 3–5 steps when DBS is on, "
            "decays over 5–10 steps when DBS is reduced. Predicts upcoming entrainment "
            "even before it appears in dbs_entrainment (useful for proactive control)."
        ),
    )

    # Timing and task info
    sim_time_s: float = Field(default=0.0, ge=0.0, description="Simulation time in seconds (10.02s–12.00s range).")
    task_id: str = Field(default="hard", description="Active task identifier.")
    grader_score: float = Field(default=-1.0, description="Final deterministic score in [0,1]. -1.0 until episode end.")
    episode_success: bool = Field(default=False, description="True if grader_score >= task success_threshold.")

    # Diagnostic surfaces (the OpenEnv envelope strips obs.metadata, so anything
    # we want clients to see at evaluation time has to be a typed field here).
    grader_components: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-component grader scores (force, beta, tremor, tracking, safety, "
            "smoothness, efficiency, terminal_stability, recovery, overall). "
            "Empty until the episode ends and the deterministic grader runs."
        ),
    )
    active_events: List[str] = Field(
        default_factory=list,
        description="Event ids active on the current step (e.g. ['dyskinesia_spike']).",
    )
    event_schedule_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-episode event firing plan, populated on reset. "
            "Each entry: {event_id, start_step, end_step, intensity}. "
            "Same value is echoed on every step so late-joining clients can recover it."
        ),
    )
