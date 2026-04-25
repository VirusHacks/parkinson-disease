"""
Stochastic clinical events for the Parkinson's Motor environment.

Real closed-loop DBS management is repeatedly disturbed by physiological and
hardware events that no deterministic deterioration curve captures:

* Dyskinesia spikes from accumulated levodopa + DBS interaction.
* Tachyphylaxis (tolerance) after sustained high-amplitude stimulation.
* Lead impedance surges that drop delivered current without operator action.
* Off-medication crises when the L-DOPA cycle reaches a deep trough.
* Voluntary motor surges (a writing task, opening a jar) demanding more force.

This module defines a small, deterministic-given-seed event scheduler that the
environment polls each step. Each task can opt into a named *event profile*
that controls which events can fire and with what probability, intensity and
duration. The scheduler is rng-driven, so a fixed task+seed pair always yields
the same event timeline — required for reproducible benchmarking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Event types and active state
# ---------------------------------------------------------------------------

EVENT_DYSKINESIA_SPIKE = "dyskinesia_spike"
EVENT_TACHYPHYLAXIS = "tachyphylaxis"
EVENT_IMPEDANCE_SURGE = "impedance_surge"
EVENT_OFF_MED_CRISIS = "off_med_crisis"
EVENT_MOTOR_SURGE = "motor_surge"
EVENT_SECOND_DETERIORATION = "second_deterioration"


@dataclass
class ActiveEvent:
    """A single event currently in effect at the active step."""

    event_type: str
    start_step: int
    end_step: int
    intensity: float  # in [0, 1]; profile-dependent meaning

    def is_active(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


@dataclass
class EventEffects:
    """Aggregated multiplicative / additive perturbations applied to one step.

    The environment composes these with the baseline calibrated dynamics. All
    fields default to a no-op so deterministic tasks pay no cost.
    """

    side_effect_burden_mult: float = 1.0   # >1 accelerates side-effect accumulation
    tremor_drive_add: float = 0.0           # additive perturbation to tremor target
    beta_drive_add: float = 0.0             # additive perturbation to beta target
    entrainment_mult: float = 1.0           # <1 reduces effective entrainment
    delivered_amp_mult: float = 1.0         # <1 reduces delivered current (impedance)
    target_output_override: Optional[float] = None  # forces motor target this step
    target_force_floor: float = 0.0         # raise force-preserved expectation
    active_event_ids: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Event profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventConfig:
    """Probabilistic spec for a single event type within a profile."""

    event_type: str
    probability: float                  # per-episode probability of triggering at all
    earliest_step_frac: float           # earliest fire window as fraction of n_steps
    latest_step_frac: float             # latest fire window as fraction of n_steps
    duration_steps_range: Tuple[int, int]
    intensity_range: Tuple[float, float]
    max_occurrences: int = 1


_PROFILES: Dict[str, Tuple[EventConfig, ...]] = {
    # Medium task: rescue phase — a manageable deterioration the agent must reverse.
    # Clinically this is one problem at a time: either a second deterioration wave
    # OR mild dyskinesia risk from early over-treatment. NOT a multi-crisis overlap.
    "rescue": (
        EventConfig(
            event_type=EVENT_SECOND_DETERIORATION,
            probability=0.55,
            earliest_step_frac=0.55,
            latest_step_frac=0.80,
            duration_steps_range=(5, 8),
            intensity_range=(0.10, 0.18),  # mild — manageable with correct response
        ),
        EventConfig(
            event_type=EVENT_DYSKINESIA_SPIKE,
            probability=0.30,
            earliest_step_frac=0.40,
            latest_step_frac=0.85,
            duration_steps_range=(3, 6),
            intensity_range=(0.08, 0.16),  # mild — medium is not a full dyskinesia crisis
        ),
    ),
    # Hard task: long horizon — events are near-guaranteed and stronger, forcing
    # the agent to manage overlapping crises across a 150-step episode.
    "long_horizon": (
        EventConfig(
            event_type=EVENT_DYSKINESIA_SPIKE,
            probability=0.80,
            earliest_step_frac=0.20,
            latest_step_frac=0.85,
            duration_steps_range=(6, 11),
            intensity_range=(0.22, 0.40),
            max_occurrences=2,
        ),
        EventConfig(
            event_type=EVENT_TACHYPHYLAXIS,
            probability=0.82,
            earliest_step_frac=0.40,
            latest_step_frac=0.78,
            duration_steps_range=(12, 20),
            intensity_range=(0.20, 0.30),
        ),
        EventConfig(
            event_type=EVENT_OFF_MED_CRISIS,
            probability=0.75,
            earliest_step_frac=0.28,
            latest_step_frac=0.72,
            duration_steps_range=(10, 15),
            intensity_range=(0.25, 0.45),
        ),
        EventConfig(
            event_type=EVENT_MOTOR_SURGE,
            probability=0.65,
            earliest_step_frac=0.15,
            latest_step_frac=0.85,
            duration_steps_range=(5, 9),
            intensity_range=(0.60, 0.90),
            max_occurrences=2,
        ),
    ),
    # Exercise bout: heavy motor-surge + moderate side-effect pressure.
    "exercise": (
        EventConfig(
            event_type=EVENT_MOTOR_SURGE,
            probability=1.00,
            earliest_step_frac=0.10,
            latest_step_frac=0.30,
            duration_steps_range=(15, 22),
            intensity_range=(0.70, 0.95),
        ),
        EventConfig(
            event_type=EVENT_DYSKINESIA_SPIKE,
            probability=0.40,
            earliest_step_frac=0.55,
            latest_step_frac=0.85,
            duration_steps_range=(4, 7),
            intensity_range=(0.20, 0.30),
        ),
    ),
    # Medication interaction: phase-coupled crises.
    "medication": (
        EventConfig(
            event_type=EVENT_OFF_MED_CRISIS,
            probability=1.00,
            earliest_step_frac=0.30,
            latest_step_frac=0.45,
            duration_steps_range=(10, 14),
            intensity_range=(0.28, 0.42),
        ),
        EventConfig(
            event_type=EVENT_DYSKINESIA_SPIKE,
            probability=0.65,
            earliest_step_frac=0.60,
            latest_step_frac=0.85,
            duration_steps_range=(5, 9),
            intensity_range=(0.22, 0.36),
        ),
    ),
    # Nocturnal: low disturbance, dominated by schedule taper rather than events.
    "nocturnal": (
        EventConfig(
            event_type=EVENT_MOTOR_SURGE,
            probability=0.30,
            earliest_step_frac=0.05,
            latest_step_frac=0.25,
            duration_steps_range=(4, 7),
            intensity_range=(0.55, 0.80),
        ),
        EventConfig(
            event_type=EVENT_DYSKINESIA_SPIKE,
            probability=0.25,
            earliest_step_frac=0.35,
            latest_step_frac=0.55,
            duration_steps_range=(4, 6),
            intensity_range=(0.18, 0.28),
        ),
    ),
    # Surgical follow-up: rare impedance surges as swelling resolves.
    "surgical": (
        EventConfig(
            event_type=EVENT_IMPEDANCE_SURGE,
            probability=0.70,
            earliest_step_frac=0.30,
            latest_step_frac=0.70,
            duration_steps_range=(6, 12),
            intensity_range=(0.20, 0.38),
            max_occurrences=2,
        ),
    ),
}


def list_profiles() -> List[str]:
    """Return all configured event profile ids."""
    return list(_PROFILES.keys())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class EventScheduler:
    """Deterministic-given-seed event timeline for one episode.

    The scheduler decides the entire episode's event timeline at construction
    time using the supplied rng. The environment then queries `effects(step)`
    at each step to obtain the aggregated perturbation. This separation keeps
    the per-step hot path branch-free and makes event traces inspectable for
    debugging or curriculum design.
    """

    def __init__(
        self,
        profile_id: Optional[str],
        n_steps: int,
        rng: random.Random,
    ) -> None:
        self._profile_id = profile_id
        self._n_steps = max(1, n_steps)
        self._rng = rng
        self._events: List[ActiveEvent] = []
        self._tachyphylaxis_intensity: float = 0.0
        if profile_id is not None and profile_id in _PROFILES:
            self._schedule(_PROFILES[profile_id])

    # ---- scheduling -------------------------------------------------------

    def _schedule(self, configs: Tuple[EventConfig, ...]) -> None:
        for cfg in configs:
            occurrences = 0
            attempts = 0
            while occurrences < cfg.max_occurrences and attempts < cfg.max_occurrences * 3:
                attempts += 1
                if self._rng.random() > cfg.probability:
                    if occurrences == 0:
                        break
                    continue
                start, end = self._sample_window(cfg)
                if start is None or end is None:
                    break
                intensity = self._rng.uniform(*cfg.intensity_range)
                self._events.append(
                    ActiveEvent(
                        event_type=cfg.event_type,
                        start_step=start,
                        end_step=end,
                        intensity=intensity,
                    )
                )
                if cfg.event_type == EVENT_TACHYPHYLAXIS:
                    self._tachyphylaxis_intensity = max(
                        self._tachyphylaxis_intensity, intensity
                    )
                occurrences += 1

    def _sample_window(
        self, cfg: EventConfig
    ) -> Tuple[Optional[int], Optional[int]]:
        earliest = max(0, int(cfg.earliest_step_frac * self._n_steps))
        latest = min(self._n_steps - 1, int(cfg.latest_step_frac * self._n_steps))
        if latest <= earliest:
            return None, None
        start = self._rng.randint(earliest, latest)
        duration = self._rng.randint(*cfg.duration_steps_range)
        end = min(self._n_steps, start + duration)
        if end <= start:
            return None, None
        return start, end

    # ---- query ------------------------------------------------------------

    def effects(
        self,
        step: int,
        recent_amp_norm: float = 0.0,
        medication_phase: float = 0.5,
    ) -> EventEffects:
        """Aggregate the perturbation effects active at `step`."""
        agg = EventEffects()
        active_ids: List[str] = []
        for ev in self._events:
            if not ev.is_active(step):
                continue
            active_ids.append(ev.event_type)
            self._apply_event(ev, agg, recent_amp_norm, medication_phase)
        if active_ids:
            agg.active_event_ids = tuple(active_ids)
        return agg

    def _apply_event(
        self,
        ev: ActiveEvent,
        agg: EventEffects,
        recent_amp_norm: float,
        medication_phase: float,
    ) -> None:
        if ev.event_type == EVENT_DYSKINESIA_SPIKE:
            # Dyskinesia is a genuine overtreatment emergency — side-effect burden
            # increases sharply; the agent cannot simply push through with more amplitude.
            agg.side_effect_burden_mult *= 1.0 + 1.65 * ev.intensity
            agg.beta_drive_add += -0.04 * ev.intensity
            agg.tremor_drive_add += 0.04 * ev.intensity
        elif ev.event_type == EVENT_TACHYPHYLAXIS:
            # Clinical tolerance: sustained high-amplitude stimulation causes axonal
            # adaptation; the same delivered current produces less therapeutic effect.
            # At max intensity (0.30), entrainment drops to 40 % of its pre-event value.
            agg.entrainment_mult *= max(0.40, 1.0 - 2.0 * ev.intensity)
        elif ev.event_type == EVENT_IMPEDANCE_SURGE:
            agg.delivered_amp_mult *= max(0.55, 1.0 - 0.55 * ev.intensity)
            agg.entrainment_mult *= max(0.65, 1.0 - 0.30 * ev.intensity)
        elif ev.event_type == EVENT_OFF_MED_CRISIS:
            # L-DOPA trough: pathological beta activity surges, tremor worsens.
            # Clinically this can double beta power (0.28 * 0.45 ≈ 0.13 drive_add at max).
            phase_floor = max(0.0, 0.30 - medication_phase)
            agg.beta_drive_add += 0.28 * ev.intensity * (1.0 + 0.8 * phase_floor)
            agg.tremor_drive_add += 0.22 * ev.intensity * (1.0 + 0.8 * phase_floor)
        elif ev.event_type == EVENT_MOTOR_SURGE:
            sign = 1.0 if (ev.start_step % 2 == 0) else -1.0
            agg.target_output_override = max(-0.95, min(0.95, sign * ev.intensity))
            agg.target_force_floor = max(agg.target_force_floor, ev.intensity * 0.65)
        elif ev.event_type == EVENT_SECOND_DETERIORATION:
            agg.beta_drive_add += 0.10 * ev.intensity
            agg.tremor_drive_add += 0.12 * ev.intensity

    # ---- introspection ----------------------------------------------------

    def schedule_summary(self) -> List[Dict[str, Any]]:
        """Return a JSON-serialisable list describing the event timeline."""
        return [
            {
                "event_type": ev.event_type,
                "start_step": ev.start_step,
                "end_step": ev.end_step,
                "intensity": round(ev.intensity, 4),
            }
            for ev in self._events
        ]

    @property
    def profile_id(self) -> Optional[str]:
        return self._profile_id


# ---------------------------------------------------------------------------
# Time-varying setpoint schedules
# ---------------------------------------------------------------------------


def schedule_overrides(
    schedule_id: Optional[str],
    step: int,
    n_steps: int,
) -> Dict[str, float]:
    """Return per-step setpoint overrides for time-varying tasks.

    Returned keys are optional and only set when this schedule modifies them:
      * target_force_floor  — minimum acceptable force_preserved
      * tremor_target_mult  — multiplier on task.target_tremor_arv
      * beta_target_mult    — multiplier on task.target_beta_arv
      * amplitude_ceiling_mult — extra amplitude ceiling beyond task max
                                  (used during e.g. surgical microlesion window)
    """
    if not schedule_id or n_steps <= 0:
        return {}
    progress = step / max(n_steps - 1, 1)

    if schedule_id == "nocturnal":
        # Phase 0: 0.00–0.40 awake (full demands)
        # Phase 1: 0.40–0.65 wind-down (force expectation drops)
        # Phase 2: 0.65–1.00 sleep (tighter beta target, lower force expectation)
        if progress < 0.40:
            return {}
        if progress < 0.65:
            blend = (progress - 0.40) / 0.25
            return {
                "tremor_target_mult": 1.0 - 0.10 * blend,
                "beta_target_mult": 1.0 - 0.15 * blend,
            }
        blend = min(1.0, (progress - 0.65) / 0.35)
        return {
            "tremor_target_mult": max(0.70, 0.90 - 0.20 * blend),
            "beta_target_mult": max(0.65, 0.85 - 0.20 * blend),
        }

    if schedule_id == "surgical_microlesion":
        # Microlesion window: amplitude ceiling drops to 0.6 mA for the first
        # 25% of the episode, then ramps linearly back to the task max.
        if progress < 0.25:
            return {"amplitude_ceiling_abs_ma": 0.60}
        if progress < 0.40:
            blend = (progress - 0.25) / 0.15
            return {"amplitude_ceiling_abs_ma": 0.60 + (1.0 - 0.60) * blend}
        return {}

    return {}
