"""
Patient profile definitions for the Parkinson's Motor environment.

Profiles let the environment vary disease severity, stimulation response,
side-effect sensitivity, and motor noise while staying anchored to the
Fleming-derived baseline trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PatientProfile:
    """Static patient-specific coefficients used by the online controller."""

    profile_id: str
    description: str
    beta_scale: float
    tremor_scale: float
    force_scale: float
    semg_scale: float
    progression_scale: float
    entrainment_scale: float
    beta_responsiveness: float
    tremor_responsiveness: float
    side_effect_sensitivity: float
    recovery_rate: float
    motor_noise_scale: float
    fatigue_rate: float
    max_amp_scale: float


PROFILE_BALANCED = PatientProfile(
    profile_id="balanced",
    description="Nominal patient close to the calibrated Fleming trajectory.",
    beta_scale=1.00,
    tremor_scale=1.00,
    force_scale=1.00,
    semg_scale=1.00,
    progression_scale=1.00,
    entrainment_scale=1.00,
    beta_responsiveness=1.00,
    tremor_responsiveness=1.00,
    side_effect_sensitivity=1.00,
    recovery_rate=0.07,
    motor_noise_scale=1.00,
    fatigue_rate=0.030,
    max_amp_scale=1.00,
)

PROFILE_FRAGILE = PatientProfile(
    profile_id="fragile",
    description="Sensitive patient with stronger side effects and lower force reserve.",
    beta_scale=1.05,
    tremor_scale=1.10,
    force_scale=0.92,
    semg_scale=1.05,
    progression_scale=1.05,
    entrainment_scale=0.95,
    beta_responsiveness=0.95,
    tremor_responsiveness=0.92,
    side_effect_sensitivity=1.40,
    recovery_rate=0.05,
    motor_noise_scale=1.10,
    fatigue_rate=0.040,
    max_amp_scale=0.95,
)

PROFILE_RESPONSIVE = PatientProfile(
    profile_id="responsive",
    description="Patient whose symptoms respond well to DBS but still needs adaptive control.",
    beta_scale=0.98,
    tremor_scale=0.96,
    force_scale=1.04,
    semg_scale=0.98,
    progression_scale=0.95,
    entrainment_scale=1.08,
    beta_responsiveness=1.08,
    tremor_responsiveness=1.06,
    side_effect_sensitivity=0.92,
    recovery_rate=0.08,
    motor_noise_scale=0.95,
    fatigue_rate=0.026,
    max_amp_scale=1.05,
)

PROFILE_REFRACTORY = PatientProfile(
    profile_id="refractory",
    description="Harder patient with weaker entrainment response and faster symptom growth.",
    beta_scale=1.08,
    tremor_scale=1.15,
    force_scale=0.95,
    semg_scale=1.08,
    progression_scale=1.10,
    entrainment_scale=0.88,
    beta_responsiveness=0.90,
    tremor_responsiveness=0.88,
    side_effect_sensitivity=1.12,
    recovery_rate=0.06,
    motor_noise_scale=1.15,
    fatigue_rate=0.038,
    max_amp_scale=1.00,
)


PROFILE_REGISTRY: Dict[str, PatientProfile] = {
    p.profile_id: p
    for p in (
        PROFILE_BALANCED,
        PROFILE_FRAGILE,
        PROFILE_RESPONSIVE,
        PROFILE_REFRACTORY,
    )
}


def get_profile(profile_id: str) -> PatientProfile:
    """Look up a patient profile by id."""
    try:
        return PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown patient profile {profile_id!r}. Available: {list(PROFILE_REGISTRY)}"
        ) from exc


def get_profiles(profile_ids: Iterable[str]) -> List[PatientProfile]:
    """Return a list of profiles preserving the provided order."""
    return [get_profile(pid) for pid in profile_ids]
