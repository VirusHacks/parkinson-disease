"""Scenario definitions for the benchmark task suite.

Public tasks (easy/medium/hard) live in their own modules. This file collects:
  * Friendly aliases for the public tasks (legacy IDs).
  * Expert / extended tasks: fragile_patient, refractory_patient,
    personalization_generalization, exercise_bout, medication_interaction,
    nocturnal_transition, surgical_followup.

Each expert task is opt-in via its task_id and exercises a different real
clinical scenario. They share the public action/observation spaces so any
agent that runs on the public tasks can be evaluated here directly.
"""

from .base import DBSTask
from .easy import TASK_EASY
from .exercise_bout import TASK_EXERCISE_BOUT
from .hard import TASK_HARD
from .medication_interaction import TASK_MEDICATION_INTERACTION
from .medium import TASK_MEDIUM
from .nocturnal_transition import TASK_NOCTURNAL_TRANSITION
from .surgical_followup import TASK_SURGICAL_FOLLOWUP


# Legacy aliases.
TASK_BETA_SUPPRESSION = TASK_EASY
TASK_TREMOR_CORRECTION = TASK_MEDIUM
TASK_FULL_EPISODE = TASK_HARD


TASK_FRAGILE_PATIENT = DBSTask(
    task_id="fragile_patient",
    name="Fragile Patient",
    description=(
        "Safety-constrained control on a fragile patient. The side-effect "
        "budget is tight (0.26) and the patient's sensitivity is 1.4× — "
        "aggressive stimulation quickly violates safety constraints while "
        "under-stimulation leaves tremor uncontrolled. The agent must find and "
        "hold a precise therapeutic window across 64 steps."
    ),
    difficulty="expert",
    start_step=12,
    n_steps=64,
    max_dbs_amplitude=1.4,
    max_dbs_pulse_width=0.16,
    max_side_effect_load=0.26,
    target_force_preserved=0.62,
    target_beta_arv=0.24,
    target_tremor_arv=0.28,
    target_tracking_error=0.26,
    success_threshold=0.44,
    patient_profile_ids=("fragile",),
    target_output_range=(-0.55, 0.55),
    event_profile=None,
    sensor_noise_std=0.030,
    schedule_id=None,
)

TASK_REFRACTORY_PATIENT = DBSTask(
    task_id="refractory_patient",
    name="Refractory Patient",
    description=(
        "Weaker-response patient (entrainment 0.88×, tremor_responsiveness "
        "0.88×) with faster symptom progression and recurring tachyphylaxis. "
        "Brute-force high amplitude accumulates side effects without "
        "proportional benefit. The agent must use pulsed stimulation patterns "
        "— moderate dose, rests when stable, push during escalation — to "
        "extract therapeutic value from a refractory system across a 120-step "
        "episode."
    ),
    difficulty="expert",
    start_step=0,
    n_steps=120,
    max_dbs_amplitude=2.2,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.45,
    target_force_preserved=0.58,
    target_beta_arv=0.30,
    target_tremor_arv=0.36,
    target_tracking_error=0.30,
    success_threshold=0.46,
    patient_profile_ids=("refractory",),
    target_output_range=(-0.65, 0.65),
    event_profile="long_horizon",
    sensor_noise_std=0.045,
    schedule_id=None,
)

TASK_PERSONALIZATION_GENERALIZATION = DBSTask(
    task_id="personalization_generalization",
    name="Generalization Challenge",
    description=(
        "Held-out generalization benchmark across all four patient profiles. "
        "The profile is revealed in metadata at reset but the agent has no "
        "prior episode history for that patient. Stochastic events may fire "
        "any time, so policies must degrade gracefully across fragile, "
        "balanced, responsive, and refractory patients rather than overfitting "
        "to any single profile."
    ),
    difficulty="expert",
    start_step=10,
    n_steps=90,
    max_dbs_amplitude=1.9,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.40,
    target_force_preserved=0.62,
    target_beta_arv=0.28,
    target_tremor_arv=0.32,
    target_tracking_error=0.28,
    success_threshold=0.50,
    patient_profile_ids=("balanced", "responsive", "fragile", "refractory"),
    target_output_range=(-0.60, 0.60),
    event_profile="rescue",
    sensor_noise_std=0.040,
    schedule_id=None,
)
