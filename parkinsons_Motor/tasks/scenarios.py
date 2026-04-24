"""Scenario definitions for the benchmark task suite."""

from .base import DBSTask


TASK_BETA_SUPPRESSION = DBSTask(
    task_id="beta_suppression",
    description=(
        "Introductory early-phase stabilization on a responsive patient. The agent "
        "starts before peak symptom onset (step 5) and has 30 steps to establish "
        "clean, low-burden DBS control. A 1.5 mA amplitude ceiling gives enough "
        "headroom for active suppression without requiring perfect efficiency. "
        "Success requires maintaining beta below target and keeping side-effect load "
        "within budget — a constant high-amplitude policy will fail on efficiency."
    ),
    difficulty="easy",
    start_step=5,
    n_steps=30,
    max_dbs_amplitude=1.5,
    max_dbs_pulse_width=0.15,
    max_side_effect_load=0.55,
    target_force_preserved=0.78,
    target_beta_arv=0.26,
    target_tremor_arv=0.20,
    target_tracking_error=0.32,
    success_threshold=0.50,
    patient_profile_ids=("responsive",),
    target_output_range=(-0.20, 0.20),
)

TASK_TREMOR_CORRECTION = DBSTask(
    task_id="tremor_correction",
    description=(
        "Acute tremor rescue starting at symptom escalation (step 16). The agent "
        "must reverse tremor growth, recover force, and manage side-effect accumulation "
        "over 48 steps. A constant high-amplitude policy will exceed the side-effect "
        "budget; a zero-stimulation policy triggers hard under-treatment penalties. "
        "The agent must modulate stimulation — reduce when symptoms stabilize, push "
        "when they escalate — to balance therapeutic benefit against safety cost."
    ),
    difficulty="medium",
    start_step=16,
    n_steps=48,
    max_dbs_amplitude=1.8,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.60,
    target_force_preserved=0.64,
    target_beta_arv=0.28,
    target_tremor_arv=0.32,
    target_tracking_error=0.28,
    success_threshold=0.36,
    patient_profile_ids=("balanced", "responsive"),
    target_output_range=(-0.50, 0.50),
)

TASK_FULL_EPISODE = DBSTask(
    task_id="full_episode",
    description=(
        "Sustained closed-loop control over the full 100-step clinical episode. "
        "The agent must manage symptom progression from onset through peak and into "
        "recovery, handle cumulative side effects (budget 0.55), maintain tracking "
        "quality, and end with stable terminal state. Episode noise means each reset "
        "has a different disease trajectory — policies that memorize the Fleming "
        "trajectory will degrade across episodes."
    ),
    difficulty="hard",
    start_step=0,
    n_steps=100,
    max_dbs_amplitude=2.4,
    max_dbs_pulse_width=0.20,
    max_side_effect_load=0.55,
    target_force_preserved=0.60,
    target_beta_arv=0.28,
    target_tremor_arv=0.36,
    target_tracking_error=0.28,
    success_threshold=0.66,
    patient_profile_ids=("balanced", "responsive", "refractory"),
    target_output_range=(-0.65, 0.65),
)

TASK_FRAGILE_PATIENT = DBSTask(
    task_id="fragile_patient",
    description=(
        "Safety-constrained control on a fragile patient. The side-effect budget is "
        "tight (0.26) and the patient's sensitivity is 1.4× — aggressive stimulation "
        "quickly violates safety constraints while under-stimulation leaves tremor "
        "uncontrolled. The agent must find and hold a precise therapeutic window "
        "across 64 steps."
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
)

TASK_REFRACTORY_PATIENT = DBSTask(
    task_id="refractory_patient",
    description=(
        "Weaker-response patient (entrainment 0.88×, tremor_responsiveness 0.88×) "
        "with faster symptom progression. Brute-force high amplitude accumulates "
        "side effects without proportional benefit due to adaptation. The agent must "
        "use pulsed stimulation patterns — moderate dose, rests when stable, push "
        "during escalation — to extract therapeutic value from a refractory system."
    ),
    difficulty="expert",
    start_step=0,
    n_steps=100,
    max_dbs_amplitude=2.2,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.48,
    target_force_preserved=0.58,
    target_beta_arv=0.32,
    target_tremor_arv=0.38,
    target_tracking_error=0.32,
    success_threshold=0.42,
    patient_profile_ids=("refractory",),
    target_output_range=(-0.65, 0.65),
)

TASK_PERSONALIZATION_GENERALIZATION = DBSTask(
    task_id="personalization_generalization",
    description=(
        "Held-out generalization benchmark across all four patient profiles. The "
        "profile is revealed in metadata at reset but the agent has no prior episode "
        "history for that patient. Success requires policies that degrade gracefully "
        "across fragile, balanced, responsive, and refractory patients rather than "
        "overfitting to any single profile."
    ),
    difficulty="expert",
    start_step=10,
    n_steps=72,
    max_dbs_amplitude=1.9,
    max_dbs_pulse_width=0.18,
    max_side_effect_load=0.40,
    target_force_preserved=0.62,
    target_beta_arv=0.28,
    target_tremor_arv=0.32,
    target_tracking_error=0.28,
    success_threshold=0.45,
    patient_profile_ids=("balanced", "responsive", "fragile", "refractory"),
    target_output_range=(-0.60, 0.60),
)
