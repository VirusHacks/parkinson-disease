"""Medication-interaction task — coupled DBS + L-DOPA dynamics.

Clinical scenario
-----------------
A guaranteed off-medication crisis occurs in the first half of the episode as
levodopa wears off. Symptoms worsen quickly — the agent must pre-empt with
extra DBS while the medication phase is still falling. After the crisis, a
likely dyskinesia spike (~65%) marks the medication coming back on; brute-force
DBS in that window over-treats the patient.

Why it is hard
--------------
* Phase-aware control is required — `medication_phase` is the dominant signal.
* The wrong response during the dyskinesia window costs more safety than
  under-stimulation during the off-med crisis.
"""

from __future__ import annotations

from .base import DBSTask


def get_medication_interaction_task() -> DBSTask:
    return DBSTask(
        task_id="medication_interaction",
        name="Medication Interaction",
        description=(
            "100-step session coupled to the L-DOPA cycle. An off-medication "
            "crisis is guaranteed in the first half — symptoms worsen as the "
            "drug wears off and the agent must compensate proactively. A "
            "dyskinesia spike commonly follows in the second half as medication "
            "returns. Phase-aware control wins this task; reactive policies "
            "either under-respond to the off-med crisis or over-treat during "
            "the dyskinesia window. Models the well-documented clinical "
            "interaction between DBS and L-DOPA dynamics."
        ),
        difficulty="expert",
        start_step=4,
        n_steps=100,
        max_dbs_amplitude=2.2,
        max_dbs_pulse_width=0.18,
        max_side_effect_load=0.42,
        target_force_preserved=0.62,
        target_beta_arv=0.26,
        target_tremor_arv=0.32,
        target_tracking_error=0.28,
        success_threshold=0.50,
        patient_profile_ids=("balanced", "responsive", "fragile"),
        target_output_range=(-0.55, 0.55),
        event_profile="medication",
        sensor_noise_std=0.045,
        schedule_id=None,
    )


TASK_MEDICATION_INTERACTION = get_medication_interaction_task()
