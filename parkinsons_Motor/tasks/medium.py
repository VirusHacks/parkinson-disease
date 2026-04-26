"""Medium task - active rescue with stochastic second wave.

Clinical intent: tremor rescue during a worsening episode, plus a real chance
that the patient deteriorates a second time after the first stabilization.
This breaks template policies that assume "ramp once, then taper". The agent
must keep enough safety budget for a possible second push and stay reactive.
"""

from __future__ import annotations

from .base import DBSTask


def get_medium_task() -> DBSTask:
    """Return the public medium benchmark task."""
    return DBSTask(
        task_id="medium",
        name="Rescue Phase",
        description=(
            "Active tremor rescue on a worsening patient over 60 steps. The "
            "patient enters mid-deterioration, demanding a corrective response "
            "that restores function while respecting safety and fatigue. Unlike "
            "a textbook rescue, there is a real chance (~55%) the patient "
            "experiences a second deterioration wave in the back half of the "
            "episode and a moderate chance (~30%) of a dyskinesia spike. The "
            "controller cannot simply ramp once and coast: it must hold safety "
            "budget in reserve, react to the second crisis, and end the episode "
            "stable. Sensor noise is moderate, mimicking a real LFP recording "
            "during arm activity."
        ),
        difficulty="medium",
        start_step=16,
        n_steps=60,
        max_dbs_amplitude=1.8,
        max_dbs_pulse_width=0.18,
        max_side_effect_load=0.55,
        target_force_preserved=0.66,
        target_beta_arv=0.26,
        target_tremor_arv=0.30,
        target_tracking_error=0.26,
        success_threshold=0.52,
        patient_profile_ids=("balanced",),
        target_output_range=(-0.50, 0.50),
        event_profile="rescue",
        sensor_noise_std=0.040,
        schedule_id=None,
    )


TASK_MEDIUM = get_medium_task()
