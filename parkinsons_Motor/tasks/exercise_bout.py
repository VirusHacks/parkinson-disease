"""Exercise-bout task — sustained high motor demand.

Clinical scenario
-----------------
The patient is performing an exercise bout (cycling, brisk walking, resistance
band work). For ~20 steps in the early-to-mid window, the motor target is
forced to a high absolute value via a motor-surge event, demanding strong
voluntary force while tremor amplifies under exertion. A late dyskinesia spike
may follow as accumulated stimulation interacts with elevated catecholamines.

Why it is hard
--------------
* Tracking error target is tight (0.20) and the surge pushes |target| ≥ 0.55.
* Force preservation must stay high under load.
* Side-effect budget is moderate, but a wrong-time push during the dyskinesia
  spike can collapse the safety score.
"""

from __future__ import annotations

from .base import DBSTask


def get_exercise_bout_task() -> DBSTask:
    return DBSTask(
        task_id="exercise_bout",
        name="Exercise Bout",
        description=(
            "Sustained high motor demand: the patient performs a 20-step "
            "exercise burst that forces |target_output| high while tremor and "
            "fatigue rise under exertion. A late dyskinesia spike is possible. "
            "The agent must preserve force under load, track tightly, and avoid "
            "over-stimulating during the post-exercise dyskinesia window. "
            "Models real clinical observations: DBS settings tuned for resting "
            "state often fail during functional activity."
        ),
        difficulty="expert",
        start_step=8,
        n_steps=70,
        max_dbs_amplitude=2.0,
        max_dbs_pulse_width=0.18,
        max_side_effect_load=0.45,
        target_force_preserved=0.72,
        target_beta_arv=0.26,
        target_tremor_arv=0.30,
        target_tracking_error=0.22,
        success_threshold=0.55,
        patient_profile_ids=("balanced", "responsive"),
        target_output_range=(-0.45, 0.45),
        event_profile="exercise",
        sensor_noise_std=0.045,
        schedule_id=None,
    )


TASK_EXERCISE_BOUT = get_exercise_bout_task()
