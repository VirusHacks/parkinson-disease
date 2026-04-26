"""Surgical-followup task - first-week post-implant programming.

Clinical scenario
-----------------
The patient is in the first programming session days after DBS implantation.
A microlesion effect from the surgery itself temporarily improves symptoms,
masking the true baseline. During the first 25% of the episode the amplitude
ceiling is hard-capped at 0.6 mA - exceeding it during the swelling window
risks tissue damage. After step 25%, the ceiling ramps back to the task max.
Lead impedance can surge mid-episode as swelling resolves, transiently
reducing delivered current.

Why it is hard
--------------
* The early amplitude cap is enforced by the scheduler, not by the action - the
  agent must learn to live within it.
* Impedance surges drop delivered current without an action change; the agent
  observes `dbs_entrainment` falling and must compensate.
* True patient response is unknown until the microlesion effect resolves.
"""

from __future__ import annotations

from .base import DBSTask


def get_surgical_followup_task() -> DBSTask:
    return DBSTask(
        task_id="surgical_followup",
        name="Surgical Follow-up",
        description=(
            "120-step first-week post-implant session. The amplitude ceiling is "
            "hard-clamped to 0.6 mA during the early microlesion window, then "
            "ramps to the task max. Lead impedance may surge mid-episode, "
            "transiently reducing delivered current - the agent observes "
            "entrainment falling without an action change and must compensate. "
            "Models the realistic clinical challenge of titrating immediately "
            "after implantation when the patient's true response curve is not "
            "yet known."
        ),
        difficulty="expert",
        start_step=0,
        n_steps=120,
        max_dbs_amplitude=2.0,
        max_dbs_pulse_width=0.16,
        max_side_effect_load=0.38,
        target_force_preserved=0.62,
        target_beta_arv=0.26,
        target_tremor_arv=0.30,
        target_tracking_error=0.28,
        success_threshold=0.50,
        patient_profile_ids=("balanced", "fragile"),
        target_output_range=(-0.40, 0.40),
        event_profile="surgical",
        sensor_noise_std=0.035,
        schedule_id="surgical_microlesion",
    )


TASK_SURGICAL_FOLLOWUP = get_surgical_followup_task()
