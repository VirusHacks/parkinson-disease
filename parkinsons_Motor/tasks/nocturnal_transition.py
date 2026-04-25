"""Nocturnal-transition task — awake → wind-down → sleep with time-varying setpoints.

Clinical scenario
-----------------
A 150-step session that traverses three phases: active awake (full motor
demands), pre-sleep wind-down, and sleep. The grader's tremor and beta targets
tighten in sleep (suppression matters more) while the force expectation drops
because the patient is not actively moving. The agent must learn time-varying
setpoints: tapering DBS too early risks REM-sleep behavioural disorder, while
holding daytime amplitude through sleep wastes battery and accumulates side
effects.

Why it is hard
--------------
* The optimal policy explicitly varies with `local_step / n_steps`.
* A constant or naive reactive policy fails the late-episode tighter targets.
* A small motor surge or dyskinesia event may still fire mid-transition.
"""

from __future__ import annotations

from .base import DBSTask


def get_nocturnal_transition_task() -> DBSTask:
    return DBSTask(
        task_id="nocturnal_transition",
        name="Nocturnal Transition",
        description=(
            "150-step session crossing awake, pre-sleep wind-down, and sleep "
            "phases. The grader's tremor and beta targets tighten in the sleep "
            "phase while force expectations drop — DBS strategy must change "
            "across the episode rather than remain constant. Tapering too early "
            "risks REM sleep behaviour disturbances; holding daytime amplitude "
            "through sleep wastes safety budget. Models a real overnight DBS "
            "session where awake and sleep optima are different."
        ),
        difficulty="expert",
        start_step=0,
        n_steps=150,
        max_dbs_amplitude=2.0,
        max_dbs_pulse_width=0.18,
        max_side_effect_load=0.40,
        target_force_preserved=0.55,
        target_beta_arv=0.24,
        target_tremor_arv=0.28,
        target_tracking_error=0.30,
        success_threshold=0.55,
        patient_profile_ids=("balanced", "responsive"),
        target_output_range=(-0.40, 0.40),
        event_profile="nocturnal",
        sensor_noise_std=0.040,
        schedule_id="nocturnal",
    )


TASK_NOCTURNAL_TRANSITION = get_nocturnal_transition_task()
