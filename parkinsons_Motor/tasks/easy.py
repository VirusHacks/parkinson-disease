"""Easy task - calm-start titration with mild sensor noise.

Clinical intent: a clean, lightly stochastic introduction to closed-loop DBS.
Rising beta with mild tremor on a responsive patient, modest motor demands,
and a short 36-step horizon. Biomarker recordings carry small Gaussian sensor
noise so the agent learns to act on trends rather than single noisy samples,
but no major disturbances fire - this task remains the gentlest of the suite.
"""

from __future__ import annotations

from .base import DBSTask


def get_easy_task() -> DBSTask:
    """Return the public easy benchmark task."""
    return DBSTask(
        task_id="easy",
        name="Calm Start",
        description=(
            "Early-session DBS titration on a responsive patient with rising "
            "beta and mild tremor. The episode is short (36 steps) and the "
            "patient is predictable, but biomarker recordings carry mild sensor "
            "noise - the agent must respond to trends, not single noisy samples. "
            "Clinically this resembles a routine morning programming session: "
            "bring symptoms under control with the lowest effective dose, hold a "
            "stable safety margin, and avoid unnecessary brute-force stimulation."
        ),
        difficulty="easy",
        start_step=5,
        n_steps=36,
        max_dbs_amplitude=1.5,
        max_dbs_pulse_width=0.16,
        max_side_effect_load=0.55,
        target_force_preserved=0.78,
        target_beta_arv=0.26,
        target_tremor_arv=0.20,
        target_tracking_error=0.32,
        success_threshold=0.55,
        patient_profile_ids=("responsive",),
        target_output_range=(-0.20, 0.20),
        event_profile=None,
        sensor_noise_std=0.025,
        schedule_id=None,
    )


TASK_EASY = get_easy_task()
