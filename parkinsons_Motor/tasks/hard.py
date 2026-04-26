"""Hard task — long-horizon full-session DBS management.

Clinical intent: 150-step extended session that traverses onset, escalation,
crisis, and late-episode stability. Multiple stochastic disturbances may fire:
dyskinesia spikes, tachyphylaxis (tolerance), off-medication crises, and
voluntary motor surges. Sensor noise is significant. The controller must pace
itself across the full horizon — a brute-force or template policy will collapse
the safety budget early or under-respond to a late crisis.
"""

from __future__ import annotations

from .base import DBSTask


def get_hard_task() -> DBSTask:
    """Return the public hard benchmark task."""
    return DBSTask(
        task_id="hard",
        name="Full Episode",
        description=(
            "Full-session closed-loop DBS management over 150 steps spanning "
            "onset, escalation, peak symptoms, and late-episode stability. "
            "Multiple stochastic disturbances may fire across the episode: "
            "dyskinesia spikes (over-treatment risk), tachyphylaxis (diminishing "
            "returns from sustained high amplitude), an off-medication crisis "
            "(symptoms worsen as L-DOPA wears off), and voluntary motor surges "
            "(transient high-force demands). Biomarkers carry realistic sensor "
            "noise. A durable policy must pace itself, hold safety budget in "
            "reserve for late crises, recover from setbacks, and finish the "
            "episode stable. Brute-force or template strategies fail: the "
            "patient's trajectory is no longer scriptable."
        ),
        difficulty="hard",
        start_step=0,
        n_steps=150,
        max_dbs_amplitude=2.4,
        max_dbs_pulse_width=0.20,
        max_side_effect_load=0.40,
        target_force_preserved=0.62,
        target_beta_arv=0.21,
        target_tremor_arv=0.27,
        target_tracking_error=0.22,
        success_threshold=0.42,
        patient_profile_ids=("refractory",),
        target_output_range=(-0.65, 0.65),
        event_profile="long_horizon",
        sensor_noise_std=0.050,
        schedule_id=None,
    )


TASK_HARD = get_hard_task()
