"""Hard-failure rules used by the deterministic benchmark grader.

Every task contributes a `task_id`-specific penalty block. Penalties model
clinically unacceptable behaviours that the smooth-component score might
otherwise undervalue (e.g. zero-stim during an exercise bout, exceeding the
microlesion-window amplitude cap, brute-force during a dyskinesia spike).
"""

from __future__ import annotations

from typing import Dict, List, Any

try:
    from ..tasks import DBSTask
    from .components import (
        clamp,
        mean,
        mean_amplitude,
        mean_constraint_violation,
    )
except ImportError:
    from tasks import DBSTask
    from graders.components import (
        clamp,
        mean,
        mean_amplitude,
        mean_constraint_violation,
    )


def _trailing_mean(values: List[float], frac: float) -> float:
    if not values:
        return 0.0
    n = max(1, int(len(values) * frac))
    return mean(values[-n:])


def hard_failure_penalty(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    details: Dict[str, float],
) -> float:
    penalty = 0.0

    # ---- Universal floors ------------------------------------------------
    if details["safety_score"] < 0.20:
        penalty += 0.12
    if details["tracking_score"] < 0.20:
        penalty += 0.08
    if details["beta_score"] < 0.40:
        penalty += 0.06
    if details["tremor_score"] < 0.22:
        penalty += 0.05
    if details["force_score"] < 0.55:
        penalty += 0.04
    if task.task_id in {"medium", "hard"} and details["safety_score"] == 0.0:
        penalty += 0.10
    if task.task_id in {"medium", "hard"}:
        if mean_constraint_violation(trajectory) > 0.20:
            penalty += 0.08

    # ---- Public tasks ----------------------------------------------------
    if task.task_id == "easy":
        if mean_amplitude(trajectory) < 0.08 and (
            details["beta_score"] < 0.65 or details["tremor_score"] < 0.60
        ):
            penalty += 0.20
        if (
            mean_amplitude(trajectory) > 0.80 * task.max_dbs_amplitude
            and details["efficiency_score"] < 0.25
        ):
            penalty += 0.14

    if task.task_id == "medium":
        mean_amp = mean_amplitude(trajectory)
        if details["tremor_score"] < 0.20:
            penalty += 0.10
        if mean_amp < 0.05 and (
            details["tremor_score"] < 0.24 or details["recovery_score"] < 0.18
        ):
            penalty += 0.14

    if task.task_id == "hard":
        # Clinical floor: DBS must actually suppress pathological activity.
        if details["beta_score"] < 0.30:
            penalty += 0.10
        if details["tremor_score"] < 0.25:
            penalty += 0.06
        if details["terminal_stability_score"] < 0.25:
            penalty += 0.08
        if details["force_score"] < 0.42 and details["terminal_stability_score"] < 0.22:
            penalty += 0.05
        # Long-horizon over-treatment: brute-force high amp with poor efficiency.
        if (
            mean_amplitude(trajectory) > 0.72 * task.max_dbs_amplitude
            and details["efficiency_score"] < 0.18
        ):
            penalty += 0.08

    # ---- Expert tasks ----------------------------------------------------
    if task.task_id == "fragile_patient":
        mean_amp = mean_amplitude(trajectory)
        if details["safety_score"] < 0.45:
            penalty += 0.12
        if mean_amp < 0.08 and (
            details["beta_score"] < 0.55 or details["tracking_score"] < 0.55
        ):
            penalty += 0.22

    if task.task_id == "refractory_patient":
        mean_amp = mean_amplitude(trajectory)
        if mean_amp < 0.18 and (
            details["tremor_score"] < 0.55 or details["recovery_score"] < 0.40
        ):
            penalty += 0.18
        if details["terminal_stability_score"] < 0.18:
            penalty += 0.08

    if task.task_id == "personalization_generalization":
        mean_amp = mean_amplitude(trajectory)
        if details["recovery_score"] < 0.35:
            penalty += 0.12
        if details["tracking_score"] < 0.50:
            penalty += 0.10
        if mean_amp < 0.10 and (
            details["force_score"] < 0.60 or details["tremor_score"] < 0.55
        ):
            penalty += 0.20

    # ---- Scenario tasks --------------------------------------------------
    if task.task_id == "exercise_bout":
        # Failing tracking during an exercise bout is the headline failure.
        if details["tracking_score"] < 0.55:
            penalty += 0.14
        if details["force_score"] < 0.60:
            penalty += 0.10
        # Zero-stim during exertion is clinically wrong.
        if mean_amplitude(trajectory) < 0.20 and details["tracking_score"] < 0.65:
            penalty += 0.16

    if task.task_id == "medication_interaction":
        # Recovery is the critical signal — must compensate the off-med crisis.
        if details["recovery_score"] < 0.40:
            penalty += 0.12
        # Over-treating during the dyskinesia window: high mean amp + poor safety.
        amps = [step.get("dbs_amplitude_ma", 0.0) for step in trajectory]
        if (
            _trailing_mean(amps, 0.4) > 0.65 * task.max_dbs_amplitude
            and details["safety_score"] < 0.55
        ):
            penalty += 0.10

    if task.task_id == "nocturnal_transition":
        # Sleep-phase tightening: terminal stability MUST be high.
        if details["terminal_stability_score"] < 0.45:
            penalty += 0.12
        # Daytime brute-force carried into sleep wastes safety budget.
        if (
            details["efficiency_score"] < 0.25
            and details["safety_score"] < 0.55
        ):
            penalty += 0.08

    if task.task_id == "surgical_followup":
        # Microlesion-window violations: any constraint_violation in the first
        # 25% of the trajectory is clinically unacceptable.
        n = len(trajectory)
        if n > 0:
            window_end = max(1, int(n * 0.25))
            window = trajectory[:window_end]
            window_violations = mean(
                [clamp(step.get("constraint_violation", 0.0)) for step in window]
            )
            if window_violations > 0.05:
                penalty += 0.20
        if details["safety_score"] < 0.55:
            penalty += 0.08
        if details["recovery_score"] < 0.30:
            penalty += 0.06

    return penalty
