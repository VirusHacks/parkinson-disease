"""Hard-failure rules used by the deterministic benchmark grader."""

from __future__ import annotations

from typing import Dict, List, Any

try:
    from ..tasks import DBSTask
    from .components import mean_amplitude, mean_constraint_violation
except ImportError:
    from tasks import DBSTask
    from graders.components import mean_amplitude, mean_constraint_violation


def hard_failure_penalty(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    details: Dict[str, float],
) -> float:
    penalty = 0.0
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
    if task.task_id in {"tremor_correction", "full_episode"} and details["safety_score"] == 0.0:
        penalty += 0.10
    if task.task_id in {"tremor_correction", "full_episode"}:
        if mean_constraint_violation(trajectory) > 0.20:
            penalty += 0.08

    if task.task_id == "beta_suppression":
        # Zero-stim failure: no DBS and symptoms not suppressed
        if mean_amplitude(trajectory) < 0.08 and (
            details["beta_score"] < 0.65 or details["tremor_score"] < 0.60
        ):
            penalty += 0.20
        # Constant max-amp with no efficiency: brute-force penalty
        if mean_amplitude(trajectory) > 0.80 * task.max_dbs_amplitude and details["efficiency_score"] < 0.25:
            penalty += 0.14
    if task.task_id == "tremor_correction":
        mean_amp = mean_amplitude(trajectory)
        # Tremor not rescued at all
        if details["tremor_score"] < 0.20:
            penalty += 0.10
        # Genuinely zero-stim attempt (< 0.05 mA mean) with no rescue
        if mean_amp < 0.05 and (
            details["tremor_score"] < 0.24 or details["recovery_score"] < 0.18
        ):
            penalty += 0.14
    if task.task_id == "full_episode":
        if details["terminal_stability_score"] < 0.12:
            penalty += 0.04
        if details["force_score"] < 0.39 and details["terminal_stability_score"] < 0.16:
            penalty += 0.04
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
    return penalty
