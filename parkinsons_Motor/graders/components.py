"""Score components used by the deterministic benchmark grader."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ..tasks import DBSTask
except ImportError:
    from tasks import DBSTask


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def weighted_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    total = 0.0
    weight_total = 0.0
    for i, value in enumerate(values):
        weight = 1.0 - 0.35 * (i / max(n - 1, 1))
        total += weight * value
        weight_total += weight
    return total / weight_total if weight_total else 0.0


def force_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [clamp(step.get("force_preserved", 0.0)) for step in trajectory]
    return clamp(weighted_mean(values) / max(task.target_force_preserved, 1e-6))


def beta_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [clamp(step.get("beta_arv", 1.0)) for step in trajectory]
    mean_beta = weighted_mean([1.0 - value for value in values])
    threshold_fraction = sum(1 for value in values if value <= task.target_beta_arv) / max(len(values), 1)
    return clamp(0.55 * mean_beta + 0.45 * threshold_fraction)


def tremor_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [clamp(step.get("tremor_arv", 1.0)) for step in trajectory]
    mean_tremor = weighted_mean([1.0 - value for value in values])
    threshold_fraction = sum(1 for value in values if value <= task.target_tremor_arv) / max(len(values), 1)
    return clamp(0.60 * mean_tremor + 0.40 * threshold_fraction)


def tracking_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    errors = [clamp(step.get("task_error", 2.0), 0.0, 2.0) for step in trajectory]
    accuracies = [clamp(step.get("tracking_accuracy", 0.0)) for step in trajectory]
    mean_error = weighted_mean(
        [1.0 - min(error / max(task.target_tracking_error, 1e-6), 1.0) for error in errors]
    )
    mean_accuracy = weighted_mean(accuracies)
    return clamp(0.45 * mean_error + 0.55 * mean_accuracy)


def safety_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    values = [clamp(step.get("side_effect_load", 0.0)) for step in trajectory]
    overloads = [
        max(0.0, (value - task.max_side_effect_load) / max(1.0 - task.max_side_effect_load, 1e-6))
        for value in values
    ]
    constraints = [clamp(step.get("constraint_violation", 0.0)) for step in trajectory]
    overload_penalty = mean(overloads)
    peak_penalty = max(overloads) if overloads else 0.0
    constraint_penalty = mean(constraints)
    return clamp(1.0 - (0.45 * overload_penalty + 0.35 * peak_penalty + 0.20 * constraint_penalty) * 1.8)


def smoothness_score(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    costs = [clamp(step.get("smoothness_cost", 0.0)) for step in trajectory]
    return clamp(1.0 - mean(costs))


def efficiency_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    mean_amp = mean(
        [clamp(step.get("dbs_amplitude_ma", 0.0), 0.0, task.max_dbs_amplitude) for step in trajectory]
    )
    mean_pw = mean(
        [clamp(step.get("dbs_pulse_width_ms", 0.06), 0.06, task.max_dbs_pulse_width) for step in trajectory]
    )
    amp_eff = 1.0 - (mean_amp / max(task.max_dbs_amplitude, 1e-6))
    pw_eff = 1.0 - ((mean_pw - 0.06) / max(task.max_dbs_pulse_width - 0.06, 1e-6))
    return clamp(0.65 * amp_eff + 0.35 * pw_eff)


def mean_amplitude(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    return mean([clamp(step.get("dbs_amplitude_ma", 0.0), 0.0, 5.0) for step in trajectory])


def mean_constraint_violation(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    return mean([clamp(step.get("constraint_violation", 0.0)) for step in trajectory])


def terminal_stability_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    tail = trajectory[max(0, len(trajectory) - 5):]
    mean_force = mean([clamp(step.get("force_preserved", 0.0)) for step in tail])
    mean_tremor = mean([clamp(step.get("tremor_arv", 1.0)) for step in tail])
    mean_error = mean([clamp(step.get("task_error", 2.0), 0.0, 2.0) for step in tail])
    force_ok = min(mean_force / max(task.target_force_preserved, 1e-6), 1.0)
    tremor_ok = 1.0 - min(mean_tremor / max(task.target_tremor_arv, 1e-6), 1.0)
    error_ok = 1.0 - min(mean_error / max(task.target_tracking_error, 1e-6), 1.0)
    return clamp(0.45 * force_ok + 0.30 * tremor_ok + 0.25 * error_ok)


def recovery_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    head = trajectory[: min(6, len(trajectory))]
    tail = trajectory[max(0, len(trajectory) - 8):]

    start_force = mean([clamp(step.get("force_preserved", 0.0)) for step in head])
    end_force = mean([clamp(step.get("force_preserved", 0.0)) for step in tail])
    start_tremor = mean([clamp(step.get("tremor_arv", 1.0)) for step in head])
    end_tremor = mean([clamp(step.get("tremor_arv", 1.0)) for step in tail])
    start_tracking = mean([clamp(step.get("tracking_accuracy", 0.0)) for step in head])
    end_tracking = mean([clamp(step.get("tracking_accuracy", 0.0)) for step in tail])

    force_recovery = clamp((end_force - start_force + 0.12) / 0.28)
    tremor_recovery = clamp((start_tremor - end_tremor + 0.08) / 0.24)
    tracking_recovery = clamp((end_tracking - start_tracking + 0.06) / 0.20)
    return clamp(0.40 * force_recovery + 0.40 * tremor_recovery + 0.20 * tracking_recovery)
