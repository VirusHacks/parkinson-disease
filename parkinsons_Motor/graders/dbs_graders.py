"""
Deterministic graders for the Parkinson's Motor environment.

The final score is benchmark-facing and clinically composite. It evaluates:
  - motor function preservation
  - suppression of pathological activity
  - movement tracking quality
  - safety burden
  - control smoothness
  - stimulation efficiency and constraint compliance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ..tasks.dbs_tasks import DBSTask
except ImportError:
    from tasks.dbs_tasks import DBSTask


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _weighted_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    total = 0.0
    weight_total = 0.0
    for i, value in enumerate(values):
        w = 1.0 - 0.35 * (i / max(n - 1, 1))
        total += w * value
        weight_total += w
    return total / weight_total if weight_total else 0.0


def _force_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [_clamp(step.get("force_preserved", 0.0)) for step in trajectory]
    return _clamp(_weighted_mean(values) / max(task.target_force_preserved, 1e-6))


def _beta_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [_clamp(step.get("beta_arv", 1.0)) for step in trajectory]
    mean_beta = _weighted_mean([1.0 - v for v in values])
    threshold_fraction = sum(1 for v in values if v <= task.target_beta_arv) / max(len(values), 1)
    return _clamp(0.55 * mean_beta + 0.45 * threshold_fraction)


def _tremor_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    values = [_clamp(step.get("tremor_arv", 1.0)) for step in trajectory]
    mean_tremor = _weighted_mean([1.0 - v for v in values])
    threshold_fraction = sum(1 for v in values if v <= task.target_tremor_arv) / max(len(values), 1)
    return _clamp(0.60 * mean_tremor + 0.40 * threshold_fraction)


def _tracking_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    errors = [_clamp(step.get("task_error", 2.0), 0.0, 2.0) for step in trajectory]
    accuracies = [_clamp(step.get("tracking_accuracy", 0.0)) for step in trajectory]
    mean_error = _weighted_mean([1.0 - min(e / max(task.target_tracking_error, 1e-6), 1.0) for e in errors])
    mean_acc = _weighted_mean(accuracies)
    return _clamp(0.45 * mean_error + 0.55 * mean_acc)


def _safety_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    values = [_clamp(step.get("side_effect_load", 0.0)) for step in trajectory]
    overloads = [
        max(0.0, (v - task.max_side_effect_load) / max(1.0 - task.max_side_effect_load, 1e-6))
        for v in values
    ]
    constraint = [_clamp(step.get("constraint_violation", 0.0)) for step in trajectory]
    overload_penalty = _mean(overloads)
    peak_penalty = max(overloads) if overloads else 0.0
    constraint_penalty = _mean(constraint)
    return _clamp(1.0 - (0.45 * overload_penalty + 0.35 * peak_penalty + 0.20 * constraint_penalty) * 1.8)


def _smoothness_score(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    costs = [_clamp(step.get("smoothness_cost", 0.0)) for step in trajectory]
    return _clamp(1.0 - _mean(costs))


def _efficiency_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    mean_amp = _mean([
        _clamp(step.get("dbs_amplitude_ma", 0.0), 0.0, task.max_dbs_amplitude)
        for step in trajectory
    ])
    mean_pw = _mean([
        _clamp(step.get("dbs_pulse_width_ms", 0.06), 0.06, task.max_dbs_pulse_width)
        for step in trajectory
    ])
    amp_eff = 1.0 - (mean_amp / max(task.max_dbs_amplitude, 1e-6))
    pw_eff = 1.0 - ((mean_pw - 0.06) / max(task.max_dbs_pulse_width - 0.06, 1e-6))
    return _clamp(0.65 * amp_eff + 0.35 * pw_eff)


def _mean_amplitude(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    return _mean([_clamp(step.get("dbs_amplitude_ma", 0.0), 0.0, 5.0) for step in trajectory])


def _mean_constraint_violation(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    return _mean([_clamp(step.get("constraint_violation", 0.0)) for step in trajectory])


def _terminal_stability_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    tail = trajectory[max(0, len(trajectory) - 5):]
    mean_force = _mean([_clamp(step.get("force_preserved", 0.0)) for step in tail])
    mean_tremor = _mean([_clamp(step.get("tremor_arv", 1.0)) for step in tail])
    mean_error = _mean([_clamp(step.get("task_error", 2.0), 0.0, 2.0) for step in tail])
    force_ok = min(mean_force / max(task.target_force_preserved, 1e-6), 1.0)
    tremor_ok = 1.0 - min(mean_tremor / max(task.target_tremor_arv, 1e-6), 1.0)
    error_ok = 1.0 - min(mean_error / max(task.target_tracking_error, 1e-6), 1.0)
    return _clamp(0.45 * force_ok + 0.30 * tremor_ok + 0.25 * error_ok)


def _recovery_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    if not trajectory:
        return 0.0
    head = trajectory[: min(6, len(trajectory))]
    tail = trajectory[max(0, len(trajectory) - 8):]

    start_force = _mean([_clamp(step.get("force_preserved", 0.0)) for step in head])
    end_force = _mean([_clamp(step.get("force_preserved", 0.0)) for step in tail])
    start_tremor = _mean([_clamp(step.get("tremor_arv", 1.0)) for step in head])
    end_tremor = _mean([_clamp(step.get("tremor_arv", 1.0)) for step in tail])
    start_tracking = _mean([_clamp(step.get("tracking_accuracy", 0.0)) for step in head])
    end_tracking = _mean([_clamp(step.get("tracking_accuracy", 0.0)) for step in tail])

    force_recovery = _clamp((end_force - start_force + 0.12) / 0.28)
    tremor_recovery = _clamp((start_tremor - end_tremor + 0.08) / 0.24)
    tracking_recovery = _clamp((end_tracking - start_tracking + 0.06) / 0.20)
    return _clamp(
        0.40 * force_recovery + 0.40 * tremor_recovery + 0.20 * tracking_recovery
    )


def compute_score_details(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return all score components plus the final deterministic score."""
    details = {
        "force_score": _force_score(trajectory, task),
        "beta_score": _beta_score(trajectory, task),
        "tremor_score": _tremor_score(trajectory, task),
        "tracking_score": _tracking_score(trajectory, task),
        "safety_score": _safety_score(trajectory, task),
        "smoothness_score": _smoothness_score(trajectory),
        "terminal_stability_score": _terminal_stability_score(trajectory, task),
        "recovery_score": _recovery_score(trajectory, task),
    }
    therapeutic_engagement = _clamp(
        0.40 * details["force_score"]
        + 0.30 * details["beta_score"]
        + 0.30 * details["tremor_score"]
    )
    details["efficiency_score"] = _efficiency_score(trajectory, task) * therapeutic_engagement

    if task.task_id == "beta_suppression":
        overall = (
            0.24 * details["beta_score"]
            + 0.16 * details["tremor_score"]
            + 0.14 * details["tracking_score"]
            + 0.18 * details["force_score"]
            + 0.16 * details["safety_score"]
            + 0.06 * details["smoothness_score"]
            + 0.06 * details["efficiency_score"]
        )
    elif task.task_id == "tremor_correction":
        overall = (
            0.16 * details["force_score"]
            + 0.06 * details["beta_score"]
            + 0.14 * details["tremor_score"]
            + 0.16 * details["tracking_score"]
            + 0.22 * details["safety_score"]
            + 0.04 * details["smoothness_score"]
            + 0.12 * details["terminal_stability_score"]
            + 0.08 * details["efficiency_score"]
            + 0.02 * details["recovery_score"]
        )
    else:
        overall = (
            0.14 * details["force_score"]
            + 0.08 * details["beta_score"]
            + 0.06 * details["tremor_score"]
            + 0.16 * details["tracking_score"]
            + 0.36 * details["safety_score"]
            + 0.05 * details["smoothness_score"]
            + 0.10 * details["efficiency_score"]
            + 0.05 * details["terminal_stability_score"]
        )

    hard_failures = 0.0
    if details["safety_score"] < 0.20:
        hard_failures += 0.12
    if details["tracking_score"] < 0.20:
        hard_failures += 0.08
    if details["beta_score"] < 0.40:
        hard_failures += 0.06
    if details["tremor_score"] < 0.22:
        hard_failures += 0.05
    if details["force_score"] < 0.55:
        hard_failures += 0.04
    if task.task_id in {"tremor_correction", "full_episode"} and details["safety_score"] == 0.0:
        hard_failures += 0.10
    if task.task_id in {"tremor_correction", "full_episode"}:
        mean_violation = _mean_constraint_violation(trajectory)
        if mean_violation > 0.20:
            hard_failures += 0.08
    if task.task_id == "beta_suppression":
        mean_amp = _mean_amplitude(trajectory)
        if mean_amp < 0.05 and details["beta_score"] < 0.35:
            hard_failures += 0.10
    if task.task_id == "tremor_correction":
        mean_amp = _mean_amplitude(trajectory)
        if details["tremor_score"] < 0.20:
            hard_failures += 0.10
        if mean_amp < 0.08 and (
            details["tremor_score"] < 0.25 or details["recovery_score"] < 0.18
        ):
            hard_failures += 0.18
    if task.task_id == "full_episode":
        if details["terminal_stability_score"] < 0.12:
            hard_failures += 0.04
        if (
            details["force_score"] < 0.39
            and details["terminal_stability_score"] < 0.16
        ):
            hard_failures += 0.04

    details["overall_score"] = _clamp(overall - hard_failures)
    return details


def compute_score(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """Deterministic benchmark-facing score in [0, 1]."""
    return compute_score_details(task, trajectory, info)["overall_score"]


def is_success(task: DBSTask, score: float) -> bool:
    """Return True if the score meets the task threshold."""
    return score >= task.success_threshold
