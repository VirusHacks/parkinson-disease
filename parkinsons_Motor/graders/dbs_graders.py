"""
Graders for the Parkinson's Motor (DBS) Environment.
=====================================================
Each grader takes the full episode trajectory (list of step dicts) and the
task specification and returns a deterministic float in [0.0, 1.0].

Score composition (all graders use the same building blocks):
  force_score     — weighted average force_preserved across the episode
  beta_score      — fraction of steps where beta_arv < task threshold
  side_eff_score  — penalty for exceeding side-effect budget
  efficiency_score— penalty for using unnecessarily high DBS amplitude
  final_bonus     — bonus if final force_preserved >= target (episode didn't collapse)

The three tasks weight these components differently to reflect their clinical
focus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ..tasks.dbs_tasks import DBSTask
except ImportError:
    from tasks.dbs_tasks import DBSTask


# ---------------------------------------------------------------------------
# Shared per-step feature helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fraction_below(values: List[float], threshold: float) -> float:
    """Fraction of steps where value stays BELOW the threshold."""
    if not values:
        return 0.0
    return sum(1 for v in values if v <= threshold) / len(values)


def _fraction_above(values: List[float], threshold: float) -> float:
    """Fraction of steps where value stays ABOVE the threshold."""
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Score building blocks — each returns a value in [0, 1]
# ---------------------------------------------------------------------------

def _force_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    """
    Mean force_preserved over the episode, normalised so that hitting the
    task's target_force_preserved scores 1.0, and zero force scores 0.0.
    Early steps in the episode get slightly higher weight (clinical importance
    of not losing force immediately).
    """
    if not trajectory:
        return 0.0
    n = len(trajectory)
    weighted_sum = 0.0
    weight_total = 0.0
    for i, step in enumerate(trajectory):
        # Linear decay: first step has weight 1.0, last step has weight 0.5
        w = 1.0 - 0.5 * (i / max(n - 1, 1))
        fp = _clamp(step.get("force_preserved", 0.0))
        weighted_sum += w * fp
        weight_total += w
    mean_fp = weighted_sum / weight_total if weight_total > 0 else 0.0
    # Normalise: 0 → 0.0, target → 1.0, above target → capped at 1.0
    return _clamp(mean_fp / max(task.target_force_preserved, 1e-6))


def _beta_score(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    """
    Fraction of steps where beta_arv < task threshold.
    Full score if the agent suppresses beta every step.
    """
    beta_values = [_clamp(s.get("beta_arv", 1.0)) for s in trajectory]
    return _fraction_below(beta_values, task.target_beta_arv)


def _side_effect_penalty(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    """
    Returns 1.0 if side-effect load NEVER exceeded the budget.
    Decays linearly toward 0.0 proportional to how often and how much
    the budget was exceeded.
    """
    if not trajectory:
        return 1.0
    violations = []
    for s in trajectory:
        se = _clamp(s.get("side_effect_load", 0.0))
        if se > task.max_side_effect_load:
            excess = (se - task.max_side_effect_load) / max(1.0 - task.max_side_effect_load, 1e-6)
            violations.append(_clamp(excess))
    if not violations:
        return 1.0
    mean_violation = _mean(violations)
    violation_fraction = len(violations) / len(trajectory)
    # Penalties compound: frequent + large violations → score near 0
    penalty = mean_violation * violation_fraction
    return _clamp(1.0 - penalty * 2.0)


def _amplitude_efficiency(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    """
    Rewards the agent for using the *minimum effective* DBS amplitude.
    An agent that blasts 3.0 mA the whole time scores poorly here.
    Score = 1 − (mean_amplitude / max_allowed)
    But clamped so it never dominates the grade.
    """
    if not trajectory:
        return 1.0
    amps = [_clamp(s.get("dbs_amplitude_ma", 0.0), 0.0, task.max_dbs_amplitude)
            for s in trajectory]
    mean_amp = _mean(amps)
    efficiency = 1.0 - (mean_amp / max(task.max_dbs_amplitude, 1e-6))
    return _clamp(efficiency)


def _final_state_bonus(trajectory: List[Dict[str, Any]], task: DBSTask) -> float:
    """
    Bonus for keeping force preserved at the END of the episode.
    This rewards the agent for not letting the patient's motor function
    collapse in the final phase.  Returns 0 or 1.
    """
    if not trajectory:
        return 0.0
    final_fp = _clamp(trajectory[-1].get("force_preserved", 0.0))
    # Bonus if final force is at least 80 % of the task's target
    return 1.0 if final_fp >= (task.target_force_preserved * 0.80) else 0.0


# ---------------------------------------------------------------------------
# Per-task graders
# ---------------------------------------------------------------------------

def grade_beta_suppression(
    trajectory: List[Dict[str, Any]],
    task: DBSTask,
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Task 1 — Easy: Static Beta Suppression.

    Focus: Did the agent successfully keep beta oscillation BELOW the clinical
    threshold during the early, relatively calm phase of the episode?

    Weights:
      beta_score    0.50  — primary: suppress the oscillation
      force_score   0.25  — secondary: don't sacrifice motor function
      side_eff      0.15  — safety: stay within side-effect budget
      efficiency    0.10  — bonus: use the minimum necessary amplitude
    """
    bs = _beta_score(trajectory, task)
    fs = _force_score(trajectory, task)
    se = _side_effect_penalty(trajectory, task)
    ef = _amplitude_efficiency(trajectory, task)

    raw = (
        0.50 * bs
        + 0.25 * fs
        + 0.15 * se
        + 0.10 * ef
    )
    return _clamp(raw)


def grade_tremor_correction(
    trajectory: List[Dict[str, Any]],
    task: DBSTask,
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Task 2 — Medium: Dynamic Tremor Correction.

    Focus: Can the agent dynamically respond to rising tremor while keeping
    the patient's muscle force above the clinical threshold?

    Weights:
      force_score   0.50  — primary: preserve motor function through tremor
      beta_score    0.25  — secondary: suppress the oscillation driver
      side_eff      0.15  — safety: manage cumulative stimulation load
      final_bonus   0.10  — reward for not collapsing at the end
    """
    fs = _force_score(trajectory, task)
    bs = _beta_score(trajectory, task)
    se = _side_effect_penalty(trajectory, task)
    fb = _final_state_bonus(trajectory, task)

    raw = (
        0.50 * fs
        + 0.25 * bs
        + 0.15 * se
        + 0.10 * fb
    )
    return _clamp(raw)


def grade_full_episode(
    trajectory: List[Dict[str, Any]],
    task: DBSTask,
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Task 3 — Hard: Full Clinical Episode.

    Focus: Over the complete 100-step episode, the agent must optimise the
    complete trade-off: sustained force preservation, beta suppression,
    side-effect management, and efficiency — all while navigating the severe
    late-episode tremor.

    Weights:
      force_score   0.40  — primary: sustained motor function
      beta_score    0.20  — oscillation suppression
      side_eff      0.20  — safety over a long horizon
      efficiency    0.10  — appropriate amplitude use
      final_bonus   0.10  — episode didn't collapse in the last window
    """
    fs = _force_score(trajectory, task)
    bs = _beta_score(trajectory, task)
    se = _side_effect_penalty(trajectory, task)
    ef = _amplitude_efficiency(trajectory, task)
    fb = _final_state_bonus(trajectory, task)

    raw = (
        0.40 * fs
        + 0.20 * bs
        + 0.20 * se
        + 0.10 * ef
        + 0.10 * fb
    )
    return _clamp(raw)


# ---------------------------------------------------------------------------
# Public grading API
# ---------------------------------------------------------------------------

_GRADER_MAP = {
    "beta_suppression":   grade_beta_suppression,
    "tremor_correction":  grade_tremor_correction,
    "full_episode":       grade_full_episode,
}


def compute_score(
    task: DBSTask,
    trajectory: List[Dict[str, Any]],
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Deterministic grader entry-point.

    Args:
        task:       The DBSTask specification for this episode.
        trajectory: List of per-step observation dicts collected during the
                    episode.  Each dict must have at minimum:
                      force_preserved  float [0, 1]
                      beta_arv         float [0, 1]
                      side_effect_load float [0, 1]
                      dbs_amplitude_ma float [0, max]
        info:       Optional extra metadata (unused by default graders).

    Returns:
        float in [0.0, 1.0]
    """
    grader_fn = _GRADER_MAP.get(task.task_id)
    if grader_fn is None:
        raise ValueError(
            f"No grader registered for task_id={task.task_id!r}. "
            f"Available: {list(_GRADER_MAP)}"
        )
    return grader_fn(trajectory, task, info)


def is_success(task: DBSTask, score: float) -> bool:
    """Return True if the score meets or exceeds the task's success threshold."""
    return score >= task.success_threshold
