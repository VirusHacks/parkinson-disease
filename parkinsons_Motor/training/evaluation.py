"""Evaluation suite for the Parkinson's-Motor closed-loop DBS environment.

Separates metrics into four families (mirrors mhtruong1031's bio-experiment
``training/evaluation.py``):

  - ``online_metrics``         — collected during training rollouts (mean
                                 return, std, episode length, success rate)
  - ``benchmark_metrics``      — computed on a fixed held-out trajectory set
                                 (final β-suppression, side-effect compliance,
                                 mean amplitude in the clinical 1.5–4.0 mA
                                 window, action smoothness, format validity)
  - ``clinical_metrics``       — compare against published adaptive-DBS
                                 literature (Little et al. 2016 — see
                                 :mod:`parkinsons_Motor.training.clinical_benchmark`
                                 for the full setup)
  - ``simulator_fidelity``     — placeholder for sim-to-real distributional
                                 distance once real iEEG traces are available

Each public method returns ``List[MetricResult]`` so the dashboard can loop
over a heterogeneous set of metrics without unpacking dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .trajectory import DBSTrajectory, DBSTrajectoryDataset


@dataclass
class MetricResult:
    """One scalar metric + arbitrary metadata for the dashboard / README."""

    name: str
    value: float
    unit: str = ""              # e.g. "mA", "%", "ratio"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "details": self.details,
        }


# ─── Pure helpers (no numpy dependency so this works in any colab cell) ───

def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])


# ─── EvaluationSuite ─────────────────────────────────────────────────────


class EvaluationSuite:
    """Aggregate metrics across a :class:`DBSTrajectoryDataset`.

    All methods are static / classmethods so the suite is cheap to call from
    notebooks: ``EvaluationSuite.online_metrics(traj_list)``.
    """

    # ── 1. online RL metrics ───────────────────────────────────────────

    @staticmethod
    def online_metrics(trajectories: Sequence[DBSTrajectory]) -> List[MetricResult]:
        """One row per training rollout — what the GRPO trainer should converge on."""
        if not trajectories:
            return []
        rewards = [t.total_reward for t in trajectories]
        graders = [t.grader_score for t in trajectories]
        lengths = [t.n_steps for t in trajectories]
        successes = [int(t.success) for t in trajectories]
        invalids = [t.invalid_count for t in trajectories]

        return [
            MetricResult("mean_return",          _mean(rewards)),
            MetricResult("median_return",        _median(rewards)),
            MetricResult("std_return",           _std(rewards)),
            MetricResult("mean_grader_score",    _mean(graders), unit="ratio"),
            MetricResult("std_grader_score",     _std(graders),  unit="ratio"),
            MetricResult("mean_episode_length",  _mean(lengths)),
            MetricResult("success_rate",         _mean(successes), unit="ratio"),
            MetricResult("mean_invalid_actions", _mean(invalids)),
        ]

    # ── 2. offline benchmark metrics ────────────────────────────────────

    @staticmethod
    def benchmark_metrics(dataset: DBSTrajectoryDataset) -> List[MetricResult]:
        """Static metrics on a held-out dataset — what the README reports."""
        if len(dataset) == 0:
            return []

        return [
            MetricResult(
                "final_beta_suppression",
                EvaluationSuite._final_beta_suppression(dataset),
                unit="ratio",
            ),
            MetricResult(
                "final_tremor_suppression",
                EvaluationSuite._final_tremor_suppression(dataset),
                unit="ratio",
            ),
            MetricResult(
                "side_effect_compliance",
                EvaluationSuite._side_effect_compliance(dataset),
                unit="ratio",
                details={"threshold": 0.45},
            ),
            MetricResult(
                "clinical_amplitude_window",
                EvaluationSuite._clinical_amplitude_window(dataset),
                unit="ratio",
                details={"window_mA": [1.5, 4.0]},
            ),
            MetricResult(
                "action_smoothness",
                EvaluationSuite._action_smoothness(dataset),
                unit="mA/step",
            ),
            MetricResult(
                "format_validity_rate",
                EvaluationSuite._format_validity_rate(dataset),
                unit="ratio",
            ),
        ]

    # ── 3. clinical / literature metrics (delegates) ────────────────────

    @staticmethod
    def clinical_metrics(
        dataset: DBSTrajectoryDataset,
        baseline: Optional[DBSTrajectoryDataset] = None,
    ) -> List[MetricResult]:
        """Compare suppression vs published adaptive-DBS literature.

        Imports lazily so :mod:`evaluation` stays usable when the optional
        :mod:`clinical_benchmark` module isn't on the path.
        """
        from .clinical_benchmark import compare_to_literature
        return compare_to_literature(dataset, baseline=baseline)

    # ── 4. simulator fidelity (stub for future sim-to-real) ─────────────

    @staticmethod
    def simulator_fidelity(
        simulated: DBSTrajectoryDataset,
        real: Optional[DBSTrajectoryDataset] = None,
    ) -> List[MetricResult]:
        if real is None or len(real) == 0:
            return [MetricResult("fidelity", 0.0, details={"note": "no real data"})]
        sim = [t.grader_score for t in simulated.trajectories]
        rea = [t.grader_score for t in real.trajectories]
        return [MetricResult(
            "grader_distribution_gap",
            abs(_mean(sim) - _mean(rea)),
            unit="ratio",
            details={"sim_mean": _mean(sim), "real_mean": _mean(rea)},
        )]

    # ── 5. all-in-one report ────────────────────────────────────────────

    @classmethod
    def full_report(
        cls,
        dataset: DBSTrajectoryDataset,
        *,
        baseline: Optional[DBSTrajectoryDataset] = None,
        include_clinical: bool = True,
    ) -> Dict[str, List[MetricResult]]:
        """Run every metric family and return them grouped — the README dump."""
        report: Dict[str, List[MetricResult]] = {
            "online":    cls.online_metrics(list(dataset)),
            "benchmark": cls.benchmark_metrics(dataset),
        }
        if include_clinical:
            try:
                report["clinical"] = cls.clinical_metrics(dataset, baseline=baseline)
            except Exception as exc:
                report["clinical"] = [MetricResult(
                    "clinical_unavailable", 0.0,
                    details={"error": repr(exc)},
                )]
        return report

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _final_beta_suppression(ds: DBSTrajectoryDataset) -> float:
        """Average β-band ARV reduction from first 5 steps to last 5 steps."""
        ratios: List[float] = []
        for t in ds.trajectories:
            beta = t.beta_arvs()
            if len(beta) < 6:
                continue
            head = _mean(beta[:5])
            tail = _mean(beta[-5:])
            if head > 1e-6:
                ratios.append(max(0.0, (head - tail) / head))
        return _mean(ratios)

    @staticmethod
    def _final_tremor_suppression(ds: DBSTrajectoryDataset) -> float:
        ratios: List[float] = []
        for t in ds.trajectories:
            tr = t.tremor_arvs()
            if len(tr) < 6:
                continue
            head = _mean(tr[:5])
            tail = _mean(tr[-5:])
            if head > 1e-6:
                ratios.append(max(0.0, (head - tail) / head))
        return _mean(ratios)

    @staticmethod
    def _side_effect_compliance(
        ds: DBSTrajectoryDataset, threshold: float = 0.45
    ) -> float:
        """Fraction of steps whose side_effect_load stays below ``threshold``."""
        below = 0
        total = 0
        for t in ds.trajectories:
            for v in t.side_effect_load_series():
                total += 1
                if v < threshold:
                    below += 1
        return below / max(1, total)

    @staticmethod
    def _clinical_amplitude_window(
        ds: DBSTrajectoryDataset, lo: float = 1.5, hi: float = 4.0
    ) -> float:
        """Fraction of all amplitudes inside the 1.5–4.0 mA clinical window."""
        in_window = 0
        total = 0
        for t in ds.trajectories:
            for a in t.amplitudes():
                total += 1
                if lo <= a <= hi:
                    in_window += 1
        return in_window / max(1, total)

    @staticmethod
    def _action_smoothness(ds: DBSTrajectoryDataset) -> float:
        """Mean |Δ amplitude| per step — lower is smoother (and clinically safer)."""
        deltas: List[float] = []
        for t in ds.trajectories:
            amps = t.amplitudes()
            for prev, cur in zip(amps[:-1], amps[1:]):
                deltas.append(abs(cur - prev))
        return _mean(deltas)

    @staticmethod
    def _format_validity_rate(ds: DBSTrajectoryDataset) -> float:
        """Fraction of LLM steps that produced parseable JSON."""
        parsed = 0
        total = 0
        for t in ds.trajectories:
            for s in t.steps:
                total += 1
                if s.parsed:
                    parsed += 1
        return parsed / max(1, total)


__all__ = ["EvaluationSuite", "MetricResult"]
