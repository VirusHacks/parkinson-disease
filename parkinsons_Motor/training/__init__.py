"""Training utilities for the Parkinson's-Motor closed-loop DBS environment.

This package mirrors the architecture used by the OpenEnv-Hackathon
bio-experiment winner ([mhtruong1031/OpenENV-Hackathon] ``training/``).
Each module is a single responsibility:

  * :mod:`parkinsons_Motor.training.trajectory`         — JSON-serialisable
    trajectories, datasets, save/load.
  * :mod:`parkinsons_Motor.training.evaluation`         — ``EvaluationSuite``
    with online / benchmark / clinical / fidelity metric families.
  * :mod:`parkinsons_Motor.training.clinical_benchmark` — comparisons against
    published adaptive-DBS literature (Little 2013/2016, Velisar 2019,
    Bronte-Stewart 2020).
  * :mod:`parkinsons_Motor.training.rollout_collection` — CLI to collect
    heuristic / constant-policy rollouts as JSON for offline analysis.
  * :mod:`parkinsons_Motor.training.plots`              — training-curve
    dashboards, base-vs-trained comparisons, before/after trajectory overlays.
    Re-exported by :mod:`parkinsons_Motor.train` so existing notebook imports
    still work.
  * :mod:`parkinsons_Motor.training.llm_eval`           — LLM-driven
    evaluation against the live env: ``sanity_check_rollout``,
    ``evaluate_model_on_task``, ``evaluate_model_suite``,
    ``eval_with_adapter_disabled``. Also re-exported from
    :mod:`parkinsons_Motor.train`.

For the **GRPO training pipeline itself** (LLM rollouts, reward composition,
GRPO trainer glue) see :mod:`parkinsons_Motor.train`. That module is the
runtime entry point used by ``colab_train_motorassist.ipynb``; this package
provides everything around it (offline analysis, evaluation, plotting,
literature benchmarking).
"""

from .evaluation import EvaluationSuite, MetricResult
from .llm_eval import (
    eval_with_adapter_disabled,
    evaluate_model_on_task,
    evaluate_model_suite,
    sanity_check_rollout,
)
from .plots import (
    compare_trajectories,
    plot_baseline_vs_trained,
    plot_training_dashboard,
    plot_training_loss,
    save_training_plots,
)
from .trajectory import DBSTrajectory, DBSTrajectoryDataset, DBSTrajectoryStep

__all__ = [
    # offline analysis surface
    "CLINICAL_TARGETS",
    "DBSTrajectory",
    "DBSTrajectoryDataset",
    "DBSTrajectoryStep",
    "EvaluationSuite",
    "LiteratureTarget",
    "MetricResult",
    "collect_trajectories",
    "compare_to_literature",
    # plots (re-exported by parkinsons_Motor.train as well)
    "compare_trajectories",
    "plot_baseline_vs_trained",
    "plot_training_dashboard",
    "plot_training_loss",
    "save_training_plots",
    # LLM-driven eval (re-exported by parkinsons_Motor.train as well)
    "eval_with_adapter_disabled",
    "evaluate_model_on_task",
    "evaluate_model_suite",
    "sanity_check_rollout",
]


def __getattr__(name: str):
    """Lazy-import optional surfaces so ``import parkinsons_Motor.training``
    stays cheap when only ``Trajectory``/``EvaluationSuite`` are needed."""
    if name in {"CLINICAL_TARGETS", "LiteratureTarget", "compare_to_literature"}:
        from .clinical_benchmark import (
            CLINICAL_TARGETS,
            LiteratureTarget,
            compare_to_literature,
        )
        return {
            "CLINICAL_TARGETS":       CLINICAL_TARGETS,
            "LiteratureTarget":       LiteratureTarget,
            "compare_to_literature":  compare_to_literature,
        }[name]
    if name == "collect_trajectories":
        from .rollout_collection import collect_trajectories
        return collect_trajectories
    raise AttributeError(f"module 'parkinsons_Motor.training' has no attribute {name!r}")
