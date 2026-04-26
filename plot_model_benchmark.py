"""
Model Benchmark Plotter
=======================
Reads outputs/benchmark/summary.json (written by run_model_benchmark.py)
and generates 8 comparison plots to plots/benchmark/.

Can also be run with --mock to generate plots with synthetic data for
layout testing before the real benchmark finishes.

Usage:
  python plot_model_benchmark.py            # reads real results
  python plot_model_benchmark.py --mock     # synthetic demo data
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent
SUMMARY_JSON = REPO_ROOT / "outputs" / "benchmark" / "summary.json"
PLOTS_DIR   = REPO_ROOT / "plots" / "benchmark"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Qwen2.5-7B":  "#4C72B0",
    "Qwen2.5-72B": "#DD8452",
    "Mistral-7B":  "#55A868",
}
TASK_COLORS = {"easy": "#2CA02C", "medium": "#FF7F0E", "hard": "#D62728"}
TASK_THRESHOLDS = {"easy": 0.55, "medium": 0.52, "hard": 0.68}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "#dddddd",
    "grid.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

SUBTITLE = "Qwen2.5-7B / Qwen2.5-72B / Mistral-7B  |  MotorAssistEnv (easy/medium/hard)"


def save(name: str) -> None:
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: plots/benchmark/{name}")


# ── Mock data (used when --mock flag is passed or summary.json missing) ────────

def _mock_data() -> list[dict]:
    rng = np.random.default_rng(42)
    models = [
        ("Qwen2.5-7B",  "Qwen/Qwen2.5-7B-Instruct",  "qwen25_7b"),
        ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", "qwen25_72b"),
        ("Mistral-7B",  "mistralai/Mistral-7B-Instruct-v0.3", "mistral_7b"),
    ]
    tasks = [
        ("easy",   36, 0.55),
        ("medium", 60, 0.52),
        ("hard",   150, 0.68),
    ]
    # Rough expected scores from docs: easy all pass, medium 72B passes, hard none
    base_scores = {
        "Qwen2.5-7B":  {"easy": 0.72, "medium": 0.51, "hard": 0.40},
        "Qwen2.5-72B": {"easy": 0.80, "medium": 0.62, "hard": 0.59},
        "Mistral-7B":  {"easy": 0.68, "medium": 0.47, "hard": 0.35},
    }
    results = []
    for display, model_id, slug in models:
        per_task = []
        for task_id, n_steps, threshold in tasks:
            base = base_scores[display][task_id]
            seeds = [
                float(np.clip(rng.normal(base, 0.04), 0, 1)) for _ in range(3)
            ]
            mean_s = np.mean(seeds)
            std_s  = np.std(seeds)
            passes = sum(1 for s in seeds if s >= threshold)

            # Synthetic per-step reward trace for this task
            def _trace(n, score):
                t = np.linspace(0, 1, n)
                base_r = score * 0.85 + 0.1 * np.sin(4 * np.pi * t)
                noise  = rng.normal(0, 0.04, n)
                return list(np.clip(base_r + noise, 0, 1).round(4))

            rollouts = [
                {
                    "seed":       i,
                    "score":      round(seeds[i], 6),
                    "success":    seeds[i] >= threshold,
                    "mean_reward": round(float(np.mean(_trace(n_steps, seeds[i]))), 4),
                    "rewards":    _trace(n_steps, seeds[i]),
                    "amplitudes": list(np.clip(rng.normal(1.2, 0.2, n_steps), 0, 5).round(3)),
                    "betas":      list(np.clip(rng.normal(0.35, 0.1, n_steps), 0, 1).round(4)),
                    "tremors":    list(np.clip(rng.normal(0.30, 0.08, n_steps), 0, 1).round(4)),
                    "forces":     list(np.clip(rng.normal(0.70, 0.08, n_steps), 0, 1).round(4)),
                    "se_loads":   list(np.clip(rng.normal(0.25, 0.07, n_steps), 0, 1).round(4)),
                    "score_details": {},
                    "error": None,
                }
                for i in range(3)
            ]
            per_task.append({
                "task_id":       task_id,
                "n_seeds":       3,
                "score_mean":    round(float(mean_s), 6),
                "score_std":     round(float(std_s), 6),
                "score_min":     round(float(min(seeds)), 6),
                "score_max":     round(float(max(seeds)), 6),
                "pass_rate":     round(passes / 3, 4),
                "passes":        passes,
                "threshold":     threshold,
                "mean_reward":   round(float(mean_s * 0.88), 4),
                "mean_beta":     round(float(rng.uniform(0.28, 0.45)), 4),
                "mean_tremor":   round(float(rng.uniform(0.22, 0.40)), 4),
                "mean_force":    round(float(rng.uniform(0.55, 0.80)), 4),
                "mean_se_load":  round(float(rng.uniform(0.18, 0.38)), 4),
                "mean_amplitude":round(float(rng.uniform(1.0, 1.8)), 4),
                "rollouts":      rollouts,
            })
        results.append({
            "model_display_name": display,
            "model_id":           model_id,
            "slug":               slug,
            "tasks":              [t[0] for t in tasks],
            "seeds":              [0, 1, 2],
            "per_task":           per_task,
            "overall_mean_score": round(float(np.mean([pt["score_mean"] for pt in per_task])), 6),
        })
    return results


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(use_mock: bool = False) -> list[dict]:
    if use_mock or not SUMMARY_JSON.exists():
        print("  [INFO] Using mock/synthetic data (pass real data via outputs/benchmark/summary.json)")
        return _mock_data()
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _model_color(name: str) -> str:
    for k, c in MODEL_COLORS.items():
        if k.lower() in name.lower():
            return c
    return "#999999"


def _task_color(task_id: str) -> str:
    return TASK_COLORS.get(task_id, "#888888")


def _mean_trace(rollouts: list[dict], key: str = "rewards", n: int | None = None) -> np.ndarray:
    """Average a per-step list across all seeds, padding/truncating to length n."""
    traces = [r.get(key, []) for r in rollouts]
    if not any(traces):
        return np.array([])
    if n is None:
        n = max(len(t) for t in traces)
    arr = np.full((len(traces), n), np.nan)
    for i, t in enumerate(traces):
        arr[i, :len(t)] = t[:n]
    return np.nanmean(arr, axis=0)


# =============================================================================
# 10 — Score bar chart: models × tasks
# =============================================================================
def plot_score_bars(data: list[dict]) -> None:
    tasks   = ["easy", "medium", "hard"]
    models  = [mr["model_display_name"] for mr in data]
    n_m, n_t = len(models), len(tasks)
    x = np.arange(n_t)
    w = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = np.linspace(-(n_m - 1) * w / 2, (n_m - 1) * w / 2, n_m)

    for i, mr in enumerate(data):
        means = [next((pt["score_mean"] for pt in mr["per_task"] if pt["task_id"] == t), 0) for t in tasks]
        stds  = [next((pt["score_std"]  for pt in mr["per_task"] if pt["task_id"] == t), 0) for t in tasks]
        bars  = ax.bar(
            x + offsets[i], means, w,
            yerr=stds, capsize=4,
            color=_model_color(mr["model_display_name"]),
            alpha=0.85,
            label=mr["model_display_name"],
            error_kw={"elinewidth": 1.2},
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8,
            )

    # Threshold lines
    for task_id, threshold in TASK_THRESHOLDS.items():
        xi = tasks.index(task_id)
        ax.hlines(threshold, xi - 0.42, xi + 0.42,
                  colors=_task_color(task_id), linewidths=1.8, linestyles="--",
                  label=f"{task_id} threshold ({threshold})")

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=12)
    ax.set_ylabel("Score (mean ± std across seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Episode Score by Model and Task\n{SUBTITLE}")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    save("10_score_by_model_task.png")


# =============================================================================
# 11 — Per-step reward curves: one panel per task
# =============================================================================
def plot_reward_curves(data: list[dict]) -> None:
    tasks = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Per-Step Reward Curves (mean across seeds)\n{SUBTITLE}", fontsize=12)

    for ax, task_id in zip(axes, tasks):
        for mr in data:
            pt = next((p for p in mr["per_task"] if p["task_id"] == task_id), None)
            if not pt:
                continue
            trace = _mean_trace(pt["rollouts"], "rewards")
            if len(trace) == 0:
                continue
            steps  = np.arange(1, len(trace) + 1)
            color  = _model_color(mr["model_display_name"])
            ax.plot(steps, trace, color=color, linewidth=2.0,
                    label=mr["model_display_name"], alpha=0.85)
            # Rolling mean overlay
            rm = np.convolve(trace, np.ones(5) / 5, mode="valid")
            ax.plot(np.arange(3, len(rm) + 3), rm, color=color,
                    linewidth=1.0, linestyle="--", alpha=0.5)

        threshold = TASK_THRESHOLDS.get(task_id, 0.5)
        ax.axhline(threshold, color="black", linewidth=1.2, linestyle=":",
                   label=f"Threshold ({threshold})")
        ax.set_title(f"{task_id.capitalize()} Task")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward" if task_id == "easy" else "")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save("11_reward_curves_per_task.png")


# =============================================================================
# 12 — Pass / fail heatmap
# =============================================================================
def plot_pass_rate_heatmap(data: list[dict]) -> None:
    tasks  = ["easy", "medium", "hard"]
    models = [mr["model_display_name"] for mr in data]

    matrix = np.zeros((len(models), len(tasks)))
    for i, mr in enumerate(data):
        for j, task_id in enumerate(tasks):
            pt = next((p for p in mr["per_task"] if p["task_id"] == task_id), None)
            if pt:
                matrix[i, j] = pt["pass_rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    for i in range(len(models)):
        for j in range(len(tasks)):
            v = matrix[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if v < 0.4 or v > 0.75 else "black")
    plt.colorbar(im, ax=ax, label="Pass Rate")
    ax.set_title(f"Pass Rate Heatmap (% seeds above threshold)\n{SUBTITLE}")
    fig.tight_layout()
    save("12_pass_rate_heatmap.png")


# =============================================================================
# 13 — Beta & Tremor suppression comparison
# =============================================================================
def plot_biomarker_comparison(data: list[dict]) -> None:
    tasks  = ["easy", "medium", "hard"]
    models = [mr["model_display_name"] for mr in data]
    x      = np.arange(len(tasks))
    w      = 0.22
    n_m    = len(models)
    offsets = np.linspace(-(n_m - 1) * w / 2, (n_m - 1) * w / 2, n_m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Mean Beta & Tremor ARV by Model and Task (lower = better)\n{SUBTITLE}")

    for ax, metric, label in [
        (axes[0], "mean_beta",   "Mean Beta ARV"),
        (axes[1], "mean_tremor", "Mean Tremor ARV"),
    ]:
        for i, mr in enumerate(data):
            vals = [
                next((pt[metric] for pt in mr["per_task"] if pt["task_id"] == t), 0)
                for t in tasks
            ]
            bars = ax.bar(
                x + offsets[i], vals, w,
                color=_model_color(mr["model_display_name"]),
                alpha=0.85, label=mr["model_display_name"],
            )
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(["Easy", "Medium", "Hard"])
        ax.set_ylabel(label)
        ax.set_ylim(0, 0.8)
        ax.legend(fontsize=8)
        ax.set_title(label)

    fig.tight_layout()
    save("13_biomarker_comparison.png")


# =============================================================================
# 14 — Amplitude traces: one panel per task
# =============================================================================
def plot_amplitude_traces(data: list[dict]) -> None:
    tasks = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Mean DBS Amplitude Trace (mean across seeds)\n{SUBTITLE}")

    for ax, task_id in zip(axes, tasks):
        for mr in data:
            pt = next((p for p in mr["per_task"] if p["task_id"] == task_id), None)
            if not pt:
                continue
            trace = _mean_trace(pt["rollouts"], "amplitudes")
            if len(trace) == 0:
                continue
            steps = np.arange(1, len(trace) + 1)
            ax.plot(steps, trace, linewidth=1.8, alpha=0.85,
                    color=_model_color(mr["model_display_name"]),
                    label=mr["model_display_name"])

        ax.set_title(f"{task_id.capitalize()} Task")
        ax.set_xlabel("Step")
        ax.set_ylabel("DBS Amplitude (mA)" if task_id == "easy" else "")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 3.0)

    fig.tight_layout()
    save("14_amplitude_traces.png")


# =============================================================================
# 15 — Overall score ranking (horizontal bar)
# =============================================================================
def plot_overall_ranking(data: list[dict]) -> None:
    models = [mr["model_display_name"] for mr in data]
    scores = [mr["overall_mean_score"] for mr in data]
    colors = [_model_color(m) for m in models]

    # Sort descending
    order  = np.argsort(scores)[::-1]
    models = [models[i] for i in order]
    scores = [scores[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(models, scores, color=colors, alpha=0.85)
    for bar, s in zip(bars, scores):
        ax.text(s + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{s:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Overall Mean Score (avg across easy/medium/hard)")
    ax.set_title(f"Overall Model Ranking\n{SUBTITLE}")
    fig.tight_layout()
    save("15_overall_ranking.png")


# =============================================================================
# 16 — Side-effect load comparison
# =============================================================================
def plot_se_load(data: list[dict]) -> None:
    tasks   = ["easy", "medium", "hard"]
    models  = [mr["model_display_name"] for mr in data]
    x       = np.arange(len(tasks))
    w       = 0.22
    n_m     = len(models)
    offsets = np.linspace(-(n_m - 1) * w / 2, (n_m - 1) * w / 2, n_m)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mr in enumerate(data):
        vals = [
            next((pt["mean_se_load"] for pt in mr["per_task"] if pt["task_id"] == t), 0)
            for t in tasks
        ]
        bars = ax.bar(
            x + offsets[i], vals, w,
            color=_model_color(mr["model_display_name"]),
            alpha=0.85, label=mr["model_display_name"],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Mean Side-Effect Load")
    ax.set_title(f"Side-Effect Load by Model and Task (lower = safer)\n{SUBTITLE}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save("16_side_effect_load.png")


# =============================================================================
# 17 — Score distribution violin / box per model
# =============================================================================
def plot_score_distribution(data: list[dict]) -> None:
    tasks = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Score Distribution Across Seeds\n{SUBTITLE}")

    for ax, task_id in zip(axes, tasks):
        scores_by_model: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []

        for mr in data:
            pt = next((p for p in mr["per_task"] if p["task_id"] == task_id), None)
            if not pt:
                continue
            seed_scores = [r["score"] for r in pt["rollouts"]]
            scores_by_model.append(seed_scores)
            labels.append(mr["model_display_name"])
            colors.append(_model_color(mr["model_display_name"]))

        if not scores_by_model:
            continue

        bplot = ax.boxplot(
            scores_by_model, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay jittered points
        rng = np.random.default_rng(99)
        for i, (scores, color) in enumerate(zip(scores_by_model, colors), start=1):
            jitter = rng.uniform(-0.08, 0.08, len(scores))
            ax.scatter(np.full(len(scores), i) + jitter, scores,
                       color=color, s=40, zorder=5, alpha=0.9)

        threshold = TASK_THRESHOLDS.get(task_id, 0.5)
        ax.axhline(threshold, color="black", linewidth=1.2, linestyle=":",
                   label=f"threshold ({threshold})")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score" if task_id == "easy" else "")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{task_id.capitalize()} Task")
        ax.legend(fontsize=8)

    fig.tight_layout()
    save("17_score_distribution.png")


# =============================================================================
# Main
# =============================================================================

def main(use_mock: bool = False) -> None:
    print(f"\nModel Benchmark Plotter")
    print(f"  Summary JSON : {SUMMARY_JSON}")
    print(f"  Output dir   : {PLOTS_DIR}")

    data = load_data(use_mock=use_mock)

    print(f"\n  Loaded results for {len(data)} model(s):")
    for mr in data:
        print(f"    {mr['model_display_name']:20s}  overall={mr['overall_mean_score']:.4f}")

    print("\nGenerating plots...")
    plot_score_bars(data)
    plot_reward_curves(data)
    plot_pass_rate_heatmap(data)
    plot_biomarker_comparison(data)
    plot_amplitude_traces(data)
    plot_overall_ranking(data)
    plot_se_load(data)
    plot_score_distribution(data)

    print(f"\nAll benchmark plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    mock = "--mock" in sys.argv
    main(use_mock=mock)
