"""
local_benchmark.py  -  Zero-dependency local environment tester
================================================================
Runs the built-in heuristic baseline policies directly against
ParkinsonsMotorEnvironment (no Docker, no server, no API key needed).

Usage (from repo root):
    uv run --project parkinsons_Motor python local_benchmark.py
    # or just:
    python local_benchmark.py          (if parkinsons_Motor is installed)

Flags:
    --tasks  beta_suppression tremor_correction full_episode
    --seeds  0 1 2
    --output outputs/benchmark/local_run.md   (where to save the report)
    --verbose                                  (print step-by-step state)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from parkinsons_Motor.server.parkinsons_Motor_environment import (
        ParkinsonsMotorEnvironment,
    )
    from parkinsons_Motor.core.models import (
        ParkinsonsMotorAction,
        ParkinsonsMotorObservation,
    )
except ImportError as e:
    print(f"[ERROR] Could not import environment: {e}")
    print("Run with:  uv run --project parkinsons_Motor python local_benchmark.py")
    sys.exit(1)


# ── policies ───────────────────────────────────────────────────────────────────

def _act(motor: float, amp: float, pw: float) -> ParkinsonsMotorAction:
    return ParkinsonsMotorAction(
        motor_command=max(-1.0, min(1.0, motor)),
        dbs_amplitude=max(0.0, min(5.0, amp)),
        dbs_pulse_width=max(0.06, min(0.20, pw)),
    )


def policy_no_dbs(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    return _act(obs.target_output, 0.0, 0.06)


def policy_const_low(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    return _act(obs.target_output, 0.30, 0.08)


def policy_const_mid(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    return _act(obs.target_output, 0.80, 0.10)


def policy_const_high(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    return _act(obs.target_output, 3.00, 0.20)


def policy_safety_aware(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    beta  = obs.beta_arv
    tremor = obs.tremor_arv
    side  = obs.side_effect_load
    task  = obs.task_id

    if task == "beta_suppression":
        amp = 0.08 + 0.08 * (beta > 0.68) + 0.04 * max(beta - 0.72, 0.0)
        amp -= 0.18 * max(side - 0.22, 0.0)
        pw  = 0.07
    elif task == "tremor_correction":
        amp = 1.60
        pw  = 0.12
    else:
        amp = 1.50 + 0.28 * beta + 0.14 * tremor
        if side > 0.31:
            amp -= 1.66 * (side - 0.31)
        if side > 0.50:
            amp *= 0.63
        pw = 0.095

    cmd_gain = 1.0 + 0.60 * beta + 0.60 * tremor - 0.08 * side
    if task == "tremor_correction":
        cmd_gain = 1.15
    elif task == "full_episode":
        cmd_gain = 1.0 + 0.49 * beta + 0.34 * tremor

    return _act(obs.target_output * cmd_gain, amp, pw)


def policy_reference(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    gt_amp = float((obs.metadata or {}).get("ground_truth_dbs_ma", 0.0))
    beta   = obs.beta_arv
    tremor = obs.tremor_arv
    side   = obs.side_effect_load
    amp    = 0.55 * gt_amp + 0.45 * (0.30 + 0.80 * beta + 0.35 * tremor) - 0.30 * side
    pw     = 0.08 + 0.02 * beta + 0.02 * tremor
    cmd    = obs.target_output * (1.0 + 0.25 * beta + 0.20 * tremor)
    return _act(cmd, amp, pw)


def policy_adaptive(obs: ParkinsonsMotorObservation) -> ParkinsonsMotorAction:
    """
    Adaptive policy: tracks trends and adjusts amplitude smoothly.
    Uses beta_trend / tremor_trend to decide whether to ramp up or back off.
    """
    beta      = obs.beta_arv
    tremor    = obs.tremor_arv
    side      = obs.side_effect_load
    b_trend   = obs.beta_trend
    t_trend   = obs.tremor_trend
    se_rate   = obs.side_effect_rate
    prev_amp  = obs.dbs_amplitude_ma
    task      = obs.task_id

    # Task-specific safe ceiling and side-effect budget
    _ceil = {"beta_suppression": 1.0, "tremor_correction": 1.8, "full_episode": 2.4}
    _budget = {"beta_suppression": 0.30, "tremor_correction": 0.46, "full_episode": 0.60}
    ceil   = _ceil.get(task, 1.5)
    budget = _budget.get(task, 0.40)

    # Start with a proportional base
    amp = 0.40 + 0.70 * beta + 0.50 * tremor

    # Trend adjustments
    if b_trend > 0.015 or t_trend > 0.015:
        amp = prev_amp + 0.12       # symptoms worsening - step up
    elif b_trend < -0.010 and t_trend < -0.010 and se_rate < 0.0:
        amp = prev_amp - 0.08       # both improving and side effects recovering - step down
    else:
        amp = 0.70 * prev_amp + 0.30 * amp   # blend toward target

    # Safety backoff
    if side > budget * 0.90:
        amp *= 0.65
    elif side > budget * 0.75:
        amp *= 0.80

    # Clamp to task ceiling
    amp = max(0.10, min(ceil, amp))

    pw = 0.10 + 0.03 * beta
    pw = max(0.06, min(0.18, pw))

    cmd_gain = 1.0 + 0.40 * beta + 0.30 * tremor - 0.10 * side
    return _act(obs.target_output * cmd_gain, amp, pw)


POLICIES: Dict[str, Callable] = {
    "no_dbs":       policy_no_dbs,
    "const_low":    policy_const_low,
    "const_mid":    policy_const_mid,
    "const_high":   policy_const_high,
    "safety_aware": policy_safety_aware,
    "reference":    policy_reference,
    "adaptive":     policy_adaptive,
}

SUCCESS_THRESHOLD = {
    "beta_suppression": 0.54,
    "tremor_correction": 0.32,
    "full_episode": 0.60,
}

# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    seed: int,
    policy_fn: Callable,
    patient_profile_id: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    env = ParkinsonsMotorEnvironment(seed=seed)
    obs = env.reset(task_id=task_id, seed=seed, patient_profile_id=patient_profile_id)

    profile_id = obs.metadata.get("patient_profile_id", "?")
    n_steps    = obs.metadata.get("episode_steps", 0)
    step = 0
    rewards: List[float] = []

    if verbose:
        print(
            f"      profile={profile_id}  steps={n_steps}  "
            f"target={obs.target_output:+.2f}"
        )
        print(
            f"      {'step':>4}  {'amp':>5}  {'pw':>5}  "
            f"{'beta':>5}  {'tremor':>5}  {'force':>5}  "
            f"{'se':>5}  {'reward':>7}"
        )

    while not obs.done:
        action = policy_fn(obs)
        obs    = env.step(action)
        step  += 1
        r = obs.reward
        rewards.append(r)

        if verbose and (step <= 5 or step % 10 == 0 or obs.done):
            print(
                f"      {step:>4}  {action.dbs_amplitude:>5.2f}  "
                f"{action.dbs_pulse_width:>5.3f}  "
                f"{obs.beta_arv:>5.3f}  {obs.tremor_arv:>5.3f}  "
                f"{obs.force_preserved:>5.3f}  "
                f"{obs.side_effect_load:>5.3f}  {r:>+7.4f}"
            )

    score    = float(obs.grader_score) if obs.grader_score >= 0 else 0.0
    success  = score >= SUCCESS_THRESHOLD.get(task_id, 0.50)
    details  = obs.metadata.get("score_details", {})

    return {
        "score":    score,
        "success":  success,
        "steps":    step,
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "profile":  profile_id,
        "details":  details,
    }


def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in runs]
    return {
        "mean":     statistics.mean(scores) if scores else 0.0,
        "std":      statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min":      min(scores) if scores else 0.0,
        "max":      max(scores) if scores else 0.0,
        "successes": sum(1 for r in runs if r["success"]),
        "n":        len(runs),
    }


# ── report builder ─────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "#" * filled + "-" * (width - filled)


def build_report(
    results: Dict[str, Dict[str, Any]],
    tasks: List[str],
    seeds: List[int],
    elapsed: float,
) -> str:
    lines = [
        "# Parkinson's Motor Environment - Local Benchmark Report",
        "",
        f"**Tasks:** {', '.join(tasks)}  |  **Seeds:** {seeds}  |  **Elapsed:** {elapsed:.1f}s",
        "",
    ]

    for task_id in tasks:
        thresh = SUCCESS_THRESHOLD.get(task_id, 0.5)
        lines += [f"## {task_id}  (success threshold: {thresh:.2f})", ""]
        lines += [
            "| Policy | Mean +/- Std | Min | Max | Successes | Bar |",
            "|---|---|---|---|---|---|",
        ]
        task_data = results.get(task_id, {})
        for policy, summary in task_data.items():
            bar  = _bar(summary["mean"])
            flag = "PASS" if summary["successes"] == summary["n"] else (
                   "PART" if summary["successes"] > 0 else "FAIL"
            )
            lines.append(
                f"| {policy} "
                f"| {summary['mean']:.3f} +/- {summary['std']:.3f} "
                f"| {summary['min']:.3f} "
                f"| {summary['max']:.3f} "
                f"| {flag} {summary['successes']}/{summary['n']} "
                f"| `{bar}` |"
            )
        lines.append("")

    # Ladder check
    lines += [
        "## Difficulty Ladder Check",
        "",
        "Naive baselines should fail; adaptive/reference should outperform.",
        "",
    ]
    for task_id in tasks:
        task_data = results.get(task_id, {})
        no_dbs_score  = task_data.get("no_dbs",    {}).get("mean", 0.0)
        ref_score     = task_data.get("reference",  {}).get("mean", 0.0)
        adap_score    = task_data.get("adaptive",   {}).get("mean", 0.0)
        thresh = SUCCESS_THRESHOLD.get(task_id, 0.5)
        ladder_ok = no_dbs_score < thresh and (ref_score > no_dbs_score or adap_score > no_dbs_score)
        lines.append(
            f"- **{task_id}**: no_dbs={no_dbs_score:.3f}  "
            f"reference={ref_score:.3f}  adaptive={adap_score:.3f}  "
            f"=> {'[LADDER OK]' if ladder_ok else '[CHECK THRESHOLDS]'}"
        )

    lines += ["", "---", "_Generated by `local_benchmark.py`_", ""]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Parkinson's DBS benchmark (no API needed)")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["beta_suppression", "tremor_correction", "full_episode"],
        choices=["beta_suppression", "tremor_correction", "full_episode"],
        help="Which tasks to evaluate",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Random seeds (each seed = one episode per policy)",
    )
    parser.add_argument(
        "--policies", nargs="+", default=list(POLICIES.keys()),
        choices=list(POLICIES.keys()),
        help="Which policies to run",
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/benchmark/local_benchmark.md",
        help="Path to save Markdown report",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print step-by-step state for every episode",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 seed, 3 policies (no_dbs / safety_aware / reference)",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds    = [0]
        args.policies = ["no_dbs", "safety_aware", "reference", "adaptive"]

    print()
    print("=" * 70)
    print("  Parkinson's Motor Environment  --  Local Benchmark")
    print("  (no API key  |  no Docker  |  no server)")
    print("=" * 70)
    print(f"  Tasks   : {args.tasks}")
    print(f"  Seeds   : {args.seeds}")
    print(f"  Policies: {args.policies}")
    print()

    t0 = time.time()
    all_results: Dict[str, Dict[str, Any]] = {}

    for task_id in args.tasks:
        print("\n" + "-"*70)
        print(f"  TASK: {task_id}   (success >= {SUCCESS_THRESHOLD.get(task_id, 0.5):.2f})")
        print("-"*70)
        all_results[task_id] = {}

        for pol_name in args.policies:
            policy_fn = POLICIES[pol_name]
            runs: List[Dict[str, Any]] = []

            for seed in args.seeds:
                if args.verbose:
                    print(f"\n    [{pol_name}] seed={seed}")
                run = run_episode(
                    task_id=task_id,
                    seed=seed,
                    policy_fn=policy_fn,
                    verbose=args.verbose,
                )
                runs.append(run)

            summary = summarize(runs)
            all_results[task_id][pol_name] = summary

            bar    = _bar(summary["mean"])
            flag   = "PASS" if summary["successes"] == summary["n"] else (
                     "PART" if summary["successes"] > 0 else "FAIL"
            )
            print(
                f"  {pol_name:<14} "
                f"score={summary['mean']:.3f}±{summary['std']:.3f}  "
                f"[{bar}]  "
                f"{flag} {summary['successes']}/{summary['n']} pass"
            )

    elapsed = time.time() - t0

    # ── overall summary ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    header = f"  {'Policy':<14}"
    for t in args.tasks:
        short = t.replace("beta_suppression","beta").replace("tremor_correction","tremor").replace("full_episode","full")
        header += f"  {short:>8}"
    header += f"  {'mean':>7}"
    print(header)
    print("  " + "-"*14 + ("  " + "-"*8) * len(args.tasks) + "  " + "-"*7)

    for pol_name in args.policies:
        row = f"  {pol_name:<14}"
        scores = []
        for t in args.tasks:
            s = all_results.get(t, {}).get(pol_name, {}).get("mean", 0.0)
            scores.append(s)
            row += f"  {s:>8.3f}"
        overall_mean = statistics.mean(scores) if scores else 0.0
        row += f"  {overall_mean:>7.3f}"
        print(row)

    # Threshold row
    thresh_row = f"  {'[threshold]':<14}"
    for t in args.tasks:
        thresh_row += f"  {SUCCESS_THRESHOLD.get(t, 0.5):>8.3f}"
    print(thresh_row)

    print(f"  Elapsed: {elapsed:.1f}s")
    print("="*70 + "\n")

    # ── write report ───────────────────────────────────────────────────────────
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report   = build_report(all_results, args.tasks, args.seeds, elapsed)
    out_path.write_text(report, encoding="utf-8")

    # Also save JSON for further analysis
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"  Saved Markdown → {out_path}")
    print(f"  Saved JSON     → {json_path}\n")


if __name__ == "__main__":
    main()
