"""Public and held-out evaluation suite for the Parkinson's Motor benchmark."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from parkinsons_Motor.core.models import ParkinsonsMotorAction
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment


TASKS = ["easy", "medium", "hard"]
EXTENDED_TASKS = [
    "fragile_patient",
    "refractory_patient",
    "personalization_generalization",
]
PUBLIC_SEEDS = [0, 1, 2, 3]
HELDOUT_CONFIG = {
    "easy": {"seed": 101, "patient_profile_id": "fragile"},
    "medium": {"seed": 102, "patient_profile_id": "fragile"},
    "hard": {"seed": 103, "patient_profile_id": "refractory"},
    "fragile_patient": {"seed": 104, "patient_profile_id": "fragile"},
    "refractory_patient": {"seed": 105, "patient_profile_id": "refractory"},
    "personalization_generalization": {"seed": 106, "patient_profile_id": "refractory"},
}

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark"
JSON_PATH = OUTPUT_DIR / "benchmark_eval.json"
MD_PATH = OUTPUT_DIR / "benchmark_eval.md"


def no_dbs_policy(obs) -> ParkinsonsMotorAction:
    return ParkinsonsMotorAction(
        motor_command=obs.target_output,
        dbs_amplitude=0.0,
        dbs_pulse_width=0.06,
    )


def constant_policy(amp: float, pw: float, gain: float = 1.0) -> Callable:
    def _policy(obs) -> ParkinsonsMotorAction:
        cmd = max(-1.0, min(1.0, obs.target_output * gain))
        return ParkinsonsMotorAction(
            motor_command=cmd,
            dbs_amplitude=amp,
            dbs_pulse_width=pw,
        )

    return _policy


def safety_aware_policy(obs) -> ParkinsonsMotorAction:
    beta = obs.beta_arv
    tremor = obs.tremor_arv
    side = obs.side_effect_load
    task = obs.task_id

    if task == "easy":
        amp = 0.08 + 0.08 * (beta > 0.68) + 0.04 * max(beta - 0.72, 0.0)
        amp -= 0.18 * max(side - 0.22, 0.0)
        pw = 0.07
    elif task == "medium":
        amp = 1.60
        pw = 0.12
    elif task == "fragile_patient":
        amp = 0.26 + 0.12 * beta + 0.08 * tremor - 0.35 * max(side - 0.18, 0.0)
        pw = 0.08
    elif task == "refractory_patient":
        amp = 0.30 + 0.14 * beta + 0.10 * tremor - 0.20 * max(side - 0.26, 0.0)
        pw = 0.08
    elif task == "hard":
        amp = 1.5018531270975986 + 0.2788883528077374 * beta + 0.1425726032821526 * tremor
        if side > 0.3055597938808898:
            amp -= 1.6639934475532732 * (side - 0.3055597938808898)
        if side > 0.5034375079931643:
            amp *= 0.6338459016108212
        pw = 0.09481476382603633
    else:
        amp = 0.50 + 0.45 * beta + 0.22 * tremor - 0.50 * max(side - 0.28, 0.0)
        pw = 0.09 + 0.015 * (beta + tremor)

    amp = max(0.0, min(5.0, amp))
    pw = max(0.06, min(0.20, pw))
    if task == "medium":
        cmd_gain = 1.15
    elif task == "fragile_patient":
        cmd_gain = 1.0 + 0.28 * beta + 0.18 * tremor - 0.08 * side
    elif task == "refractory_patient":
        cmd_gain = 1.0 + 0.30 * beta + 0.20 * tremor - 0.04 * side
    elif task == "hard":
        cmd_gain = 1.0 + 0.4894260933407289 * beta + 0.3446252762938653 * tremor
    else:
        cmd_gain = 1.0 + 0.45 * beta + 0.35 * tremor - 0.08 * side
    cmd = max(-1.0, min(1.0, obs.target_output * cmd_gain))
    return ParkinsonsMotorAction(
        motor_command=cmd,
        dbs_amplitude=amp,
        dbs_pulse_width=pw,
    )


def reference_policy(obs) -> ParkinsonsMotorAction:
    gt_amp = float(obs.metadata.get("ground_truth_dbs_ma", 0.0))
    beta = obs.beta_arv
    tremor = obs.tremor_arv
    side = obs.side_effect_load
    amp = 0.55 * gt_amp + 0.45 * (0.30 + 0.80 * beta + 0.35 * tremor) - 0.30 * side
    pw = 0.08 + 0.02 * beta + 0.02 * tremor
    cmd = max(-1.0, min(1.0, obs.target_output * (1.0 + 0.25 * beta + 0.20 * tremor)))
    return ParkinsonsMotorAction(
        motor_command=cmd,
        dbs_amplitude=max(0.0, min(5.0, amp)),
        dbs_pulse_width=max(0.06, min(0.20, pw)),
    )


POLICIES: Dict[str, Callable] = {
    "no_dbs": no_dbs_policy,
    "const_low": constant_policy(0.3, 0.08),
    "const_mid": constant_policy(0.8, 0.10),
    "const_high": constant_policy(3.0, 0.20),
    "safety_aware": safety_aware_policy,
    "reference": reference_policy,
}


def run_episode(
    task_id: str,
    seed: int,
    policy_fn: Callable,
    patient_profile_id: str | None = None,
) -> Dict[str, object]:
    env = ParkinsonsMotorEnvironment()
    obs = env.reset(task_id=task_id, seed=seed, patient_profile_id=patient_profile_id)
    profile_id = obs.metadata["patient_profile_id"]

    while not obs.done:
        obs = env.step(policy_fn(obs))

    return {
        "score": float(obs.grader_score),
        "success": bool(obs.episode_success),
        "patient_profile_id": profile_id,
        "score_details": obs.metadata.get("score_details", {}),
    }


def summarize_scores(scores: Iterable[float]) -> Dict[str, float]:
    values = list(scores)
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def format_summary(summary: Dict[str, float]) -> str:
    return (
        f"{summary['mean']:.3f} +- {summary['std']:.3f} "
        f"(min={summary['min']:.3f}, max={summary['max']:.3f})"
    )


def _evaluate_task_set(task_ids: List[str]) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for task_id in task_ids:
        task_results: Dict[str, object] = {}
        for name, policy_fn in POLICIES.items():
            runs = [run_episode(task_id, seed, policy_fn) for seed in PUBLIC_SEEDS]
            scores = [float(run["score"]) for run in runs]
            task_results[name] = {
                "summary": summarize_scores(scores),
                "successes": sum(1 for run in runs if run["success"]),
                "num_runs": len(runs),
                "profiles": [str(run["patient_profile_id"]) for run in runs],
                "runs": runs,
            }
        results[task_id] = task_results
    return results


def evaluate_public() -> Dict[str, Dict[str, object]]:
    return _evaluate_task_set(TASKS)


def evaluate_extended() -> Dict[str, Dict[str, object]]:
    return _evaluate_task_set(EXTENDED_TASKS)


def evaluate_heldout() -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for task_id in TASKS + EXTENDED_TASKS:
        cfg = HELDOUT_CONFIG[task_id]
        task_results: Dict[str, object] = {}
        for name, policy_fn in POLICIES.items():
            run = run_episode(
                task_id,
                int(cfg["seed"]),
                policy_fn,
                patient_profile_id=str(cfg["patient_profile_id"]),
            )
            task_results[name] = run
        results[task_id] = task_results
    return results


def build_markdown(
    public: Dict[str, Dict[str, object]],
    extended: Dict[str, Dict[str, object]],
    heldout: Dict[str, Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# Benchmark Evaluation Report")
    lines.append("")
    lines.append("This report is auto-generated by `parkinsons_Motor/evaluation/eval_suite.py`.")
    lines.append("")
    lines.append("## Public Benchmark")
    lines.append("")
    for task_id in TASKS:
        lines.append(f"### {task_id}")
        lines.append("")
        lines.append("| Policy | Score | Success | Profiles |")
        lines.append("|---|---|---:|---|")
        for name, block in public[task_id].items():
            summary = format_summary(block["summary"])  # type: ignore[index]
            successes = block["successes"]  # type: ignore[index]
            num_runs = block["num_runs"]  # type: ignore[index]
            profiles = ", ".join(block["profiles"])  # type: ignore[index]
            lines.append(f"| {name} | {summary} | {successes}/{num_runs} | {profiles} |")
        lines.append("")

    lines.append("## Extended Scenario Tasks")
    lines.append("")
    lines.append("These are named benchmark-extension tasks requested in the implementation plan.")
    lines.append("")
    for task_id in EXTENDED_TASKS:
        lines.append(f"### {task_id}")
        lines.append("")
        lines.append("| Policy | Score | Success | Profiles |")
        lines.append("|---|---|---:|---|")
        for name, block in extended[task_id].items():
            summary = format_summary(block["summary"])  # type: ignore[index]
            successes = block["successes"]  # type: ignore[index]
            num_runs = block["num_runs"]  # type: ignore[index]
            profiles = ", ".join(block["profiles"])  # type: ignore[index]
            lines.append(f"| {name} | {summary} | {successes}/{num_runs} | {profiles} |")
        lines.append("")

    lines.append("## Held-Out Hard Cases")
    lines.append("")
    for task_id in TASKS + EXTENDED_TASKS:
        cfg = HELDOUT_CONFIG[task_id]
        lines.append(f"### {task_id} (`{cfg['patient_profile_id']}`, seed `{cfg['seed']}`)")
        lines.append("")
        lines.append("| Policy | Score | Success |")
        lines.append("|---|---:|---:|")
        for name, run in heldout[task_id].items():
            score = float(run["score"])  # type: ignore[index]
            success = bool(run["success"])  # type: ignore[index]
            lines.append(f"| {name} | {score:.3f} | {int(success)} |")
        lines.append("")

    lines.append("## Ladder Check")
    lines.append("")
    lines.append("- `no_dbs` should fail all core tasks.")
    lines.append("- Constant policies should mostly fail or remain marginal.")
    lines.append("- `safety_aware` and `reference` should outperform simple constants on at least some tasks.")
    lines.append("- Fragile, refractory, and personalization tasks should remain clearly harder than the public core ladder.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    public = evaluate_public()
    extended = evaluate_extended()
    heldout = evaluate_heldout()
    payload = {
        "public": public,
        "extended": extended,
        "heldout": heldout,
        "public_seeds": PUBLIC_SEEDS,
        "heldout_config": HELDOUT_CONFIG,
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    MD_PATH.write_text(build_markdown(public, extended, heldout), encoding="utf-8")

    print("=" * 72)
    print("Parkinson's Motor Environment Evaluation Suite")
    print("=" * 72)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")
    print("")

    for task_id in TASKS:
        print(f"Task: {task_id}")
        for name, block in public[task_id].items():
            summary = format_summary(block["summary"])  # type: ignore[index]
            successes = block["successes"]  # type: ignore[index]
            num_runs = block["num_runs"]  # type: ignore[index]
            profiles = block["profiles"]  # type: ignore[index]
            print(
                f"  {name:<12} score={summary:<30} "
                f"success={successes}/{num_runs} profiles={profiles}"
            )
        print("")

    print("Extended tasks:")
    for task_id in EXTENDED_TASKS:
        print(f"Task: {task_id}")
        for name, block in extended[task_id].items():
            summary = format_summary(block["summary"])  # type: ignore[index]
            successes = block["successes"]  # type: ignore[index]
            num_runs = block["num_runs"]  # type: ignore[index]
            profiles = block["profiles"]  # type: ignore[index]
            print(
                f"  {name:<12} score={summary:<30} "
                f"success={successes}/{num_runs} profiles={profiles}"
            )
        print("")


if __name__ == "__main__":
    main()
