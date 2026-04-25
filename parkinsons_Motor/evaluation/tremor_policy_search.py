"""Search utilities for medium-task tremor rescue baselines."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from parkinsons_Motor.core.models import ParkinsonsMotorAction
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "search"
JSON_PATH = OUTPUT_DIR / "tremor_policy_search.json"
MD_PATH = OUTPUT_DIR / "tremor_policy_search.md"
PUBLIC_SEEDS = [0, 1, 2, 3]
HELDOUT = {"seed": 102, "patient_profile_id": "fragile"}


def constant_policy_factory(amp: float, pw: float, gain: float):
    def _policy(obs):
        cmd = max(-1.0, min(1.0, obs.target_output * gain))
        return ParkinsonsMotorAction(
            motor_command=cmd,
            dbs_amplitude=amp,
            dbs_pulse_width=pw,
        )

    return _policy


def run_episode(policy_fn, seed: int, patient_profile_id: str | None = None):
    env = ParkinsonsMotorEnvironment()
    obs = env.reset(
        task_id="medium",
        seed=seed,
        patient_profile_id=patient_profile_id,
    )
    while not obs.done:
        obs = env.step(policy_fn(obs))
    return {
        "score": float(obs.grader_score),
        "success": bool(obs.episode_success),
        "patient_profile_id": obs.metadata["patient_profile_id"],
        "score_details": obs.metadata.get("score_details", {}),
    }


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def search():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    for amp in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        for pw in [0.06, 0.08, 0.10, 0.12]:
            for gain in [1.0, 1.15, 1.3]:
                policy = constant_policy_factory(amp, pw, gain)
                public_runs = [run_episode(policy, seed) for seed in PUBLIC_SEEDS]
                heldout_run = run_episode(
                    policy,
                    HELDOUT["seed"],
                    patient_profile_id=HELDOUT["patient_profile_id"],
                )
                public_scores = [run["score"] for run in public_runs]
                public_successes = sum(1 for run in public_runs if run["success"])
                heldout_score = heldout_run["score"]
                objective = (
                    statistics.mean(public_scores)
                    - 0.25 * statistics.pstdev(public_scores)
                    + 0.15 * heldout_score
                )
                candidates.append(
                    {
                        "amp": amp,
                        "pw": pw,
                        "gain": gain,
                        "objective": objective,
                        "public_summary": summarize(public_scores),
                        "public_successes": public_successes,
                        "heldout_score": heldout_score,
                        "heldout_success": heldout_run["success"],
                    }
                )
    candidates.sort(key=lambda item: item["objective"], reverse=True)
    payload = {
        "search_space": {
            "amp": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
            "pw": [0.06, 0.08, 0.10, 0.12],
            "gain": [1.0, 1.15, 1.3],
        },
        "top_candidates": candidates[:12],
        "best_candidate": candidates[0],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tremor Policy Search",
        "",
        "Simple constant-policy sweep for `medium`.",
        "",
        "## Best Candidate",
        "",
        f"- amplitude: `{candidates[0]['amp']:.2f}` mA",
        f"- pulse width: `{candidates[0]['pw']:.2f}` ms",
        f"- motor gain: `{candidates[0]['gain']:.2f}`",
        f"- public mean score: `{candidates[0]['public_summary']['mean']:.3f}`",
        f"- public successes: `{candidates[0]['public_successes']}/4`",
        f"- held-out score: `{candidates[0]['heldout_score']:.3f}`",
        "",
        "## Top Candidates",
        "",
        "| Rank | Amp | PW | Gain | Public Mean | Public Success | Held-Out | Objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, candidate in enumerate(candidates[:12], start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"{candidate['amp']:.2f} | "
            f"{candidate['pw']:.2f} | "
            f"{candidate['gain']:.2f} | "
            f"{candidate['public_summary']['mean']:.3f} | "
            f"{candidate['public_successes']}/4 | "
            f"{candidate['heldout_score']:.3f} | "
            f"{candidate['objective']:.3f} |"
        )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")
    print(json.dumps(candidates[0], indent=2))


if __name__ == "__main__":
    search()
