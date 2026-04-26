"""Collect trajectories with direct OpenEnv environment access.

Mirrors the bio-experiment winner's ``training/rollout_collection.py`` -
runs episodes against the live HF Space (or a local env), persists each
trajectory as JSON, and prints summary metrics. Lets reviewers reproduce
the policy's behaviour offline without re-running GRPO.

Two policies are built in (no LLM required):

  * ``constant``   - emits the clinical default (1.5 mA, 0.13 ms, 130 Hz)
                     every step. Acts as the "always-on cDBS" baseline used
                     by the clinical benchmark.
  * ``heuristic``  - uses the rule-based policy from
                     :func:`parkinsons_Motor.train.heuristic_action`, which
                     is roughly the strategy a clinician trained on the
                     literature would deploy.

Usage::

    python -m parkinsons_Motor.training.rollout_collection \
        --env-url https://virustechhacks-parkinsons-motor.hf.space \
        --episodes 24 --policy heuristic \
        --output-dir artifacts/baseline_traj
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .evaluation import EvaluationSuite
from .trajectory import DBSTrajectory, DBSTrajectoryDataset

logger = logging.getLogger(__name__)


# ─── Policies ────────────────────────────────────────────────────────────


def _clinical_default_action() -> Dict[str, float]:
    """Always-on continuous-DBS clinical default - ~midpoint of the safe window."""
    return {
        "dbs_amplitude":   1.5,
        "dbs_pulse_width": 0.13,
        "dbs_frequency":   130.0,
        "motor_command":   0.0,
    }


def _heuristic_action(obs: Dict[str, Any], task_id: str, last_amp: Optional[float]) -> Dict[str, float]:
    """Lazy-import the heuristic from parkinsons_Motor.train so we share its rules."""
    from parkinsons_Motor.train import heuristic_action as _heur
    a = _heur(obs, task_id=task_id, last_amp=last_amp)
    return {
        "dbs_amplitude":   float(a.dbs_amplitude),
        "dbs_pulse_width": float(a.dbs_pulse_width),
        "dbs_frequency":   float(a.dbs_frequency),
        "motor_command":   float(getattr(a, "motor_command", 0.0)),
    }


# ─── Single-episode driver ───────────────────────────────────────────────


async def _run_one_episode(
    env_url: str,
    task_id: str,
    seed: Optional[int],
    *,
    policy: str,
    max_steps: Optional[int],
    episode_id: str,
) -> DBSTrajectory:
    from parkinsons_Motor.train import _obs_to_dict, ParkinsonsMotorEnv
    from parkinsons_Motor.client.parkinsons_motor_environment import ParkinsonsMotorAction

    traj = DBSTrajectory(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        policy=policy,
        metadata={"env_url": env_url},
    )

    env = ParkinsonsMotorEnv(base_url=env_url)
    await env.__aenter__()
    last_amp: Optional[float] = None
    try:
        reset_kwargs: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            reset_kwargs["seed"] = seed
        result = await env.reset(**reset_kwargs)
        obs = result.observation

        step = 0
        while True:
            if max_steps is not None and step >= max_steps:
                break
            obs_dict = _obs_to_dict(obs)

            if policy == "constant":
                action_dict = _clinical_default_action()
            elif policy == "heuristic":
                action_dict = _heuristic_action(obs_dict, task_id=task_id, last_amp=last_amp)
            else:
                raise ValueError(f"Unknown policy: {policy!r}. Use 'constant' or 'heuristic'.")
            last_amp = action_dict["dbs_amplitude"]

            action = ParkinsonsMotorAction(**action_dict)
            try:
                result = await env.step(action)
            except Exception as exc:
                traj.env_error = str(exc)
                logger.warning("env.step raised: %s", exc)
                break
            obs = result.observation
            step += 1
            traj.add_step(
                action=action_dict,
                observation=_obs_to_dict(obs),
                reward=float(result.reward or 0.0),
                done=bool(result.done),
                parsed=True,
            )
            if result.done:
                break
    finally:
        await env.__aexit__(None, None, None)
    return traj


def _run_one_episode_sync(*args: Any, **kwargs: Any) -> DBSTrajectory:
    return asyncio.run(_run_one_episode(*args, **kwargs))


# ─── Public surface (importable from the notebook) ───────────────────────


def collect_trajectories(
    env_url: str,
    *,
    episodes: int = 12,
    policy: str = "heuristic",
    tasks: Sequence[str] = ("easy", "medium", "hard"),
    seeds: Optional[Sequence[int]] = None,
    max_steps_per_task: Optional[Dict[str, int]] = None,
    output_dir: Optional[Path] = None,
) -> DBSTrajectoryDataset:
    """Run heuristic/constant rollouts and persist them to disk.

    Returns a :class:`DBSTrajectoryDataset` ready to feed into
    :class:`parkinsons_Motor.training.evaluation.EvaluationSuite`.
    """
    seeds = list(seeds) if seeds is not None else list(range(episodes))
    tasks = list(tasks)
    max_steps_per_task = max_steps_per_task or {}
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DBSTrajectoryDataset()
    rng = random.Random(0xDB5)
    for ep in range(episodes):
        task_id = tasks[ep % len(tasks)]
        seed = seeds[ep % len(seeds)] if seeds else rng.randrange(10**6)
        ep_id = f"{policy}-{task_id}-{seed}-{uuid.uuid4().hex[:6]}"
        max_steps = max_steps_per_task.get(task_id)
        logger.info("Episode %d/%d | task=%s seed=%s policy=%s",
                    ep + 1, episodes, task_id, seed, policy)
        traj = _run_one_episode_sync(
            env_url=env_url,
            task_id=task_id,
            seed=seed,
            policy=policy,
            max_steps=max_steps,
            episode_id=ep_id,
        )
        dataset.add(traj)
        if out_dir is not None:
            traj.save(out_dir / f"{traj.episode_id}.json")

    return dataset


# ─── CLI ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect DBS trajectories with a fixed policy.")
    p.add_argument("--env-url", required=True, help="Base URL of the HF Space env.")
    p.add_argument("--episodes", type=int, default=12, help="Number of episodes to roll out.")
    p.add_argument("--policy", choices=("constant", "heuristic"), default="heuristic",
                   help="Which non-LLM policy to use.")
    p.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"],
                   help="Curriculum tasks to cycle through.")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Optional explicit seeds (cycled through).")
    p.add_argument("--max-steps-easy", type=int, default=36)
    p.add_argument("--max-steps-medium", type=int, default=60)
    p.add_argument("--max-steps-hard", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_traj"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = _parse_args()
    max_steps = {
        "easy":   args.max_steps_easy,
        "medium": args.max_steps_medium,
        "hard":   args.max_steps_hard,
    }

    print(f"Collecting {args.episodes} episodes | policy={args.policy} | tasks={args.tasks}")
    print(f"  env URL : {args.env_url}")
    print(f"  output  : {args.output_dir}")

    dataset = collect_trajectories(
        env_url=args.env_url,
        episodes=args.episodes,
        policy=args.policy,
        tasks=args.tasks,
        seeds=args.seeds,
        max_steps_per_task=max_steps,
        output_dir=args.output_dir,
    )

    print("\nDataset summary:")
    for k, v in dataset.summary().items():
        print(f"  {k:>22} : {v}")

    print("\nOnline metrics:")
    for m in EvaluationSuite.online_metrics(list(dataset)):
        print(f"  {m.name:>24} : {m.value:.4f} {m.unit}")


if __name__ == "__main__":
    main()
