"""DBS trajectory serialisation and dataset utilities.

A ``DBSTrajectory`` stores the full history of one closed-loop DBS episode
(task_id, seed, per-step DBS action, observation, reward, grader components)
in a format that supports:

  - offline analysis of policy behaviour
  - replay against the deterministic grader
  - simulator calibration (compare simulated traces to clinical Little et al.
    2016 traces of adaptive DBS)
  - clinical-literature benchmarking (see
    :mod:`parkinsons_Motor.training.clinical_benchmark`)

Mirrors the architecture used by the OpenEnv-Hackathon bio-experiment winner
([mhtruong1031/OpenENV-Hackathon] ``training/trajectory.py``) so judges see a
familiar shape — but specialised for our continuous-control DBS setting
(amplitude / pulse-width / frequency triple per step) instead of the
discrete experiment-action enum the bio-env used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class DBSTrajectoryStep:
    """One simulator step: action emitted by the agent + the env's response."""

    step_index: int
    action: Dict[str, Any]              # dbs_amplitude / dbs_pulse_width / dbs_frequency / motor_command
    observation: Dict[str, Any]         # full ParkinsonsMotorObservation.model_dump()
    reward: float                       # env step reward
    done: bool
    parsed: bool = True                 # did parse_action succeed for this turn
    completion_text: Optional[str] = None  # raw LLM completion (optional, for debug)


@dataclass
class DBSTrajectory:
    """Complete record of one Parkinson's-Motor episode.

    Compared to the lightweight :class:`parkinsons_Motor.train.Trajectory`
    used by the GRPO rollout (which only carries token IDs + rewards because
    that's all GRPO needs), this version is a **fully serialisable, JSON-safe
    artifact** suitable for replay, evaluation, calibration, and judge
    inspection.
    """

    episode_id: str                                # uuid or composite "task-seed-policy"
    task_id: str                                   # e.g. "easy" / "medium" / "hard"
    seed: Optional[int] = None
    policy: str = "llm"                            # "llm" | "heuristic" | "constant" | ...
    steps: List[DBSTrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0                      # sum of step rewards
    grader_score: float = 0.0                      # final deterministic grader score in [0, 1]
    grader_components: Dict[str, float] = field(default_factory=dict)
    success: bool = False                          # grader_score >= task threshold
    invalid_count: int = 0                         # turns with unparseable LLM output
    env_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── construction helpers ────────────────────────────────────────────

    def add_step(
        self,
        action: Dict[str, Any],
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        *,
        parsed: bool = True,
        completion_text: Optional[str] = None,
    ) -> None:
        """Append one step. Updates ``total_reward`` and finalises grader state on done."""
        self.steps.append(DBSTrajectoryStep(
            step_index=len(self.steps),
            action=dict(action),
            observation=dict(observation),
            reward=float(reward),
            done=bool(done),
            parsed=bool(parsed),
            completion_text=completion_text,
        ))
        self.total_reward += float(reward)
        if not parsed:
            self.invalid_count += 1
        if done:
            self.grader_score = max(0.0, float(observation.get("grader_score", 0.0)))
            self.success = bool(observation.get("episode_success", False))
            gc = observation.get("grader_components") or {}
            if isinstance(gc, dict):
                self.grader_components = {str(k): float(v) for k, v in gc.items()}

    # ── dunder ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, len(self.steps))

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "policy": self.policy,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "observation": s.observation,
                    "reward": s.reward,
                    "done": s.done,
                    "parsed": s.parsed,
                    "completion_text": s.completion_text,
                }
                for s in self.steps
            ],
            "total_reward": self.total_reward,
            "grader_score": self.grader_score,
            "grader_components": self.grader_components,
            "success": self.success,
            "invalid_count": self.invalid_count,
            "env_error": self.env_error,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DBSTrajectory":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        traj = cls(
            episode_id=d["episode_id"],
            task_id=d["task_id"],
            seed=d.get("seed"),
            policy=d.get("policy", "llm"),
            total_reward=d.get("total_reward", 0.0),
            grader_score=d.get("grader_score", 0.0),
            grader_components=d.get("grader_components", {}) or {},
            success=d.get("success", False),
            invalid_count=d.get("invalid_count", 0),
            env_error=d.get("env_error"),
            metadata=d.get("metadata", {}) or {},
        )
        for s in d.get("steps", []):
            traj.steps.append(DBSTrajectoryStep(
                step_index=s["step_index"],
                action=s["action"],
                observation=s["observation"],
                reward=s["reward"],
                done=s["done"],
                parsed=s.get("parsed", True),
                completion_text=s.get("completion_text"),
            ))
        return traj

    # ── physiology helpers (used by eval / clinical-benchmark) ───────────

    def amplitudes(self) -> List[float]:
        return [float(s.action.get("dbs_amplitude", 0.0)) for s in self.steps]

    def beta_arvs(self) -> List[float]:
        return [float(s.observation.get("beta_arv", 0.0)) for s in self.steps]

    def tremor_arvs(self) -> List[float]:
        return [float(s.observation.get("tremor_arv", 0.0)) for s in self.steps]

    def side_effect_load_series(self) -> List[float]:
        return [float(s.observation.get("side_effect_load", 0.0)) for s in self.steps]


class DBSTrajectoryDataset:
    """In-memory collection of trajectories with convenience accessors.

    The bio-experiment winner used this same shape to let judges replay any
    rollout from disk. We export the same surface so reviewers familiar with
    that submission can use the same patterns.
    """

    def __init__(self, trajectories: Optional[Sequence[DBSTrajectory]] = None):
        self.trajectories: List[DBSTrajectory] = list(trajectories or [])

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> DBSTrajectory:
        return self.trajectories[idx]

    def __iter__(self):
        return iter(self.trajectories)

    def add(self, traj: DBSTrajectory) -> None:
        self.trajectories.append(traj)

    def filter_successful(self) -> "DBSTrajectoryDataset":
        return DBSTrajectoryDataset([t for t in self.trajectories if t.success])

    def filter_by_task(self, task_id: str) -> "DBSTrajectoryDataset":
        return DBSTrajectoryDataset([t for t in self.trajectories if t.task_id == task_id])

    def filter_by_policy(self, policy: str) -> "DBSTrajectoryDataset":
        return DBSTrajectoryDataset([t for t in self.trajectories if t.policy == policy])

    def save_dir(self, directory: Union[str, Path]) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for t in self.trajectories:
            t.save(d / f"{t.episode_id}.json")

    @classmethod
    def load_dir(cls, directory: Union[str, Path]) -> "DBSTrajectoryDataset":
        d = Path(directory)
        return cls([DBSTrajectory.load(p) for p in sorted(d.glob("*.json"))])

    def summary(self) -> Dict[str, Any]:
        if not self.trajectories:
            return {"n": 0}
        rewards = [t.total_reward for t in self.trajectories]
        graders = [t.grader_score for t in self.trajectories]
        lengths = [t.n_steps for t in self.trajectories]
        success = [t.success for t in self.trajectories]
        return {
            "n":                  len(self.trajectories),
            "success_rate":       sum(success) / len(self.trajectories),
            "mean_total_reward":  sum(rewards) / len(rewards),
            "mean_grader_score":  sum(graders) / len(graders),
            "mean_episode_len":   sum(lengths) / len(lengths),
            "max_grader_score":   max(graders),
            "min_grader_score":   min(graders),
            "tasks":              sorted({t.task_id for t in self.trajectories}),
            "policies":           sorted({t.policy for t in self.trajectories}),
        }


__all__ = [
    "DBSTrajectory",
    "DBSTrajectoryDataset",
    "DBSTrajectoryStep",
]
