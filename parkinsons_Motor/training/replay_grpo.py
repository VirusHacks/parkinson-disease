# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay-based GRPO training surface (no custom ``rollout_func``).

This module is the simple, honest path:

    1. **Before training** — roll out the deterministic heuristic policy against
       an *in-process* :class:`ParkinsonsMotorEnvironment` for a few episodes
       per training task. For each step we record the *exact* prompt the LLM
       would see (system + per-step user block, chat-templated with thinking
       OFF) plus the JSON-serialised history of heuristic actions taken so far,
       the task id, and the episode seed. The result is a dataset whose rows
       cover the **full distribution of states** a real episode visits — first
       step, mid-crisis, late-episode taper — not just a synthetic toy prompt.

    2. **During GRPO** — TRL's standard generation path produces ``num_generations``
       completions per prompt (group). For every completion we

         a. parse the JSON action,
         b. spin up a *fresh* in-process env, ``reset(task_id, seed)``, replay
            the recorded history actions to recreate the exact state the LLM
            saw at training time,
         c. apply the LLM's parsed action,
         d. read the env's ``reward`` field, mix in a tiny format bonus, and
            return one scalar per completion.

       Group-relative advantages are now informative because different
       completions land different actions on the *same* state, so their rewards
       genuinely differ. No custom ``rollout_func``, no ``asyncio``, no
       WebSocket keepalives, no cursor scheduling, no kwarg plumbing.

The env code being replayed is the same ``ParkinsonsMotorEnvironment`` we
deploy on the HF Space — so this is the same env, just called in-process for
speed. Evaluation still goes against the *remote* Space so the demo end-to-end
is real.

Public API
----------
:class:`LocalEnvFactory`
    Picklable, no-arg callable that builds a fresh
    :class:`ParkinsonsMotorEnvironment`. Pass an instance into both
    :func:`collect_prompt_dataset` and :func:`make_replay_reward_fn` so the
    same env code is exercised on both ends.

:func:`collect_prompt_dataset`
    Roll out the heuristic policy against an in-process env and emit a list of
    dicts ready to feed into :class:`datasets.Dataset.from_list`.

:func:`make_replay_reward_fn`
    Build the GRPO-compatible ``reward_funcs`` callable. The returned function
    accepts the standard TRL kwargs (``completions``, ``prompts``, plus every
    extra dataset column) and returns one scalar per completion.

The module is import-safe with or without ``torch``/``unsloth`` installed; the
only hard dependencies are :mod:`parkinsons_Motor` itself and the standard
library.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation
from ..server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
from ..train import (
    SYSTEM_PROMPT,
    apply_chat_template,
    build_user_prompt,
    heuristic_action,
    make_action,
    parse_action,
)

logger = logging.getLogger("parkinsons_Motor.training.replay_grpo")


# ---------------------------------------------------------------------------
# Reward weights — tuned so a typical group of 6 completions sees rewards in
# roughly [-0.2, +1.2] and a 4B policy can drive variance up reliably.
# ---------------------------------------------------------------------------

DEFAULT_REPLAY_REWARD_WEIGHTS: Dict[str, float] = {
    "env":           1.0,   # raw env step reward (already shaped)
    "format":        0.2,   # bonus for emitting parseable JSON
    "invalid":       0.5,   # penalty when JSON parse fails (subtracted)
}
"""Default mix between the env-supplied step reward, a format bonus, and the
invalid-action penalty.

Why the format bonus is small: GRPO converges faster when most of the signal
comes from the *task* reward, not from a constant format prior. A 0.2 bonus is
enough to make a parsed-but-bad action strictly better than a parse failure
without dominating the gradient. The bio-experiment hackathon winner used a
similar mix (small format bonus, large env-shaped reward).
"""


# ---------------------------------------------------------------------------
# 1. Picklable env factory
# ---------------------------------------------------------------------------

@dataclass
class LocalEnvFactory:
    """No-arg callable that builds a fresh :class:`ParkinsonsMotorEnvironment`.

    Stored as a dataclass so it's trivially picklable — TRL may serialise the
    reward function across worker processes when ``dataloader_num_workers > 0``,
    and a closure over an env instance would die at pickle time.

    Example::

        env_factory = LocalEnvFactory(base_seed=7)
        env = env_factory()           # fresh ParkinsonsMotorEnvironment
        obs = env.reset(task_id='easy', seed=42)
    """

    base_seed: int = 7

    def __call__(self) -> ParkinsonsMotorEnvironment:
        return ParkinsonsMotorEnvironment(seed=self.base_seed)


# ---------------------------------------------------------------------------
# 2. Dataset collection — heuristic rollouts → (prompt, history, task, seed)
# ---------------------------------------------------------------------------

def _serialise_actions(actions: Sequence[ParkinsonsMotorAction]) -> str:
    """JSON-serialise a list of actions so it survives the HF Datasets schema.

    HF Datasets requires every cell in a column to share a stable
    ``pyarrow``-compatible type. Lists of pydantic objects break this; a flat
    JSON string is the cheapest stable representation.
    """
    return json.dumps([a.model_dump() for a in actions])


def _deserialise_actions(blob: str) -> List[ParkinsonsMotorAction]:
    """Inverse of :func:`_serialise_actions`. Returns ``[]`` on empty/missing."""
    if not blob:
        return []
    try:
        items = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out: List[ParkinsonsMotorAction] = []
    for it in items:
        if isinstance(it, Mapping):
            try:
                out.append(ParkinsonsMotorAction(**it))
            except (TypeError, ValueError):
                # Skip malformed entries — the replay just gets one fewer step.
                continue
    return out


def _episode_done(obs: ParkinsonsMotorObservation) -> bool:
    """Robust ``done`` check (the field is set in ``_make_obs``)."""
    return bool(getattr(obs, "done", False))


def collect_prompt_dataset(
    env_factory: Callable[[], ParkinsonsMotorEnvironment],
    *,
    tasks: Sequence[str] = ("easy", "medium", "hard"),
    episodes_per_task: int = 5,
    max_steps_per_episode: int = 20,
    seed: int = 42,
    tokenizer: Any = None,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """Build a GRPO-ready prompt dataset by rolling the heuristic policy.

    Each emitted row corresponds to one (state, history) snapshot the policy
    will be asked to act on. The fields are:

    ``prompt`` (str)
        The chat-templated string the model will be conditioned on. This is
        what TRL feeds straight into ``model.generate``. Pre-rendering means
        TRL does **not** apply its own chat template — we control the
        ``enable_thinking`` flag exactly.

    ``task_id`` (str), ``seed`` (int), ``step_idx`` (int)
        The deterministic key needed to replay this state during reward
        computation.

    ``history_actions`` (str, JSON)
        ``json.dumps([action.model_dump(), …])`` — the heuristic actions taken
        on this episode *before* this step. The reward function replays them
        to recreate the exact observation the LLM saw.

    ``ref_amplitude`` (float)
        The dose the heuristic chose at this step. Used only for diagnostics
        and plotting, never as a target.

    Args:
        env_factory: Callable returning a fresh env. Use :class:`LocalEnvFactory`
            in normal use.
        tasks: Task ids to roll out. Defaults to the ``easy/medium/hard``
            curriculum used everywhere else in this project.
        episodes_per_task: Number of heuristic episodes per task. The default
            (5) gives ~5 × 20 × 3 = 300 prompts, matching the 200-400 prompt
            sweet spot the hackathon winners used.
        max_steps_per_episode: Hard cap on per-episode length. Real episodes
            may end earlier via ``obs.done``. Keeping this small (≤ 20) keeps
            replay cost bounded — the reward function replays at most this many
            steps per completion.
        seed: Deterministic seed for the per-episode seed sampler. Same seed
            ⇒ same dataset every time.
        tokenizer: A Hugging Face-style tokenizer with ``apply_chat_template``.
            Required — we render prompts ahead of time so TRL doesn't second
            guess us.
        enable_thinking: Whether to leave Qwen3's ``<think>`` block on at
            train-time. Keep this **False** for training; flip on only for
            demo / sample-trajectory eval.

    Returns:
        A list of plain dicts ready for ``datasets.Dataset.from_list``.

    Raises:
        ValueError: If ``tokenizer`` is None.
    """
    if tokenizer is None:
        raise ValueError(
            "collect_prompt_dataset requires a tokenizer with apply_chat_template; "
            "pass the same one you'll hand to GRPOTrainer."
        )

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for task_id in tasks:
        for ep in range(episodes_per_task):
            ep_seed = rng.randrange(1, 1_000_000)
            env = env_factory()
            obs = env.reset(task_id=task_id, seed=ep_seed)
            history: List[ParkinsonsMotorAction] = []
            last_amp: Optional[float] = None
            for step_idx in range(max_steps_per_episode):
                user_text = build_user_prompt(
                    step=step_idx + 1,
                    obs=obs,
                    task_id=task_id,
                    history=(),  # transcript-style history is too noisy here;
                                 # the JSON history_actions captures the same info
                )
                prompt = apply_chat_template(
                    tokenizer,
                    SYSTEM_PROMPT,
                    user_text,
                    enable_thinking=enable_thinking,
                )
                next_action = heuristic_action(obs, task_id=task_id, last_amp=last_amp)
                rows.append(
                    {
                        "prompt":           prompt,
                        "task_id":          task_id,
                        "seed":             ep_seed,
                        "step_idx":         step_idx,
                        "history_actions":  _serialise_actions(history),
                        "ref_amplitude":    float(next_action.dbs_amplitude),
                    }
                )
                history.append(next_action)
                last_amp = next_action.dbs_amplitude
                obs = env.step(next_action)
                if _episode_done(obs):
                    break

    logger.info(
        "collect_prompt_dataset: %d rows from %d tasks × %d episodes (max %d steps)",
        len(rows), len(tasks), episodes_per_task, max_steps_per_episode,
    )
    return rows


# ---------------------------------------------------------------------------
# 3. Reward function factory — replay env, score LLM action, return scalar
# ---------------------------------------------------------------------------

def _completion_text(completion: Any) -> str:
    """Extract the assistant-turn text from a TRL completion.

    TRL hands reward functions completions in one of two shapes depending on
    the dataset / version:

      * ``str`` — the raw decoded completion text.
      * ``list[dict]`` — a single-message chat list, ``[{"role": "assistant",
        "content": "..."}]``.

    Be defensive about both so we don't silently zero out rewards if TRL
    changes the convention between releases.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", ""))
    return ""


@dataclass
class _ReplayStats:
    """Cheap counters surfaced in the reward function's logger output."""
    total_calls:    int = 0
    total_invalid:  int = 0
    total_envfails: int = 0
    last_rewards:   List[float] = field(default_factory=list)


def make_replay_reward_fn(
    env_factory: Callable[[], ParkinsonsMotorEnvironment],
    *,
    weights: Optional[Mapping[str, float]] = None,
    log_every: int = 5,
) -> Callable[..., List[float]]:
    """Build a TRL-compatible reward function using local env replay.

    The returned function has the standard TRL ``reward_funcs`` signature::

        def reward_fn(completions, **kwargs) -> list[float]: ...

    where ``kwargs`` carries every column of the dataset (so we receive
    ``task_id``, ``seed``, ``step_idx``, ``history_actions`` as parallel lists
    one entry per completion).

    For each completion we:

      1. Decode → :func:`parse_action`. On failure the reward is
         ``-weights['invalid'] + (env reward of fallback action)``: we still
         apply the heuristic-default action so the env never crashes and the
         group still has *some* signal, but the policy gets a clear negative
         relative to a parse-success peer.
      2. Reset a fresh env to ``(task_id, seed)``, replay the recorded
         ``history_actions`` (deterministic) to recreate the exact state the
         training prompt described.
      3. Apply the LLM's parsed action and return
         ``weights['env'] * env.reward + weights['format']``.

    Args:
        env_factory: Same factory used by :func:`collect_prompt_dataset`. The
            reward function holds a *reference* and creates a fresh env per
            completion (cheap — just python-object construction, not a process).
        weights: Override :data:`DEFAULT_REPLAY_REWARD_WEIGHTS`. Only the keys
            you supply are overridden.
        log_every: Throttle factor for the per-step INFO log. ``log_every=5``
            prints aggregate stats once every five reward-function calls (one
            "call" = one full GRPO group of completions).

    Returns:
        A callable suitable for ``GRPOTrainer(reward_funcs=[reward_fn])``.
    """
    w = dict(DEFAULT_REPLAY_REWARD_WEIGHTS)
    if weights:
        w.update(weights)
    stats = _ReplayStats()

    def _replay_one(
        completion: Any,
        task_id: str,
        seed: int,
        history_actions_json: str,
    ) -> Tuple[float, bool, bool]:
        """Score a single completion. Returns (reward, parsed_ok, env_ok)."""
        text = _completion_text(completion)
        parsed = parse_action(text)
        history = _deserialise_actions(history_actions_json)

        try:
            env = env_factory()
            obs = env.reset(task_id=task_id, seed=int(seed))
            for past_action in history:
                obs = env.step(past_action)
            target = float(getattr(obs, "target_output", 0.0))
            llm_action = make_action(parsed, target_output=target)
            obs = env.step(llm_action)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "replay reward: env failed task=%s seed=%s history_len=%d err=%r",
                task_id, seed, len(history), exc,
            )
            # Env crashed → return a constant negative so this row doesn't
            # dominate the group; flag for stats.
            return -float(w["invalid"]), parsed is not None, False

        env_reward = float(getattr(obs, "reward", 0.0) or 0.0)
        if parsed is None:
            # Heuristic-default action was applied; charge the format penalty.
            return float(w["env"]) * env_reward - float(w["invalid"]), False, True
        return float(w["env"]) * env_reward + float(w["format"]), True, True

    def reward_fn(
        completions: Sequence[Any],
        prompts: Optional[Sequence[Any]] = None,           # noqa: ARG001 - TRL passes this
        completion_ids: Optional[Sequence[Any]] = None,    # noqa: ARG001
        task_id: Optional[Sequence[str]] = None,
        seed: Optional[Sequence[int]] = None,
        history_actions: Optional[Sequence[str]] = None,
        step_idx: Optional[Sequence[int]] = None,          # noqa: ARG001
        **_extra: Any,
    ) -> List[float]:
        n = len(completions)
        # All these dataset-derived kwargs should be parallel to ``completions``.
        # If any are missing it means somebody called this function without the
        # dataset wiring (e.g. a unit test) — fail loudly rather than silently
        # returning zeros (which is exactly the bug we're fixing).
        if task_id is None or seed is None or history_actions is None:
            raise ValueError(
                "replay reward_fn requires `task_id`, `seed`, and `history_actions` "
                "in the dataset (every column is forwarded to the reward function "
                "as a parallel list). Got: "
                f"task_id={'present' if task_id is not None else 'missing'}, "
                f"seed={'present' if seed is not None else 'missing'}, "
                f"history_actions={'present' if history_actions is not None else 'missing'}."
            )

        rewards: List[float] = []
        n_invalid = 0
        n_envfail = 0
        for i in range(n):
            r, parsed_ok, env_ok = _replay_one(
                completions[i],
                str(task_id[i]),
                int(seed[i]),
                str(history_actions[i] or ""),
            )
            rewards.append(r)
            if not parsed_ok:
                n_invalid += 1
            if not env_ok:
                n_envfail += 1

        stats.total_calls += 1
        stats.total_invalid += n_invalid
        stats.total_envfails += n_envfail
        stats.last_rewards = list(rewards)

        if stats.total_calls % max(1, log_every) == 0:
            mean_r = sum(rewards) / max(1, len(rewards))
            std_r = (
                (sum((r - mean_r) ** 2 for r in rewards) / max(1, len(rewards))) ** 0.5
            )
            logger.info(
                "replay reward call=%d  group_size=%d  invalid=%d  env_fail=%d  "
                "mean=%+.3f  std=%.3f  cum_invalid=%d/%d  cum_env_fail=%d",
                stats.total_calls, n, n_invalid, n_envfail, mean_r, std_r,
                stats.total_invalid, stats.total_calls * n, stats.total_envfails,
            )
        return rewards

    reward_fn.__name__ = "replay_env_reward"
    reward_fn.weights = dict(w)        # type: ignore[attr-defined]
    reward_fn.stats = stats            # type: ignore[attr-defined]
    return reward_fn


__all__ = [
    "DEFAULT_REPLAY_REWARD_WEIGHTS",
    "LocalEnvFactory",
    "collect_prompt_dataset",
    "make_replay_reward_fn",
]
