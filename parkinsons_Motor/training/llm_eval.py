# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM-driven evaluation utilities for the Parkinson's-Motor environment.

These were originally hosted in :mod:`parkinsons_Motor.train` but split out
so the runtime training module stays focused on rollout + reward + GRPO
glue. The LLM-evaluation surface is separate from the offline-trajectory
:class:`~parkinsons_Motor.training.evaluation.EvaluationSuite` because:

  * :class:`EvaluationSuite` is **policy-agnostic** — it takes a
    :class:`~parkinsons_Motor.training.trajectory.DBSTrajectory` dataset and
    computes metrics on the recorded actions/observations.
  * The functions here actually **drive an LLM through the env** to produce
    those trajectories (or to spot-check its output before training).

Imports from :mod:`parkinsons_Motor.train` are lazy/inline to avoid the
circular-import chain ``train -> training.llm_eval -> train``: ``train``
re-exports these functions from this module after defining the rollout
primitives (``rollout_episode``, ``llm_generate``, ``parse_action``,
``apply_chat_template``, ``build_user_prompt``, ``SYSTEM_PROMPT``).
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger("parkinsons_Motor.training.llm_eval")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tiny parsing / warm-up helpers used by the eval functions below
# ─────────────────────────────────────────────────────────────────────────────

def _amp_from_history(line: str) -> Optional[float]:
    """Recover the DBS amplitude from a ``Trajectory.history`` line.

    History lines have the shape ``"step=3 amp=1.42 => beta=0.31 ..."`` —
    we just slice out the ``amp=<float>`` substring.
    """
    if "amp=" not in line:
        return None
    try:
        return float(line.split("amp=", 1)[1].split(" ", 1)[0])
    except Exception:  # pragma: no cover
        return None


def _warm_up_generation(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 16,
) -> None:
    """One throwaway generation so the env doesn't pay the CUDA-compile cost
    on its very first step (which can add 8-15 s and cause WebSocket timeouts).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    try:
        prompt = "ping"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:  # pragma: no cover
        logger.debug("GPU warm-up generation skipped (%s)", exc)


# ─────────────────────────────────────────────────────────────────────────────
# 2. eval_with_adapter_disabled  — base-model evaluation on the same seeds
# ─────────────────────────────────────────────────────────────────────────────

def eval_with_adapter_disabled(
    model: Any,
    tokenizer: Any,
    env_url: str,
    tasks: Sequence[str],
    seeds: Sequence[int],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Evaluate the **base** model (LoRA disabled) on the same tasks/seeds.

    This is what makes the "before vs after" comparison meaningful: same
    weights, same seeds, only the LoRA adapter is toggled off via PEFT's
    ``disable_adapter`` context. If the model has no PEFT adapter (e.g. a
    fresh ``FastLanguageModel.from_pretrained`` before LoRA is attached),
    this falls back to a plain evaluate so the function is always safe to call.
    """
    disable_ctx = getattr(model, "disable_adapter", None)
    if callable(disable_ctx):
        with disable_ctx():
            return evaluate_model_suite(
                model, tokenizer, env_url, tasks, seeds, **kwargs
            )
    return evaluate_model_suite(model, tokenizer, env_url, tasks, seeds, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. sanity_check_rollout — pre-flight check before training
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_rollout(
    model: Any,
    tokenizer: Any,
    env_url: str,
    *,
    task_id: str = "easy",
    seed: int = 0,
    max_turns: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1280,
    raise_on_failure: bool = True,
    warm_up: bool = True,
    retry_on_env_error: bool = True,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Run ONE short rollout and verify the LLM→env loop is healthy.

    Use this BEFORE training to catch the five most common failure modes:
      1. completions never produce parseable JSON (cap too small / wrong template)
      2. env never returns a non-zero step reward (URL wrong / task broken)
      3. all rollouts emit identical actions (no reward variance => GRPO can't learn)
      4. env errors silently (ENVIRONMENT_ERROR_PENALTY appears in every step)
      5. **WebSocket keepalive timeouts during long generations** (sync call
         in async coroutine — fixed in ``rollout_episode_async`` via
         ``asyncio.to_thread``; this check verifies the fix actually landed)

    Prints raw completion + parsed action + per-step env reward + final grader,
    and (by default) raises ``RuntimeError`` on a hard failure so the notebook
    cell turns red instead of letting you waste an hour on a flat training run.

    ``warm_up`` runs one throwaway generation first so CUDA kernels are
    compiled before we open the env (otherwise the first real step pays
    +8-15 s of compile time and the env's keepalive can fire).

    ``retry_on_env_error`` will re-attempt the rollout once if the first
    attempt failed with an env error (HF Spaces sometimes 503 on cold start).
    """
    # Lazy imports break the circular dep on parkinsons_Motor.train.
    from parkinsons_Motor.train import (
        SYSTEM_PROMPT,
        apply_chat_template,
        build_user_prompt,
        llm_generate,
        parse_action,
        rollout_episode,
    )

    if warm_up:
        print("[warm-up] compiling CUDA kernels with one throwaway generation ...")
        _warm_up_generation(model, tokenizer)

    # ── Pre-rollout: trace ONE completion offline so we can show the raw text
    # if parsing fails. This is the only way to tell apart "thinking ate the
    # budget" vs "model wrote markdown around the JSON" vs "wrong template".
    print(f"\n=== sanity_check_rollout  task={task_id}  seed={seed}  turns={max_turns} ===")
    print("\n[trace] generating one completion offline (no env call) to inspect raw text ...")
    try:
        _trace_obs = {
            "beta_arv": 0.55, "tremor_arv": 0.45, "side_effect_load": 0.10,
            "gamma_arv": 0.30, "beta_trend": 0.02, "tremor_trend": 0.01,
            "target_output": 0.0, "side_effect_rate": 0.0,
        }
        _trace_user = build_user_prompt(1, _trace_obs, task_id, [])
        _trace_prompt = apply_chat_template(
            tokenizer, SYSTEM_PROMPT, _trace_user, enable_thinking=enable_thinking
        )
        _trace_text, _trace_p, _trace_c = llm_generate(
            model, tokenizer, _trace_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_prompt_length=max_prompt_length,
        )
        _trace_parsed = parse_action(_trace_text)
        _completion_token_count = len(_trace_c) if _trace_c is not None else 0
        _hit_cap = _completion_token_count >= max_new_tokens
        _has_think_open = "<think>" in _trace_text
        _has_think_close = "</think>" in _trace_text
        _has_json_brace = "{" in _trace_text and "}" in _trace_text
        _preview = _trace_text if len(_trace_text) <= 800 else (_trace_text[:600] + " ... [truncated] ... " + _trace_text[-200:])

        print(f"  completion tokens       : {_completion_token_count} / {max_new_tokens}  "
              f"({'HIT CAP' if _hit_cap else 'ok'})")
        print(f"  has <think> ... </think>: open={_has_think_open}  close={_has_think_close}")
        print(f"  has any JSON-like {{...}}: {_has_json_brace}")
        print(f"  parse_action result     : {_trace_parsed}")
        print("  --- raw completion (preview) ---")
        for line in _preview.split("\n"):
            print(f"  | {line}")
        print("  --- end preview ---")

        if _trace_parsed is None:
            print("\n  [diagnosis] parse failed. Most likely cause:")
            if _hit_cap and _has_think_open and not _has_think_close:
                print("    * The thinking block did NOT finish before max_new_tokens. "
                      "Bump MAX_COMPLETION_LENGTH (try 1024 or 1536) or "
                      "set enable_thinking=False in apply_chat_template.")
            elif _has_json_brace and not _trace_parsed:
                print("    * Model wrote JSON-like text but the regex/json.loads rejected it. "
                      "Check for code fences, smart quotes, trailing commas, or comments.")
            elif not _has_json_brace:
                print("    * Model emitted prose with no JSON object at all. "
                      "Check the chat template / SYSTEM_PROMPT — the model isn't following the format.")
            else:
                print("    * Unclear — inspect the preview above.")
    except Exception as exc:
        print(f"  [trace] generation crashed: {exc!r}  — falling through to env rollout anyway")

    traj = rollout_episode(
        model, tokenizer, env_url,
        task_id=task_id, seed=seed,
        max_turns=max_turns,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_prompt_length,
        fallback_to_heuristic_on_invalid=False,  # we WANT to see parse failures here
        enable_thinking=enable_thinking,
    )

    if retry_on_env_error and traj.env_error and len(traj.rewards) == 0:
        print(f"\n[retry] first attempt failed with env_error={traj.env_error!r}; retrying once ...")
        import time as _time
        _time.sleep(3.0)
        traj = rollout_episode(
            model, tokenizer, env_url,
            task_id=task_id, seed=seed,
            max_turns=max_turns,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            fallback_to_heuristic_on_invalid=False,
            enable_thinking=enable_thinking,
        )

    n_steps        = len(traj.rewards)
    n_invalid      = traj.invalid_count
    parseable_pct  = (sum(traj.parsed) / max(1, len(traj.parsed))) * 100
    rewards_nonzero = sum(1 for r in traj.rewards if abs(r) > 1e-9)

    print(f"  steps run               : {n_steps} / {max_turns}")
    print(f"  parseable JSON          : {sum(traj.parsed)} / {len(traj.parsed)}  ({parseable_pct:.0f}%)")
    print(f"  invalid_count           : {n_invalid}")
    print(f"  steps with non-0 reward : {rewards_nonzero} / {n_steps}")
    print(f"  dense reward (mean)     : {traj.dense_reward_mean:+.4f}")
    print(f"  grader_score            : {traj.grader_score:.4f}")
    print(f"  episode_success         : {traj.episode_success}")
    print(f"  env_error               : {traj.env_error!r}")
    if traj.history:
        print(f"  first step trace        : {traj.history[0]}")
        print(f"  last  step trace        : {traj.history[-1]}")

    failures: List[str] = []
    if traj.env_error:
        failures.append(f"env raised: {traj.env_error}")
    if n_steps == 0:
        failures.append("zero env steps completed")
    if parseable_pct < 50.0:
        failures.append(
            f"only {parseable_pct:.0f}% of completions parsed as JSON — "
            "increase max_new_tokens, tighten SYSTEM_PROMPT, or check the chat template"
        )
    if n_steps > 0 and rewards_nonzero == 0:
        failures.append(
            "every env step returned reward=0 — the env may be broken, the seed may be invalid, "
            "or the task_id may not be registered server-side"
        )

    if failures:
        msg = "SANITY CHECK FAILED:\n  - " + "\n  - ".join(failures)
        print("\n" + msg)
        if raise_on_failure:
            raise RuntimeError(msg)
    else:
        print("\nSANITY CHECK PASSED — LLM produces parseable JSON, env returns rewards, no errors.")

    return {
        "n_steps":         n_steps,
        "parseable_pct":   parseable_pct,
        "invalid_count":   n_invalid,
        "rewards_nonzero": rewards_nonzero,
        "grader_score":    traj.grader_score,
        "env_error":       traj.env_error,
        "history":         list(traj.history),
        "failures":        failures,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. evaluate_model_on_task / evaluate_model_suite
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_on_task(
    model: Any,
    tokenizer: Any,
    env_url: str,
    task_id: str,
    seeds: Sequence[int],
    *,
    max_turns: int = 30,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1024,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Run multiple seeds against one task and return aggregated stats.

    ``enable_thinking`` defaults to ``False`` (matches training conditions).
    Pass ``True`` for the demo / judges' inspection: completions will then
    include a ``<think>...</think>`` block so the chain-of-thought is
    visible. Bump ``max_new_tokens`` to 1024+ when thinking is on, otherwise
    the budget will be consumed before the JSON action is emitted.
    """
    from parkinsons_Motor.train import rollout_episode

    scores: List[float] = []
    successes = 0
    mean_amps: List[float] = []
    raw: List[Dict[str, Any]] = []
    for s in seeds:
        traj = rollout_episode(
            model, tokenizer, env_url,
            task_id=task_id, seed=int(s),
            max_turns=max_turns,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            enable_thinking=enable_thinking,
        )
        scores.append(traj.grader_score)
        successes += int(traj.episode_success)
        amps = [a for a in (_amp_from_history(ln) for ln in traj.history) if a is not None]
        mean_amps.append(sum(amps) / len(amps) if amps else 0.0)
        raw.append(traj.to_dict())
    return {
        "task_id":      task_id,
        "n_seeds":      len(seeds),
        "mean_score":   statistics.mean(scores) if scores else 0.0,
        "std_score":    statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min_score":    min(scores) if scores else 0.0,
        "max_score":    max(scores) if scores else 0.0,
        "pass_rate":    (successes / len(seeds)) if seeds else 0.0,
        "successes":    successes,
        "mean_amp_ma":  statistics.mean(mean_amps) if mean_amps else 0.0,
        "rollouts":     raw,
    }


def evaluate_model_suite(
    model: Any,
    tokenizer: Any,
    env_url: str,
    tasks: Sequence[str],
    seeds: Sequence[int],
    *,
    max_turns_per_task: Optional[Mapping[str, int]] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1024,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate one model on a set of (task, seed) pairs and return per-task results."""
    max_turns_per_task = max_turns_per_task or {}
    out: List[Dict[str, Any]] = []
    for task_id in tasks:
        result = evaluate_model_on_task(
            model, tokenizer, env_url, task_id, seeds,
            max_turns=max_turns_per_task.get(task_id, 30),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            enable_thinking=enable_thinking,
        )
        out.append(result)
        logger.info(
            "EVAL %-10s n=%d  mean=%.3f +/- %.3f  pass=%.0f%%  amp=%.2f mA",
            task_id, result["n_seeds"], result["mean_score"], result["std_score"],
            result["pass_rate"] * 100, result["mean_amp_ma"],
        )
    return out


__all__ = [
    "eval_with_adapter_disabled",
    "evaluate_model_on_task",
    "evaluate_model_suite",
    "sanity_check_rollout",
]
