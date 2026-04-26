# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Training utilities for the Parkinson's Motor (DBS) OpenEnv environment.

Public surface (mirrors the winning OpenEnv hackathon notebooks
`kube_sre_gym.train` and `training_script` from the Bio Experiment env):

    from parkinsons_Motor.train import (
        # constants
        SYSTEM_PROMPT,
        TASK_CONTEXT,
        INVALID_ACTION_PENALTY,
        ENVIRONMENT_ERROR_PENALTY,
        DEFAULT_REWARD_WEIGHTS,

        # prompt + chat helpers
        build_user_prompt,
        apply_chat_template,

        # action helpers
        parse_action,
        make_action,
        heuristic_action,

        # generation + rollout
        llm_generate,
        rollout_episode,
        rollout_episode_async,

        # reward
        compute_reward,
        MotorAssistReward,
        reward_total,
        reward_grader,
        reward_dense,
        reward_format,

        # GRPO glue
        make_rollout_func,
        make_episode_logger,

        # plots
        plot_training_dashboard,
        plot_training_loss,
        plot_baseline_vs_trained,
        compare_trajectories,
        save_training_plots,

        # eval
        evaluate_model_on_task,
        evaluate_model_suite,
        eval_with_adapter_disabled,   # base model on the same seeds, no retraining
    )

The module is import-safe with or without torch / matplotlib / pandas installed.
LLM-only helpers (rollout, evaluate) require torch + transformers + a live
ParkinsonsMotorEnv server. Reward and prompt helpers are pure-Python and can be
unit-tested without any heavy dependency.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import statistics
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .client import ParkinsonsMotorEnv
from .core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation

# torch / pandas / matplotlib are optional at import time so this module can
# be imported on a machine without a GPU stack (e.g. for unit tests).
try:  # pragma: no cover - thin import guard
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger("parkinsons_Motor.train")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants & prompt
# ─────────────────────────────────────────────────────────────────────────────

INVALID_ACTION_PENALTY: float = -1.0
"""Penalty (per turn) when the LLM emits a completion we can't parse as JSON."""

ENVIRONMENT_ERROR_PENALTY: float = -2.0
"""Penalty (per episode) when the env raises during step()."""

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "grader":  1.0,   # episode-end deterministic grader_score in [0, 1]
    "dense":   0.5,   # mean of per-step env rewards, smooths early training
    "format":  0.2,   # fraction of turns that produced parseable JSON
    "invalid": 1.0,   # multiplier on INVALID_ACTION_PENALTY * (invalid_turns / total)
}

SYSTEM_PROMPT: str = textwrap.dedent("""
You are an expert closed-loop DBS controller managing Parkinsonian motor symptoms in real time.
Every step is a short clinical control decision: suppress pathological activity, preserve movement,
avoid overstimulation, and keep enough safety budget for the remainder of the episode.

You MAY think step by step inside <think>...</think> if helpful, but your response MUST end with
exactly one valid JSON object on its own final line, in this exact shape:

{"dbs_amplitude": <float>, "dbs_pulse_width": <float>, "dbs_frequency": <float>}

Hard constraints (the deterministic grader will reject violations):
- amplitude in mA [0.0, 5.0]; pulse_width in ms [0.06, 0.20]; frequency in Hz [60, 185].
- Use pulse_width=0.13 and frequency=130 unless safety demands otherwise.
- Smooth amplitude changes; avoid jumps > 0.3 mA per step.
- The JSON object MUST be the last non-empty line. No prose, markdown, or fences after it.
- Do NOT include motor_command — that field is ignored.

Clinical priorities (ranked):
1. Prevent overstimulation: gamma_arv > 0.55 OR side_effect_load near budget => reduce.
2. Don't undertreat: tremor_arv > 0.55 OR beta_arv > 0.60 => at least 1.2 mA.
3. Symptoms worsening + safety acceptable => +0.10-0.15 mA.
4. Symptoms improving + side_effect_rate > 0 => hold or reduce.
5. Stable control => taper toward minimum effective amplitude.
""").strip()

TASK_CONTEXT: Dict[str, str] = {
    "easy":   "EASY — responsive patient, no events. Stabilize quickly, taper to maintenance.",
    "medium": "MEDIUM — balanced patient, mid-episode deterioration possible. Decisive rescue, then taper.",
    "hard":   "HARD — refractory patient, multi-crisis 150-step run. Pace safety budget; expect tachyphylaxis + off-med crisis.",
    "fragile_patient":              "FRAGILE — narrow safety window. Stay conservative; gamma rises fast.",
    "refractory_patient":           "REFRACTORY — low responsiveness. Push amplitude harder while watching gamma.",
    "personalization_generalization": "PERSONALIZATION — patient drawn from a held-out cohort.",
    "exercise_bout":                "EXERCISE BOUT — transient motor demand spike mid-episode.",
    "medication_interaction":       "MEDICATION INTERACTION — L-DOPA on/off cycle modulates baseline.",
    "nocturnal_transition":         "NOCTURNAL — setpoint slowly tapers; avoid late-episode overstim.",
    "surgical_followup":            "SURGICAL FOLLOWUP — recent lead repositioning; entrainment is noisy.",
}

# Keys we cherry-pick from the obs dict for the "STATE" block in the prompt.
# Kept short so the LLM context stays small and deterministic.
_PROMPT_STATE_KEYS: Tuple[str, ...] = (
    "beta_arv", "tremor_arv", "force_preserved", "side_effect_load",
    "beta_trend", "tremor_trend", "side_effect_rate", "gamma_arv",
    "dbs_entrainment", "stim_washout", "target_output",
)

# Compact JSON object regex tolerant of internal whitespace / floats / negatives.
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Replace ```json ... ``` blocks with their inner content (in-place).

    Qwen3-4B sometimes wraps its final action in a markdown code fence even when
    we tell it not to. Returning the inner content makes the JSON regex find it.
    """
    if "```" not in text:
        return text
    return _CODE_FENCE_RE.sub(lambda m: " " + m.group(1) + " ", text)


def _find_balanced_json(text: str) -> Optional[str]:
    """Scan for the LAST balanced ``{...}`` substring (handles nested objects).

    The flat ``_JSON_RE`` rejects any nested brace, so we keep this as a
    backup for the (rare) case where the model emits ``{"a": {"b": 1}}``.
    Returns the JSON text or None.
    """
    last: Optional[str] = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = text[start : i + 1]
                    start = -1
    return last


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Coerce a ParkinsonsMotorObservation (pydantic) or dict into a plain dict."""
    if obs is None:
        return {}
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if isinstance(obs, dict):
        return obs
    return getattr(obs, "__dict__", {}) or {}


def build_user_prompt(
    step: int,
    obs: Union[Dict[str, Any], ParkinsonsMotorObservation],
    task_id: str,
    history: Sequence[str],
) -> str:
    """Build the per-turn user prompt.

    Includes:
      * task name + one-line task context
      * explicit ALERT lines (priority cues from the system prompt)
      * the 11 most decision-relevant observation fields
      * the last 3 history lines
    """
    obs = _obs_to_dict(obs)
    flags: List[str] = []
    if obs.get("tremor_arv", 0) > 0.55 or obs.get("beta_arv", 0) > 0.60:
        flags.append("[!] symptoms HIGH — maintain >= 1.2 mA")
    if obs.get("side_effect_load", 0) > 0.36:
        flags.append("[!] safety budget being spent — reduce amp")
    if obs.get("gamma_arv", 0) > 0.55:
        flags.append("[!] gamma high — overstimulation")
    if obs.get("beta_trend", 0) > 0.01 and obs.get("tremor_trend", 0) > 0.01:
        flags.append("[^] both signals worsening — increase")
    # Use 6-space joins so the interpolated multi-line blocks keep the same
    # leading whitespace as the surrounding template; otherwise textwrap.dedent
    # re-anchors to col 0 and second/third lines come out flush-left.
    flag_str = "\n      ".join(flags) if flags else "normal — adaptive control"
    recent = "\n      ".join(list(history)[-3:]) if history else "(first step)"

    state_lines = "\n      ".join(
        f"{k}={obs.get(k, 0):+.3f}" if k in {"beta_trend", "tremor_trend", "side_effect_rate"}
        else f"{k}={obs.get(k, 0):.3f}"
        for k in _PROMPT_STATE_KEYS
    )
    context = TASK_CONTEXT.get(task_id, "")

    return textwrap.dedent(f"""
    TASK: {task_id} | STEP: {step}
    {context}

    ALERTS:
      {flag_str}

    STATE:
      {state_lines}

    RECENT:
      {recent}

    Output JSON only.
    """).strip()


def apply_chat_template(
    tokenizer: Any,
    system: str,
    user: str,
    *,
    enable_thinking: bool = True,
) -> str:
    """Render a Qwen-style chat template ready to feed to .generate().

    Qwen3 templates support an ``enable_thinking`` kwarg that controls whether
    the assistant turn opens with ``<think>``; we pass it through and silently
    fall back for tokenizers (Qwen2, Llama, ...) that don't accept it.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Action parsing & construction
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_AMP = 1.0
_DEFAULT_PW  = 0.13
_DEFAULT_FQ  = 130.0


def parse_action(text: str) -> Optional[Dict[str, float]]:
    """Extract the **last** JSON object from a completion. Returns None on failure.

    We deliberately scan back-to-front (and skip anything inside
    ``<think>...</think>`` blocks) because Qwen3-style models often write
    intermediate JSON examples while reasoning, then commit to the real action
    on the final line. We want the commit, not the first scratchpad sample.

    Robustness layers, applied in order:
      1. Strip ``<think>...</think>`` blocks.
      2. Strip ```` ```json ... ``` ```` markdown fences (Qwen3 sometimes adds
         them even when told not to).
      3. Try the flat ``_JSON_RE`` (matches non-nested ``{...}``).
      4. Fall back to a balanced-brace scan to recover nested or
         multi-line JSON the simple regex misses.
    """
    if not text:
        return None
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if not answer.strip():
        answer = text  # all of it was inside <think> — fall back to full text
    answer = _strip_code_fences(answer)

    candidates = list(_JSON_RE.finditer(answer))
    for match in reversed(candidates):
        try:
            d = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            return d

    nested = _find_balanced_json(answer)
    if nested:
        try:
            d = json.loads(nested)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            return None
    return None


def make_action(
    parsed: Optional[Mapping[str, Any]],
    target_output: float = 0.0,
    *,
    fallback_amp: float = _DEFAULT_AMP,
    fallback_pw: float = _DEFAULT_PW,
    fallback_fq: float = _DEFAULT_FQ,
) -> ParkinsonsMotorAction:
    """Coerce a parsed dict into a clipped, well-formed action.

    Always sets ``motor_command = clip(target_output, -1, 1)`` — the LLM is
    only responsible for the three DBS knobs.
    """
    target = float(max(-1.0, min(1.0, target_output)))
    if not parsed:
        return ParkinsonsMotorAction(
            motor_command=target,
            dbs_amplitude=fallback_amp,
            dbs_pulse_width=fallback_pw,
            dbs_frequency=fallback_fq,
        )
    return ParkinsonsMotorAction(
        motor_command=target,
        dbs_amplitude=float(max(0.0,  min(5.0,   parsed.get("dbs_amplitude",   fallback_amp)))),
        dbs_pulse_width=float(max(0.06, min(0.20, parsed.get("dbs_pulse_width", fallback_pw)))),
        dbs_frequency=float(max(60.0, min(185.0, parsed.get("dbs_frequency",   fallback_fq)))),
    )


def heuristic_action(
    obs: Union[Dict[str, Any], ParkinsonsMotorObservation],
    task_id: str = "easy",
    last_amp: Optional[float] = None,
) -> ParkinsonsMotorAction:
    """A simple clinical-priority heuristic — useful as a teacher / baseline.

    Mirrors the priorities encoded in `SYSTEM_PROMPT`:
      1. Reduce on overstimulation (gamma high or safety budget burning).
      2. Push to >= 1.2 mA when symptoms are clearly high.
      3. Small bump when both trends point up.
      4. Otherwise hold near a moderate maintenance dose.

    The ``last_amp`` arg, if provided, is used as the smooth starting point so
    successive heuristic actions don't jump by more than 0.3 mA.
    """
    obs = _obs_to_dict(obs)
    base_amp = last_amp if last_amp is not None else 1.2
    pw, fq = _DEFAULT_PW, _DEFAULT_FQ

    gamma   = obs.get("gamma_arv", 0)
    se_load = obs.get("side_effect_load", 0)
    tremor  = obs.get("tremor_arv", 0)
    beta    = obs.get("beta_arv", 0)
    beta_tr = obs.get("beta_trend", 0)
    trem_tr = obs.get("tremor_trend", 0)
    se_rate = obs.get("side_effect_rate", 0)

    if gamma > 0.55 or se_load > 0.36:
        amp = max(0.4, base_amp - 0.3)
    elif tremor > 0.55 or beta > 0.60:
        amp = min(2.5, max(1.2, base_amp + 0.2))
    elif beta_tr > 0.01 and trem_tr > 0.01:
        amp = min(2.0, base_amp + 0.15)
    elif beta_tr < -0.01 and trem_tr < -0.01 and se_rate > 0:
        amp = max(0.4, base_amp - 0.1)
    else:
        amp = base_amp

    if last_amp is not None:
        amp = max(last_amp - 0.3, min(last_amp + 0.3, amp))

    return ParkinsonsMotorAction(
        motor_command=float(max(-1.0, min(1.0, obs.get("target_output", 0.0)))),
        dbs_amplitude=float(amp),
        dbs_pulse_width=pw,
        dbs_frequency=fq,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generation + rollout
# ─────────────────────────────────────────────────────────────────────────────

def llm_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_prompt_length: int = 1024,
) -> Tuple[str, "torch.Tensor", "torch.Tensor"]:
    """Run one greedy/sampling generation. Returns (text, prompt_ids, completion_ids).

    ``top_p`` defaults to 0.95 to match the value GRPOConfig uses during
    training; aligning rollout-time and trainer-time samplers prevents an
    importance-sampling mismatch in TRL's per-token loss.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for llm_generate; install transformers + torch.")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            pad_token_id=pad_id,
        )
    prompt_ids = inputs["input_ids"][0].detach().cpu()
    completion_ids = out[0][inputs["input_ids"].shape[1]:].detach().cpu()
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, prompt_ids, completion_ids


@dataclass
class Trajectory:
    """One full episode rolled out against a live ParkinsonsMotorEnv server."""
    task_id: str
    seed: Optional[int]
    prompts: List[Any] = field(default_factory=list)         # list[Tensor] of prompt token ids
    completions: List[Any] = field(default_factory=list)     # list[Tensor] of completion ids
    parsed: List[bool] = field(default_factory=list)         # did the LLM emit valid JSON?
    rewards: List[float] = field(default_factory=list)       # per-step env reward
    history: List[str] = field(default_factory=list)         # short human-readable transcript
    invalid_count: int = 0
    grader_score: float = 0.0
    episode_success: bool = False
    dense_reward_mean: float = 0.0
    env_error: Optional[str] = None
    n_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "n_steps": self.n_steps,
            "invalid_count": self.invalid_count,
            "grader_score": self.grader_score,
            "episode_success": self.episode_success,
            "dense_reward_mean": self.dense_reward_mean,
            "rewards": list(self.rewards),
            "history": list(self.history),
            "env_error": self.env_error,
        }


async def rollout_episode_async(
    model: Any,
    tokenizer: Any,
    env_url: str,
    task_id: str,
    seed: Optional[int] = None,
    *,
    max_turns: int = 30,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1024,
    fallback_to_heuristic_on_invalid: bool = False,
) -> Trajectory:
    """Play one full episode against a remote ParkinsonsMotorEnv (async)."""
    traj = Trajectory(task_id=task_id, seed=seed)
    env = ParkinsonsMotorEnv(base_url=env_url)
    await env.__aenter__()
    last_amp: Optional[float] = None
    try:
        reset_kwargs: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            reset_kwargs["seed"] = seed
        result = await env.reset(**reset_kwargs)
        obs = result.observation

        for step in range(1, max_turns + 1):
            obs_dict = _obs_to_dict(obs)
            user = build_user_prompt(step, obs_dict, task_id, traj.history)
            prompt = apply_chat_template(tokenizer, SYSTEM_PROMPT, user)

            # CRITICAL: model.generate() is a blocking sync call. Running it
            # directly inside this coroutine starves the asyncio event loop, so
            # the EnvClient's persistent-WebSocket keepalive task can't PONG
            # within ~20 s and the server closes with code 1011. Hand the
            # generation to a worker thread so the loop stays responsive.
            text, p_ids, c_ids = await asyncio.to_thread(
                llm_generate,
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_prompt_length=max_prompt_length,
            )

            parsed = parse_action(text)
            if parsed is None:
                traj.invalid_count += 1
                if fallback_to_heuristic_on_invalid:
                    action = heuristic_action(obs_dict, task_id=task_id, last_amp=last_amp)
                else:
                    action = make_action(None, target_output=obs_dict.get("target_output", 0.0))
            else:
                action = make_action(parsed, target_output=obs_dict.get("target_output", 0.0))
            last_amp = action.dbs_amplitude

            try:
                result = await env.step(action)
            except Exception as exc:
                traj.env_error = str(exc)
                logger.warning("env.step raised: %s", exc)
                break
            obs = result.observation
            obs_dict = _obs_to_dict(obs)
            r = float(result.reward or 0.0)

            traj.prompts.append(p_ids)
            traj.completions.append(c_ids)
            traj.parsed.append(parsed is not None)
            traj.rewards.append(r)
            traj.history.append(
                f"step={step} amp={action.dbs_amplitude:.2f} "
                f"=> beta={obs_dict.get('beta_arv', 0):.3f} "
                f"tremor={obs_dict.get('tremor_arv', 0):.3f} "
                f"se={obs_dict.get('side_effect_load', 0):.3f} r={r:+.2f}"
            )

            if result.done:
                traj.grader_score = max(0.0, float(obs_dict.get("grader_score", 0.0)))
                traj.episode_success = bool(obs_dict.get("episode_success", False))
                break
        traj.n_steps = len(traj.rewards)
        if traj.rewards:
            traj.dense_reward_mean = sum(traj.rewards) / len(traj.rewards)
    finally:
        await env.__aexit__(None, None, None)
    return traj


def rollout_episode(*args, **kwargs) -> Trajectory:
    """Sync wrapper around :func:`rollout_episode_async` (uses ``asyncio.run``)."""
    return asyncio.run(rollout_episode_async(*args, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reward composition
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    trajectory: Union[Trajectory, Dict[str, Any]],
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Combine grader / dense / format / invalid signals into a single scalar.

    Returns a dict so per-component rewards can be plotted separately during
    training (mirrors the bio-env winner's `OpenEnvReward` decomposition).
    """
    if isinstance(trajectory, Trajectory):
        n_turns       = max(1, trajectory.n_steps or len(trajectory.rewards))
        n_valid       = sum(trajectory.parsed)
        invalid_count = trajectory.invalid_count
        grader_score  = trajectory.grader_score
        dense_mean    = trajectory.dense_reward_mean
        env_error     = trajectory.env_error
        success       = trajectory.episode_success
    else:
        n_turns       = max(1, int(trajectory.get("n_steps") or len(trajectory.get("rewards", []))))
        n_valid       = int(trajectory.get("n_valid", n_turns - int(trajectory.get("invalid_count", 0))))
        invalid_count = int(trajectory.get("invalid_count", 0))
        grader_score  = float(trajectory.get("grader_score", 0.0))
        dense_mean    = float(trajectory.get("dense_reward_mean", 0.0))
        env_error     = trajectory.get("env_error")
        success       = bool(trajectory.get("episode_success", False))

    w = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        w.update(weights)

    r_grader  = w["grader"] * grader_score
    r_dense   = w["dense"]  * max(0.0, min(1.0, dense_mean))
    r_format  = w["format"] * (n_valid / n_turns)
    r_invalid = w["invalid"] * INVALID_ACTION_PENALTY * (invalid_count / n_turns)
    r_env_err = ENVIRONMENT_ERROR_PENALTY if env_error else 0.0
    r_total   = r_grader + r_dense + r_format + r_invalid + r_env_err

    return {
        "reward_total":            r_total,
        "reward_grader":           r_grader,
        "reward_dense":            r_dense,
        "reward_format":           r_format,
        "reward_invalid_penalty":  r_invalid,
        "reward_env_error":        r_env_err,
        "grader_score":            grader_score,
        "dense_reward_mean":       dense_mean,
        "n_steps":                 n_turns,
        "invalid_count":           invalid_count,
        "episode_success":         success,
    }


@dataclass
class MotorAssistReward:
    """Lookup-style GRPO reward (mirrors `OpenEnvReward` from the bio-env winner).

    The actual env replay happens inside the rollout function — this callable
    just surfaces precomputed reward components from kwargs back to GRPO. If
    nothing was precomputed (e.g. the user is testing the reward in isolation),
    it returns zeros instead of crashing.
    """

    component: str = "reward_total"

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> List[float]:
        values = kwargs.get(self.component)
        if values is None:
            return [0.0] * len(completions)
        return [float(v) for v in values]


# Convenience GRPO-style reward callables. Pass these into `GRPOTrainer(reward_funcs=[...])`.
def reward_total(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    return MotorAssistReward("reward_total")(completions, **kwargs)


def reward_grader(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    return MotorAssistReward("reward_grader")(completions, **kwargs)


def reward_dense(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    return MotorAssistReward("reward_dense")(completions, **kwargs)


def reward_format(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    return MotorAssistReward("reward_format")(completions, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CSV episode logger
# ─────────────────────────────────────────────────────────────────────────────

_LOG_COLUMNS: Tuple[str, ...] = (
    "step", "task_id", "seed",
    "reward_total", "reward_grader", "reward_dense",
    "reward_format", "reward_invalid_penalty", "reward_env_error",
    "grader_score", "dense_reward_mean", "n_steps",
    "invalid_count", "episode_success",
)


def make_episode_logger(csv_path: Union[str, Path]) -> Callable[[str, Optional[int], Dict[str, float]], int]:
    """Return a function ``log_episode(task_id, seed, reward_breakdown)`` that
    appends to ``csv_path``. The CSV header is written on first call.

    The returned callable also logs to ``parkinsons_Motor.train`` at INFO and
    keeps a running mean / mean-of-last-10 for human-friendly progress.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_LOG_COLUMNS)

    counter = [0]
    all_rewards: List[float] = []

    def log_episode(task_id: str, seed: Optional[int], breakdown: Dict[str, float]) -> int:
        counter[0] += 1
        all_rewards.append(breakdown["reward_total"])
        row = [
            counter[0], task_id, seed if seed is not None else "",
            breakdown.get("reward_total", 0.0),
            breakdown.get("reward_grader", 0.0),
            breakdown.get("reward_dense", 0.0),
            breakdown.get("reward_format", 0.0),
            breakdown.get("reward_invalid_penalty", 0.0),
            breakdown.get("reward_env_error", 0.0),
            breakdown.get("grader_score", 0.0),
            breakdown.get("dense_reward_mean", 0.0),
            breakdown.get("n_steps", 0),
            breakdown.get("invalid_count", 0),
            int(breakdown.get("episode_success", False)),
        ]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        last10 = all_rewards[-10:]
        logger.info(
            "ep=%03d  task=%-10s seed=%s  R=%+.3f  grader=%.3f  pass=%s  "
            "mean=%+.3f  mean(10)=%+.3f",
            counter[0], task_id, seed,
            breakdown["reward_total"], breakdown["grader_score"],
            bool(breakdown.get("episode_success", False)),
            sum(all_rewards) / len(all_rewards),
            sum(last10) / len(last10),
        )
        return counter[0]

    return log_episode


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRPO rollout factory
# ─────────────────────────────────────────────────────────────────────────────

def make_rollout_func(
    env_url: str,
    episodes: Sequence[Tuple[str, Optional[int]]],
    *,
    max_turns_per_task: Mapping[str, int],
    num_generations: int,
    log_episode: Optional[Callable[[str, Optional[int], Dict[str, float]], Any]] = None,
    temperature: float = 0.7,
    reward_weights: Optional[Mapping[str, float]] = None,
    fallback_to_heuristic_on_invalid: bool = True,
    max_new_tokens: int = 256,
    max_prompt_length: int = 1024,
) -> Callable[..., Dict[str, List[Any]]]:
    """Build the GRPO ``rollout_func`` callable.

    The returned function plays one full episode against the remote env per
    incoming "prompt", in the order given by ``episodes`` (each replicated
    ``num_generations`` times so a GRPO group sees identical env conditions
    with different policy samples — the same trick the kube-sre-gym winner
    used).
    """
    schedule: List[Tuple[str, Optional[int]]] = []
    for task_id, seed in episodes:
        for _ in range(num_generations):
            schedule.append((task_id, seed))
    cursor = [0]

    def rollout_func(prompts: Sequence[Any], trainer: Any) -> Dict[str, List[Any]]:
        results: Dict[str, List[Any]] = {
            "prompt_ids":             [],
            "completion_ids":         [],
            "logprobs":               [],
            "reward_total":           [],
            "reward_grader":          [],
            "reward_dense":           [],
            "reward_format":          [],
            "reward_invalid_penalty": [],
            "reward_env_error":       [],
            "grader_score":           [],
            "episode_success":        [],
            "n_steps":                [],
        }
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for rollout_func.")
        for _ in prompts:
            task_id, seed = schedule[cursor[0] % len(schedule)]
            cursor[0] += 1
            max_turns = max_turns_per_task.get(task_id, 30)
            traj = rollout_episode(
                trainer.model, trainer.processing_class,
                env_url, task_id=task_id, seed=seed,
                max_turns=max_turns,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_prompt_length=max_prompt_length,
                fallback_to_heuristic_on_invalid=fallback_to_heuristic_on_invalid,
            )
            breakdown = compute_reward(traj, weights=reward_weights)
            if log_episode:
                log_episode(task_id, seed, breakdown)
            if not traj.prompts:
                # Env failed before *any* successful step. We still need to return
                # one row per incoming prompt so GRPO doesn't get a length mismatch
                # (which silently zero-pads the group and kills the gradient).
                logger.warning(
                    "ROLLOUT FAILED  task=%s seed=%s env_error=%r — emitting "
                    "placeholder row with reward_total=%.2f so GRPO group stays aligned.",
                    task_id, seed, traj.env_error, ENVIRONMENT_ERROR_PENALTY,
                )
                # Tokenize a 1-token "[FAIL]" so prompt/completion lists are non-empty.
                fail_ids = tokenizer("[FAIL]", return_tensors="pt").input_ids[0]
                results["prompt_ids"].append(fail_ids.tolist())
                results["completion_ids"].append(fail_ids.tolist())
                results["logprobs"].append([0.0] * len(fail_ids))
                results["reward_total"].append(float(ENVIRONMENT_ERROR_PENALTY))
                results["reward_grader"].append(0.0)
                results["reward_dense"].append(0.0)
                results["reward_format"].append(0.0)
                results["reward_invalid_penalty"].append(0.0)
                results["reward_env_error"].append(float(ENVIRONMENT_ERROR_PENALTY))
                results["grader_score"].append(0.0)
                results["episode_success"].append(0)
                results["n_steps"].append(0)
                continue
            p_ids = torch.cat(traj.prompts)
            c_ids = torch.cat(traj.completions)
            results["prompt_ids"].append(p_ids.tolist())
            results["completion_ids"].append(c_ids.tolist())
            results["logprobs"].append([0.0] * len(c_ids))      # GRPO recomputes
            results["reward_total"].append(breakdown["reward_total"])
            results["reward_grader"].append(breakdown["reward_grader"])
            results["reward_dense"].append(breakdown["reward_dense"])
            results["reward_format"].append(breakdown["reward_format"])
            results["reward_invalid_penalty"].append(breakdown["reward_invalid_penalty"])
            results["reward_env_error"].append(breakdown["reward_env_error"])
            results["grader_score"].append(breakdown["grader_score"])
            results["episode_success"].append(int(breakdown["episode_success"]))
            results["n_steps"].append(breakdown["n_steps"])
        return results

    return rollout_func


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────

_TASK_THRESHOLDS: Dict[str, float] = {
    "easy": 0.55, "medium": 0.52, "hard": 0.68,
}


def plot_training_dashboard(
    csv_path: Union[str, Path],
    png_path: Union[str, Path],
    train_tasks: Optional[Sequence[str]] = None,
    title: str = "MotorAssistEnv — GRPO training",
) -> Path:
    """Render a 2x2 dashboard (total / grader / per-task / decomposition).

    Mirrors `plot_rewards` from the kube-sre-gym winner but with one extra
    axis (per-task curves) since our env is a curriculum.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib + pandas required for plot_training_dashboard") from exc
    csv_path = Path(csv_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if train_tasks is None:
        train_tasks = list(dict.fromkeys(df["task_id"].astype(str).tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    axes[0].plot(df["step"], df["reward_total"], marker="o", linewidth=1, label="total")
    axes[0].plot(df["step"], df["reward_total"].rolling(5, min_periods=1).mean(),
                 label="5-ep mean", color="black")
    axes[0].set(title="Total reward", xlabel="episode", ylabel="reward")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["step"], df["grader_score"], marker="o", linewidth=1, color="tab:green")
    axes[1].plot(df["step"], df["grader_score"].rolling(5, min_periods=1).mean(), color="black")
    for tid, thr in _TASK_THRESHOLDS.items():
        if tid in train_tasks:
            axes[1].axhline(thr, ls=":", alpha=0.6, label=f"{tid} threshold ({thr:.2f})")
    axes[1].set(title="Grader score (deterministic, [0,1])", xlabel="episode", ylabel="score")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    palette = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for i, task_id in enumerate(train_tasks):
        sub = df[df["task_id"] == task_id]
        if len(sub):
            axes[2].plot(sub["step"], sub["grader_score"].rolling(3, min_periods=1).mean(),
                         label=task_id, color=palette[i % len(palette)],
                         marker="o", linewidth=1.4)
    axes[2].set(title="Grader by task (3-ep rolling)", xlabel="episode", ylabel="grader score")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    bar_x = df["step"]
    axes[3].bar(bar_x, df["reward_grader"],          label="grader",  alpha=0.85)
    axes[3].bar(bar_x, df["reward_dense"],           label="dense",   alpha=0.55)
    axes[3].bar(bar_x, df["reward_format"],          label="format",  alpha=0.45)
    axes[3].bar(bar_x, df["reward_invalid_penalty"], label="invalid penalty", alpha=0.45)
    axes[3].set(title="Reward decomposition per episode", xlabel="episode", ylabel="component")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def plot_training_loss(
    log_history: Sequence[Mapping[str, Any]],
    png_path: Union[str, Path],
    title: str = "MotorAssistEnv — GRPO training (loss & policy stats)",
) -> Path:
    """Plot loss + reward + KL + grad_norm from ``trainer.state.log_history``.

    Judges explicitly ask for "loss AND reward plots" — the
    ``training_dashboard`` covers reward decomposition; this one covers the
    optimization side (policy loss, KL anchor, gradient norm).

    ``log_history`` is the list TRL writes after every logging step:
    each row is a dict that may contain ``loss``, ``reward``, ``kl``,
    ``grad_norm``, ``learning_rate`` and a ``step`` key.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required for plot_training_loss") from exc
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    def _series(key: str) -> Tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []
        for i, row in enumerate(log_history):
            if not isinstance(row, Mapping):
                continue
            v = row.get(key)
            if v is None:
                continue
            try:
                ys.append(float(v))
                xs.append(int(row.get("step", i)))
            except (TypeError, ValueError):
                continue
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, (key, color, title_) in zip(
        axes,
        [
            ("loss",        "tab:blue",   "Policy loss"),
            ("reward",      "tab:green",  "Mean reward (per logging step)"),
            ("kl",          "tab:purple", "KL divergence to reference"),
            ("grad_norm",   "tab:orange", "Gradient norm"),
        ],
    ):
        xs, ys = _series(key)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.2, color=color)
            ax.set(title=title_, xlabel="training step", ylabel=key)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"no '{key}' values in log_history",
                    ha="center", va="center", color="grey", transform=ax.transAxes)
            ax.set(title=title_); ax.set_axis_off()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def plot_baseline_vs_trained(
    baseline_results: Sequence[Mapping[str, Any]],
    trained_results: Sequence[Mapping[str, Any]],
    png_path: Union[str, Path],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    title: str = "Base vs trained — grader score by task",
) -> Path:
    """Side-by-side bar chart comparing two ``evaluate_model_suite`` outputs.

    Judges explicitly ask for "multiple runs on the same axes so the comparison
    is obvious". This is that plot.

    Each input is a list of per-task dicts with at least ``task_id``,
    ``mean_score``, ``std_score`` (the shape ``evaluate_model_suite`` returns).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib + numpy required for plot_baseline_vs_trained") from exc

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = dict(thresholds or _TASK_THRESHOLDS)

    base_by_task    = {r["task_id"]: r for r in baseline_results}
    trained_by_task = {r["task_id"]: r for r in trained_results}
    tasks = [t for t in (list(base_by_task.keys()) + list(trained_by_task.keys()))]
    seen: List[str] = []
    for t in tasks:
        if t not in seen:
            seen.append(t)
    tasks = seen

    base_means    = [float(base_by_task.get(t, {}).get("mean_score", 0.0)) for t in tasks]
    base_stds     = [float(base_by_task.get(t, {}).get("std_score", 0.0))  for t in tasks]
    trained_means = [float(trained_by_task.get(t, {}).get("mean_score", 0.0)) for t in tasks]
    trained_stds  = [float(trained_by_task.get(t, {}).get("std_score", 0.0))  for t in tasks]
    base_pass     = [float(base_by_task.get(t, {}).get("pass_rate", 0.0)) for t in tasks]
    trained_pass  = [float(trained_by_task.get(t, {}).get("pass_rate", 0.0)) for t in tasks]

    x = np.arange(len(tasks))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.bar(x - w / 2, base_means,    w, yerr=base_stds,    capsize=4,
            label="base",    color="tab:gray", alpha=0.85)
    ax1.bar(x + w / 2, trained_means, w, yerr=trained_stds, capsize=4,
            label="trained", color="tab:green", alpha=0.85)
    for i, t in enumerate(tasks):
        thr = thresholds.get(t)
        if thr is not None:
            ax1.hlines(thr, i - w, i + w, colors="black", linestyles=":", linewidth=1.2)
            ax1.text(i, thr + 0.015, f"thr {thr:.2f}", ha="center", fontsize=8, color="black")
    for i, (b, tr) in enumerate(zip(base_means, trained_means)):
        delta = tr - b
        ax1.annotate(
            f"{delta:+.2f}",
            xy=(i + w / 2, max(b, tr) + 0.04),
            ha="center", fontsize=9,
            color="tab:green" if delta > 0 else "tab:red", weight="bold",
        )
    ax1.set(xlabel="task", ylabel="grader score (deterministic, [0,1])",
            title="Mean grader score ± std (5 seeds per bar)")
    ax1.set_xticks(x); ax1.set_xticklabels(tasks)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x - w / 2, [p * 100 for p in base_pass],    w,
            label="base",    color="tab:gray",  alpha=0.85)
    ax2.bar(x + w / 2, [p * 100 for p in trained_pass], w,
            label="trained", color="tab:green", alpha=0.85)
    ax2.set(xlabel="task", ylabel="pass rate (%)",
            title="Pass rate by task (success_threshold from TASKS.md)")
    ax2.set_xticks(x); ax2.set_xticklabels(tasks)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def compare_trajectories(
    base_traj: Union[Trajectory, Mapping[str, Any]],
    trained_traj: Union[Trajectory, Mapping[str, Any]],
    png_path: Union[str, Path],
    *,
    title: Optional[str] = None,
) -> Path:
    """Overlay base-vs-trained per-step traces from one episode each.

    Judges explicitly ask for "before/after behavior" as qualitative evidence —
    this is that plot. We extract amplitude / β / tremor / side-effect-load
    from the ``history`` strings written by ``rollout_episode_async``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required for compare_trajectories") from exc
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    def _parse(traj: Union[Trajectory, Mapping[str, Any]]) -> Dict[str, List[float]]:
        if isinstance(traj, Trajectory):
            history = traj.history
            task_id = traj.task_id
            seed    = traj.seed
            grader  = traj.grader_score
            success = traj.episode_success
        else:
            history = list(traj.get("history") or [])
            task_id = str(traj.get("task_id", ""))
            seed    = traj.get("seed")
            grader  = float(traj.get("grader_score", 0.0))
            success = bool(traj.get("episode_success", False))

        out: Dict[str, List[float]] = {
            "step": [], "amp": [], "beta": [], "tremor": [], "se": [], "reward": [],
        }
        for line in history:
            try:
                parts = dict(p.split("=", 1) for p in line.replace("=>", "").split() if "=" in p)
            except Exception:
                continue
            if "step" not in parts:
                continue
            try:
                out["step"].append(int(parts["step"]))
                out["amp"].append(float(parts.get("amp", 0)))
                out["beta"].append(float(parts.get("beta", 0)))
                out["tremor"].append(float(parts.get("tremor", 0)))
                out["se"].append(float(parts.get("se", 0)))
                out["reward"].append(float(parts.get("r", 0)))
            except (TypeError, ValueError):
                continue
        out["task_id"] = task_id  # type: ignore[assignment]
        out["seed"]    = seed     # type: ignore[assignment]
        out["grader"]  = grader   # type: ignore[assignment]
        out["success"] = success  # type: ignore[assignment]
        return out

    base = _parse(base_traj)
    tr   = _parse(trained_traj)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, key, ylabel, ylim in [
        (axes[0], "amp",    "DBS amplitude (mA)",      None),
        (axes[1], "beta",   "β-band ARV [0,1]",        (0, 1)),
        (axes[2], "tremor", "tremor ARV [0,1]",        (0, 1)),
        (axes[3], "se",     "side-effect load [0,1]",  (0, 1)),
    ]:
        if base["step"]:
            ax.plot(base["step"], base[key], color="tab:gray",
                    label=f"base (grader={base['grader']:.2f})", linewidth=1.6)
        if tr["step"]:
            ax.plot(tr["step"], tr[key], color="tab:green",
                    label=f"trained (grader={tr['grader']:.2f})", linewidth=1.8)
        ax.set(xlabel="step (20 ms each)", ylabel=ylabel, title=ylabel.split(" (")[0])
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    task_id = base.get("task_id") or tr.get("task_id") or "?"   # type: ignore[assignment]
    seed    = base.get("seed") if base.get("seed") is not None else tr.get("seed")
    full_title = title or f"Before vs after training — task=`{task_id}` seed={seed}"
    fig.suptitle(full_title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def save_training_plots(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_tasks: Optional[Sequence[str]] = None,
    *,
    log_history: Optional[Sequence[Mapping[str, Any]]] = None,
    baseline_results: Optional[Sequence[Mapping[str, Any]]] = None,
    trained_results: Optional[Sequence[Mapping[str, Any]]] = None,
    base_trajectory: Optional[Union[Trajectory, Mapping[str, Any]]] = None,
    trained_trajectory: Optional[Union[Trajectory, Mapping[str, Any]]] = None,
) -> Dict[str, Path]:
    """Save every plot we know how to draw. Returns ``{name: path}``.

    Name kept identical to the bio-env winner's ``save_training_plots`` so the
    notebook surface looks the same. Optional inputs let you add the loss
    panel, the base-vs-trained bars, and a sample trajectory overlay in a
    single call.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    paths["dashboard"] = plot_training_dashboard(
        csv_path,
        output_dir / "training_dashboard.png",
        train_tasks=train_tasks,
    )
    if log_history:
        paths["loss"] = plot_training_loss(
            log_history,
            output_dir / "training_loss.png",
        )
    if baseline_results and trained_results:
        paths["comparison"] = plot_baseline_vs_trained(
            baseline_results,
            trained_results,
            output_dir / "eval_comparison.png",
        )
    if base_trajectory is not None and trained_trajectory is not None:
        paths["trajectory"] = compare_trajectories(
            base_trajectory,
            trained_trajectory,
            output_dir / "trajectory_compare.png",
        )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 8. Eval
# ─────────────────────────────────────────────────────────────────────────────

def _amp_from_history(line: str) -> Optional[float]:
    if "amp=" not in line:
        return None
    try:
        return float(line.split("amp=", 1)[1].split(" ", 1)[0])
    except Exception:  # pragma: no cover
        return None


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


def _warm_up_generation(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 16,
) -> None:
    """One throwaway generation so the env doesn't pay the CUDA-compile cost
    on its very first step (which can add 8-15 s and cause WebSocket timeouts).
    """
    if torch is None:  # pragma: no cover
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
        _trace_prompt = apply_chat_template(tokenizer, SYSTEM_PROMPT, _trace_user)
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
) -> Dict[str, Any]:
    """Run multiple seeds against one task and return aggregated stats."""
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
        )
        out.append(result)
        logger.info(
            "EVAL %-10s n=%d  mean=%.3f +/- %.3f  pass=%.0f%%  amp=%.2f mA",
            task_id, result["n_seeds"], result["mean_score"], result["std_score"],
            result["pass_rate"] * 100, result["mean_amp_ma"],
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 9. Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # constants
    "SYSTEM_PROMPT", "TASK_CONTEXT",
    "INVALID_ACTION_PENALTY", "ENVIRONMENT_ERROR_PENALTY",
    "DEFAULT_REWARD_WEIGHTS",
    # prompt + chat helpers
    "build_user_prompt", "apply_chat_template",
    # action helpers
    "parse_action", "make_action", "heuristic_action",
    # generation + rollout
    "llm_generate", "rollout_episode", "rollout_episode_async",
    "Trajectory",
    # reward
    "compute_reward", "MotorAssistReward",
    "reward_total", "reward_grader", "reward_dense", "reward_format",
    # GRPO glue
    "make_rollout_func", "make_episode_logger",
    # plots
    "plot_training_dashboard", "plot_training_loss",
    "plot_baseline_vs_trained", "compare_trajectories",
    "save_training_plots",
    # eval
    "evaluate_model_on_task", "evaluate_model_suite",
    "eval_with_adapter_disabled",
    "sanity_check_rollout",
]
