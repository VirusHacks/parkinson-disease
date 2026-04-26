#!/usr/bin/env python
"""End-to-end dry-run of the replay-based GRPO training surface.

Verifies — without a GPU / transformers / TRL — that:

  1. ``LocalEnvFactory`` builds a fresh in-process env.
  2. ``collect_prompt_dataset`` rolls the heuristic policy and emits
     well-formed dataset rows with every column the reward function needs.
  3. ``make_replay_reward_fn`` returns a callable with the standard TRL
     ``reward_funcs`` signature.
  4. The reward function:
       * gives a *strictly higher* reward to a valid-JSON completion than to
         a malformed one (the whole point of the format bonus),
       * is deterministic — same inputs ⇒ same scalar,
       * never crashes when it sees ill-formed completions.
  5. The module's public API is exposed via ``parkinsons_Motor.training``.
  6. ``parkinsons_Motor.train.make_rollout_func`` still imports (the legacy
     surface stays functional for anyone holding a reference to it).

Run this BEFORE touching the notebook so we know the module works in
isolation. If anything fails here, the Colab run won't get any further.

Usage:
    python scripts/dry_run_replay_grpo.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, List

# Make ``parkinsons_Motor`` importable when this script is run from anywhere.
# The repo root is the parent of the ``scripts/`` folder this file lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [fail] {msg}")


# ---------------------------------------------------------------------------
# Stub tokenizer: mimics tokenizer.apply_chat_template tightly enough that
# ``apply_chat_template(...)`` in train.py picks the right control path. We
# don't tokenize anything — the dataset rows just need a string ``prompt``.
# ---------------------------------------------------------------------------

class StubTokenizer:
    """Minimal stand-in that supports ``apply_chat_template(messages, ...)``."""

    eos_token = "<|im_end|>"
    pad_token = "<|im_end|>"

    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **_: Any,
    ) -> str:
        # Render a Qwen-3 style chat template. The actual content of the token
        # markers doesn't matter for this dry-run; we just need a deterministic,
        # non-empty string.
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not enable_thinking:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _test_local_env_factory() -> None:
    print("== LocalEnvFactory ==")
    from parkinsons_Motor.training.replay_grpo import LocalEnvFactory

    fac = LocalEnvFactory()
    e1 = fac()
    e2 = fac()
    assert e1 is not e2, "Factory must return a fresh env each call"
    o = e1.reset(task_id="easy", seed=7)
    assert hasattr(o, "reward") and hasattr(o, "done"), \
        "Observation must expose reward + done fields"
    _ok(f"factory builds fresh envs; reset(easy, seed=7) → reward={o.reward:.3f} done={o.done}")

    # The factory should be picklable — TRL may cross-process serialise it.
    import pickle
    blob = pickle.dumps(fac)
    fac2 = pickle.loads(blob)
    assert isinstance(fac2(), type(e1)), "Pickled factory must round-trip"
    _ok("LocalEnvFactory round-trips through pickle")


def _test_collect_prompt_dataset() -> List[dict]:
    print("== collect_prompt_dataset ==")
    from parkinsons_Motor.training.replay_grpo import (
        LocalEnvFactory,
        collect_prompt_dataset,
    )

    rows = collect_prompt_dataset(
        env_factory          = LocalEnvFactory(),
        tasks                = ("easy", "medium"),
        episodes_per_task    = 1,
        max_steps_per_episode= 4,
        seed                 = 123,
        tokenizer            = StubTokenizer(),
        enable_thinking      = False,
    )
    assert rows, "Expected at least one row"
    _ok(f"collected {len(rows)} rows from 2 tasks × 1 episode × ≤4 steps")

    expected_cols = {"prompt", "task_id", "seed", "step_idx",
                     "history_actions", "ref_amplitude"}
    have = set(rows[0])
    missing = expected_cols - have
    assert not missing, f"missing dataset columns: {missing}"
    _ok(f"every row has the expected columns: {sorted(have)}")

    # history_actions must be a JSON-decodable list of dicts (one entry per
    # heuristic action that came BEFORE this step).
    for r in rows:
        history = json.loads(r["history_actions"])
        assert isinstance(history, list), "history_actions must decode to a list"
        assert len(history) == r["step_idx"], (
            f"step_idx={r['step_idx']} but history has {len(history)} entries — "
            "they must match for replay to recreate the right state."
        )
    _ok("history_actions is JSON-decodable and length matches step_idx for every row")

    sample_prompt = rows[0]["prompt"]
    assert "system" in sample_prompt and "user" in sample_prompt and "assistant" in sample_prompt, \
        "prompt must contain the rendered system+user+assistant chat sections"
    assert "<think>" in sample_prompt, \
        "with enable_thinking=False, Qwen3 template injects an empty <think></think>"
    _ok(f"prompt[0] is a chat-templated string ({len(sample_prompt)} chars); "
        f"first row task={rows[0]['task_id']} seed={rows[0]['seed']} step_idx={rows[0]['step_idx']}")

    return rows


def _test_replay_reward_fn(rows: List[dict]) -> None:
    print("== make_replay_reward_fn ==")
    from parkinsons_Motor.training.replay_grpo import (
        LocalEnvFactory,
        make_replay_reward_fn,
    )

    reward_fn = make_replay_reward_fn(LocalEnvFactory(), log_every=1000)
    assert callable(reward_fn), "reward_fn must be callable"
    assert reward_fn.__name__ == "replay_env_reward", \
        f"expected reward_fn.__name__ == 'replay_env_reward', got {reward_fn.__name__!r}"
    _ok("reward_fn is callable, named 'replay_env_reward'")

    # Use the FIRST dataset row for replay; build two completions:
    #   * one valid JSON action (the heuristic's reference dose),
    #   * one obvious garbage string.
    row = rows[0]
    valid_completion = (
        '{"dbs_amplitude": '
        f'{row["ref_amplitude"]:.2f}'
        ', "dbs_pulse_width": 0.13, "dbs_frequency": 130.0}'
    )
    garbage_completion = "I'm not going to give you JSON, deal with it."

    rewards = reward_fn(
        completions=[valid_completion, garbage_completion],
        task_id=[row["task_id"], row["task_id"]],
        seed=[row["seed"], row["seed"]],
        history_actions=[row["history_actions"], row["history_actions"]],
        step_idx=[row["step_idx"], row["step_idx"]],
    )
    assert len(rewards) == 2, f"expected 2 rewards, got {len(rewards)}"
    r_valid, r_invalid = rewards
    _ok(f"rewards (valid, garbage) = ({r_valid:+.3f}, {r_invalid:+.3f})")

    assert r_valid > r_invalid, (
        f"valid completion ({r_valid:+.3f}) must score strictly higher than "
        f"garbage ({r_invalid:+.3f}) — otherwise the format bonus isn't biting."
    )
    _ok("valid JSON outranks garbage completion (format bonus is biting)")

    # Determinism — same inputs must give the same scalar twice.
    rewards_again = reward_fn(
        completions=[valid_completion, garbage_completion],
        task_id=[row["task_id"], row["task_id"]],
        seed=[row["seed"], row["seed"]],
        history_actions=[row["history_actions"], row["history_actions"]],
        step_idx=[row["step_idx"], row["step_idx"]],
    )
    assert rewards == rewards_again, (
        f"reward_fn is non-deterministic: {rewards} vs {rewards_again}"
    )
    _ok("reward_fn is deterministic across repeated calls")

    # The list-of-chat-messages format that some TRL versions use.
    chat_completion = [{"role": "assistant", "content": valid_completion}]
    rewards_chat = reward_fn(
        completions=[chat_completion],
        task_id=[row["task_id"]],
        seed=[row["seed"]],
        history_actions=[row["history_actions"]],
        step_idx=[row["step_idx"]],
    )
    assert abs(rewards_chat[0] - r_valid) < 1e-6, (
        "list-of-messages and string completions must produce identical rewards"
    )
    _ok("reward_fn handles both string and list-of-messages completion shapes")

    # Missing mandatory dataset kwargs must fail loudly — a silent zero-reward
    # would re-introduce the dead-policy collapse we're trying to avoid.
    try:
        reward_fn(completions=[valid_completion])
    except ValueError as exc:
        _ok(f"missing dataset kwargs raises ValueError as expected: {exc.__class__.__name__}")
    else:  # pragma: no cover
        raise AssertionError(
            "reward_fn silently accepted a call with no task_id/seed/history_actions; "
            "this would silently zero out rewards and re-introduce dead-policy collapse."
        )


def _test_package_reexport() -> None:
    print("== package re-exports ==")
    import parkinsons_Motor.training as training_pkg

    for name in (
        "LocalEnvFactory",
        "collect_prompt_dataset",
        "make_replay_reward_fn",
        "DEFAULT_REPLAY_REWARD_WEIGHTS",
    ):
        assert hasattr(training_pkg, name), f"parkinsons_Motor.training is missing {name!r}"
    _ok("parkinsons_Motor.training re-exports the replay_grpo public surface")

    # Legacy back-compat: train.make_rollout_func should still import (we
    # haven't deleted it; the notebook just stops using it).
    from parkinsons_Motor.train import make_rollout_func, make_episode_logger
    assert callable(make_rollout_func) and callable(make_episode_logger)
    _ok("legacy train.make_rollout_func / make_episode_logger still importable")


def main() -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    print(textwrap.dedent("""
    ╭─────────────────────────────────────────────────────────────────────╮
    │  dry-run: replay-based GRPO surface (parkinsons_Motor/training/)    │
    │                                                                     │
    │  exercises LocalEnvFactory + collect_prompt_dataset +               │
    │  make_replay_reward_fn end-to-end with no GPU / transformers / TRL  │
    ╰─────────────────────────────────────────────────────────────────────╯
    """).strip())
    print()

    try:
        _test_local_env_factory()
        rows = _test_collect_prompt_dataset()
        _test_replay_reward_fn(rows)
        _test_package_reexport()
    except AssertionError as exc:
        _fail(f"ASSERTION: {exc}")
        traceback.print_exc()
        return 1
    except Exception as exc:  # pragma: no cover
        _fail(f"UNCAUGHT: {exc!r}")
        traceback.print_exc()
        return 2

    print()
    print("ALL DRY-RUN CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
