#!/usr/bin/env python3
"""
Local dry-run: prove ``rollout_episode_async`` keeps the asyncio loop alive
during slow sync ``llm_generate`` (via ``asyncio.to_thread``), matching the
WebSocket keepalive fix for HF Spaces.

Requires (once per machine):

  pip install "openenv-core[core]>=0.2.2"

Torch is NOT required — we mock the model, tokenizer, and env.

Run from repo root:

  python scripts/dry_run_rollout_eventloop.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fake_observation(*, done: bool) -> Any:
    from parkinsons_Motor.models import ParkinsonsMotorObservation

    return ParkinsonsMotorObservation(
        beta_arv=0.4,
        tremor_arv=0.35,
        grader_score=0.75 if done else -1.0,
        episode_success=done,
    )


class _FakeStepResult:
    def __init__(self, observation: Any, reward: float, done: bool) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done


class _FakeParkinsonsMotorEnv:
    """Minimal async env matching the calls ``rollout_episode_async`` makes."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self._n = 0

    async def __aenter__(self) -> "_FakeParkinsonsMotorEnv":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def reset(self, **kwargs: Any) -> _FakeStepResult:
        self._n = 0
        return _FakeStepResult(_fake_observation(done=False), reward=0.0, done=False)

    async def step(self, action: Any) -> _FakeStepResult:
        self._n += 1
        done = self._n >= 2
        return _FakeStepResult(_fake_observation(done=done), reward=0.25 * self._n, done=done)


def _slow_llm_generate(*_args: Any, **_kwargs: Any) -> tuple[str, List[int], List[int]]:
    """Blocking sync work (simulates ``model.generate``). Runs inside ``to_thread``."""
    time.sleep(1.2)
    text = (
        "<think>short</think>\n"
        '{"dbs_amplitude": 1.2, "dbs_pulse_width": 0.13, "dbs_frequency": 130}'
    )
    return text, [1, 2, 3], [4, 5]


def _test_warm_up_skips_without_torch() -> None:
    import parkinsons_Motor.train as train

    if train.torch is not None:
        print("NOTE: torch installed — _warm_up_generation not validated here (needs real model).")
        return
    train._warm_up_generation(None, None)  # type: ignore[arg-type]
    print("OK: _warm_up_generation no-ops when torch is absent")


async def _main_async() -> None:
    import parkinsons_Motor.train as train

    train.ParkinsonsMotorEnv = _FakeParkinsonsMotorEnv  # type: ignore[misc, assignment]
    train.llm_generate = _slow_llm_generate  # type: ignore[assignment]

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
            return "USER:" + str(msgs)

    ticks: list[int] = [0]

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(0.05)
                ticks[0] += 1
        except asyncio.CancelledError:
            return

    hb = asyncio.create_task(_heartbeat())
    t0 = time.perf_counter()
    try:
        traj = await train.rollout_episode_async(
            model=None,
            tokenizer=_Tok(),
            env_url="http://fake",
            task_id="easy",
            seed=0,
            max_turns=5,
            max_new_tokens=256,
            max_prompt_length=2048,
            fallback_to_heuristic_on_invalid=False,
        )
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass

    elapsed = time.perf_counter() - t0

    assert traj.env_error is None, traj.env_error
    assert traj.n_steps == 2, f"expected 2 steps, got {traj.n_steps}"
    assert all(traj.parsed), traj.parsed
    assert ticks[0] >= 20, (
        f"event loop starved: only {ticks[0]} heartbeat ticks in {elapsed:.1f}s — "
        "asyncio.to_thread may be missing around llm_generate"
    )
    print(f"OK: event-loop responsiveness — {ticks[0]} heartbeats during ~{elapsed:.1f}s rollout")
    print(f"OK: trajectory — n_steps={traj.n_steps}, rewards={traj.rewards}, parsed={traj.parsed}")


def _test_parse_action_last_json() -> None:
    import parkinsons_Motor.train as train

    text = (
        "<think>{\"dbs_amplitude\": 9.0}</think>\n"
        '{"dbs_amplitude": 1.2, "dbs_pulse_width": 0.13, "dbs_frequency": 130}'
    )
    d = train.parse_action(text)
    assert d is not None and d["dbs_amplitude"] == 1.2, d
    print("OK: parse_action prefers last JSON outside thinking")


def _test_parse_action_strips_code_fence() -> None:
    import parkinsons_Motor.train as train

    text = (
        "Here is the action:\n"
        "```json\n"
        '{"dbs_amplitude": 2.5, "dbs_pulse_width": 0.13, "dbs_frequency": 130}\n'
        "```"
    )
    d = train.parse_action(text)
    assert d is not None and d["dbs_amplitude"] == 2.5, d
    print("OK: parse_action strips ```json``` markdown fence")


def _test_parse_action_handles_only_thinking() -> None:
    import parkinsons_Motor.train as train

    # All content is inside the thinking block — fall back to scanning the
    # full text rather than returning None on an empty answer.
    text = (
        "<think>I need to set "
        '{"dbs_amplitude": 1.5, "dbs_pulse_width": 0.13, "dbs_frequency": 130}'
        " for safety</think>"
    )
    d = train.parse_action(text)
    assert d is not None and d["dbs_amplitude"] == 1.5, d
    print("OK: parse_action falls back to thinking content when answer is empty")


def main() -> int:
    print("--- dry_run_rollout_eventloop ---")
    _test_parse_action_last_json()
    _test_parse_action_strips_code_fence()
    _test_parse_action_handles_only_thinking()
    _test_warm_up_skips_without_torch()
    asyncio.run(_main_async())
    print("--- all dry-run checks passed ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
