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
    from parkinsons_Motor.training import llm_eval

    if train.torch is not None:
        print("NOTE: torch installed — _warm_up_generation not validated here (needs real model).")
        return
    # _warm_up_generation now lives in parkinsons_Motor.training.llm_eval
    # (re-exported from train via sanity_check_rollout). It must still be a
    # no-op when torch isn't available.
    llm_eval._warm_up_generation(None, None)  # type: ignore[arg-type]
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


def _test_apply_chat_template_thinking_off() -> None:
    """Default ``enable_thinking=False`` should not crash; should pass kwarg
    when supported and silently fall back when not."""
    import parkinsons_Motor.train as train

    class _TokSupports:
        seen: dict = {}
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
            _TokSupports.seen = dict(kw)
            return "TEMPL:" + str(msgs)

    out = train.apply_chat_template(_TokSupports(), "sys", "user")
    assert "TEMPL:" in out, out
    assert _TokSupports.seen.get("enable_thinking") is False, _TokSupports.seen
    print("OK: apply_chat_template defaults to enable_thinking=False")

    out = train.apply_chat_template(_TokSupports(), "sys", "user", enable_thinking=True)
    assert _TokSupports.seen.get("enable_thinking") is True, _TokSupports.seen
    print("OK: apply_chat_template threads enable_thinking=True for demo")

    class _TokRejects:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            return "FALLBACK:" + str(msgs)

    out = train.apply_chat_template(_TokRejects(), "sys", "user")
    assert "FALLBACK:" in out, out
    print("OK: apply_chat_template falls back when tokenizer rejects enable_thinking")


def _test_training_package_imports() -> None:
    """The new training/ package mirrors mhtruong1031's bio-experiment layout.

    Just importing the public surface and round-tripping a tiny trajectory
    is enough to catch typos, wrong types, broken paths, etc.
    """
    from parkinsons_Motor.training import (
        DBSTrajectory,
        DBSTrajectoryDataset,
        EvaluationSuite,
        MetricResult,
    )
    from parkinsons_Motor.training import (
        CLINICAL_TARGETS,
        compare_to_literature,
    )

    traj = DBSTrajectory(episode_id="test-ep", task_id="easy", seed=0, policy="constant")
    for i in range(5):
        traj.add_step(
            action={"dbs_amplitude": 1.5, "dbs_pulse_width": 0.13, "dbs_frequency": 130.0},
            observation={
                "beta_arv": 0.6 - 0.05 * i,
                "tremor_arv": 0.5 - 0.05 * i,
                "side_effect_load": 0.3,
                "grader_score": 0.0 if i < 4 else 0.78,
                "episode_success": (i == 4),
                "grader_components": {"beta": 0.4, "tremor": 0.3, "safety": 0.08},
            },
            reward=0.2 * (i + 1),
            done=(i == 4),
            parsed=True,
        )

    assert traj.n_steps == 5, traj.n_steps
    assert abs(traj.total_reward - sum(0.2 * (i + 1) for i in range(5))) < 1e-9
    assert traj.success is True
    assert traj.grader_score == 0.78
    assert traj.grader_components.get("beta") == 0.4

    ds = DBSTrajectoryDataset([traj])
    summary = ds.summary()
    assert summary["n"] == 1, summary
    assert summary["success_rate"] == 1.0, summary

    online = EvaluationSuite.online_metrics([traj])
    assert any(m.name == "mean_grader_score" and m.value == 0.78 for m in online), online

    bench = EvaluationSuite.benchmark_metrics(ds)
    bench_names = {m.name for m in bench}
    assert {"final_beta_suppression", "clinical_amplitude_window",
            "format_validity_rate"} <= bench_names, bench_names

    clin = compare_to_literature(ds)
    clin_names = {m.name for m in clin}
    assert "median_amplitude" in clin_names and "mean_frequency" in clin_names, clin_names
    assert all(isinstance(m, MetricResult) for m in clin)
    assert any(t.unit == "mA" for t in CLINICAL_TARGETS)

    print(f"OK: parkinsons_Motor.training imports + 1-traj round-trip "
          f"(online={len(online)}, bench={len(bench)}, clinical={len(clin)})")


def _test_train_reexports_plots_and_eval() -> None:
    """train.py must re-export every plot_* / eval_* / sanity_check_rollout
    symbol so the existing notebook import block keeps working unchanged.

    The functions actually live in parkinsons_Motor.training.{plots,llm_eval}
    after the refactor, but `from parkinsons_Motor.train import …` must still
    succeed for backwards compatibility.
    """
    from parkinsons_Motor.train import (
        compare_trajectories,
        eval_with_adapter_disabled,
        evaluate_model_on_task,
        evaluate_model_suite,
        plot_baseline_vs_trained,
        plot_training_dashboard,
        plot_training_loss,
        sanity_check_rollout,
        save_training_plots,
    )
    from parkinsons_Motor.training import llm_eval as _ll
    from parkinsons_Motor.training import plots as _pl

    # Identity check — re-exports must point to the SAME function objects so a
    # monkeypatch on `train.sanity_check_rollout` can't get out of sync with
    # `training.llm_eval.sanity_check_rollout` and silently diverge.
    assert sanity_check_rollout is _ll.sanity_check_rollout
    assert evaluate_model_on_task is _ll.evaluate_model_on_task
    assert evaluate_model_suite is _ll.evaluate_model_suite
    assert eval_with_adapter_disabled is _ll.eval_with_adapter_disabled
    assert plot_training_dashboard is _pl.plot_training_dashboard
    assert plot_training_loss is _pl.plot_training_loss
    assert plot_baseline_vs_trained is _pl.plot_baseline_vs_trained
    assert compare_trajectories is _pl.compare_trajectories
    assert save_training_plots is _pl.save_training_plots
    print("OK: train.py re-exports plot_* / eval_* / sanity_check_rollout from training/")


def main() -> int:
    print("--- dry_run_rollout_eventloop ---")
    _test_parse_action_last_json()
    _test_parse_action_strips_code_fence()
    _test_parse_action_handles_only_thinking()
    _test_apply_chat_template_thinking_off()
    _test_warm_up_skips_without_torch()
    _test_training_package_imports()
    _test_train_reexports_plots_and_eval()
    asyncio.run(_main_async())
    print("--- all dry-run checks passed ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
