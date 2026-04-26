"""
Model Benchmark Runner — MotorAssistEnv
=======================================
Runs Qwen2.5-7B, Qwen2.5-72B, and Mistral-7B-Instruct as zero-shot baseline
agents against the easy / medium / hard tasks via the HF Router API.

Results are written to:
  outputs/benchmark/<model_slug>/  — one JSON per task
  outputs/benchmark/summary.json   — all models × tasks
  outputs/benchmark/summary.csv    — spreadsheet-friendly view

Usage:
  python run_model_benchmark.py

Override env vars:
  BENCHMARK_ENV_URL     environment server URL  (default: HF Space)
  BENCHMARK_TASKS       comma-separated tasks   (default: easy,medium,hard)
  BENCHMARK_SEEDS       comma-separated seeds   (default: 0,1,2)
  BENCHMARK_MAX_STEPS   step cap per task       (default: full episode length)
  HF_TOKEN              HuggingFace router token (required)
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows UTF-8 fix
for _s in ("stdout", "stderr"):
    _stream = getattr(sys, _s, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv
from parkinsons_Motor.tasks import get_task


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ.setdefault(k, v)


_load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_BASE_URL = "https://router.huggingface.co/v1"
ENV_URL     = os.getenv(
    "BENCHMARK_ENV_URL",
    "https://virustechhacks-parkinsons-motor.hf.space",
)

TASKS: List[str] = [
    t.strip()
    for t in os.getenv("BENCHMARK_TASKS", "easy,medium,hard").split(",")
    if t.strip()
]

# Seeds per task (more seeds = more reliable mean, but slower)
SEEDS: List[int] = [
    int(s.strip())
    for s in os.getenv("BENCHMARK_SEEDS", "0,1,2").split(",")
    if s.strip()
]

TEMPERATURE = float(os.getenv("BENCHMARK_TEMPERATURE", "0.2"))
MAX_TOKENS  = int(os.getenv("BENCHMARK_MAX_TOKENS", "256"))
TIMEOUT_S   = float(os.getenv("BENCHMARK_TIMEOUT_S", "120.0"))
MAX_RETRIES = int(os.getenv("BENCHMARK_MAX_RETRIES", "3"))
STEP_SLEEP  = float(os.getenv("BENCHMARK_STEP_SLEEP", "0.3"))

OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark"

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (display_name, hf_model_id, slug_for_filenames)
MODELS: List[tuple[str, str, str]] = [
    ("Qwen2.5-7B",  "Qwen/Qwen2.5-7B-Instruct",               "qwen25_7b"),
    ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct",              "qwen25_72b"),
    ("Mistral-7B",  "mistralai/Mistral-7B-Instruct-v0.3",      "mistral_7b"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert closed-loop Deep Brain Stimulation (DBS) controller for a
Parkinson's patient. Every step you receive the patient's brain biomarkers and
return three control knobs as JSON. No prose. No markdown. JSON only.

Output format (exactly):
{"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}

Ranges:
  dbs_amplitude:   0.0 – 5.0 mA   (typical therapeutic: 0.8 – 2.0 mA)
  dbs_pulse_width: 0.06 – 0.20 ms (default 0.13)
  dbs_frequency:   60 – 185 Hz    (default 130)

Key signals:
  beta_arv        pathological beta oscillation — lower is better
  tremor_arv      tremor severity — lower is better
  force_preserved motor function — keep high
  side_effect_load  cumulative stimulation burden — stay below budget
  gamma_arv       overstimulation marker — if >0.55 reduce amplitude fast
  beta_trend / tremor_trend  positive = worsening, negative = improving
  side_effect_rate  positive = burden still rising

Rules:
  1. If gamma_arv > 0.55 OR side_effect_load > 0.9*budget → reduce amplitude
  2. If beta_arv > 0.60 OR tremor_arv > 0.55 → increase amplitude (>= 1.2 mA)
  3. Once symptoms stable, taper toward lowest effective dose
  4. Avoid jumps larger than 0.3 mA per step
  5. Never output zero amplitude when symptoms are elevated
""").strip()

_TASK_CONTEXT = {
    "easy":   "EASY — Calm Start. Responsive patient, mild early symptoms. Ceiling 1.5 mA. Budget 0.55.",
    "medium": "MEDIUM — Rescue Phase. Active deterioration; rescue without triggering dyskinesia. Ceiling 1.8 mA. Budget 0.60.",
    "hard":   "HARD — Full Episode. Four overlapping crises. Refractory patient. Ceiling 2.4 mA. Budget 0.40.",
}


def _build_user_prompt(step: int, obs: dict, task_id: str, history: List[str]) -> str:
    recent = "\n  ".join(history[-4:]) if history else "(first step)"
    ctx    = _TASK_CONTEXT.get(task_id, f"Task: {task_id}")
    return textwrap.dedent(f"""
        {ctx}
        Step: {step}

        State:
          beta_arv         = {obs.get('beta_arv', 0):.4f}
          tremor_arv       = {obs.get('tremor_arv', 0):.4f}
          force_preserved  = {obs.get('force_preserved', 0):.4f}
          side_effect_load = {obs.get('side_effect_load', 0):.4f}
          gamma_arv        = {obs.get('gamma_arv', 0):.4f}
          beta_trend       = {obs.get('beta_trend', 0):+.4f}
          tremor_trend     = {obs.get('tremor_trend', 0):+.4f}
          side_effect_rate = {obs.get('side_effect_rate', 0):+.4f}
          dbs_entrainment  = {obs.get('dbs_entrainment', 0):.4f}
          stim_washout     = {obs.get('stim_washout', 0):.4f}
          tracking_accuracy= {obs.get('tracking_accuracy', 0):.4f}
          target_output    = {obs.get('target_output', 0):.4f}

        Recent history:
          {recent}

        Respond with JSON only.
    """).strip()


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(
    client: OpenAI,
    model_id: str,
    step: int,
    obs: dict,
    task_id: str,
    history: List[str],
) -> str:
    prompt = _build_user_prompt(step, obs, task_id, history)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT_S,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            delay = min(2 ** attempt, 10)
            print(f"    [WARN] LLM error attempt {attempt+1}/{MAX_RETRIES}: {exc}. Retry in {delay}s", flush=True)
            time.sleep(delay)
    return ""


def _parse_action(text: str) -> Optional[dict]:
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None


def _make_action(d: Optional[dict], target_output: float = 0.0) -> ParkinsonsMotorAction:
    if not d:
        return ParkinsonsMotorAction(
            motor_command=float(max(-1.0, min(1.0, target_output))),
            dbs_amplitude=1.0, dbs_pulse_width=0.13, dbs_frequency=130.0,
        )
    return ParkinsonsMotorAction(
        motor_command=float(max(-1.0, min(1.0, target_output))),
        dbs_amplitude=float(max(0.0,  min(5.0,   d.get("dbs_amplitude",   1.0)))),
        dbs_pulse_width=float(max(0.06, min(0.20, d.get("dbs_pulse_width", 0.13)))),
        dbs_frequency=float(max(60.0, min(185.0,  d.get("dbs_frequency",   130.0)))),
    )


# ── Single rollout ─────────────────────────────────────────────────────────────

async def _run_rollout(
    env: ParkinsonsMotorEnv,
    client: OpenAI,
    model_id: str,
    task_id: str,
    seed: int,
    max_steps: int,
    threshold: float,
) -> Dict[str, Any]:
    """Run one full episode and return all diagnostics."""
    rewards:    List[float] = []
    amplitudes: List[float] = []
    betas:      List[float] = []
    tremors:    List[float] = []
    forces:     List[float] = []
    se_loads:   List[float] = []
    history:    List[str]   = []

    steps_taken  = 0
    score        = 0.0
    success      = False
    grader_ran   = False
    score_details: Dict[str, Any] = {}
    error: Optional[str] = None

    print(f"    rollout task={task_id} seed={seed} max_steps={max_steps}", flush=True)

    try:
        result   = await env.reset(task_id=task_id, seed=seed)
        obs      = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw    = await asyncio.to_thread(_call_llm, client, model_id, step, obs_dict, task_id, history)
            parsed = _parse_action(raw)
            action = _make_action(parsed, target_output=obs_dict.get("target_output", 0.0))
            amplitudes.append(action.dbs_amplitude)

            try:
                result   = await env.step(action)
                obs      = result.observation
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
                reward   = float(result.reward or 0.0)
                done     = result.done
            except Exception as exc:
                error = str(exc)
                print(f"    [ERR] step={step} env.step failed: {exc}", flush=True)
                steps_taken = step
                break

            rewards.append(reward)
            betas.append(obs_dict.get("beta_arv", 0.0))
            tremors.append(obs_dict.get("tremor_arv", 0.0))
            forces.append(obs_dict.get("force_preserved", 0.0))
            se_loads.append(obs_dict.get("side_effect_load", 0.0))
            steps_taken = step

            history.append(
                f"step={step} amp={action.dbs_amplitude:.2f} "
                f"beta={obs_dict.get('beta_arv',0):.3f} "
                f"tremor={obs_dict.get('tremor_arv',0):.3f} "
                f"force={obs_dict.get('force_preserved',0):.3f} "
                f"se={obs_dict.get('side_effect_load',0):.3f} "
                f"reward={reward:+.3f}"
            )

            if STEP_SLEEP > 0 and not done:
                await asyncio.sleep(STEP_SLEEP)

            if done:
                gs = obs_dict.get("grader_score", -1.0)
                if gs >= 0:
                    score = gs
                    grader_ran = True
                score_details = dict(obs_dict.get("grader_components") or {})
                break

        if not grader_ran and rewards:
            score = min(max(sum(rewards) / len(rewards), 0.0), 1.0)

        success = error is None and score >= threshold

    except Exception as outer:
        error = str(outer)
        print(f"    [ERR] rollout failed: {outer}", flush=True)

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"    -> score={score:.4f} pass={success} "
        f"steps={steps_taken} mean_reward={mean_r:.4f}",
        flush=True,
    )

    return {
        "task_id":       task_id,
        "seed":          seed,
        "score":         round(score, 6),
        "success":       success,
        "grader_ran":    grader_ran,
        "steps_taken":   steps_taken,
        "threshold":     threshold,
        "mean_reward":   round(mean_r, 6),
        "mean_amplitude": round(sum(amplitudes)/len(amplitudes), 4) if amplitudes else 0.0,
        "max_amplitude":  round(max(amplitudes), 4) if amplitudes else 0.0,
        "mean_beta":      round(sum(betas)/len(betas), 4) if betas else 0.0,
        "mean_tremor":    round(sum(tremors)/len(tremors), 4) if tremors else 0.0,
        "mean_force":     round(sum(forces)/len(forces), 4) if forces else 0.0,
        "mean_se_load":   round(sum(se_loads)/len(se_loads), 4) if se_loads else 0.0,
        "rewards":        [round(r, 4) for r in rewards],
        "amplitudes":     [round(a, 4) for a in amplitudes],
        "betas":          [round(b, 4) for b in betas],
        "tremors":        [round(t, 4) for t in tremors],
        "forces":         [round(f, 4) for f in forces],
        "se_loads":       [round(s, 4) for s in se_loads],
        "score_details":  {k: round(float(v), 6) for k, v in score_details.items()},
        "error":          error,
    }


# ── Model runner ──────────────────────────────────────────────────────────────

async def _run_model(
    display_name: str,
    model_id: str,
    slug: str,
) -> Dict[str, Any]:
    """Run one model across all tasks and seeds. Returns full result dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"  MODEL: {display_name}  ({model_id})", flush=True)
    print(f"{'='*60}", flush=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env or export it.")

    client = OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN)

    model_dir = OUTPUT_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    all_rollouts: List[Dict[str, Any]] = []

    for task_id in TASKS:
        task      = get_task(task_id)
        max_steps = task.n_steps
        threshold = task.success_threshold

        print(f"\n  Task: {task_id}  (n_steps={max_steps}, threshold={threshold})", flush=True)

        task_rollouts: List[Dict[str, Any]] = []

        for seed in SEEDS:
            env = ParkinsonsMotorEnv(base_url=ENV_URL)
            await env.__aenter__()
            try:
                rollout = await _run_rollout(
                    env, client, model_id, task_id, seed, max_steps, threshold
                )
                task_rollouts.append(rollout)
                all_rollouts.append(rollout)
            finally:
                await env.__aexit__(None, None, None)

            await asyncio.sleep(2.0)  # brief pause between seeds

        # Save per-task JSON
        task_path = model_dir / f"{task_id}.json"
        task_path.write_text(json.dumps(task_rollouts, indent=2), encoding="utf-8")
        print(f"  Saved {task_path.name}", flush=True)

    # Aggregate per task
    per_task: List[Dict[str, Any]] = []
    for task_id in TASKS:
        runs   = [r for r in all_rollouts if r["task_id"] == task_id]
        scores = [r["score"] for r in runs]
        n      = len(scores)
        mean_s = sum(scores) / n
        std_s  = (sum((s - mean_s) ** 2 for s in scores) / n) ** 0.5
        passes = sum(1 for r in runs if r["success"])
        per_task.append({
            "task_id":       task_id,
            "n_seeds":       n,
            "score_mean":    round(mean_s, 6),
            "score_std":     round(std_s, 6),
            "score_min":     round(min(scores), 6),
            "score_max":     round(max(scores), 6),
            "pass_rate":     round(passes / n, 4),
            "passes":        passes,
            "threshold":     get_task(task_id).success_threshold,
            "mean_reward":   round(sum(r["mean_reward"] for r in runs) / n, 6),
            "mean_beta":     round(sum(r["mean_beta"] for r in runs) / n, 4),
            "mean_tremor":   round(sum(r["mean_tremor"] for r in runs) / n, 4),
            "mean_force":    round(sum(r["mean_force"] for r in runs) / n, 4),
            "mean_se_load":  round(sum(r["mean_se_load"] for r in runs) / n, 4),
            "mean_amplitude": round(sum(r["mean_amplitude"] for r in runs) / n, 4),
            "rollouts":      runs,
        })

    result = {
        "model_display_name": display_name,
        "model_id":           model_id,
        "slug":               slug,
        "tasks":              TASKS,
        "seeds":              SEEDS,
        "env_url":            ENV_URL,
        "per_task":           per_task,
        "overall_mean_score": round(
            sum(pt["score_mean"] for pt in per_task) / len(per_task), 6
        ),
    }

    # Save full model JSON
    model_json = model_dir / "model_summary.json"
    model_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Full model summary -> {model_json}", flush=True)

    return result


# ── Summary writers ───────────────────────────────────────────────────────────

def _write_summary(all_model_results: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(json.dumps(all_model_results, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON: {json_path}", flush=True)

    # CSV — one row per (model, task)
    csv_path = OUTPUT_DIR / "summary.csv"
    rows: List[Dict[str, Any]] = []
    for mr in all_model_results:
        for pt in mr["per_task"]:
            rows.append({
                "model":          mr["model_display_name"],
                "model_id":       mr["model_id"],
                "task":           pt["task_id"],
                "score_mean":     pt["score_mean"],
                "score_std":      pt["score_std"],
                "score_min":      pt["score_min"],
                "score_max":      pt["score_max"],
                "pass_rate":      pt["pass_rate"],
                "passes":         pt["passes"],
                "n_seeds":        pt["n_seeds"],
                "threshold":      pt["threshold"],
                "mean_reward":    pt["mean_reward"],
                "mean_beta":      pt["mean_beta"],
                "mean_tremor":    pt["mean_tremor"],
                "mean_force":     pt["mean_force"],
                "mean_se_load":   pt["mean_se_load"],
                "mean_amplitude": pt["mean_amplitude"],
            })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved summary CSV:  {csv_path}", flush=True)

    # Console table
    print(f"\n{'='*70}", flush=True)
    print(f"{'BENCHMARK SUMMARY':^70}", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'Model':<18} {'Task':<10} {'Score':>8} {'±Std':>7} {'Pass%':>7} {'Threshold':>10}"
    print(header, flush=True)
    print("-" * 70, flush=True)
    for mr in all_model_results:
        for pt in mr["per_task"]:
            print(
                f"{mr['model_display_name']:<18} {pt['task_id']:<10} "
                f"{pt['score_mean']:>8.4f} {pt['score_std']:>7.4f} "
                f"{pt['pass_rate']*100:>6.0f}%  {pt['threshold']:>10.2f}",
                flush=True,
            )
        print(
            f"  {'-> overall mean':>26} {mr['overall_mean_score']:>8.4f}",
            flush=True,
        )
        print("-" * 70, flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("\nMotorAssistEnv — Multi-Model Baseline Benchmark", flush=True)
    print(f"  Env URL : {ENV_URL}", flush=True)
    print(f"  Models  : {[m[0] for m in MODELS]}", flush=True)
    print(f"  Tasks   : {TASKS}", flush=True)
    print(f"  Seeds   : {SEEDS}", flush=True)
    print(f"  Outputs : {OUTPUT_DIR}", flush=True)

    if not HF_TOKEN:
        print("\n[ERROR] HF_TOKEN not set. Add it to .env file.", flush=True)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for display_name, model_id, slug in MODELS:
        result = await _run_model(display_name, model_id, slug)
        all_results.append(result)
        # Brief pause between models
        await asyncio.sleep(5.0)

    _write_summary(all_results)
    print("\nBenchmark complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
