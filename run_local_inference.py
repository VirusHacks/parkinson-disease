"""
Run a live LLM agent against the local Parkinson's DBS environment.

Connects to the locally-running FastAPI server (localhost:8000), runs the
LLM through all 3 tasks, and emits the required OpenEnv stdout format.

Usage:
    uv run --project parkinsons_Motor python run_local_inference.py

Reads HF_TOKEN, API_BASE_URL, MODEL_NAME from .env or environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

# path setup
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv  # noqa: E402


# ── env config ────────────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(__file__).resolve().parent / path
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


_load_dotenv(".env")

LOCAL_SERVER_URL = "http://localhost:8000"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "parkinsons_Motor"

TASKS = ["beta_suppression", "tremor_correction", "full_episode"]
MAX_STEPS = {"beta_suppression": 20, "tremor_correction": 50, "full_episode": 100}
SUCCESS_THRESHOLD = {"beta_suppression": 0.60, "tremor_correction": 0.55, "full_episode": 0.50}
TEMPERATURE = 0.3
MAX_TOKENS  = 200


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an autonomous Deep Brain Stimulation (DBS) programmer for a Parkinson's patient.
At each step you see their brain state and must output a JSON action.

Brain state fields:
  beta_arv         — STN beta oscillation (0=suppressed, 1=peak Parkinson's)
  tremor_arv       — Tremor amplitude (0=none, 1=maximum observed)
  force_preserved  — Fraction of healthy muscle force (1.0=fully healthy, 0.0=none)
  side_effect_load — Cumulative DBS side-effect load (keep below task budget)
  dbs_entrainment  — Fraction of cortical axons entrained by DBS last step
  sim_time_s       — Simulation time in seconds

Goal: maximise force_preserved every step by tuning DBS to suppress beta without draining the side effect budget.

Output ONLY valid JSON — no explanation, no markdown:
{
  "motor_command":   <float -1.0 to 1.0>,
  "dbs_amplitude":   <float 0.0 to 3.0 mA>,
  "dbs_pulse_width": <float 0.06 to 0.20 ms>
}

Clinical rules:
- START LOW: If tremor is still extremely low (e.g., sim_time < 10.5), use very low amplitude (0.0 - 0.5) to explicitly preserve the side-effect budget!
- beta_arv > 0.5 → raise dbs_amplitude
- side_effect_load > 0.4 → aggressively lower dbs_amplitude  
- force_preserved < 0.3 → critical — use maximum safe DBS
- Good starting pulse width: 0.13 ms
- motor_command near 0.4 is a reasonable default
""").strip()


def _build_user_prompt(step: int, obs: dict, task_id: str, history: list) -> str:
    recent = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
    Task: {task_id}  |  Step {step}

    Brain state:
      beta_arv:         {obs.get('beta_arv', 0):.3f}
      tremor_arv:       {obs.get('tremor_arv', 0):.3f}
      force_preserved:  {obs.get('force_preserved', 0):.3f}
      side_effect_load: {obs.get('side_effect_load', 0):.3f}
      dbs_entrainment:  {obs.get('dbs_entrainment', 0):.3f}
      disease_severity: {obs.get('disease_severity', 0):.3f}
      sim_time_s:       {obs.get('sim_time_s', 0):.2f}

    Recent history:
    {recent}

    Output your JSON action now.
    """).strip()


def _call_llm(client: OpenAI, step: int, obs: dict, task_id: str, history: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(step, obs, task_id, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            delay = 2 ** attempt
            print(f"[DEBUG] LLM call failed (attempt {attempt+1}/{max_retries}): {exc}. Retrying in {delay}s...", flush=True)
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


def _make_action(d: Optional[dict]) -> ParkinsonsMotorAction:
    if not d:
        return ParkinsonsMotorAction(motor_command=0.3, dbs_amplitude=0.8, dbs_pulse_width=0.13)
    return ParkinsonsMotorAction(
        motor_command   = float(max(-1.0, min(1.0,  d.get("motor_command",   0.3)))),
        dbs_amplitude   = float(max(0.0,  min(5.0,  d.get("dbs_amplitude",   0.8)))),
        dbs_pulse_width = float(max(0.06, min(0.20, d.get("dbs_pulse_width", 0.13)))),
    )


# ── logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error or "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)


# ── task runner ───────────────────────────────────────────────────────────────

async def run_task(env, client: OpenAI, task_id: str) -> tuple[float, bool]:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    max_steps = MAX_STEPS[task_id]
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None

    try:
        result = await env.reset()
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw = _call_llm(client, step, obs_dict, task_id, history)
            parsed = _parse_action(raw)
            action = _make_action(parsed)
            action_str = json.dumps({
                "motor_command":   round(action.motor_command, 3),
                "dbs_amplitude":   round(action.dbs_amplitude, 3),
                "dbs_pulse_width": round(action.dbs_pulse_width, 3),
            })

            err_this_step: Optional[str] = None
            try:
                result = await env.step(action)
                obs = result.observation
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
                reward = float(result.reward or 0.0)
                done = result.done
            except Exception as exc:
                reward, done = 0.0, True
                err_this_step = final_error = str(exc)
                log_step(step, action_str, reward, done, err_this_step)
                steps_taken = step
                break

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, err_this_step)

            history.append(
                f"step={step} amp={action.dbs_amplitude:.2f} pw={action.dbs_pulse_width:.3f}"
                f" → beta={obs_dict.get('beta_arv',0):.3f} force={obs_dict.get('force_preserved',0):.3f}"
                f" se={obs_dict.get('side_effect_load',0):.3f} reward={reward:+.3f}"
            )

            if done:
                gs = obs_dict.get("grader_score", -1.0)
                if gs >= 0:
                    score = gs
                break

        if score <= 0 and rewards:
            score = min(max(sum(rewards) / len(rewards), 0.0), 1.0)

        success = final_error is None and score >= SUCCESS_THRESHOLD[task_id]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN / API_KEY in .env")

    print(f"\n{'='*60}", flush=True)
    print(f"Parkinson's DBS Environment - LLM Baseline Run", flush=True)
    print(f"Model  : {MODEL_NAME}", flush=True)
    print(f"Server : {LOCAL_SERVER_URL}", flush=True)
    print(f"Tasks  : {TASKS}", flush=True)
    print(f"{'='*60}\n", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to local server via URL (no Docker needed)
    env = ParkinsonsMotorEnv(base_url=LOCAL_SERVER_URL)
    await env.__aenter__()

    all_scores: dict[str, float] = {}
    all_success: dict[str, bool] = {}

    try:
        for task_id in TASKS:
            print(f"\n{'-'*50}", flush=True)
            score, success = await run_task(env, client, task_id)
            all_scores[task_id] = score
            all_success[task_id] = success
            print(f"  -> {task_id}: score={score:.4f} success={success}", flush=True)
    finally:
        await env.__aexit__(None, None, None)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    for tid in TASKS:
        s = all_scores.get(tid, 0.0)
        ok = all_success.get(tid, False)
        bar = "#" * int(s * 20) + "-" * (20 - int(s * 20))
        result_flag = "PASS" if ok else "FAIL"
        print(f"  {tid:<22} [{bar}] {s:.4f}  {result_flag}", flush=True)
    mean = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"\n  Mean score: {mean:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
