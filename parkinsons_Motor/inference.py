"""
Inference Script — Parkinson's Motor (DBS) Environment
=======================================================
Runs an LLM agent against all three tasks and emits the required stdout:

  [START] task=<task_id> env=parkinsons_Motor model=<model_name>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,...>

Environment variables required:
  HF_TOKEN (or API_KEY / OPENAI_API_KEY) — API key
  API_BASE_URL                            — inference endpoint
  MODEL_NAME                              — model identifier
  LOCAL_IMAGE_NAME                        — Docker image for the environment

Runtime: < 20 min on 2 vCPU / 8 GB RAM (all three tasks combined).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

# ── path setup ────────────────────────────────────────────────────────────────
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
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


_load_dotenv()

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "parkinsons-motor:latest"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "parkinsons_Motor"

# Task execution config
MAX_STEPS_PER_TASK = {
    "beta_suppression":  20,   # Easy  — 20-step window
    "tremor_correction": 50,   # Medium — 50-step window
    "full_episode":      100,  # Hard   — full 100 steps
}
TEMPERATURE = 0.3
MAX_TOKENS  = 200

SUCCESS_SCORE_THRESHOLD = {
    "beta_suppression":  0.60,
    "tremor_correction": 0.55,
    "full_episode":      0.50,
}

TASKS = list(MAX_STEPS_PER_TASK.keys())  # run in order: easy → medium → hard


# ── LLM prompts ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an autonomous Deep Brain Stimulation (DBS) programmer for a Parkinson's patient.

At each step you receive the patient's current brain state and must output a JSON action.

The brain state contains:
  beta_arv        — STN beta oscillation (0=suppressed, 1=peak Parkinson's)
  tremor_arv      — Tremor amplitude (0=none, 1=maximum observed)
  force_preserved — Fraction of healthy muscle force (1.0=fully healthy, 0.0=no motor output)
  side_effect_load — Cumulative DBS side-effect load (keep below task budget)
  dbs_entrainment — Fraction of cortical axons entrained by current DBS (0–1)
  sim_time_s      — Current simulation time in seconds

Your goal: preserve the patient's motor function (force_preserved as high as possible)
by tuning DBS to suppress the beta oscillation, without exceeding the side-effect budget.

Output ONLY valid JSON with these fields (no extra text):
{
  "motor_command":  <float, -1.0 to 1.0>,
  "dbs_amplitude":  <float, 0.0 to 3.0, in mA>,
  "dbs_pulse_width": <float, 0.06 to 0.20, in ms>
}

Clinical guidance:
- beta_arv > 0.5 → increase dbs_amplitude (more suppression needed)
- side_effect_load > 0.4 → reduce dbs_amplitude (side-effect budget at risk)
- force_preserved < 0.3 → this is a critical failure state; use maximum DBS
- dbs_pulse_width of 0.13 ms is a good balanced default
- motor_command should match the target_output (around ±0.5) if available
""").strip()


def _build_user_prompt(step: int, obs_dict: dict, task_id: str, step_history: list) -> str:
    recent = "\n".join(step_history[-5:]) if step_history else "None"
    return textwrap.dedent(f"""
    Task: {task_id}
    Step: {step}

    Current brain state:
      beta_arv:        {obs_dict.get('beta_arv', 0):.3f}
      tremor_arv:      {obs_dict.get('tremor_arv', 0):.3f}
      force_preserved: {obs_dict.get('force_preserved', 0):.3f}
      side_effect_load:{obs_dict.get('side_effect_load', 0):.3f}
      dbs_entrainment: {obs_dict.get('dbs_entrainment', 0):.3f}
      disease_severity:{obs_dict.get('disease_severity', 0):.3f}
      sim_time_s:      {obs_dict.get('sim_time_s', 0):.2f}

    Recent actions (last 5):
    {recent}

    Output your DBS action as JSON now.
    """).strip()


def _parse_action(text: str) -> Optional[dict]:
    """Extract JSON action dict from model response."""
    text = text.strip()
    # Try to find a JSON block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _action_from_dict(d: Optional[dict]) -> ParkinsonsMotorAction:
    if not d:
        # Safe default: low-amplitude DBS, neutral motor command
        return ParkinsonsMotorAction(motor_command=0.0, dbs_amplitude=0.5, dbs_pulse_width=0.13)
    return ParkinsonsMotorAction(
        motor_command  = float(max(-1.0, min(1.0,  d.get("motor_command",  0.0)))),
        dbs_amplitude  = float(max(0.0,  min(5.0,  d.get("dbs_amplitude",  0.5)))),
        dbs_pulse_width= float(max(0.06, min(0.20, d.get("dbs_pulse_width", 0.13)))),
    )


# ── logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── single task runner ────────────────────────────────────────────────────────

async def run_task(env, client: OpenAI, task_id: str) -> tuple[float, bool]:
    """
    Run one complete task episode.  Returns (final_score, success).
    """
    max_steps = MAX_STEPS_PER_TASK[task_id]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    step_history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None

    try:
        # Reset with chosen task
        result = await env.reset()
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Get model action
            user_prompt = _build_user_prompt(step, obs_dict, task_id, step_history)
            raw_text = ""
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw_text = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                raw_text = ""
                final_error = str(exc)

            action_dict = _parse_action(raw_text)
            action = _action_from_dict(action_dict)
            action_str = json.dumps(action_dict or {
                "motor_command": action.motor_command,
                "dbs_amplitude": action.dbs_amplitude,
                "dbs_pulse_width": action.dbs_pulse_width,
            })

            error_for_step: Optional[str] = None
            try:
                result = await env.step(action)
                obs = result.observation
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
                reward = float(result.reward or 0.0)
                done = result.done
            except Exception as exc:
                reward = 0.0
                done = True
                error_for_step = str(exc)
                final_error = error_for_step
                log_step(step, action_str, reward, done, error_for_step)
                steps_taken = step
                break

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error_for_step)
            step_history.append(
                f"Step {step}: amp={action.dbs_amplitude:.2f}mA pw={action.dbs_pulse_width:.3f}ms "
                f"→ beta={obs_dict.get('beta_arv', 0):.3f} force={obs_dict.get('force_preserved', 0):.3f} "
                f"reward={reward:+.3f}"
            )

            if done:
                # Grader score is in observation at episode end
                grader_score = obs_dict.get("grader_score", -1.0)
                if grader_score >= 0:
                    score = grader_score
                break

        # If score wasn't emitted by grader (short episode), normalise rewards
        if score <= 0 and rewards:
            score = sum(rewards) / len(rewards)
            score = min(max(score, 0.0), 1.0)

        success = final_error is None and score >= SUCCESS_SCORE_THRESHOLD[task_id]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Missing API key. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY before running."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await ParkinsonsMotorEnv.from_docker_image(IMAGE_NAME)

    all_scores: dict[str, float] = {}
    all_success: dict[str, bool] = {}

    try:
        for task_id in TASKS:
            score, success = await run_task(env, client, task_id)
            all_scores[task_id] = score
            all_success[task_id] = success

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    # Summary across all tasks
    print("\n[SUMMARY]", flush=True)
    for tid in TASKS:
        print(
            f"  task={tid} score={all_scores.get(tid, 0):.2f} "
            f"success={str(all_success.get(tid, False)).lower()}",
            flush=True,
        )
    mean_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  mean_score={mean_score:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
