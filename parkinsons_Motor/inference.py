"""
Inference Script - Parkinson's Motor (DBS) Environment.

Runs an LLM agent against all three public tasks and emits OpenEnv-style logs.

Environment variables:
  LOCAL_IMAGE_NAME / IMAGE_NAME          Docker image for the environment
  LLM_PROVIDER                          auto | openai | hf | huggingface
  API_KEY / OPENAI_API_KEY              OpenAI-compatible API key
  OPENAI_BASE_URL / API_BASE_URL        OpenAI-compatible base URL
  OPENAI_MODEL / MODEL_NAME             OpenAI-compatible model name
  HF_TOKEN                              Hugging Face router token
  HF_API_BASE_URL / API_BASE_URL        Hugging Face router base URL
  HF_MODEL_NAME / MODEL_NAME            Hugging Face model name
  OPENAI_REQUEST_TIMEOUT_SECONDS        LLM request timeout (default: 120)
  OPENAI_MAX_RETRIES                    LLM retry count (default: 4)
  INFERENCE_REQUEST_SLEEP_SECONDS       Delay between successful steps (default: 0.5)
  INFERENCE_TASK_SLEEP_SECONDS          Delay between tasks (default: 2.0)
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

# -- path setup ---------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv  # noqa: E402


# -- env config ---------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    env_path = REPO_ROOT / path
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(".env")

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "parkinsons-motor:latest"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
HF_ROUTER_HOST = "huggingface.co"


def _looks_like_hf_router(url: Optional[str]) -> bool:
    return bool(url and HF_ROUTER_HOST in url.lower())


def _looks_like_openai_model(name: Optional[str]) -> bool:
    if not name:
        return False
    normalized = name.strip().lower()
    return "/" not in normalized and normalized.startswith(("gpt", "o", "text-embedding"))


def _resolve_llm_config() -> tuple[str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    hf_key = os.getenv("HF_TOKEN")

    generic_base_url = os.getenv("API_BASE_URL")
    generic_model = os.getenv("MODEL_NAME")

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL")

    hf_base_url = os.getenv("HF_API_BASE_URL")
    hf_model = os.getenv("HF_MODEL_NAME")

    if provider in {"auto", "openai"} and openai_key:
        base_url = openai_base_url or generic_base_url
        if provider == "openai" or _looks_like_hf_router(base_url) or not base_url:
            base_url = OPENAI_DEFAULT_BASE_URL

        model_name = openai_model
        if not model_name and generic_model and _looks_like_openai_model(generic_model):
            model_name = generic_model
        if not model_name:
            model_name = OPENAI_DEFAULT_MODEL

        return "openai", openai_key, base_url, model_name

    if provider in {"auto", "hf", "huggingface"} and hf_key:
        base_url = hf_base_url or generic_base_url or "https://router.huggingface.co/v1"
        model_name = hf_model or generic_model or "Qwen/Qwen2.5-72B-Instruct"
        return "huggingface", hf_key, base_url, model_name

    raise RuntimeError(
        "Missing LLM credentials. Set OPENAI_API_KEY/API_KEY for OpenAI or HF_TOKEN for the Hugging Face router."
    )


LLM_PROVIDER, API_KEY, API_BASE_URL, MODEL_NAME = _resolve_llm_config()
BENCHMARK = "parkinsons_Motor"

DEFAULT_TASKS = ["beta_suppression", "tremor_correction", "full_episode"]
TASKS = [
    task.strip()
    for task in os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS)).split(",")
    if task.strip()
]
MAX_STEPS_PER_TASK = {"beta_suppression": 30, "tremor_correction": 48, "full_episode": 100}
SUCCESS_SCORE_THRESHOLD = {"beta_suppression": 0.50, "tremor_correction": 0.36, "full_episode": 0.66}

TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
REQUEST_SLEEP_SECONDS = float(os.getenv("INFERENCE_REQUEST_SLEEP_SECONDS", "0.5"))
TASK_SLEEP_SECONDS = float(os.getenv("INFERENCE_TASK_SLEEP_SECONDS", "2.0"))


# -- LLM prompts --------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a closed-loop Deep Brain Stimulation (DBS) controller for a Parkinson's patient.
    Every 20ms you read the patient's brain state and output DBS settings as JSON.

    KEY OBSERVATIONS
      beta_arv         STN beta oscillation. Lower is better.
      tremor_arv       Tremor envelope. Lower is better.
      force_preserved  Fraction of healthy muscle force. Higher is better.
      side_effect_load Cumulative stimulation burden. Stay inside task budget.
      beta_trend       Negative means beta is improving.
      tremor_trend     Negative means tremor is improving.
      side_effect_rate Positive means burden is rising.
      gamma_arv        Over-stimulation alarm. If high, reduce amplitude.
      target_output    Copy this exactly into motor_command every step.
      dbs_entrainment  How much DBS is currently suppressing beta.

    OUTPUT FORMAT
    Return JSON only:
    {"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}

    RULES
    - If tremor_arv > 0.55 or beta_arv > 0.60, use at least 1.2 mA until symptoms improve.
    - If side_effect_load is near budget, reduce amplitude by 20-40%.
    - If gamma_arv > 0.55, reduce amplitude immediately.
    - Increase amplitude gradually when symptoms worsen.
    - Reduce amplitude slowly once both beta and tremor are improving.
    - Keep pulse width near 0.13 ms and frequency near 130 Hz unless there is a strong reason not to.
    - Avoid abrupt amplitude jumps larger than 0.3 mA per step.
    """
).strip()


_TASK_CONTEXT = {
    "beta_suppression": (
        "EASY. Ceiling: 1.5 mA. Side-effect budget: 0.55. "
        "Prefer 0.9-1.1 mA early, then taper toward 0.7-0.8 mA once beta and tremor are controlled."
    ),
    "tremor_correction": (
        "MEDIUM. Ceiling: 1.8 mA. Side-effect budget: 0.60. "
        "Rescue tremor quickly with 1.2-1.5 mA, then drop to 0.7-1.0 mA maintenance."
    ),
    "full_episode": (
        "HARD. Ceiling: 2.4 mA. Side-effect budget: 0.55. "
        "Phase 1 build entrainment, phase 2 rescue symptoms, phase 3 preserve budget for stable maintenance."
    ),
}


def _build_user_prompt(step: int, obs: dict, task_id: str, history: list[str]) -> str:
    recent = "\n".join(history[-4:]) if history else "(first step)"
    return textwrap.dedent(
        f"""
        Task: {task_id}
        Step: {step}
        Context: {_TASK_CONTEXT.get(task_id, "")}

        Brain state:
          beta_arv:         {obs.get('beta_arv', 0.0):.4f}
          tremor_arv:       {obs.get('tremor_arv', 0.0):.4f}
          force_preserved:  {obs.get('force_preserved', 0.0):.4f}
          side_effect_load: {obs.get('side_effect_load', 0.0):.4f}
          beta_trend:       {obs.get('beta_trend', 0.0):+.4f}
          tremor_trend:     {obs.get('tremor_trend', 0.0):+.4f}
          side_effect_rate: {obs.get('side_effect_rate', 0.0):+.4f}
          gamma_arv:        {obs.get('gamma_arv', 0.0):.4f}
          dbs_entrainment:  {obs.get('dbs_entrainment', 0.0):.4f}
          stim_washout:     {obs.get('stim_washout', 0.0):.4f}
          target_output:    {obs.get('target_output', 0.0):.4f}
          tracking_accuracy:{obs.get('tracking_accuracy', 0.0):.4f}

        Recent steps:
        {recent}

        Output JSON only now.
        """
    ).strip()


def _parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _action_from_dict(d: Optional[dict], target_output: float = 0.0) -> ParkinsonsMotorAction:
    if not d:
        return ParkinsonsMotorAction(
            motor_command=float(max(-1.0, min(1.0, target_output))),
            dbs_amplitude=1.0,
            dbs_pulse_width=0.13,
            dbs_frequency=130.0,
        )
    return ParkinsonsMotorAction(
        motor_command=float(max(-1.0, min(1.0, target_output))),
        dbs_amplitude=float(max(0.0, min(5.0, d.get("dbs_amplitude", 1.0)))),
        dbs_pulse_width=float(max(0.06, min(0.20, d.get("dbs_pulse_width", 0.13)))),
        dbs_frequency=float(max(60.0, min(185.0, d.get("dbs_frequency", 130.0)))),
    )


def _call_llm(client: OpenAI, step: int, obs: dict, task_id: str, history: list[str]) -> str:
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(step, obs, task_id, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            delay = min(2 ** attempt, 12)
            print(
                f"[DEBUG] LLM call failed (attempt {attempt + 1}/{OPENAI_MAX_RETRIES}): {exc}. Retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)
    return ""


# -- logging helpers ----------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error or "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# -- single task runner -------------------------------------------------------

async def run_task(env, client: OpenAI, task_id: str) -> tuple[float, bool]:
    max_steps = MAX_STEPS_PER_TASK[task_id]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw_text = _call_llm(client, step, obs_dict, task_id, history)
            action_dict = _parse_action(raw_text)
            action = _action_from_dict(action_dict, target_output=obs_dict.get("target_output", 0.0))
            action_str = json.dumps(
                {
                    "motor_command": round(action.motor_command, 3),
                    "dbs_amplitude": round(action.dbs_amplitude, 3),
                    "dbs_pulse_width": round(action.dbs_pulse_width, 3),
                    "dbs_frequency": round(action.dbs_frequency, 1),
                }
            )

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

            history.append(
                f"step={step} amp={action.dbs_amplitude:.2f} pw={action.dbs_pulse_width:.3f} freq={action.dbs_frequency:.0f}"
                f" -> beta={obs_dict.get('beta_arv', 0):.3f} tremor={obs_dict.get('tremor_arv', 0):.3f}"
                f" force={obs_dict.get('force_preserved', 0):.3f} se={obs_dict.get('side_effect_load', 0):.3f}"
                f" reward={reward:+.3f}"
            )

            if not done and REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)

            if done:
                grader_score = obs_dict.get("grader_score", -1.0)
                if grader_score >= 0:
                    score = grader_score
                break

        if score <= 0 and rewards:
            score = min(max(sum(rewards) / len(rewards), 0.0), 1.0)

        success = final_error is None and score >= SUCCESS_SCORE_THRESHOLD[task_id]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success


# -- main ---------------------------------------------------------------------

async def main() -> None:
    print(f"Provider: {LLM_PROVIDER}", flush=True)
    print(f"Model   : {MODEL_NAME}", flush=True)
    print(f"Image   : {IMAGE_NAME}", flush=True)
    print(f"Tasks   : {TASKS}", flush=True)
    print(f"Timeout : {REQUEST_TIMEOUT_SECONDS}s", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await ParkinsonsMotorEnv.from_docker_image(IMAGE_NAME)

    all_scores: dict[str, float] = {}
    all_success: dict[str, bool] = {}

    try:
        for idx, task_id in enumerate(TASKS):
            score, success = await run_task(env, client, task_id)
            all_scores[task_id] = score
            all_success[task_id] = success
            if idx < len(TASKS) - 1 and TASK_SLEEP_SECONDS > 0:
                time.sleep(TASK_SLEEP_SECONDS)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

    print("\n[SUMMARY]", flush=True)
    for task_id in TASKS:
        print(
            f"  task={task_id} score={all_scores.get(task_id, 0.0):.2f} success={str(all_success.get(task_id, False)).lower()}",
            flush=True,
        )
    mean_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  mean_score={mean_score:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
