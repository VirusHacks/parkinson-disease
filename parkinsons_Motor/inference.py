"""
Inference Script - Parkinson's Motor (DBS) Environment.

Runs an LLM agent against the public tasks via the live OpenEnv-compatible
environment server (Hugging Face Space or local URL).  No Docker required.

Required environment variables
───────────────────────────────
  ENV_URL          Base URL of the environment server.
                   Default: https://virustechhacks-parkinsons-motor.hf.space

LLM credentials (one of):
  OPENAI_API_KEY   OpenAI-compatible key  →  uses OPENAI_BASE_URL / OPENAI_MODEL
  HF_TOKEN         Hugging Face router    →  uses Qwen/Qwen2.5-72B-Instruct

Optional
────────
  LLM_PROVIDER              auto | openai | hf  (default: auto)
  OPENAI_BASE_URL           OpenAI-compatible base URL
  OPENAI_MODEL              Model name for OpenAI provider
  HF_API_BASE_URL           HF router base URL
  HF_MODEL_NAME             HF model name
  INFERENCE_TASKS           Comma-separated task IDs  (default: easy,medium,hard)
  OPENAI_TEMPERATURE        Sampling temperature      (default: 0.2)
  OPENAI_MAX_TOKENS         Max tokens per LLM call   (default: 300)
  OPENAI_REQUEST_TIMEOUT_SECONDS  (default: 120)
  OPENAI_MAX_RETRIES              (default: 4)
  INFERENCE_REQUEST_SLEEP_SECONDS Delay between steps  (default: 0.5)
  INFERENCE_TASK_SLEEP_SECONDS    Delay between tasks  (default: 2.0)
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

# ── path setup ────────────────────────────────────────────────────────────────
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv  # noqa: E402
from parkinsons_Motor.tasks import get_task                              # noqa: E402


# ── .env loader ───────────────────────────────────────────────────────────────

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


_load_dotenv()


# ── config ────────────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "https://virustechhacks-parkinsons-motor.hf.space")

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL    = "gpt-4o-mini"
HF_ROUTER_DEFAULT_URL   = "https://router.huggingface.co/v1"
HF_DEFAULT_MODEL        = "Qwen/Qwen2.5-72B-Instruct"


def _resolve_llm_config() -> tuple[str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    hf_key     = os.getenv("HF_TOKEN")
    base_url   = os.getenv("HF_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL") or ""
    model_name = os.getenv("HF_MODEL_NAME") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME") or ""

    # If the base URL points to HF router, always use HF mode regardless of
    # which other keys happen to be set.
    url_is_hf = "huggingface.co" in base_url.lower()

    if provider in {"hf", "huggingface"} or (provider == "auto" and (url_is_hf or not openai_key)):
        if not hf_key:
            raise RuntimeError("HF_TOKEN is required for the Hugging Face router.")
        base_url   = base_url or HF_ROUTER_DEFAULT_URL
        model_name = model_name or HF_DEFAULT_MODEL
        return "huggingface", hf_key, base_url, model_name

    if provider in {"auto", "openai"} and openai_key:
        base_url   = base_url or OPENAI_DEFAULT_BASE_URL
        model_name = model_name or OPENAI_DEFAULT_MODEL
        return "openai", openai_key, base_url, model_name

    raise RuntimeError(
        "Missing LLM credentials. "
        "Set OPENAI_API_KEY (or API_KEY) for OpenAI, or HF_TOKEN for the Hugging Face router."
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


LLM_PROVIDER, API_KEY, API_BASE_URL, MODEL_NAME = _resolve_llm_config()

DEFAULT_TASKS = ["easy", "medium", "hard"]
TASKS: List[str] = [
    t.strip()
    for t in os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS)).split(",")
    if t.strip()
] or DEFAULT_TASKS

TEMPERATURE             = _env_float("OPENAI_TEMPERATURE", 0.2)
MAX_TOKENS              = _env_int("OPENAI_MAX_TOKENS", 300)
REQUEST_TIMEOUT_SECONDS = _env_float("OPENAI_REQUEST_TIMEOUT_SECONDS", 120.0)
MAX_RETRIES             = _env_int("OPENAI_MAX_RETRIES", 4)
REQUEST_SLEEP_SECONDS   = _env_float("INFERENCE_REQUEST_SLEEP_SECONDS", 0.5)
TASK_SLEEP_SECONDS      = _env_float("INFERENCE_TASK_SLEEP_SECONDS", 2.0)

# per-task step limits and thresholds from task definitions
_MAX_STEPS: dict[str, int]   = {}
_THRESHOLDS: dict[str, float] = {}
for _task_id in TASKS:
    _t = get_task(_task_id)
    _MAX_STEPS[_task_id]   = _env_int(f"INFERENCE_MAX_STEPS_{_task_id.upper()}", _t.n_steps)
    _THRESHOLDS[_task_id]  = _env_float(f"INFERENCE_SUCCESS_THRESHOLD_{_task_id.upper()}", _t.success_threshold)


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert closed-loop DBS controller managing Parkinsonian motor symptoms in real time.
    Every step is a short clinical control decision: suppress pathological beta activity, preserve
    movement, avoid overstimulation, and maintain enough safety budget for the rest of the episode.

    Return JSON only - no explanations, no markdown:
    {"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}

    Clinical meaning of key signals:
    - beta_arv / tremor_arv: pathological activity - lower is better.
    - force_preserved: motor function - keep high.
    - side_effect_load / gamma_arv: overstimulation risk - watch carefully.
    - positive beta_trend / tremor_trend: symptoms worsening - act.
    - positive side_effect_rate: burden still rising - consider reducing.

    Control priorities (in order):
    1. Prevent unsafe overstimulation (gamma_arv high → reduce amplitude).
    2. Treat elevated symptoms actively (beta > 0.60 or tremor > 0.55 → increase).
    3. Restore useful motor function.
    4. Taper toward the lowest effective dose once stable.
""").strip()

_TASK_CONTEXT = {
    "easy": (
        "EASY / Calm Start. Responsive patient, mild early symptoms. "
        "Stabilize quickly without wasting safety budget. Ceiling: 1.5 mA."
    ),
    "medium": (
        "MEDIUM / Rescue Phase. Worsening symptoms, force at risk. "
        "Rescue actively, then taper once recovery begins. Ceiling: 1.8 mA."
    ),
    "hard": (
        "HARD / Full Episode. Long-horizon control through onset, escalation, "
        "peak, and recovery. Keep the patient functional throughout. Ceiling: 2.4 mA."
    ),
}


def _build_user_prompt(step: int, obs: dict, task_id: str, history: list[str]) -> str:
    recent = "\n".join(history[-4:]) if history else "(first step)"
    ctx = _TASK_CONTEXT.get(task_id, f"Task: {task_id}")
    return textwrap.dedent(f"""
        Task: {task_id}  |  Step: {step}
        Context: {ctx}

        Brain state:
          beta_arv:          {obs.get('beta_arv', 0.0):.4f}
          tremor_arv:        {obs.get('tremor_arv', 0.0):.4f}
          force_preserved:   {obs.get('force_preserved', 0.0):.4f}
          side_effect_load:  {obs.get('side_effect_load', 0.0):.4f}
          gamma_arv:         {obs.get('gamma_arv', 0.0):.4f}
          beta_trend:        {obs.get('beta_trend', 0.0):+.4f}
          tremor_trend:      {obs.get('tremor_trend', 0.0):+.4f}
          side_effect_rate:  {obs.get('side_effect_rate', 0.0):+.4f}
          dbs_entrainment:   {obs.get('dbs_entrainment', 0.0):.4f}
          stim_washout:      {obs.get('stim_washout', 0.0):.4f}
          tracking_accuracy: {obs.get('tracking_accuracy', 0.0):.4f}
          target_output:     {obs.get('target_output', 0.0):.4f}

        Recent steps:
        {recent}

        Output JSON only.
    """).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def _parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}") + 1
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
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(step, obs, task_id, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            delay = min(2 ** attempt, 16)
            print(f"[WARN] LLM attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}. Retrying in {delay}s…", flush=True)
            time.sleep(delay)
    return ""


# ── logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env_url: str, model: str) -> None:
    print(f"[START] task={task} env={env_url} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error or "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards=[{rewards_str}]", flush=True)


# ── task runner ───────────────────────────────────────────────────────────────

async def run_task(env: ParkinsonsMotorEnv, client: OpenAI, task_id: str) -> tuple[float, bool]:
    max_steps = _MAX_STEPS[task_id]
    threshold = _THRESHOLDS[task_id]
    log_start(task=task_id, env_url=ENV_URL, model=MODEL_NAME)

    rewards:      List[float] = []
    history:      List[str]   = []
    steps_taken   = 0
    score         = 0.0
    success       = False
    final_error:  Optional[str] = None

    try:
        result  = await env.reset(task_id=task_id)
        obs     = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw_text    = _call_llm(client, step, obs_dict, task_id, history)
            action_dict = _parse_action(raw_text)
            action      = _action_from_dict(action_dict, target_output=obs_dict.get("target_output", 0.0))
            action_str  = json.dumps({
                "dbs_amplitude":   round(action.dbs_amplitude, 3),
                "dbs_pulse_width": round(action.dbs_pulse_width, 3),
                "dbs_frequency":   round(action.dbs_frequency, 1),
                "motor_command":   round(action.motor_command, 3),
            })

            try:
                result   = await env.step(action)
                obs      = result.observation
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
                reward   = float(result.reward or 0.0)
                done     = result.done
                error_for_step: Optional[str] = None
            except Exception as exc:
                reward         = 0.0
                done           = True
                error_for_step = str(exc)
                final_error    = error_for_step
                log_step(step, action_str, reward, done, error_for_step)
                steps_taken = step
                break

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error_for_step)

            history.append(
                f"step={step} amp={action.dbs_amplitude:.2f} pw={action.dbs_pulse_width:.3f} "
                f"freq={action.dbs_frequency:.0f} → "
                f"beta={obs_dict.get('beta_arv', 0):.3f} tremor={obs_dict.get('tremor_arv', 0):.3f} "
                f"force={obs_dict.get('force_preserved', 0):.3f} se={obs_dict.get('side_effect_load', 0):.3f} "
                f"reward={reward:+.4f}"
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

        success = (final_error is None) and (score >= threshold)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"[INFO] Provider : {LLM_PROVIDER}", flush=True)
    print(f"[INFO] Model    : {MODEL_NAME}", flush=True)
    print(f"[INFO] Env URL  : {ENV_URL}", flush=True)
    print(f"[INFO] Tasks    : {TASKS}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect directly to the live HF Space (or local server) - no Docker needed.
    env = ParkinsonsMotorEnv(base_url=ENV_URL)
    await env.connect()

    all_scores:  dict[str, float] = {}
    all_success: dict[str, bool]  = {}

    try:
        for idx, task_id in enumerate(TASKS):
            score, success      = await run_task(env, client, task_id)
            all_scores[task_id] = score
            all_success[task_id] = success
            if idx < len(TASKS) - 1 and TASK_SLEEP_SECONDS > 0:
                time.sleep(TASK_SLEEP_SECONDS)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[WARN] env.close() raised: {exc}", flush=True)

    print("\n[SUMMARY]", flush=True)
    for task_id in TASKS:
        s = all_scores.get(task_id, 0.0)
        ok = all_success.get(task_id, False)
        print(f"  task={task_id:<30} score={s:.4f}  success={str(ok).lower()}", flush=True)
    mean_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"  mean_score={mean_score:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
