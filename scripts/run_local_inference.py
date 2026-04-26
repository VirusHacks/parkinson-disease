"""
Run a live LLM agent against the local Parkinson's DBS environment.

Connects to the locally-running FastAPI server (localhost:8000), runs the
LLM through all 3 tasks, and emits the required OpenEnv stdout format.

Usage:
    uv run --project parkinsons_Motor python run_local_inference.py

Reads OpenAI or Hugging Face router settings from `.env` or the environment.
When the config clearly points at the HF router, `auto` mode will prefer
`HF_TOKEN` over `API_KEY`.

Environment variables:
  LLM_PROVIDER                          auto | openai | hf | huggingface
  API_KEY / OPENAI_API_KEY              OpenAI-compatible API key
  OPENAI_BASE_URL / API_BASE_URL        OpenAI-compatible base URL
  OPENAI_MODEL / MODEL_NAME             OpenAI-compatible model name
  HF_TOKEN                              Hugging Face router token
  HF_API_BASE_URL / HF_ROUTE / API_BASE_URL
                                        Hugging Face router base URL
  HF_MODEL_NAME / HF_MODEL / MODEL_NAME Hugging Face model name
  OPENAI_REQUEST_TIMEOUT_SECONDS        LLM request timeout (default: 120)
  OPENAI_MAX_RETRIES                    LLM retry count (default: 4)
  OPENAI_TEMPERATURE                    Sampling temperature (default: 0.2)
  OPENAI_MAX_TOKENS                     Completion token cap (default: 300)
  INFERENCE_REQUEST_SLEEP_SECONDS       Delay between successful steps (default: 0.5)
  INFERENCE_TASK_SLEEP_SECONDS          Delay between tasks (default: 4.0)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows consoles default to cp1252; the LLM prompt text and report contain
# arrows/emojis that would otherwise crash with UnicodeEncodeError on Windows.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from openai import OpenAI

# path setup
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinsons_Motor import ParkinsonsMotorAction, ParkinsonsMotorEnv  # noqa: E402
from parkinsons_Motor.tasks import get_task  # noqa: E402


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
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
HF_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
HF_DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
HF_ROUTER_HOST = "huggingface.co"


def _looks_like_hf_router(url: Optional[str]) -> bool:
    return bool(url and HF_ROUTER_HOST in url.lower())


def _looks_like_openai_model(name: Optional[str]) -> bool:
    if not name:
        return False
    normalized = name.strip().lower()
    return "/" not in normalized and normalized.startswith(("gpt", "o", "text-embedding"))


def _looks_like_hf_model(name: Optional[str]) -> bool:
    if not name:
        return False
    normalized = name.strip().lower()
    return (
        "/" in normalized
        or normalized.startswith(("qwen", "meta-llama", "mistral", "deepseek", "google/"))
    )


def _resolve_llm_config() -> tuple[str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    hf_key = os.getenv("HF_TOKEN")

    generic_base_url = os.getenv("API_BASE_URL")
    generic_model = os.getenv("MODEL_NAME")

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL")

    hf_base_url = (
        os.getenv("HF_API_BASE_URL")
        or os.getenv("HF_ROUTE")
        or os.getenv("HF_ROUTER_URL")
        or os.getenv("HUGGINGFACE_API_BASE_URL")
    )
    hf_model = os.getenv("HF_MODEL_NAME") or os.getenv("HF_MODEL") or os.getenv("HUGGINGFACE_MODEL")

    hf_base_candidate = hf_base_url or generic_base_url
    hf_model_candidate = hf_model or generic_model
    prefer_hf_auto = (
        provider == "auto"
        and hf_key
        and (
            _looks_like_hf_router(hf_base_candidate)
            or _looks_like_hf_model(hf_model_candidate)
        )
    )

    if provider in {"hf", "huggingface"} or prefer_hf_auto:
        base_url = hf_base_candidate or HF_DEFAULT_BASE_URL
        model_name = hf_model_candidate or HF_DEFAULT_MODEL
        return "huggingface", hf_key, base_url, model_name

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
        base_url = hf_base_candidate or HF_DEFAULT_BASE_URL
        model_name = hf_model_candidate or HF_DEFAULT_MODEL
        return "huggingface", hf_key, base_url, model_name

    raise RuntimeError(
        "Missing LLM credentials. Set OPENAI_API_KEY/API_KEY for OpenAI or HF_TOKEN for the Hugging Face router."
    )


LLM_PROVIDER, API_KEY, API_BASE_URL, MODEL_NAME = _resolve_llm_config()
BENCHMARK    = "parkinsons_Motor"

DEFAULT_TASKS = ["easy", "medium", "hard"]

# Per-task max-step caps for cost control. Override via INFERENCE_MAX_STEPS_<TASK>.
# IMPORTANT: when a cap is < task.n_steps, env.done never fires, so the
# deterministic grader (`overall_score`) is NOT computed and the reported score
# falls back to a per-step reward mean. The report flags this with
# `grader_invoked=False` so you know the score is a soft proxy, not the
# benchmark grader.
DEFAULT_MAX_STEPS = {
    "easy": 50,
    "medium": 50,
    "hard": 30,
}


def _parse_task_list() -> List[str]:
    raw = os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS))
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    if not tasks:
        raise RuntimeError("INFERENCE_TASKS resolved to an empty task list.")
    return tasks


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default


def _resolve_task_runtime_config(tasks: List[str]) -> tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    """Resolve per-task max_steps, success thresholds, and safety budgets.

    Defaults come from the task definition itself (task.n_steps,
    task.success_threshold, task.max_side_effect_load) so the LLM context,
    grader, and pass/fail bar can never silently disagree.
    """
    max_steps: Dict[str, int] = {}
    success_thresholds: Dict[str, float] = {}
    side_effect_budgets: Dict[str, float] = {}

    for task_id in tasks:
        task = get_task(task_id)
        env_key = task.task_id.upper()
        default_max_steps = DEFAULT_MAX_STEPS.get(task_id, task.n_steps)
        max_steps[task_id] = _env_int(f"INFERENCE_MAX_STEPS_{env_key}", default_max_steps)
        success_thresholds[task_id] = _env_float(
            f"INFERENCE_SUCCESS_THRESHOLD_{env_key}",
            task.success_threshold,
        )
        side_effect_budgets[task_id] = _env_float(
            f"INFERENCE_SIDE_EFFECT_BUDGET_{env_key}",
            task.max_side_effect_load,
        )

    return max_steps, success_thresholds, side_effect_budgets


def _resolve_seeds() -> List[Optional[int]]:
    """Resolve the seed list for repeated rollouts.

    Set INFERENCE_SEEDS=0,1,2,3,4 for a 5-seed sweep. Default = single rollout
    with whatever seed the env chose (None -> server picks).
    """
    raw = os.getenv("INFERENCE_SEEDS")
    if not raw:
        return [None]
    seeds: List[Optional[int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError:
            print(f"[WARN] Ignoring non-integer seed token: {token!r}", flush=True)
    return seeds or [None]


TASKS = _parse_task_list()
MAX_STEPS, SUCCESS_THRESHOLD, SIDE_EFFECT_BUDGETS = _resolve_task_runtime_config(TASKS)
SEEDS = _resolve_seeds()
TEMPERATURE = _env_float("OPENAI_TEMPERATURE", 0.2)
MAX_TOKENS = _env_int("OPENAI_MAX_TOKENS", 300)
REQUEST_TIMEOUT_SECONDS = _env_float("OPENAI_REQUEST_TIMEOUT_SECONDS", 120.0)
OPENAI_MAX_RETRIES = _env_int("OPENAI_MAX_RETRIES", 4)
REQUEST_SLEEP_SECONDS = _env_float("INFERENCE_REQUEST_SLEEP_SECONDS", 0.5)
TASK_SLEEP_SECONDS = _env_float("INFERENCE_TASK_SLEEP_SECONDS", 4.0)
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs"
OUTPUT_BASENAME = os.getenv("INFERENCE_OUTPUT_BASENAME", "local_inference_report")
OUTPUT_JSON = OUTPUT_DIR / f"{OUTPUT_BASENAME}.json"
OUTPUT_MD = OUTPUT_DIR / f"{OUTPUT_BASENAME}.md"


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert closed-loop DBS controller managing Parkinsonian motor symptoms in real time.
Every step is a short clinical control decision: suppress pathological activity, preserve movement,
avoid overstimulation, and keep enough safety budget for the remainder of the episode.

Return JSON only:
{"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}

Important:
- Do not output prose, markdown, or explanation.
- Do not include motor_command. It is handled separately.
- Use pulse_width = 0.13 and frequency = 130 unless there is a strong safety reason to deviate.
- Prefer smooth amplitude changes. Avoid jumps larger than about 0.3 mA in one step.

Clinical meaning of observations:
- beta_arv: pathological beta oscillation. Lower is better.
- tremor_arv: tremor severity. Lower is better.
- force_preserved: retained motor force. Higher is better.
- side_effect_load: cumulative stimulation burden. High values mean budget is being exhausted.
- beta_trend / tremor_trend: positive means worsening, negative means improving.
- side_effect_rate: positive means burden is still rising.
- gamma_arv: overstimulation warning. If high, back off quickly.
- dbs_entrainment: how much current DBS is suppressing beta.
- stim_washout: residual stimulation carry-over. High washout means prior DBS is still acting.

Control priorities in order:
1. Prevent unsafe overstimulation.
2. Do not leave the patient undertreated when symptoms are clearly high or worsening.
3. Restore usable function by reducing tremor/beta while preserving force.
4. Once stable, taper toward the lowest effective amplitude.

Decision policy:
- If gamma_arv > 0.55, or side_effect_load is near budget, reduce amplitude promptly.
- If tremor_arv > 0.55 or beta_arv > 0.60, treat actively: usually at least 1.2 mA unless safety signals are severe.
- If symptoms are worsening and safety is acceptable, increase gradually by about 0.1-0.15 mA.
- If symptoms are improving and side effects are still rising, hold or reduce slightly.
- If symptoms are controlled and stable, taper slowly toward an efficient maintenance level.
- Brief rescue bursts are acceptable; sustained high amplitude is not.
- Never sit near zero amplitude while symptoms remain elevated.

Good controller behavior:
- Early tasks: stabilize without wasting budget.
- Rescue tasks: intervene decisively, then taper once recovery begins.
- Long tasks: pace stimulation so the episode stays controllable all the way to the end.
""").strip()


_TASK_CONTEXT_TEMPLATES = {
    "easy": (
        "EASY / Calm Start. Responsive patient early in symptom build-up. "
        "Clinical goal: calm rising beta and mild tremor without creating unnecessary side effects. "
        "Ceiling: {amp_ceiling:.2f} mA. Side-effect budget: {se_budget:.2f}. "
        "Preferred pattern: start in a moderate therapeutic range, stabilize quickly, then taper toward a low maintenance dose."
    ),
    "medium": (
        "MEDIUM / Rescue Phase. Symptoms are already escalating and force is at risk. "
        "Clinical goal: interrupt deterioration, restore usable movement, then step down to maintenance. "
        "Ceiling: {amp_ceiling:.2f} mA. Side-effect budget: {se_budget:.2f}. "
        "Preferred pattern: decisive rescue early, then gradual taper once tremor and beta stop worsening."
    ),
    "hard": (
        "HARD / Full Episode. Long closed-loop management across onset, escalation, peak symptoms, and recovery. "
        "Clinical goal: keep the patient functional through the whole session, not just one short rescue. "
        "Ceiling: {amp_ceiling:.2f} mA. Side-effect budget: {se_budget:.2f}. "
        "Preferred pattern: build entrainment, rescue when needed, then preserve enough budget for late stability."
    ),
}


def _build_task_context(task_id: str) -> str:
    """Render the task context with the actual task budgets, not stale literals."""
    template = _TASK_CONTEXT_TEMPLATES.get(task_id, "")
    if not template:
        return ""
    task = get_task(task_id)
    return template.format(
        amp_ceiling=task.max_dbs_amplitude,
        se_budget=SIDE_EFFECT_BUDGETS.get(task_id, task.max_side_effect_load),
    )


_TASK_CONTEXT = {tid: _build_task_context(tid) for tid in TASKS}


def _build_user_prompt(step: int, obs: dict, task_id: str, history: list) -> str:
    recent = "\n".join(history[-4:]) if history else "None"
    context = _TASK_CONTEXT.get(task_id, "")

    beta = obs.get('beta_arv', 0)
    tremor = obs.get('tremor_arv', 0)
    force = obs.get('force_preserved', 0)
    se_load = obs.get('side_effect_load', 0)
    beta_trend = obs.get('beta_trend', 0)
    tremor_trend = obs.get('tremor_trend', 0)
    se_rate = obs.get('side_effect_rate', 0)
    gamma = obs.get('gamma_arv', 0)
    target = obs.get('target_output', 0)

    se_budget = SIDE_EFFECT_BUDGETS.get(task_id, 0.44)
    se_warn = round(se_budget * 0.85, 3)
    se_crit = round(se_budget * 0.95, 3)

    # Urgency flags to help agent prioritise
    flags = []
    if tremor > 0.55 or beta > 0.60:
        flags.append("⚠ EMERGENCY: symptoms high - Priority 2 applies, maintain >= 1.2 mA")
    if se_load > se_crit:
        flags.append(f"⚠ SAFETY CRITICAL: side_effect_load={se_load:.3f} > {se_crit} - reduce amp 40%")
    elif se_load > se_warn:
        flags.append(f"⚠ SAFETY WARNING: side_effect_load={se_load:.3f} > {se_warn} - reduce amp 20-30%")
    if gamma > 0.55:
        flags.append(f"⚠ GAMMA HIGH ({gamma:.3f}): over-stimulation detected - reduce amplitude")
    if tremor_trend > 0.015 and tremor > 0.35:
        flags.append("↑ Tremor escalating - increase amplitude (Priority 4)")
    if beta_trend < -0.010 and tremor_trend < -0.010:
        flags.append("↓ Both signals improving - hold or very slowly reduce (Priority 5)")

    flag_str = "\n  ".join(flags) if flags else "No alerts - normal adaptive control (Priority 4)"

    return textwrap.dedent(f"""
    TASK: {task_id} | STEP: {step}
    {context}

    ALERTS:
      {flag_str}

    CURRENT STATE:
      beta_arv={beta:.4f}
      tremor_arv={tremor:.4f}
      force_preserved={force:.4f}
      side_effect_load={se_load:.4f}
      beta_trend={beta_trend:+.4f}
      tremor_trend={tremor_trend:+.4f}
      side_effect_rate={se_rate:+.4f}
      gamma_arv={gamma:.4f}
      dbs_entrainment={obs.get('dbs_entrainment', 0):.4f}
      stim_washout={obs.get('stim_washout', 0):.4f}
      tracking_accuracy={obs.get('tracking_accuracy', 0):.4f}
      target_output={target:.4f}

    INTERPRETATION HINTS:
    - worsening symptoms + acceptable safety => increase a little
    - high gamma or high side_effect_load => reduce
    - improving symptoms + positive side_effect_rate => taper or hold
    - stable control => maintain the lowest effective dose

    RECENT HISTORY:
      {recent if recent != "None" else "(first step)"}

    Output JSON only now.
    """).strip()


def _call_llm(client: OpenAI, step: int, obs: dict, task_id: str, history: list) -> str:
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(step, obs, task_id, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            delay = min(2 ** attempt, 12)
            print(
                f"[DEBUG] LLM call failed (attempt {attempt + 1}/{OPENAI_MAX_RETRIES}): {exc}. Retrying in {delay}s...",
                flush=True,
            )
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
    # motor_command is always target_output - LLM only decides DBS settings
    if not d:
        return ParkinsonsMotorAction(motor_command=target_output, dbs_amplitude=1.0, dbs_pulse_width=0.13, dbs_frequency=130.0)
    return ParkinsonsMotorAction(
        motor_command   = float(max(-1.0, min(1.0, target_output))),
        dbs_amplitude   = float(max(0.0,  min(5.0,   d.get("dbs_amplitude",   1.0)))),
        dbs_pulse_width = float(max(0.06, min(0.20,  d.get("dbs_pulse_width", 0.13)))),
        dbs_frequency   = float(max(60.0, min(185.0, d.get("dbs_frequency",   130.0)))),
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

def _amp_from_action_str(action_str: str) -> float:
    try:
        return float(json.loads(action_str).get("dbs_amplitude", 0.0))
    except Exception:
        return 0.0


async def run_task(
    env,
    client: OpenAI,
    task_id: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single rollout. Returns a dict with all diagnostics for the report."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    max_steps = MAX_STEPS[task_id]
    task = get_task(task_id)
    rewards: List[float] = []
    history: List[str] = []
    amplitudes: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None
    score_details: Dict[str, Any] = {}
    event_schedule: List[Dict[str, Any]] = []
    grader_invoked = False

    try:
        reset_kwargs: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            reset_kwargs["seed"] = seed
        result = await env.reset(**reset_kwargs)
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        # Prefer the typed observation field; fall back to obs.metadata for older
        # builds. (The OpenEnv server's serialize_observation strips obs.metadata
        # off the wire, so the typed field is what actually reaches us.)
        event_schedule = (
            list(obs_dict.get("event_schedule_summary") or [])
            or list((obs_dict.get("metadata") or {}).get("event_schedule") or [])
        )
        episode_steps_full = (obs_dict.get("metadata") or {}).get("episode_steps", task.n_steps)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw = await asyncio.to_thread(_call_llm, client, step, obs_dict, task_id, history)
            parsed = _parse_action(raw)
            action = _make_action(parsed, target_output=obs_dict.get("target_output", 0.0))
            action_str = json.dumps({
                "motor_command":   round(action.motor_command, 3),
                "dbs_amplitude":   round(action.dbs_amplitude, 3),
                "dbs_pulse_width": round(action.dbs_pulse_width, 3),
                "dbs_frequency":   round(action.dbs_frequency, 1),
            })
            amplitudes.append(action.dbs_amplitude)

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
                f"step={step} amp={action.dbs_amplitude:.2f} pw={action.dbs_pulse_width:.3f} freq={action.dbs_frequency:.0f}Hz"
                f" → beta={obs_dict.get('beta_arv',0):.3f} tremor={obs_dict.get('tremor_arv',0):.3f}"
                f" force={obs_dict.get('force_preserved',0):.3f} se={obs_dict.get('side_effect_load',0):.3f}"
                f" gamma={obs_dict.get('gamma_arv',0):.3f} reward={reward:+.3f}"
            )

            if not done and REQUEST_SLEEP_SECONDS > 0:
                await asyncio.sleep(REQUEST_SLEEP_SECONDS)

            if done:
                gs = obs_dict.get("grader_score", -1.0)
                if gs >= 0:
                    score = gs
                    grader_invoked = True
                # grader_components is a typed obs field; metadata.score_details
                # is the legacy path that gets stripped by the server envelope.
                score_details = (
                    dict(obs_dict.get("grader_components") or {})
                    or dict((obs_dict.get("metadata") or {}).get("score_details") or {})
                )
                # If event_schedule wasn't captured at reset (e.g. server never
                # populated metadata), pick it up from the final step.
                if not event_schedule:
                    event_schedule = list(obs_dict.get("event_schedule_summary") or [])
                break

        if not grader_invoked and rewards:
            score = min(max(sum(rewards) / len(rewards), 0.0), 1.0)

        success = final_error is None and score >= SUCCESS_THRESHOLD[task_id]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": float(score),
        "success": bool(success),
        "steps_taken": int(steps_taken),
        "max_steps": int(max_steps),
        "episode_steps_full": int(episode_steps_full),
        "grader_invoked": bool(grader_invoked),
        "success_threshold": float(SUCCESS_THRESHOLD[task_id]),
        "side_effect_budget": float(SIDE_EFFECT_BUDGETS[task_id]),
        "mean_amplitude_ma": float(sum(amplitudes) / len(amplitudes)) if amplitudes else 0.0,
        "max_amplitude_ma": float(max(amplitudes)) if amplitudes else 0.0,
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "rewards": [round(r, 4) for r in rewards],
        "event_schedule": event_schedule,
        "score_details": {k: float(v) for k, v in score_details.items()} if score_details else {},
        "error": final_error,
    }


def _aggregate_per_task(
    rollouts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group rollouts by task and compute mean/std/pass-rate."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in rollouts:
        by_task.setdefault(r["task_id"], []).append(r)

    aggregated: List[Dict[str, Any]] = []
    for task_id, runs in by_task.items():
        scores = [r["score"] for r in runs]
        n = len(scores)
        mean_score = sum(scores) / n
        var = sum((s - mean_score) ** 2 for s in scores) / n
        std_score = var ** 0.5
        successes = sum(1 for r in runs if r["success"])
        # Component means across seeds (only over rollouts where the grader ran).
        graded = [r["score_details"] for r in runs if r.get("score_details")]
        component_means: Dict[str, float] = {}
        if graded:
            keys = sorted({k for d in graded for k in d.keys()})
            for k in keys:
                vals = [d.get(k, 0.0) for d in graded]
                component_means[k] = sum(vals) / len(vals)
        aggregated.append({
            "task_id": task_id,
            "n_seeds": n,
            "score_mean": mean_score,
            "score_std": std_score,
            "score_min": min(scores),
            "score_max": max(scores),
            "success_rate": successes / n,
            "successes": successes,
            "success_threshold": runs[0]["success_threshold"],
            "any_grader_invoked": any(r["grader_invoked"] for r in runs),
            "component_means": component_means,
            "rollouts": runs,
        })
    return aggregated


def _write_report(summary: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Local Inference Report",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Server: `{summary['server_url']}`",
        f"- Tasks: `{', '.join(summary['tasks'])}`",
        f"- Seeds per task: `{summary['seeds']}`",
        f"- Request sleep: `{summary['request_sleep_seconds']}` s",
        f"- Inter-task sleep: `{summary['task_sleep_seconds']}` s",
        f"- Mean score (all rollouts): `{summary['mean_score']:.4f}`",
        "",
        "## Task Results (aggregated across seeds)",
        "",
        "| Task | n | Mean ± Std | Min | Max | Pass | Threshold | Grader ran? |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for agg in summary["per_task"]:
        lines.append(
            f"| `{agg['task_id']}` | {agg['n_seeds']} | "
            f"{agg['score_mean']:.4f} ± {agg['score_std']:.4f} | "
            f"{agg['score_min']:.4f} | {agg['score_max']:.4f} | "
            f"{agg['successes']}/{agg['n_seeds']} | "
            f"{agg['success_threshold']:.2f} | "
            f"{'yes' if agg['any_grader_invoked'] else 'NO (mean-reward fallback)'} |"
        )

    # Component-mean breakdown (only for tasks where the grader actually ran).
    has_components = any(agg["component_means"] for agg in summary["per_task"])
    if has_components:
        lines += [
            "",
            "## Grader component means (only for tasks where grader ran)",
            "",
        ]
        for agg in summary["per_task"]:
            cm = agg["component_means"]
            if not cm:
                continue
            lines.append(f"### `{agg['task_id']}`")
            lines.append("")
            lines.append("| Component | Value |")
            lines.append("|---|---:|")
            for k, v in cm.items():
                lines.append(f"| `{k}` | {v:.4f} |")
            lines.append("")

    # Per-seed detail with event timeline so we can see which crises actually fired.
    lines += [
        "",
        "## Per-seed detail",
        "",
    ]
    for agg in summary["per_task"]:
        lines.append(f"### `{agg['task_id']}`")
        lines.append("")
        lines.append("| Seed | Steps | Score | Pass | Mean amp (mA) | Max amp (mA) | Events fired |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for r in agg["rollouts"]:
            ev_summary = (
                ", ".join(
                    f"{e['event_type']}@{e['start_step']}-{e['end_step']}"
                    for e in r.get("event_schedule", [])
                )
                or "-"
            )
            lines.append(
                f"| {r['seed'] if r['seed'] is not None else '·'} | "
                f"{r['steps_taken']}/{r['episode_steps_full']} | "
                f"{r['score']:.4f} | "
                f"{'PASS' if r['success'] else 'FAIL'} | "
                f"{r['mean_amplitude_ma']:.3f} | "
                f"{r['max_amplitude_ma']:.3f} | "
                f"{ev_summary} |"
            )
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Parkinson's DBS Environment - LLM Baseline Run", flush=True)
    print(f"Provider: {LLM_PROVIDER}", flush=True)
    print(f"Model  : {MODEL_NAME}", flush=True)
    print(f"Server : {LOCAL_SERVER_URL}", flush=True)
    print(f"Tasks  : {TASKS}", flush=True)
    print(f"Steps  : {MAX_STEPS}", flush=True)
    print(f"Pass@  : {SUCCESS_THRESHOLD}", flush=True)
    print(f"SE bud : {SIDE_EFFECT_BUDGETS}", flush=True)
    print(f"Seeds  : {SEEDS}", flush=True)
    print(f"Timeout: {REQUEST_TIMEOUT_SECONDS}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Warn if any task is being capped below its full horizon: the deterministic
    # grader requires done=True (i.e. step == n_steps) to compute overall_score.
    for tid in TASKS:
        full = get_task(tid).n_steps
        if MAX_STEPS[tid] < full:
            print(
                f"[WARN] task={tid}: max_steps={MAX_STEPS[tid]} < n_steps={full} → "
                f"grader will NOT run, score is mean-reward fallback.",
                flush=True,
            )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rollouts: List[Dict[str, Any]] = []

    for task_id in TASKS:
        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n{'-'*50}", flush=True)
            print(
                f"Rollout: task={task_id} seed={seed} ({seed_idx + 1}/{len(SEEDS)})",
                flush=True,
            )
            env = ParkinsonsMotorEnv(base_url=LOCAL_SERVER_URL)
            await env.__aenter__()
            try:
                rollout = await run_task(env, client, task_id, seed=seed)
                rollouts.append(rollout)
                print(
                    f"  -> {task_id} seed={seed}: score={rollout['score']:.4f} "
                    f"success={rollout['success']} grader_ran={rollout['grader_invoked']}",
                    flush=True,
                )
            finally:
                await env.__aexit__(None, None, None)

            is_last = (task_id == TASKS[-1]) and (seed_idx == len(SEEDS) - 1)
            if not is_last and TASK_SLEEP_SECONDS > 0:
                print(f"  -> sleeping {TASK_SLEEP_SECONDS:.1f}s before next rollout", flush=True)
                await asyncio.sleep(TASK_SLEEP_SECONDS)

    per_task = _aggregate_per_task(rollouts)
    mean = sum(r["score"] for r in rollouts) / len(rollouts) if rollouts else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY (mean ± std across {len(SEEDS)} seed(s))", flush=True)
    for agg in per_task:
        s = agg["score_mean"]
        bar = "#" * int(s * 20) + "-" * (20 - int(s * 20))
        flag = f"{agg['successes']}/{agg['n_seeds']} PASS"
        print(
            f"  {agg['task_id']:<10} [{bar}] {s:.4f} ± {agg['score_std']:.4f}  {flag}",
            flush=True,
        )
    print(f"\n  Mean across all rollouts: {mean:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {
        "model_name": MODEL_NAME,
        "server_url": LOCAL_SERVER_URL,
        "tasks": TASKS,
        "seeds": SEEDS,
        "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
        "task_sleep_seconds": TASK_SLEEP_SECONDS,
        "mean_score": mean,
        "per_task": per_task,
    }
    _write_report(summary)
    print(f"Saved reports to {OUTPUT_JSON} and {OUTPUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
