"""
Run a live LLM agent against the local Parkinson's DBS environment.

Connects to the locally-running FastAPI server (localhost:8000), runs the
LLM through all 3 tasks, and emits the required OpenEnv stdout format.

Usage:
    uv run --project parkinsons_Motor python run_local_inference.py

Reads OpenAI or Hugging Face router settings from .env or environment.
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
BENCHMARK    = "parkinsons_Motor"

DEFAULT_TASKS = ["beta_suppression", "tremor_correction", "full_episode"]
TASKS = [
    task.strip()
    for task in os.getenv("INFERENCE_TASKS", ",".join(DEFAULT_TASKS)).split(",")
    if task.strip()
]
# Must match task n_steps exactly so grader runs at episode end
MAX_STEPS = {"beta_suppression": 30, "tremor_correction": 48, "full_episode": 100}
# Must match task success_threshold in scenarios.py
SUCCESS_THRESHOLD = {"beta_suppression": 0.50, "tremor_correction": 0.36, "full_episode": 0.66}
TEMPERATURE = 0.2
MAX_TOKENS  = 300
REQUEST_SLEEP_SECONDS = float(os.getenv("INFERENCE_SLEEP_SECONDS", "0.5"))
TASK_SLEEP_SECONDS = float(os.getenv("INFERENCE_TASK_SLEEP_SECONDS", "4.0"))
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs"
OUTPUT_BASENAME = os.getenv("INFERENCE_OUTPUT_BASENAME", "local_inference_report")
OUTPUT_JSON = OUTPUT_DIR / f"{OUTPUT_BASENAME}.json"
OUTPUT_MD = OUTPUT_DIR / f"{OUTPUT_BASENAME}.md"


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a closed-loop Deep Brain Stimulation (DBS) controller for a Parkinson's patient.
Every 20ms you read the patient's brain state and output DBS settings as JSON.

━━━ KEY OBSERVATIONS ━━━
  beta_arv        [0→1]  STN beta oscillation. 0=suppressed(good), 1=peak pathology(bad).
  tremor_arv      [0→1]  Tremor envelope. 0=no tremor(good), 1=max tremor(bad).
  force_preserved [0→1]  Fraction of healthy muscle force. MAXIMISE THIS.
  side_effect_load[0→1]  Cumulative stimulation burden. Budget varies per task (shown in context).
  beta_trend      [±]    Negative = beta improving. Positive = worsening → increase amplitude.
  tremor_trend    [±]    Negative = tremor improving. Positive = worsening → maintain/increase amplitude.
  side_effect_rate[±]    Positive = load rising. Negative = recovering (safe to hold current amplitude).
  gamma_arv       [0→1]  Over-stimulation alarm. >0.55 = reduce amplitude now.
  target_output   [±1]   Motor goal. Copy this EXACTLY into motor_command every step.
  dbs_entrainment [0→1]  How much DBS is suppressing beta right now (one step lag).

━━━ ACTION FORMAT (JSON only, no explanation) ━━━
{"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}
Note: motor_command is handled automatically — do NOT include it.

━━━ DECISION RULES (apply in order) ━━━

RULE 1 — EMERGENCY: If tremor_arv > 0.55 OR beta_arv > 0.60:
  → Set amplitude ≥ 1.2 mA. Keep this level until symptoms drop below 0.45.

RULE 2 — SAFETY: If side_effect_load alert is shown (⚠ SAFETY):
  → WARNING: reduce amplitude by 20-30%. CRITICAL: reduce by 40%.

RULE 3 — ADAPTIVE (normal):
  • beta_trend > 0.015 or tremor_trend > 0.015 → increase amplitude by 0.1-0.15 mA
  • Both trends negative AND side_effect_rate negative → reduce by 0.05-0.10 mA slowly
  • Otherwise → hold current amplitude

RULE 4 — EFFICIENCY: When symptoms stable (both arv < target, 3+ consecutive steps improving):
  → Slowly reduce toward 0.6-0.9 mA. Change ≤0.15 mA per step.

━━━ AMPLITUDE GUIDANCE ━━━
  Sustainable range (won't exhaust budget): 0.5-1.0 mA for most tasks
  Rescue range (brief bursts only): 1.2-1.8 mA — reduce after symptoms stabilize
  Never exceed task ceiling. Large abrupt changes (>0.3 mA/step) → smoothness penalty.

━━━ FIXED PARAMETERS ━━━
  pulse_width: 0.13 ms (standard — do not change)
  frequency: 130 Hz (optimal for beta suppression — do not change)

━━━ WHAT KILLS YOUR SCORE ━━━
  ✗ Letting amplitude stay at 0 while symptoms are elevated → hard grader penalty (-0.20)
  ✗ Sustained high amplitude past safe range → side_effect_load exceeds budget → safety collapse
  ✗ Abrupt amplitude changes >0.3 mA per step → smoothness penalty
  ✗ Never reducing amplitude even when symptoms improve → efficiency score collapses
""").strip()


_TASK_CONTEXT = {
    "beta_suppression": (
        "EASY — Early stabilization (30 steps). Patient is responsive. "
        "Ceiling: 1.5 mA. Side-effect BUDGET: 0.55 (trigger safety reduction if load > 0.47). "
        "STRATEGY: Hold 0.9-1.1 mA. Reduce to 0.7-0.8 mA when BOTH beta_arv<0.22 AND tremor_arv<0.18. "
        "Changes ≤0.15 mA/step. Avoid exceeding 1.3 mA — approaching 1.5 mA (max) risks budget."
    ),
    "tremor_correction": (
        "MEDIUM — Active tremor rescue (48 steps). Episode starts during tremor escalation. "
        "Ceiling: 1.8 mA. Side-effect BUDGET: 0.60 (trigger safety reduction if load > 0.51). "
        "TWO-PHASE STRATEGY: "
        "Phase 1 (steps 1-18): Use 1.2-1.5 mA to arrest escalation. This is urgent. "
        "Phase 2 (steps 19+): Drop to 0.7-1.0 mA maintenance once tremor_trend turns negative. "
        "Sustaining >1.5 mA past step 20 risks budget. Grader rewards recovery from bad to good."
    ),
    "full_episode": (
        "HARD — Full 100-step closed-loop episode. Patient may be refractory. "
        "Ceiling: 2.4 mA. Side-effect BUDGET: 0.55 (trigger safety reduction if load > 0.47). "
        "THREE-PHASE STRATEGY (ESSENTIAL): "
        "Phase 1 (steps 1-15): Build entrainment at 0.8-1.2 mA. "
        "Phase 2 (steps 16-45): Push 1.2-1.8 mA during peak escalation — watch side_effect_load. "
        "Phase 3 (steps 46-100): Reduce to 0.5-0.9 mA maintenance. Budget must survive 100 steps. "
        "Max amp (2.4 mA) is NEVER sustainable — budget exhausts quickly at full power."
    ),
}


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

    # Task-specific safety thresholds
    _se_budgets = {"beta_suppression": 0.55, "tremor_correction": 0.60, "full_episode": 0.55}
    se_budget = _se_budgets.get(task_id, 0.44)
    se_warn = round(se_budget * 0.85, 3)
    se_crit = round(se_budget * 0.95, 3)

    # Urgency flags to help agent prioritise
    flags = []
    if tremor > 0.55 or beta > 0.60:
        flags.append("⚠ EMERGENCY: symptoms high — Priority 2 applies, maintain >= 1.2 mA")
    if se_load > se_crit:
        flags.append(f"⚠ SAFETY CRITICAL: side_effect_load={se_load:.3f} > {se_crit} — reduce amp 40%")
    elif se_load > se_warn:
        flags.append(f"⚠ SAFETY WARNING: side_effect_load={se_load:.3f} > {se_warn} — reduce amp 20-30%")
    if gamma > 0.55:
        flags.append(f"⚠ GAMMA HIGH ({gamma:.3f}): over-stimulation detected — reduce amplitude")
    if tremor_trend > 0.015 and tremor > 0.35:
        flags.append("↑ Tremor escalating — increase amplitude (Priority 4)")
    if beta_trend < -0.010 and tremor_trend < -0.010:
        flags.append("↓ Both signals improving — hold or very slowly reduce (Priority 5)")

    flag_str = "\n  ".join(flags) if flags else "No alerts — normal adaptive control (Priority 4)"

    return textwrap.dedent(f"""
    TASK: {task_id} | Step {step}
    {context}

    ┌─ ALERTS ────────────────────────────────────────────
      {flag_str}

    ┌─ BRAIN STATE ───────────────────────────────────────
      beta_arv:         {beta:.4f}   tremor_arv:       {tremor:.4f}
      force_preserved:  {force:.4f}   side_effect_load: {se_load:.4f}
      beta_trend:      {beta_trend:+.4f}   tremor_trend:    {tremor_trend:+.4f}
      side_effect_rate:{se_rate:+.4f}   gamma_arv:        {gamma:.4f}
      dbs_entrainment:  {obs.get('dbs_entrainment', 0):.4f}   stim_washout:     {obs.get('stim_washout', 0):.4f}

    ┌─ MOTOR ─────────────────────────────────────────────
      target_output:    {target:.4f}  ← copy this into motor_command
      tracking_accuracy:{obs.get('tracking_accuracy', 0):.4f}

    ┌─ RECENT HISTORY ────────────────────────────────────
      {recent if recent != "None" else "(first step)"}

    Output JSON now:
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


def _make_action(d: Optional[dict], target_output: float = 0.0) -> ParkinsonsMotorAction:
    # motor_command is always target_output — LLM only decides DBS settings
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
        result = await env.reset(task_id=task_id)
        obs = result.observation
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        for step in range(1, max_steps + 1):
            if result.done:
                break

            raw = _call_llm(client, step, obs_dict, task_id, history)
            parsed = _parse_action(raw)
            action = _make_action(parsed, target_output=obs_dict.get("target_output", 0.0))
            action_str = json.dumps({
                "motor_command":   round(action.motor_command, 3),
                "dbs_amplitude":   round(action.dbs_amplitude, 3),
                "dbs_pulse_width": round(action.dbs_pulse_width, 3),
                "dbs_frequency":   round(action.dbs_frequency, 1),
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
                f"step={step} amp={action.dbs_amplitude:.2f} pw={action.dbs_pulse_width:.3f} freq={action.dbs_frequency:.0f}Hz"
                f" → beta={obs_dict.get('beta_arv',0):.3f} tremor={obs_dict.get('tremor_arv',0):.3f}"
                f" force={obs_dict.get('force_preserved',0):.3f} se={obs_dict.get('side_effect_load',0):.3f}"
                f" gamma={obs_dict.get('gamma_arv',0):.3f} reward={reward:+.3f}"
            )

            if not done and REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)

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


def _write_report(summary: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Local Inference Report",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Server: `{summary['server_url']}`",
        f"- Tasks: `{', '.join(summary['tasks'])}`",
        f"- Request sleep: `{summary['request_sleep_seconds']}` s",
        f"- Inter-task sleep: `{summary['task_sleep_seconds']}` s",
        f"- Mean score: `{summary['mean_score']:.4f}`",
        "",
        "## Task Results",
        "",
        "| Task | Score | Success |",
        "|---|---:|---:|",
    ]
    for task in summary["task_results"]:
        lines.append(
            f"| `{task['task_id']}` | {task['score']:.4f} | {'PASS' if task['success'] else 'FAIL'} |"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Parkinson's DBS Environment - LLM Baseline Run", flush=True)
    print(f"Provider: {LLM_PROVIDER}", flush=True)
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
            if task_id != TASKS[-1] and TASK_SLEEP_SECONDS > 0:
                print(f"  -> sleeping {TASK_SLEEP_SECONDS:.1f}s before next task", flush=True)
                time.sleep(TASK_SLEEP_SECONDS)
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

    summary = {
        "model_name": MODEL_NAME,
        "server_url": LOCAL_SERVER_URL,
        "tasks": TASKS,
        "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
        "task_sleep_seconds": TASK_SLEEP_SECONDS,
        "mean_score": mean,
        "task_results": [
            {
                "task_id": tid,
                "score": all_scores.get(tid, 0.0),
                "success": all_success.get(tid, False),
            }
            for tid in TASKS
        ],
    }
    _write_report(summary)
    print(f"Saved reports to {OUTPUT_JSON} and {OUTPUT_MD}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
