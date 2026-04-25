"""Live demo runner for the Parkinson's Motor viewer.

The runner keeps API keys on the server, steps the real environment, and emits
small JSON snapshots that the browser can turn into brain/body visuals.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from openai import OpenAI

from parkinsons_Motor.core.models import ParkinsonsMotorAction
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment


TASK_LIMITS = {
    # Public tier — keep demo runs short enough to feel snappy.
    "easy": 36,
    "beta_suppression": 36,
    "calm_start": 36,
    "medium": 60,
    "tremor_correction": 60,
    "rescue_phase": 60,
    "hard": 100,
    "full_episode": 100,
    # Expert tier — capped to keep demo episodes bounded.
    "fragile_patient": 64,
    "refractory_patient": 90,
    "personalization_generalization": 90,
    "exercise_bout": 70,
    "medication_interaction": 90,
    "nocturnal_transition": 90,
    "surgical_followup": 90,
}

DEFAULT_DEMO_TASK = "hard"


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a closed-loop Deep Brain Stimulation controller for a Parkinson's patient.
    Return JSON only: {"dbs_amplitude": X, "dbs_pulse_width": X, "dbs_frequency": X}.

    Keep beta_arv and tremor_arv low, force_preserved high, and side_effect_load inside budget.
    Raise amplitude when symptoms are high or worsening. Back off when side effects or gamma rise.
    Prefer smooth changes, pulse width near 0.13 ms, and frequency near 130 Hz.
    """
).strip()


TASK_CONTEXT = {
    "easy": "Easy early stabilization. Ceiling 1.5 mA. Keep DBS gentle and efficient.",
    "beta_suppression": "Easy early stabilization. Ceiling 1.5 mA. Keep DBS gentle and efficient.",
    "calm_start": "Easy early stabilization. Ceiling 1.5 mA. Keep DBS gentle and efficient.",
    "medium": "Medium tremor rescue. Use a short rescue push, then taper to maintenance.",
    "tremor_correction": "Medium tremor rescue. Use a short rescue push, then taper to maintenance.",
    "rescue_phase": "Medium tremor rescue. Use a short rescue push, then taper to maintenance.",
    "hard": "Hard full episode. Preserve safety budget while moving through rescue and maintenance phases.",
    "full_episode": "Hard full episode. Preserve safety budget while moving through rescue and maintenance phases.",
    "fragile_patient": "Fragile window: tight side-effect budget. Stay below ~1.4 mA, recover quickly from spikes.",
    "refractory_patient": "Drug-resistant patient: high baseline beta and tremor; needs sustained but smooth higher-amplitude DBS.",
    "personalization_generalization": "Mixed patient profile per episode: read responses early, adapt amplitude to that patient.",
    "exercise_bout": "Exercise burst: motor demand spikes — be ready for tracking surges and wider pulse widths briefly.",
    "medication_interaction": "L-DOPA cycle interacts with DBS: lower amplitude near medication peaks, raise during troughs.",
    "nocturnal_transition": "Sleep transition: very low amplitude required; avoid driving over-stimulation overnight.",
    "surgical_followup": "Post-implant follow-up: tolerance and impedance shifts; favor smooth, conservative changes.",
}


@dataclass
class DemoConfig:
    task_id: str
    agent_type: str = "auto"
    step_delay_ms: int = 450
    max_steps: Optional[int] = None


def _load_dotenv() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_llm_client(agent_type: str) -> tuple[Optional[OpenAI], str, str]:
    if agent_type == "heuristic":
        return None, "heuristic", "Local heuristic controller"

    _load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    hf_key = os.getenv("HF_TOKEN")

    if agent_type == "qwen":
        if hf_key:
            base_url = os.getenv("HF_API_BASE_URL") or os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
            model = os.getenv("HF_MODEL_NAME") or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
            return OpenAI(base_url=base_url, api_key=hf_key), model, "HF Qwen via router"
        return None, "heuristic", "HF token missing - local heuristic fallback"

    if provider in {"auto", "openai"} and openai_key:
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
        return OpenAI(base_url=base_url, api_key=openai_key), model, "OpenAI-compatible controller"

    if provider in {"auto", "hf", "huggingface", "qwen"} and hf_key:
        base_url = os.getenv("HF_API_BASE_URL") or os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
        model = os.getenv("HF_MODEL_NAME") or os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
        return OpenAI(base_url=base_url, api_key=hf_key), model, "HF router controller"

    return None, "heuristic", "No API token found - local heuristic fallback"


def _obs_dict(obs) -> dict:
    return obs.model_dump() if hasattr(obs, "model_dump") else dict(obs.__dict__)


def _build_prompt(step: int, task_id: str, obs: dict, history: list[str]) -> str:
    recent = "\n".join(history[-4:]) if history else "(first step)"
    return textwrap.dedent(
        f"""
        Task: {task_id}
        Context: {TASK_CONTEXT.get(task_id, "")}
        Step: {step}

        beta_arv={obs.get("beta_arv", 0):.4f}
        tremor_arv={obs.get("tremor_arv", 0):.4f}
        force_preserved={obs.get("force_preserved", 0):.4f}
        side_effect_load={obs.get("side_effect_load", 0):.4f}
        beta_trend={obs.get("beta_trend", 0):+.4f}
        tremor_trend={obs.get("tremor_trend", 0):+.4f}
        side_effect_rate={obs.get("side_effect_rate", 0):+.4f}
        gamma_arv={obs.get("gamma_arv", 0):.4f}
        dbs_entrainment={obs.get("dbs_entrainment", 0):.4f}
        target_output={obs.get("target_output", 0):.4f}

        Recent steps:
        {recent}
        """
    ).strip()


def _parse_json_action(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _heuristic_action(obs: dict, task_id: str) -> ParkinsonsMotorAction:
    beta = float(obs.get("beta_arv", 0.0))
    tremor = float(obs.get("tremor_arv", 0.0))
    side = float(obs.get("side_effect_load", 0.0))
    beta_trend = float(obs.get("beta_trend", 0.0))
    tremor_trend = float(obs.get("tremor_trend", 0.0))
    prev_amp = float(obs.get("dbs_amplitude_ma", 0.0))

    if task_id in ("easy", "beta_suppression", "calm_start"):
        amp = 0.72 + 0.38 * beta + 0.18 * tremor
        ceiling = 1.5
    elif task_id in ("medium", "tremor_correction", "rescue_phase"):
        amp = 1.15 + 0.32 * tremor + 0.18 * beta
        ceiling = 1.8
    elif task_id == "fragile_patient":
        amp = 0.65 + 0.30 * beta + 0.18 * tremor
        ceiling = 1.4
    elif task_id == "refractory_patient":
        amp = 1.40 + 0.48 * beta + 0.32 * tremor
        ceiling = 2.6
    elif task_id == "exercise_bout":
        amp = 1.10 + 0.34 * tremor + 0.20 * beta
        ceiling = 2.2
    elif task_id == "nocturnal_transition":
        amp = 0.40 + 0.18 * beta + 0.10 * tremor
        ceiling = 1.0
    elif task_id == "medication_interaction":
        med = float(obs.get("medication_phase", 0.5))
        amp = (1.20 - 0.30 * med) + 0.32 * beta + 0.20 * tremor
        ceiling = 2.0
    elif task_id == "surgical_followup":
        amp = 0.85 + 0.30 * beta + 0.20 * tremor
        ceiling = 1.7
    else:
        amp = 0.95 + 0.38 * beta + 0.28 * tremor
        ceiling = 2.4

    if beta_trend > 0.015 or tremor_trend > 0.015:
        amp = max(amp, prev_amp + 0.12)
    if side > 0.48:
        amp *= 0.72
    elif side > 0.38:
        amp *= 0.86

    if prev_amp > 0:
        amp = max(prev_amp - 0.22, min(prev_amp + 0.22, amp))
    amp = max(0.05, min(ceiling, amp))
    return ParkinsonsMotorAction(
        motor_command=float(max(-1.0, min(1.0, obs.get("target_output", 0.0)))),
        dbs_amplitude=amp,
        dbs_pulse_width=0.13,
        dbs_frequency=130.0,
    )


async def _llm_action(
    client: Optional[OpenAI],
    model: str,
    step: int,
    task_id: str,
    obs: dict,
    history: list[str],
) -> ParkinsonsMotorAction:
    if client is None:
        return _heuristic_action(obs, task_id)

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(step, task_id, obs, history)},
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "220")),
            timeout=float(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "60")),
        )
        return response.choices[0].message.content or ""

    try:
        raw = await asyncio.to_thread(_call)
        parsed = _parse_json_action(raw)
    except Exception:
        parsed = None

    if parsed is None:
        return _heuristic_action(obs, task_id)

    return ParkinsonsMotorAction(
        motor_command=float(max(-1.0, min(1.0, obs.get("target_output", 0.0)))),
        dbs_amplitude=float(max(0.0, min(5.0, parsed.get("dbs_amplitude", 1.0)))),
        dbs_pulse_width=float(max(0.06, min(0.20, parsed.get("dbs_pulse_width", 0.13)))),
        dbs_frequency=float(max(60.0, min(185.0, parsed.get("dbs_frequency", 130.0)))),
    )


def _action_dict(action: ParkinsonsMotorAction) -> dict:
    return {
        "motor_command": round(action.motor_command, 4),
        "dbs_amplitude": round(action.dbs_amplitude, 4),
        "dbs_pulse_width": round(action.dbs_pulse_width, 4),
        "dbs_frequency": round(action.dbs_frequency, 2),
    }


def _visual_phase(task_id: str, step: int, obs: dict) -> str:
    if obs.get("side_effect_load", 0.0) > 0.48:
        return "safety backoff"
    if obs.get("tremor_arv", 0.0) > 0.52 or obs.get("beta_arv", 0.0) > 0.58:
        return "rescue"
    if task_id in ("hard", "full_episode") and step > 45:
        return "maintenance"
    if task_id == "nocturnal_transition" and step > 30:
        return "low-drive sleep"
    return "stabilizing"


def _build_rationale(task_id: str, obs: dict, action: ParkinsonsMotorAction) -> str:
    beta = float(obs.get("beta_arv", 0.0))
    tremor = float(obs.get("tremor_arv", 0.0))
    side = float(obs.get("side_effect_load", 0.0))
    beta_trend = float(obs.get("beta_trend", 0.0))
    tremor_trend = float(obs.get("tremor_trend", 0.0))

    if side > 0.48:
        return f"Side-effect load is elevated, so stimulation is backing off to {action.dbs_amplitude:.2f} mA."
    if tremor > 0.55 or beta > 0.60:
        return f"Symptoms are high, so the agent is pushing a rescue pulse at {action.dbs_amplitude:.2f} mA."
    if beta_trend > 0.015 or tremor_trend > 0.015:
        return f"Signals are worsening, so DBS is stepping up to {action.dbs_amplitude:.2f} mA."
    if task_id in ("hard", "full_episode") and action.dbs_amplitude < 1.0:
        return f"The episode is in a maintenance window, holding a lighter dose at {action.dbs_amplitude:.2f} mA."
    if task_id == "nocturnal_transition":
        return f"Sleep window — keeping DBS gentle at {action.dbs_amplitude:.2f} mA to avoid over-stimulation."
    if task_id == "exercise_bout":
        return f"Exercise burst — pushing {action.dbs_amplitude:.2f} mA to keep tracking through the motor demand."
    return f"Control is staying smooth at {action.dbs_amplitude:.2f} mA while tracking the target movement."


def _snapshot(
    task_id: str,
    step: int,
    obs: dict,
    action: ParkinsonsMotorAction,
    model: str,
    agent_runtime: str,
) -> dict:
    metadata = obs.get("metadata") or {}
    active_events = list(metadata.get("active_events", [])) if isinstance(metadata, dict) else []
    return {
        "type": "step",
        "step": step,
        "task_id": task_id,
        "agent_model": model,
        "agent_runtime": agent_runtime,
        "action": _action_dict(action),
        "observation": obs,
        "active_events": active_events,
        "rationale": _build_rationale(task_id, obs, action),
        "derived_visuals": {
            "active_region": "stn",
            "brain_intensity": max(float(obs.get("beta_arv", 0.0)), float(obs.get("tremor_arv", 0.0))),
            "tremor_level": float(obs.get("tremor_arv", 0.0)),
            "safe": float(obs.get("side_effect_load", 0.0)) < 0.5,
            "phase": _visual_phase(task_id, step, obs),
            "active_events": active_events,
        },
    }


async def stream_demo_episode(config: DemoConfig) -> AsyncIterator[dict]:
    task_id = config.task_id if config.task_id in TASK_LIMITS else DEFAULT_DEMO_TASK
    delay = max(0, min(config.step_delay_ms, 2500)) / 1000
    max_steps = min(config.max_steps or TASK_LIMITS[task_id], TASK_LIMITS[task_id])
    client, model, agent_runtime = _resolve_llm_client(config.agent_type)

    env = ParkinsonsMotorEnvironment()
    obs = env.reset(task_id=task_id)
    obs_data = _obs_dict(obs)
    history: list[str] = []

    reset_metadata = obs_data.get("metadata") or {}
    reset_events = list(reset_metadata.get("active_events", [])) if isinstance(reset_metadata, dict) else []
    yield {
        "type": "reset",
        "task_id": task_id,
        "agent_model": model,
        "agent_runtime": agent_runtime,
        "observation": obs_data,
        "active_events": reset_events,
        "rationale": "The environment is ready. Start the episode to watch the controller adapt DBS in real time.",
        "derived_visuals": {
            "active_region": "stn",
            "brain_intensity": max(obs_data.get("beta_arv", 0.0), obs_data.get("tremor_arv", 0.0)),
            "tremor_level": obs_data.get("tremor_arv", 0.0),
            "active_events": reset_events,
            "safe": True,
            "phase": "ready",
        },
    }

    for step in range(1, max_steps + 1):
        action = await _llm_action(client, model, step, task_id, obs_data, history)
        obs = env.step(action)
        obs_data = _obs_dict(obs)
        yield _snapshot(task_id, step, obs_data, action, model, agent_runtime)

        history.append(
            f"step={step} amp={action.dbs_amplitude:.2f} beta={obs_data.get('beta_arv', 0):.3f} "
            f"tremor={obs_data.get('tremor_arv', 0):.3f} force={obs_data.get('force_preserved', 0):.3f}"
        )
        if obs.done:
            break
        if delay:
            await asyncio.sleep(delay)

    final_obs = obs_data
    yield {
        "type": "done",
        "task_id": task_id,
        "agent_model": model,
        "agent_runtime": agent_runtime,
        "final_score": final_obs.get("grader_score", -1.0),
        "success": final_obs.get("episode_success", False),
        "observation": final_obs,
        "rationale": "Episode complete. The final score reflects motor function, tremor suppression, and safety.",
    }
