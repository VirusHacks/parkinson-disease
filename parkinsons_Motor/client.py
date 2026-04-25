"""Parkinsons Motor Environment Client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ParkinsonsMotorAction, ParkinsonsMotorObservation


class ParkinsonsMotorEnv(
    EnvClient[ParkinsonsMotorAction, ParkinsonsMotorObservation, State]
):
    """
    Client for the Parkinson's Motor (DBS) Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session.

    Example — local server:
        >>> env = ParkinsonsMotorEnv(base_url="http://localhost:8000")
        >>> result = await env.reset()
        >>> result = await env.step(ParkinsonsMotorAction(dbs_amplitude=1.0))
        >>> await env.close()

    Example — Docker image:
        >>> env = await ParkinsonsMotorEnv.from_docker_image("parkinsons-motor:latest")
        >>> result = await env.reset()
        >>> await env.close()

    Example — Hugging Face Space:
        >>> env = await ParkinsonsMotorEnv.from_url("https://<your-space>.hf.space")
        >>> result = await env.reset()
        >>> await env.close()
    """

    def _step_payload(self, action: ParkinsonsMotorAction) -> Dict[str, Any]:
        """Convert ParkinsonsMotorAction to JSON payload for the step message."""
        return {
            "motor_command":   action.motor_command,
            "dbs_amplitude":   action.dbs_amplitude,
            "dbs_pulse_width": action.dbs_pulse_width,
            "task_id":         action.task_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ParkinsonsMotorObservation]:
        """Parse the server response into a typed StepResult."""
        obs_data = payload.get("observation", {})

        observation = ParkinsonsMotorObservation(
            # neural state
            beta_arv=obs_data.get("beta_arv", 0.0),
            tremor_arv=obs_data.get("tremor_arv", 0.0),
            semg_arv=obs_data.get("semg_arv", 0.0),
            # disease state
            disease_severity=obs_data.get("disease_severity", 0.0),
            beta_suppression=obs_data.get("beta_suppression", 0.0),
            beta_trend=obs_data.get("beta_trend", 0.0),
            tremor_trend=obs_data.get("tremor_trend", 0.0),
            side_effect_rate=obs_data.get("side_effect_rate", 0.0),
            # motor output
            force_amplitude=obs_data.get("force_amplitude", 0.0),
            force_preserved=obs_data.get("force_preserved", 0.0),
            target_output=obs_data.get("target_output", 0.0),
            effective_motor_output=obs_data.get("effective_motor_output", 0.0),
            task_error=obs_data.get("task_error", 0.0),
            tracking_accuracy=obs_data.get("tracking_accuracy", 0.0),
            # DBS state
            dbs_amplitude_ma=obs_data.get("dbs_amplitude_ma", 0.0),
            dbs_pulse_width_ms=obs_data.get("dbs_pulse_width_ms", 0.06),
            dbs_entrainment=obs_data.get("dbs_entrainment", 0.0),
            recent_dbs_avg_ma=obs_data.get("recent_dbs_avg_ma", 0.0),
            recent_dbs_avg_pw_ms=obs_data.get("recent_dbs_avg_pw_ms", 0.06),
            side_effect_load=obs_data.get("side_effect_load", 0.0),
            action_smoothness_cost=obs_data.get("action_smoothness_cost", 0.0),
            dbs_constraint_violation=obs_data.get("dbs_constraint_violation", 0.0),
            sim_time_s=obs_data.get("sim_time_s", 0.0),
            # extended physiological signals
            gamma_arv=obs_data.get("gamma_arv", 0.0),
            medication_phase=obs_data.get("medication_phase", 0.5),
            battery_drain_rate=obs_data.get("battery_drain_rate", 0.0),
            stim_washout=obs_data.get("stim_washout", 0.0),
            # task & grader
            task_id=obs_data.get("task_id", "hard"),
            grader_score=obs_data.get("grader_score", -1.0),
            episode_success=obs_data.get("episode_success", False),
            # diagnostic surfaces (the OpenEnv envelope strips obs.metadata,
            # so these have to be typed observation fields).
            grader_components=obs_data.get("grader_components") or {},
            active_events=list(obs_data.get("active_events") or []),
            event_schedule_summary=list(obs_data.get("event_schedule_summary") or []),
            # base fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse the server response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
