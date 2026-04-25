"""End-to-end smoke test for new scenario tasks (events + schedules)."""

from parkinsons_Motor.core.models import ParkinsonsMotorAction
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment


SCENARIO_TASKS = [
    "exercise_bout",
    "medication_interaction",
    "nocturnal_transition",
    "surgical_followup",
]


def main() -> None:
    env = ParkinsonsMotorEnvironment()

    for task_id in SCENARIO_TASKS:
        obs = env.reset(task_id=task_id, seed=11)
        n_steps = obs.metadata["episode_steps"]
        events = obs.metadata.get("event_schedule", [])
        schedule = obs.metadata.get("schedule_id")
        print(
            f"Task: {task_id} | n_steps={n_steps} | profile={obs.metadata['patient_profile_id']} "
            f"| schedule={schedule} | scheduled_events={len(events)}"
        )
        for ev in events:
            print(f"    event: {ev}")

        total_reward = 0.0
        observed_event_steps = []
        for i in range(n_steps):
            action = ParkinsonsMotorAction(
                motor_command=obs.target_output,
                dbs_amplitude=1.0,
                dbs_pulse_width=0.13,
            )
            obs = env.step(action)
            total_reward += obs.reward or 0.0
            active = obs.metadata.get("active_events", [])
            if active:
                observed_event_steps.append((i, active))
            if obs.done:
                break

        print(
            f"  -> total_reward={total_reward:.2f} | grader_score={obs.grader_score:.4f} "
            f"| success={obs.episode_success} | observed_event_steps={len(observed_event_steps)}"
        )

    print("\nScenario smoke test PASSED!")


if __name__ == "__main__":
    main()
