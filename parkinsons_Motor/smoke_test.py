import sys
sys.path.insert(0, ".")
from parkinsons_Motor.server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
from parkinsons_Motor.models import ParkinsonsMotorAction

env = ParkinsonsMotorEnvironment()

for task_id in ["beta_suppression", "tremor_correction", "full_episode"]:
    obs = env.reset(task_id=task_id, seed=11)
    n = obs.metadata["episode_steps"]
    print(
        f"Task: {task_id} | n_steps={n} | profile={obs.metadata['patient_profile_id']} "
        f"| force={obs.force_preserved:.3f} | beta={obs.beta_arv:.3f}"
    )

    total_reward = 0.0
    for _ in range(n):
        action = ParkinsonsMotorAction(
            motor_command=obs.target_output,
            dbs_amplitude=1.0,
            dbs_pulse_width=0.13,
        )
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        if obs.done:
            break

    print(
        f"  -> total_reward={total_reward:.2f} | grader_score={obs.grader_score:.4f} "
        f"| success={obs.episode_success} | details={obs.metadata.get('score_details', {})}"
    )

print("\nEnd-to-end smoke test PASSED!")
