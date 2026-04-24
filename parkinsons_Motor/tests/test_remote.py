"""Simple connectivity test for the hosted remote environment."""

from parkinsons_Motor import ParkinsonsMotorEnv


def test_remote() -> None:
    print("Connecting to HF Space...")
    with ParkinsonsMotorEnv(base_url="https://virustechhacks-parkinsons-motor.hf.space").sync() as env:
        print("Connected! Resetting beta_suppression task...")
        result = env.reset(task_id="beta_suppression")
        print(f"Success! Remote beta_arv: {result.observation.beta_arv}")
        print(f"Success! Remote tremor_arv: {result.observation.tremor_arv}")


if __name__ == "__main__":
    test_remote()
