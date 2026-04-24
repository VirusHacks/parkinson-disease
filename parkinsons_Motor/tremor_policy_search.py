"""Backward-compatible entrypoint for tremor policy search."""

from parkinsons_Motor.evaluation.tremor_policy_search import *  # noqa: F401,F403

if __name__ == "__main__":
    search()
