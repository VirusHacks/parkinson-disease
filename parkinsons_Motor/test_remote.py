"""Backward-compatible entrypoint for the remote smoke test."""

from parkinsons_Motor.tests.test_remote import test_remote


if __name__ == "__main__":
    test_remote()
