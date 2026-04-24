"""Compatibility entrypoint for the environment calibration layer."""

try:
    from ..brain_calibrator import *  # noqa: F401,F403
except ImportError:
    from brain_calibrator import *  # noqa: F401,F403
