"""Compatibility entrypoint for patient-profile definitions."""

try:
    from ..patient_profiles import *  # noqa: F401,F403
except ImportError:
    from patient_profiles import *  # noqa: F401,F403
