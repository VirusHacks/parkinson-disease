"""Compatibility entrypoint for environment action and observation models."""

try:
    from ..models import *  # noqa: F401,F403
except ImportError:
    from models import *  # noqa: F401,F403
