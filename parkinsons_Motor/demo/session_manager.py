"""Tiny in-memory session registry for the live viewer demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .agent_runner import DemoConfig


@dataclass
class DemoSession:
    session_id: str
    config: DemoConfig
    stopped: bool = False
    events_sent: int = 0


@dataclass
class DemoSessionManager:
    sessions: dict[str, DemoSession] = field(default_factory=dict)

    def create(self, config: DemoConfig) -> DemoSession:
        session = DemoSession(session_id=str(uuid4()), config=config)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> DemoSession | None:
        return self.sessions.get(session_id)

    def stop(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        session.stopped = True
        return True

    def remove(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

