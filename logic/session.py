"""
session.py — in-memory session cache.

Stores everything needed for an LLM summary at the end of a session.
Error codes (not full messages) are stored — the LLM uses these codes
to generate its own natural-language feedback.

    from session import session_cache

    session_cache.create(session_id, user_id, exercise)
    session_cache.add_rep(session_id, rep_result)
    summary = session_cache.close_session(session_id)   # -> dict for LLM
    session_cache.remove_session(session_id)
"""

import time
from dataclasses import dataclass, field
from logic.rep_tracker import RepResult

@dataclass
class RepRecord:
    rep_id:        int
    duration_sec:  float
    is_correct:    bool
    speed_warning: str | None                  # "TOO_FAST" | "TOO_SLOW" | None
    phase_errors:  dict[str, list[str]]        # {phase_name: [error_codes]}


@dataclass
class SessionData:
    session_id: str
    user_id:    str
    exercise:   str
    started_at: float = field(default_factory=time.time)
    ended_at:   float | None = None
    reps:       list[RepRecord] = field(default_factory=list)

    def add_rep(self, result: RepResult):
        self.reps.append(RepRecord(
            rep_id        = result.rep_id,
            duration_sec  = result.duration_sec,
            is_correct    = result.is_correct,
            speed_warning = result.speed_warning,
            phase_errors  = {p.phase: p.errors for p in result.phases if p.errors},
        ))

    def to_summary(self) -> dict:
        """
        Structured dict passed to the LLM.
        Contains only codes — no verbose strings.
        """
        total   = len(self.reps)
        correct = sum(1 for r in self.reps if r.is_correct)
        avg_dur = round(sum(r.duration_sec for r in self.reps) / total, 2) if total else 0

        return {
            "session_id":       self.session_id,
            "user_id":          self.user_id,
            "exercise":         self.exercise,
            "started_at":       self.started_at,
            "ended_at":         self.ended_at or time.time(),
            "total_reps":       total,
            "correct_reps":     correct,
            "avg_rep_time_sec": avg_dur,
            "reps": [
                {
                    "rep_id":        r.rep_id,
                    "duration_sec":  r.duration_sec,
                    "is_correct":    r.is_correct,
                    "speed_warning": r.speed_warning,
                    "phase_errors":  r.phase_errors,
                }
                for r in self.reps
            ],
        }


class SessionCache:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    def create(self, session_id: str, user_id: str, exercise: str) -> SessionData:
        if session_id in self._sessions:
            raise ValueError(f"Session '{session_id}' already exists.")
        data = SessionData(session_id=session_id, user_id=user_id, exercise=exercise)
        self._sessions[session_id] = data
        return data

    def get(self, session_id: str) -> SessionData:
        s = self._sessions.get(session_id)
        if s is None:
            raise ValueError(f"Session '{session_id}' not found.")
        return s

    def add_rep(self, session_id: str, result: RepResult):
        self.get(session_id).add_rep(result)

    def get_summary(self, session_id: str) -> dict:
        return self.get(session_id).to_summary()

    def close_session(self, session_id: str) -> dict:
        s = self.get(session_id)
        s.ended_at = time.time()
        return s.to_summary()

    def remove_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        return [sid for sid, s in self._sessions.items() if s.ended_at is None]


# Module-level singleton
session_cache = SessionCache()