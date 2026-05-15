"""
pose_checker.py — single input/output entry point.

────────────────────────────────────────────────────────────
SETUP
────────────────────────────────────────────────────────────
    col     = db.collection("poses")
    checker = PoseChecker(col)

────────────────────────────────────────────────────────────
PER VIDEO
────────────────────────────────────────────────────────────
    checker.start_session(
        session_id = "sess_abc",
        user_id    = "user_42",
        exercise   = "squat",
    )

    # inside your frame loop:
    result = checker.process_frame(
        session_id   = "sess_abc",
        landmarks    = detection_result,   # PoseLandmarkerResult
        timestamp_ms = timestamp_ms,       # int(frame_idx / fps * 1000)
    )

    if result:                             # None until a rep completes
        result.rep_id                      # int
        result.phase                       # "bottom"
        result.duration_sec                # float
        result.messages                    # ["KNEE_CAVE_LEFT", "TOO_FAST", ...]
        result.is_correct                  # bool

    summary = checker.end_session("sess_abc")   # dict → pass to LLM
    checker.remove_session("sess_abc")

────────────────────────────────────────────────────────────
SUMMARY STRUCTURE (for LLM)
────────────────────────────────────────────────────────────
    {
      "session_id":       "sess_abc",
      "user_id":          "user_42",
      "exercise":         "squat",
      "started_at":       1717000000.0,
      "ended_at":         1717000120.0,
      "total_reps":       5,
      "correct_reps":     3,
      "avg_rep_time_sec": 3.2,
      "reps": [
        {
          "rep_id":        1,
          "duration_sec":  3.1,
          "is_correct":    false,
          "speed_warning": null,
          "phase_errors": {
            "descending": ["TORSO_LEAN"],
            "bottom":     ["KNEE_CAVE_LEFT", "SQUAT_DEEPER"]
          }
        },
        ...
      ]
    }
"""

from dataclasses import dataclass
from logic.pose_utils import get_joint_angles
from logic.pose_loader import PoseFrame, PoseLoader
from logic.phase_tracker import PhaseTracker
from logic.rep_tracker import RepTracker
from logic.session import session_cache

@dataclass
class FrameResult:
    rep_id:       int
    phase:        str
    duration_sec: float
    messages:     list[str]   # error codes for video overlay
    is_correct:   bool


class _ActiveSession:
    def __init__(self, exercise: str, frame: PoseFrame):
        self.exercise      = exercise
        self.frame         = frame
        self.phase_tracker = PhaseTracker(frame)
        self.rep_tracker   = RepTracker(frame)


class PoseChecker:
    def __init__(self, poses: dict):
        self._loader   = PoseLoader(poses)
        self._sessions: dict[str, _ActiveSession] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, session_id: str, user_id: str, exercise: str):
        frame = self._loader.get(exercise)
        self._sessions[session_id] = _ActiveSession(exercise, frame)
        session_cache.create(session_id, user_id, exercise)

    def end_session(self, session_id: str) -> dict:
        self._sessions.pop(session_id, None)
        return session_cache.close_session(session_id)

    def remove_session(self, session_id: str):
        self._sessions.pop(session_id, None)
        session_cache.remove_session(session_id)

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def process_frame(
        self,
        session_id:   str,
        landmarks,
        timestamp_ms: float,
    ) -> FrameResult | None:

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not started.")

        angles = get_joint_angles(landmarks)
        phase  = session.phase_tracker.update(angles, timestamp_ms / 1000.0)
        errors = self._check_rules(session.frame, phase, angles)

        rep_result = session.rep_tracker.update(phase, errors, timestamp_ms)
        if rep_result is None:
            return None

        session_cache.add_rep(session_id, rep_result)

        return FrameResult(
            rep_id       = rep_result.rep_id,
            phase        = phase,
            duration_sec = rep_result.duration_sec,
            messages     = rep_result.messages,
            is_correct   = rep_result.is_correct,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_summary(self, session_id: str) -> dict:
        return session_cache.get_summary(session_id)

    def reload_pose(self, exercise: str):
        self._loader.reload(exercise)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_rules(frame: PoseFrame, phase: str, angles: dict) -> list[str]:
        errors = []
        for rule in frame.rules_for(phase):
            value = angles.get(rule.joint)
            if value is None:
                continue
            msg = ""
            if value < rule.min_angle:
                msg = rule.too_low_msg
            elif value > rule.max_angle:
                msg = rule.too_high_msg
            if msg:   # empty string = no error, skip
                errors.append(msg)
        return errors