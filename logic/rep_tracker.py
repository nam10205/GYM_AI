from dataclasses import dataclass, field
from logic.pose_loader import PoseFrame, PhaseTiming

SPEED_REPEAT_THRESHOLD = 2      # consecutive reps before surfacing a speed warning
ERROR_MIN_DURATION_MS  = 200.0  # error must persist this long to be registered


@dataclass
class PhaseResult:
    phase:        str
    duration_sec: float
    errors:       list[str]

@dataclass
class RepResult:
    rep_id:        int
    duration_sec:  float
    phases:        list[PhaseResult]
    speed_warning: str | None   # "TOO_FAST" | "TOO_SLOW" | None
    is_correct:    bool

    @property
    def messages(self) -> list[str]:
        msgs = []
        for p in self.phases:
            msgs.extend(p.errors)
        if self.speed_warning:
            msgs.append(self.speed_warning)
        return msgs


class RepTracker:
    def __init__(self, frame: PoseFrame):
        self._frame      = frame
        self._rep_phases = frame.rep_phases

        self._rep_count       = 0
        self._rep_start_ms:   float | None = None
        self._phase_start_ms: float | None = None
        self._current_phase:  str | None   = None

        self._visited_phases:          list[str]         = []
        self._completed_phase_results: list[PhaseResult] = []

        # Per-error streak tracking within the current phase.
        # error_code -> timestamp_ms when it first appeared in current streak
        self._error_streak_start: dict[str, float] = {}

        # Streak tracking for speed: key -> {"dir": str|None, "count": int}
        self._speed_streak: dict[str, dict] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        phase: str,
        errors: list[str],
        timestamp_ms: float,
    ) -> RepResult | None:

        if phase != self._current_phase:
            self._on_phase_exit(self._current_phase, timestamp_ms)
            self._on_phase_enter(phase, timestamp_ms)

        # Update per-error streaks
        active_errors = set(errors)
        for code in list(self._error_streak_start.keys()):
            if code not in active_errors:
                # Error disappeared — reset its streak
                del self._error_streak_start[code]
        for code in active_errors:
            if code not in self._error_streak_start:
                self._error_streak_start[code] = timestamp_ms

        # Check rep complete every frame once all phases visited
        if (self._rep_start_ms is not None
                and self._visited_phases == self._rep_phases
                and phase == self._rep_phases[-1]
                and self._phase_start_ms is not None):
            elapsed_last = (timestamp_ms - self._phase_start_ms) / 1000.0
            timing       = self._frame.timing_for(phase)
            min_hold     = timing.min_sec if timing else 0.0
            if elapsed_last >= min_hold:
                return self._close_rep(timestamp_ms)

        return None

    def reset(self):
        self._rep_count               = 0
        self._rep_start_ms            = None
        self._phase_start_ms          = None
        self._current_phase           = None
        self._visited_phases          = []
        self._completed_phase_results = []
        self._error_streak_start      = {}
        self._speed_streak            = {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_phase_enter(self, phase: str, timestamp_ms: float):
        self._current_phase      = phase
        self._phase_start_ms     = timestamp_ms
        self._error_streak_start = {}   # reset error streaks for the new phase

        if self._rep_phases and phase == self._rep_phases[0] and self._rep_start_ms is None:
            self._rep_start_ms            = timestamp_ms
            self._visited_phases          = []
            self._completed_phase_results = []

        if phase in self._rep_phases:
            expected_idx = len(self._visited_phases)
            if expected_idx < len(self._rep_phases) and self._rep_phases[expected_idx] == phase:
                self._visited_phases.append(phase)

    def _on_phase_exit(self, phase: str | None, timestamp_ms: float):
        if phase is None or self._phase_start_ms is None:
            return
        if phase not in self._rep_phases:
            return

        duration_sec = (timestamp_ms - self._phase_start_ms) / 1000.0

        # Only register errors that persisted for ≥ ERROR_MIN_DURATION_MS
        confirmed = [
            code for code, start in self._error_streak_start.items()
            if (timestamp_ms - start) >= ERROR_MIN_DURATION_MS
        ]
        # Deduplicate while preserving insertion order
        errors = list(dict.fromkeys(confirmed))

        self._completed_phase_results.append(PhaseResult(
            phase        = phase,
            duration_sec = round(duration_sec, 3),
            errors       = errors,
        ))

    def _close_rep(self, timestamp_ms: float) -> RepResult:
        self._on_phase_exit(self._current_phase, timestamp_ms)

        rep_sec    = (timestamp_ms - self._rep_start_ms) / 1000.0
        direction  = self._speed_direction(rep_sec, self._frame.rep_timing)
        speed_code = self._resolve_speed_code("rep:overall", direction)

        self._rep_count += 1
        result = RepResult(
            rep_id        = self._rep_count,
            duration_sec  = round(rep_sec, 3),
            phases        = list(self._completed_phase_results),
            speed_warning = speed_code,
            is_correct    = (
                speed_code is None and
                all(not p.errors for p in self._completed_phase_results)
            ),
        )

        self._rep_start_ms            = None
        self._visited_phases          = []
        self._completed_phase_results = []
        self._error_streak_start      = {}
        return result

    # ── Speed helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _speed_direction(duration_sec: float, timing: PhaseTiming | None) -> str | None:
        if timing is None:
            return None
        if duration_sec < timing.min_sec:
            return "too_fast"
        if duration_sec > timing.max_sec:
            return "too_slow"
        return None

    def _resolve_speed_code(self, key: str, direction: str | None) -> str | None:
        streak = self._speed_streak.setdefault(key, {"dir": None, "count": 0})

        if direction is None:
            streak["dir"]   = None
            streak["count"] = 0
            return None

        if streak["dir"] != direction:
            streak["dir"]   = direction
            streak["count"] = 0

        streak["count"] += 1

        if streak["count"] >= SPEED_REPEAT_THRESHOLD:
            return "TOO_FAST" if direction == "too_fast" else "TOO_SLOW"
        return None