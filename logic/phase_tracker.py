from logic.pose_loader import PhaseDetectionRule, PoseFrame
MIN_PHASE_SECONDS = 0.2
HYSTERESIS_DEG    = 5.0


class PhaseTracker:
    def __init__(self, frame: PoseFrame):
        self._frame          = frame
        self.confirmed_phase = frame.phase_order[0] if frame.phase_order else "active"
        self._candidate      = self.confirmed_phase
        self._candidate_since: float | None = None

    def update(self, angles: dict, timestamp: float) -> str:
        """
        timestamp — wall-clock seconds (pass timestamp_ms / 1000.0).
        Returns the current confirmed phase.
        """
        raw = self._detect_raw(angles)

        if raw != self._candidate:
            self._candidate       = raw
            self._candidate_since = timestamp
        elif raw != self.confirmed_phase:
            since   = self._candidate_since if self._candidate_since is not None else timestamp
            elapsed = timestamp - since
            if elapsed >= MIN_PHASE_SECONDS:
                self.confirmed_phase = raw

        return self.confirmed_phase

    def reset(self):
        self.confirmed_phase  = self._frame.phase_order[0] if self._frame.phase_order else "active"
        self._candidate       = self.confirmed_phase
        self._candidate_since = None

    def _detect_raw(self, angles: dict) -> str:
        for rule in self._frame.phase_detection:
            value = angles.get(rule.joint)
            if value is None:
                continue
            threshold = self._hysteresis_threshold(rule)
            if   rule.op == "gt"  and value >  threshold: return rule.phase
            elif rule.op == "lt"  and value <  threshold: return rule.phase
            elif rule.op == "gte" and value >= threshold: return rule.phase
            elif rule.op == "lte" and value <= threshold: return rule.phase
        return self.confirmed_phase

    def _hysteresis_threshold(self, rule: PhaseDetectionRule) -> float:
        already_in = (self.confirmed_phase == rule.phase)
        if rule.op in ("gt", "gte"):
            return rule.threshold - HYSTERESIS_DEG if already_in else rule.threshold + HYSTERESIS_DEG
        else:
            return rule.threshold + HYSTERESIS_DEG if already_in else rule.threshold - HYSTERESIS_DEG