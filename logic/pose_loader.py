from dataclasses import dataclass


@dataclass
class JointRule:
    joint:        str
    min_angle:    float
    max_angle:    float
    too_low_msg:  str
    too_high_msg: str


@dataclass
class PhaseDetectionRule:
    joint:     str
    op:        str    # "gt" | "lt" | "gte" | "lte"
    threshold: float
    phase:     str


@dataclass
class PhaseTiming:
    min_sec: float
    max_sec: float


@dataclass
class PoseFrame:
    key:             str
    phase_order:     list[str]
    phase_detection: list[PhaseDetectionRule]
    phase_timing:    dict[str, PhaseTiming]
    rep_phases:      list[str]
    rep_timing:      PhaseTiming
    phases:          dict[str, list[JointRule]]

    def rules_for(self, phase: str) -> list[JointRule]:
        return self.phases.get(phase, [])

    def timing_for(self, phase: str) -> PhaseTiming | None:
        return self.phase_timing.get(phase)


class PoseLoader:
    def __init__(self, poses: dict):
        self._poses = poses
        self._cache: dict[str, PoseFrame] = {}

    def get(self, key: str) -> PoseFrame:
        if key not in self._cache:
            self._cache[key] = self._fetch(key)
        return self._cache[key]

    def reload(self, key: str):
        self._cache.pop(key, None)

    def reload_all(self):
        self._cache.clear()

    def _fetch(self, key: str) -> PoseFrame:
        doc = self._poses.get(key)
        if doc is None:
            raise ValueError(f"Pose '{key}' not found.")

        phase_detection = [
            PhaseDetectionRule(
                joint=r["joint"], op=r["op"],
                threshold=r["threshold"], phase=r["phase"],
            )
            for r in doc.get("phase_detection", [])
        ]

        phase_timing = {
            name: PhaseTiming(min_sec=t["min_sec"], max_sec=t["max_sec"])
            for name, t in doc.get("phase_timing", {}).items()
        }

        rt = doc.get("rep_timing", {"min_sec": 1.0, "max_sec": 10.0})
        rep_timing = PhaseTiming(min_sec=rt["min_sec"], max_sec=rt["max_sec"])

        phases = {
            phase_name: [
                JointRule(
                    joint        = r["joint"],
                    min_angle    = r["min"],
                    max_angle    = r["max"],
                    too_low_msg  = r["too_low_msg"],
                    too_high_msg = r["too_high_msg"],
                )
                for r in phase_data.get("rules", [])
            ]
            for phase_name, phase_data in doc.get("phases", {}).items()
        }

        return PoseFrame(
            key             = doc["_key"],
            phase_order     = doc.get("phase_order", []),
            phase_detection = phase_detection,
            phase_timing    = phase_timing,
            rep_phases      = doc.get("rep_phases", []),
            rep_timing      = rep_timing,
            phases          = phases,
        )