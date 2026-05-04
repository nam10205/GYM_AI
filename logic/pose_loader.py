"""
pose_loader.py — loads pose documents from ArangoDB.

DB document structure (collection: "poses"):
{
  "_key": "squat",
  "phase_order": ["standing", "descending", "bottom"],
  "phase_detection": [
    {"joint": "left_knee", "op": "gt",  "threshold": 150, "phase": "standing"},
    {"joint": "left_knee", "op": "lt",  "threshold": 100, "phase": "bottom"},
    {"joint": "left_knee", "op": "lte", "threshold": 150, "phase": "descending"}
  ],
  "phase_timing": {
    "descending": {"min_sec": 0.5, "max_sec": 3.0},
    "bottom":     {"min_sec": 0.1, "max_sec": 1.5},
    "standing":   {"min_sec": 0.3, "max_sec": 3.0}
  },
  "rep_phases":  ["descending", "bottom", "standing"],
  "rep_timing":  {"min_sec": 1.5, "max_sec": 6.0},
  "phases": {
    "bottom": {
      "rules": [
        {
          "joint":        "left_knee",
          "min":          80,
          "max":          100,
          "too_low_msg":  "KNEE_CAVE_LEFT",
          "too_high_msg": "SQUAT_DEEPER"
        }
      ]
    }
  }
}

Rules:
- phase_detection thresholds get ±5° hysteresis from PhaseTracker automatically.
- too_low_msg / too_high_msg are short error codes shown on video and stored in summary.
- phase_timing / rep_timing define speed bounds for warnings.
"""

from dataclasses import dataclass


@dataclass
class JointRule:
    joint:        str
    min_angle:    float
    max_angle:    float
    too_low_msg:  str   # short code — shown on video overlay AND stored in summary
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
    def __init__(self, collection):
        self._col   = collection
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
        doc = self._col.get(key)
        if doc is None:
            raise ValueError(f"Pose '{key}' not found in DB.")

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