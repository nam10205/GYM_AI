"""
pose_utils.py — angle math + joint extraction from MediaPipe PoseLandmarkerResult
"""
import numpy as np

LANDMARK = {
    "left_shoulder":    11, "right_shoulder":   12,
    "left_elbow":       13, "right_elbow":      14,
    "left_wrist":       15, "right_wrist":      16,
    "left_hip":         23, "right_hip":        24,
    "left_knee":        25, "right_knee":       26,
    "left_ankle":       27, "right_ankle":      28,
    "left_foot_index":  31, "right_foot_index": 32,
}


def _extract_landmarks(result):
    """
    Normalise PoseLandmarkerResult into a flat list of landmarks.
      - PoseLandmarkerResult  → result.pose_landmarks[0]  (first person)
      - flat list/tuple       → used as-is (legacy fallback)
    """
    if hasattr(result, "pose_landmarks"):
        poses = result.pose_landmarks
        if not poses:
            return None
        return poses[0]
    return result


def _coord(result, name):
    landmarks = _extract_landmarks(result)
    if landmarks is None:
        return None
    lm = landmarks[LANDMARK[name]]
    if getattr(lm, "visibility", 1.0) < 0.5:
        return None
    if getattr(lm, "presence", 1.0) < 0.5:
        return None
    return np.array([lm.x, lm.y, lm.z])


def _angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _torso(result):
    pts = [_coord(result, n) for n in
           ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(p is None for p in pts):
        return None
    ls, rs, lh, rh = pts
    vec = (ls + rs) / 2 - (lh + rh) / 2
    cos = np.dot(vec, np.array([0, -1, 0])) / (np.linalg.norm(vec) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def get_joint_angles(result) -> dict:
    """
    Pass in a PoseLandmarkerResult or a flat landmark list.
    Returns {joint_name: angle_degrees}. None if landmark not visible.
    """
    def safe(a, b, c):
        pts = [_coord(result, x) for x in (a, b, c)]
        return _angle(*pts) if all(p is not None for p in pts) else None

    return {
        "left_elbow":     safe("left_shoulder",  "left_elbow",    "left_wrist"),
        "right_elbow":    safe("right_shoulder", "right_elbow",   "right_wrist"),
        "left_shoulder":  safe("left_elbow",     "left_shoulder", "left_hip"),
        "right_shoulder": safe("right_elbow",    "right_shoulder","right_hip"),
        "left_hip":       safe("left_shoulder",  "left_hip",      "left_knee"),
        "right_hip":      safe("right_shoulder", "right_hip",     "right_knee"),
        "left_knee":      safe("left_hip",       "left_knee",     "left_ankle"),
        "right_knee":     safe("right_hip",      "right_knee",    "right_ankle"),
        "left_ankle":     safe("left_knee",      "left_ankle",    "left_foot_index"),
        "right_ankle":    safe("right_knee",     "right_ankle",   "right_foot_index"),
        "torso":          _torso(result),
    }