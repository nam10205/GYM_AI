import json

def pose_result_to_dict(result, timestamp_ms=None):
    data = {
        "timestamp_ms": timestamp_ms,
        "poses": []
    }

    if not result.pose_landmarks:
        return data

    for i, pose in enumerate(result.pose_landmarks):
        pose_data = {
            "landmarks": [],
            "world_landmarks": []
        }

        for lm in pose:
            pose_data["landmarks"].append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        if result.pose_world_landmarks:
            for wlm in result.pose_world_landmarks[i]:
                pose_data["world_landmarks"].append({
                    "x": wlm.x,
                    "y": wlm.y,
                    "z": wlm.z
                })

        data["poses"].append(pose_data)

    return data
