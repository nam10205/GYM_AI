import time
import cv2
from POSE_CONNECTIONS import POSE_CONNECTIONS

def drawing(pose_result, result, frame):
    # persistent state
    if not hasattr(drawing, "messages"):
        drawing.messages = []
        drawing.expire_time = 0

    now = time.time()

    # --- update messages when new result comes ---
    if result is not None:
        drawing.messages = result.messages
        drawing.expire_time = now + 18.0  # keep for 3 seconds

    # --- clear if expired ---
    if now > drawing.expire_time:
        drawing.messages = []

    h, w, _ = frame.shape

    if pose_result and pose_result.pose_landmarks:

        for landmarks in pose_result.pose_landmarks:

            # draw joints
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # draw skeleton lines
            for start_idx, end_idx in POSE_CONNECTIONS:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                x1 = int(start_lm.x * w)
                y1 = int(start_lm.y * h)

                x2 = int(end_lm.x * w)
                y2 = int(end_lm.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

    x, y = 10, 30
    for i, text in enumerate(drawing.messages):
        position = (x, y + i * 30)

        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame