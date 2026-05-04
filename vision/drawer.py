import time
import cv2

def drawing(result, frame):
    # persistent state
    if not hasattr(drawing, "messages"):
        drawing.messages = []
        drawing.expire_time = 0

    now = time.time()

    # --- update messages when new result comes ---
    if result is not None:
        drawing.messages = result.messages
        drawing.expire_time = now + 3.0  # keep for 3 seconds

    # --- clear if expired ---
    if now > drawing.expire_time:
        drawing.messages = []

    # --- draw current messages ---
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