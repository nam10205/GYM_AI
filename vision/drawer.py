import cv2
from config import Pose_Connections

def drawing(pose_landmarker_result, frame, stop_signal):
    if pose_landmarker_result.pose_landmarks:
        landmarks = pose_landmarker_result.pose_landmarks[0]
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        for a, b in Pose_Connections.POSE_CONNECTIONS:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("Pose Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        stop_signal.stop = True