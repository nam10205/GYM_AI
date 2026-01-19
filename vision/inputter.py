import cv2
import mediapipe as mp
import time

def feeding_frame(mode, video_path = None):
    if mode == 'video':
        if video_path is None:
            raise ValueError("video path required")
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        while True:
            read, frame = vid.read()
            if not read:
                return 0
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
            yield mp_image, timestamp_ms, frame

    elif mode == 'live':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Can not open webcam')
        while True:
            read, frame = cap.read()
            if not read:
                return 0
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
            yield mp_image, timestamp_ms, frame
