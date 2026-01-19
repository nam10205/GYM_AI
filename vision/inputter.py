import cv2
import mediapipe as mp

def feeding_frame(mode):
    if mode == 'video':
        vid = cv2.VideoCapture('test/test.mp4')
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

    # elif mode == 'live':
