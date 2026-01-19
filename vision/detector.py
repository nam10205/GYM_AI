import mediapipe as mp
from vision.inputter import feeding_frame
from vision.drawer import drawing
import json
from vision.to_Json import pose_result_to_dict
import cv2
from config import Pose_Connections
import os

class StopSignal:
    def __init__(self):
        self.stop = False


def detect(mode, video_path = None):

    model_path = 'pose_landmarker_full.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    stop_signal = StopSignal()

    all_frames = []

    if mode == 'video':
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)

        with PoseLandmarker.create_from_options(options) as landmarker:
            for mp_image, timestamp_ms, frame in feeding_frame(mode, video_path):
                if stop_signal.stop:
                    break
                pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                drawing(pose_landmarker_result, frame, stop_signal)
                frame_data = pose_result_to_dict(pose_landmarker_result, timestamp_ms) # for json file
                all_frames.append(frame_data)

    elif mode == 'live':
        latest_result = None

        def print_result(result, output_image, timestamp_ms):
            nonlocal latest_result
            latest_result = result

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)

        with PoseLandmarker.create_from_options(options) as landmarker:
            for mp_image, timestamp_ms, frame in feeding_frame(mode):
                if stop_signal.stop:
                    break
                landmarker.detect_async(mp_image, timestamp_ms)
                if latest_result:
                    drawing(latest_result, frame, timestamp_ms)

    # file_path = os.path.join("json_outputs", "pose_video.json")
    # with open(file_path, "w") as f:
    #     json.dump(all_frames, f, indent=2)