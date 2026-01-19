import mediapipe as mp
from vision.inputter import feeding_frame
from vision.drawer import drawing
import json
from vision.to_Json import pose_result_to_dict

class StopSignal:
    def __init__(self):
        self.stop = False


def detect(mode):

    model_path = 'pose_landmarker_full.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # for livestream mode only
        print('pose landmarker result: {}'.format(result))

    if mode == 'video':
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)
    elif mode == 'live':
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)

    stop_signal = StopSignal()

    all_frames = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for mp_image, timestamp_ms, frame in feeding_frame(mode):
            if stop_signal.stop:
                break
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # drawing(pose_landmarker_result, frame, stop_signal)
            frame_data = pose_result_to_dict(pose_landmarker_result, timestamp_ms)
            all_frames.append(frame_data)

    with open("pose_video.json", "w") as f:
        json.dump(all_frames, f, indent=2)