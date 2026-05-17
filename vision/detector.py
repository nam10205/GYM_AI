import subprocess
import cv2
import mediapipe as mp
from vision.inputter import feeding_frame
from vision.drawer import drawing
from logic.pose_checker import PoseChecker
from dotenv import load_dotenv
from CONSTANTS import POSES
from vision.AI_summary import get_summary

load_dotenv()


def detect(mode, video_path, exercise1, user_id1, session_id1):

    model_path = 'model/pose_landmarker_full.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    # for saving result
    output_path = f"/tmp/tmp_res_of_{session_id1}.mp4"
    real_output_path = f"/tmp/result_for_{user_id1}.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # safe fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    if mode == 'video':
        checker = PoseChecker(POSES)
        checker.start_session(
            session_id=session_id1,
            user_id=user_id1,
            exercise=exercise1
        )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)

        if not out.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        with PoseLandmarker.create_from_options(options) as landmarker:
            for mp_image, timestamp_ms, frame in feeding_frame(mode, video_path):
                pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                checking_result = checker.process_frame(
                    session_id=session_id1,
                    landmarks=pose_landmarker_result,
                    timestamp_ms=timestamp_ms
                )
                drawn_frame = drawing(pose_landmarker_result, checking_result, frame)
                out.write(drawn_frame)
        cap.release()
        out.release()

        subprocess.run([
            "ffmpeg", "-i", output_path,
            "-vcodec", "libx264",
            "-preset", "ultrafast",  # much less CPU, slightly larger file
            "-crf", "28",  # lower quality but faster (23 is default, 28 is fine for web)
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-threads", "1",  # limit to 1 thread to avoid Render CPU spike
            "-y", real_output_path
        ], check=True)

        summary = checker.end_session(session_id1)
        llm_response = get_summary(summary)
        checker.remove_session(session_id1)

    return real_output_path, llm_response

    # elif mode == 'live':
    #     latest_result = None
    #
    #     def print_result(result, output_image, timestamp_ms):
    #         nonlocal latest_result
    #         latest_result = result
    #
    #     options = PoseLandmarkerOptions(
    #         base_options=BaseOptions(model_asset_path=model_path),
    #         running_mode=VisionRunningMode.LIVE_STREAM,
    #         result_callback=print_result)
    #
    #     with PoseLandmarker.create_from_options(options) as landmarker:
    #         for mp_image, timestamp_ms, frame in feeding_frame(mode):
    #             if stop_signal.stop:
    #                 break
    #             landmarker.detect_async(mp_image, timestamp_ms)
    #             if latest_result:
    #                 drawing(latest_result, frame, timestamp_ms)
