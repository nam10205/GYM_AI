import os
import cv2
import mediapipe as mp
from vision.inputter import feeding_frame
from vision.drawer import drawing
from arango import ArangoClient
from logic.pose_checker import PoseChecker
from dotenv import load_dotenv

import psycopg
from psycopg.rows import dict_row

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_PW = os.getenv('SUPABASE_PW')
SUPABASE_USER = os.getenv('SUPABASE_USER')
SUPABASE_DB = os.getenv('SUPABASE_DB')

def detect(mode, video_path, exercise1, user_id1, session_id1):

    conn = psycopg.connect(
        host=SUPABASE_URL,
        port=5432,
        dbname=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PW,
        sslmode="require",
        row_factory=dict_row,
    )
    print("Connected successfully", flush=True)

    POSES = {}

    with conn.cursor() as cur:
        cur.execute("SELECT data FROM poses")

        rows = cur.fetchall()
        print(f"Total rows fetched: {len(rows)}", flush=True)

        # print first row
        if rows:
            print("First row:", flush=True)
            print(rows[0], flush=True)

        for row in rows:
            pose = row["data"]
            POSES[pose["_key"]] = pose

    print(f"POSES loaded: {len(POSES)}",flush=True)

    # print one sample item
    if POSES:
        first_key = next(iter(POSES))
        print("Sample key:", first_key, flush=True)
        print("Sample value:", POSES[first_key], flush=True)

    model_path = 'model/pose_landmarker_full.task'
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    # for saving result
    output_path = f"/tmp/res_of_{session_id1}.mp4"
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
                drawn_frame = drawing(checking_result, frame)
                out.write(drawn_frame)
        cap.release()
        out.release()

        checker.remove_session(session_id1)

    return output_path

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
