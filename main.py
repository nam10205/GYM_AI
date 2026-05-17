from CONSTANTS import POSES
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
from vision.uploader import upload
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks
import psycopg
from psycopg.rows import dict_row

load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_PW = os.getenv('SUPABASE_PW')
SUPABASE_USER = os.getenv('SUPABASE_USER')
SUPABASE_DB = os.getenv('SUPABASE_DB')

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("=========SERVER STARTED=========")

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


    with conn.cursor() as cur:
        cur.execute("SELECT data FROM poses")

        rows = cur.fetchall()
        print(f"Total rows fetched: {len(rows)}", flush=True)
        for row in rows:
            pose = row["data"]
            POSES[pose["_key"]] = pose

    print(f"POSES loaded: {len(POSES)}", flush=True)
    conn.close()

    model = 'loaded'

    yield

    print("=========SERVER STOPPED=========")

from cuh import inference

app = FastAPI(lifespan=lifespan)


def run_processing(video_path, exercise, mode, user_id):
    output_path, llm_response = inference(video_path, exercise, mode, user_id)
    res_url = upload(output_path, f"result_for_{user_id}")
    return res_url, llm_response


class ProcessRequest(BaseModel):
    video_url: str
    exercise: str
    mode: str
    user_id: str
    job_id: str
    callback_url: str


@app.post("/process")
async def process_video(data: ProcessRequest, background_tasks: BackgroundTasks):
    print("=== HIT PROCESS ===")
    print(f"Job ID: {data.job_id}")
    print(f"Exercise: {data.exercise}, Mode: {data.mode}")
    
    background_tasks.add_task(
        process_in_background,
        data.video_url, data.exercise, data.mode,
        data.user_id, data.job_id, data.callback_url
    )

    return {"job_id": data.job_id}


def process_in_background(video_url, exercise, mode, user_id, job_id, callback_url):
    try:

        video_path = f"/tmp/{user_id}_input.mp4"
        print(f"Downloading video from: {video_url}", flush=True)
        
        try:
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()
            print(f"Video response received, status: {response.status_code}", flush=True)
        except Exception as e:
            print(f"ERROR downloading video: {str(e)}", flush=True)
            raise
        
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Video downloaded to: {video_path}", flush=True)

        print(f"Processing video with exercise: {exercise}, mode: {mode}", flush=True)
        res, llm = run_processing(video_path, exercise, mode, user_id)
        print(f"Processing complete. Result URL: {res}", flush=True)
        
        os.remove(video_path)

        print(f"Sending callback to: {callback_url}", flush=True)
        try:
            callback_response = requests.post(callback_url, json={"result_url": res, "llm_response": llm}, timeout=30)
            print(f"Callback response status: {callback_response.status_code}", flush=True)
            print(f"Callback response body: {callback_response.text}", flush=True)
            callback_response.raise_for_status()
            print(f"=== BACKGROUND PROCESSING DONE: {job_id} ===\n", flush=True)
        except Exception as callback_err:
            print(f"ERROR sending callback: {str(callback_err)}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
        
    except Exception as e:
        print(f"Background processing failed for {job_id}: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)