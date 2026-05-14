from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
from cuh import inference
from vision.uploader import upload
from contextlib import asynccontextmanager
import psycopg
from fastapi import BackgroundTasks

load_dotenv()

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("=========SERVER STARTED=========")

    model = 'loaded'

    yield

    print("=========SERVER STOPPED=========")

app = FastAPI(lifespan=lifespan)


def run_processing(video_path, exercise, mode, user_id):
    output_path = inference(video_path, exercise, mode, user_id)
    res_url = upload(output_path, f"result_for_{user_id}")
    return res_url


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
    
    # Start background processing
    background_tasks.add_task(
        process_in_background,
        data.video_url, data.exercise, data.mode,
        data.user_id, data.job_id, data.callback_url
    )

    # Return immediately - don't wait for processing
    return {"job_id": data.job_id}


def process_in_background(video_url, exercise, mode, user_id, job_id, callback_url):
    """Process video in background and callback to Django when done"""
    try:
        print(f"\n=== BACKGROUND PROCESSING START: {job_id} ===", flush=True)
        
        # Download video
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

        # Process video
        print(f"Processing video with exercise: {exercise}, mode: {mode}", flush=True)
        res = run_processing(video_path, exercise, mode, user_id)
        print(f"Processing complete. Result URL: {res}", flush=True)
        
        # Clean up
        os.remove(video_path)

        # Callback to Django
        print(f"Sending callback to: {callback_url}", flush=True)
        try:
            callback_response = requests.post(callback_url, json={"result_url": res}, timeout=30)
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