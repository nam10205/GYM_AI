from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
from cuh import inference
from vision.uploader import upload
from contextlib import asynccontextmanager
import psycopg

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
async def process_video(data: ProcessRequest, request: Request):
    print("=== HIT PROCESS ===")

    # raw request body
    raw_body = await request.body()
    print("RAW BODY:", raw_body.decode())

    # parsed pydantic object
    print("PARSED DATA:", data)

    video_path = f"/tmp/{data.user_id}_input.mp4"

    response = requests.get(data.video_url, stream=True)

    with open(video_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    res = run_processing(
        video_path,
        data.exercise,
        data.mode,
        data.user_id
    )

    os.remove(video_path)

    try:
        requests.post(data.callback_url, json={"result_url": res})
    except Exception as e:
        print(f"Callback failed: {e}")

    return {"status": "processing", "job_id": data.job_id}