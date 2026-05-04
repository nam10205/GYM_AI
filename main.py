from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from cuh import inference
from vision.uploader import upload

def run_processing(video_path, exercise, mode, user_id):
    output_path = inference(mode, video_path, exercise, user_id)
    res_url = upload(output_path, f"result_for_{user_id}")
    return res_url
app = FastAPI()

class ProcessRequest(BaseModel):
    video_url: str
    exercise: str
    mode: str
    user_id: int

@app.post("/process")
def process_video(data: ProcessRequest):
    print("=== HIT PROCESS ===")
    video_path = f"/tmp/{data.user_id}_input.mp4"
    response = requests.get(data.video_url, stream=True)

    with open(video_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    res = run_processing(video_path, data.exercise, data.mode, data.user_id)

    os.remove(video_path)

    return {
        "result_url": res
    }
