import os

import boto3
from dotenv import load_dotenv

load_dotenv()
R2_id = os.getenv('R2_ACCESS_KEY_ID')
R2_pass = os.getenv('R2_SECRET_ACCESS_KEY')
R2_url = os.getenv('R2_URL')

s3 = boto3.client(
    "s3",
    endpoint_url="https://377cd1b9f45bc4ba22459f510676b999.r2.cloudflarestorage.com",
    aws_access_key_id=R2_id,
    aws_secret_access_key=R2_pass
)

def upload(video_path, key):
    s3.upload_file(video_path, "fitness-video", key)
    return f"{R2_url}/{key}"
