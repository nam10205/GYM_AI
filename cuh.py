from vision.detector import detect

def inference(video_path, exercise, mode, user_id):
    session_id = f"sess_of_{user_id}"
    output_path, llm_response = detect(mode, video_path, exercise, user_id, session_id)
    return output_path, llm_response