from vision.detector import detect

# mode = input('choose mode: ')
# video = None
# if mode == 'video':
#
#     video = input('video path: ')
# detect(mode, video, 'squat', '123', 'ses123')

def inference(video_path, exercise, mode, user_id):
    session_id = f"sess_of_{user_id}"
    output_path = detect(mode, video_path, exercise, user_id, session_id)
    return output_path