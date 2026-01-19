from vision.detector import detect

mode = input('choose mode: ')
video = None
if mode == 'video':
    video = input('video path: ')
detect(mode, video)