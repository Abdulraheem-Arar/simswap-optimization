import cv2

video_path = "/scratch/aa10947/SimSwap/demo_file/multi_people_1080p.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video!")
else:
    print("Video opened successfully.")
    ret, frame = cap.read()
    if ret:
        print("Successfully read one frame.")
    else:
        print("Failed to read a frame.")
cap.release()
