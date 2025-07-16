import cv2
import numpy as np
from tqdm import tqdm
from insightface_func.face_detect_crop_single import Face_detect_crop
from options.test_options import TestOptions

def extract_aligned_faces(opt):
    # Initialize InsightFace
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode='None')

    # Open input video
    cap = cv2.VideoCapture(opt.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {opt.video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = opt.crop_size, opt.crop_size

    # Prepare output video with proper settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        opt.output_path,
        fourcc,
        fps,
        (width, height),
        isColor=True  # Explicitly specify color video
    )

    # Process frames
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        detect_results = app.get(frame, opt.crop_size)
        if detect_results is not None:
            for face in detect_results[0]:
                # Ensure frame is proper BGR format
                if face.dtype != np.uint8:
                    face = (face * 255).astype(np.uint8)
                if len(face.shape) == 2:  # If grayscale
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
                out.write(face)

    # Proper cleanup
    cap.release()
    out.release()
    print(f"Successfully saved to: {opt.output_path}")

if __name__ == "__main__":
    opt = TestOptions().parse()
    if not hasattr(opt, 'output_path') or not opt.output_path:
        opt.output_path = "./aligned_faces.mp4"
    extract_aligned_faces(opt)