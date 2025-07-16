import cv2
import numpy as np
from tqdm import tqdm
from insightface_func.face_detect_crop_single import Face_detect_crop
from options.test_options import TestOptions
import onnxruntime
import os

# Patch ONNXRuntime to prefer GPU
original_InferenceSession = onnxruntime.InferenceSession
def patched_InferenceSession(*args, **kwargs):
    if 'providers' not in kwargs:
        kwargs['providers'] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return original_InferenceSession(*args, **kwargs)
onnxruntime.InferenceSession = patched_InferenceSession

def extract_aligned_faces(opt):
    """
    Extract the most confident aligned face per frame from video using InsightFace (single-face mode).
    """
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode='None')

    cap = cv2.VideoCapture(opt.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {opt.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = height = opt.crop_size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opt.output_path, fourcc, fps, (width, height), isColor=True)

    written_frames = 0  # ✅ Track number of successfully written frames

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Extracting faces"):
        ret, frame = cap.read()
        if not ret:
            break

        detect_results = app.get(frame, opt.crop_size)
        if detect_results is None:
            out.write(np.zeros((height, width, 3), dtype=np.uint8))
            continue

        face_crop_list, bboxes = detect_results

        if len(face_crop_list) == 0:
            out.write(np.zeros((height, width, 3), dtype=np.uint8))
            continue

        # Get the most confident face (only one expected in single-face mode)
        face_crop = face_crop_list[0]

        if face_crop.dtype != np.uint8:
            face_crop = (face_crop * 255).astype(np.uint8)
        if len(face_crop.shape) == 2:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)

        # ✅ Ensure resolution matches expected video writer dimensions
        if face_crop.shape[:2] != (height, width):
            face_crop = cv2.resize(face_crop, (width, height))

        out.write(face_crop)
        written_frames += 1  # ✅ Count it

    cap.release()
    out.release()

    # ✅ Report how many frames were written
    print(f"Successfully saved aligned faces to: {opt.output_path}")
    print(f"Total frames written: {written_frames}")

if __name__ == "__main__":
    opt = TestOptions().parse()
    if not hasattr(opt, 'output_path') or not opt.output_path:
        opt.output_path = "./aligned_faces.mp4"
    extract_aligned_faces(opt)
