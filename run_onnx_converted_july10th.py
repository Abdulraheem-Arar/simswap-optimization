import cv2
import torch
import numpy as np
import time
import onnxruntime as ort
from tqdm import tqdm
from torchvision import transforms

def load_id_vector(npy_path):
    id_vector = np.load(npy_path).astype(np.float32)
    norm = np.linalg.norm(id_vector, axis=1, keepdims=True)
    id_vector = id_vector / norm  # Normalize to unit vector
    return id_vector

def process_video_frames(video_path, session, id_vector, output_path, target_size=224):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
    
    transform = transforms.ToTensor()

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_start_time = time.time()

    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    with tqdm(total=frame_count, desc="Swapping frames (ONNX)") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # --- Preprocessing ---
            pre_start = time.time()
            if frame.shape[:2] != (target_size, target_size):
                frame = cv2.resize(frame, (target_size, target_size))

            # Check what resolution your model expects
            print(f"PyTorch model input shape: {frame.shape}")  # Should be near your test image
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).numpy().astype(np.float32)  # shape: (1,3,224,224)
            pre_time = time.time() - pre_start
            preprocess_times.append(pre_time)

            # --- Inference ---
            inf_start = time.time()
            inputs = {
                input_names[0]: frame_tensor,
                input_names[1]: id_vector
            }

            ort_outs = session.run([output_name], inputs)
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)

            # --- Postprocessing ---
            post_start = time.time()
            swapped = ort_outs[0][0].transpose(1, 2, 0)  # Convert from (3,224,224) to (224,224,3)
            swapped = (swapped * 255).clip(0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR))
            post_time = time.time() - post_start
            postprocess_times.append(post_time)

            pbar.update(1)

    total_time = time.time() - total_start_time
    out.release()
    video.release()

    if inference_times:
        avg_pre = 1000 * np.mean(preprocess_times)
        avg_inf = 1000 * np.mean(inference_times)
        avg_post = 1000 * np.mean(postprocess_times)
        avg_model_fps = 1 / np.mean(inference_times)
        avg_total_fps = frame_count / total_time

        print("\n--- Timing Results (ONNX) ---")
        print(f" Avg time per frame (preprocess): {avg_pre:.2f} ms")
        print(f" Avg time per frame (inference):  {avg_inf:.2f} ms")
        print(f" Avg time per frame (postprocess): {avg_post:.2f} ms")
        print(f"\n Inference-only FPS (model):      {avg_model_fps:.2f}")
        print(f" Total FPS (full pipeline):        {avg_total_fps:.2f}")
        print(f" Avg total time per frame:         {1000 * total_time / frame_count:.2f} ms")
    else:
        print("\n No frames were processed.")


if __name__ == '__main__':
    onnx_model_path = '/scratch/aa10947/SimSwap/simswap_224.onnx'
    video_path = '/scratch/aa10947/SimSwap/test_files/aligned_faces.mp4'
    id_path = '/scratch/aa10947/SimSwap/test_files/ironman_embedding_july9th.py.npy'
    output_path = '/scratch/aa10947/SimSwap/test_files/swapped_faces_onnx_july10th.mp4'

    # Load ONNX session
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)


    print("Model inputs:")
    for i, inp in enumerate(session.get_inputs()):
        print(f"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}")

    # Load ID embedding
    id_vector = load_id_vector(id_path)  # shape: (1, 512)

    process_video_frames(
        video_path=video_path,
        session=session,
        id_vector=id_vector,
        output_path=output_path,
        target_size=224
    )
    
    print("Actual provider being used:", session.get_providers())
