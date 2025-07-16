import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions

def load_model(opt):
    model = create_model(opt)
    model.eval()
    return model

def load_id_vector(npy_path):
    id_vector = torch.from_numpy(np.load(npy_path)).cuda()
    id_vector = id_vector / id_vector.norm(dim=1, keepdim=True)  # Just to be safe
    return id_vector

def process_video_frames(video_path, model, id_vector, output_path, target_size=224):
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

    with tqdm(total=frame_count, desc="Swapping frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # --- Preprocessing ---
            pre_start = time.time()
            if frame.shape[:2] != (target_size, target_size):
                frame = cv2.resize(frame, (target_size, target_size))

            # Check what resolution your model expects
            # print(f"PyTorch model input shape: {frame.shape}")  # Should be near your test image

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).cuda()
            torch.cuda.synchronize()  # Ensure .cuda() completes before timing ends
            pre_time = time.time() - pre_start
            preprocess_times.append(pre_time)

            # --- Inference ---
            torch.cuda.synchronize()
            inf_start = time.time()
            with torch.no_grad():
                swapped = model(None, frame_tensor, id_vector, id_vector, True)[0]
            torch.cuda.synchronize()
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)

            # --- Postprocessing ---
            post_start = time.time()
            swapped = swapped.squeeze().cpu().numpy().transpose(1, 2, 0)
            torch.cuda.synchronize()  # Ensure .cpu() is done
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

        print("\n--- Timing Results ---")
        print(f" Avg time per frame (preprocess): {avg_pre:.2f} ms")
        print(f" Avg time per frame (inference):  {avg_inf:.2f} ms")
        print(f" Avg time per frame (postprocess): {avg_post:.2f} ms")
        print(f"\n Inference-only FPS (model):      {avg_model_fps:.2f}")
        print(f" Total FPS (full pipeline):        {avg_total_fps:.2f}")
        print(f" Avg total time per frame:         {1000 * total_time / frame_count:.2f} ms")
    else:
        print("\n No frames were processed.")



if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.which_epoch = 'latest'

    model = load_model(opt)

    # Load precomputed ID embedding from .npy
    id_vector = load_id_vector('/scratch/aa10947/SimSwap/test_files/ironman_embedding_july9th.py.npy')

    process_video_frames(
        video_path=opt.video_path,       # e.g., ./aligned_faces.mp4
        model=model,
        id_vector=id_vector,
        output_path=opt.output_path,     # e.g., ./swapped_faces.mp4
        target_size=opt.crop_size        # e.g., 224
    )
