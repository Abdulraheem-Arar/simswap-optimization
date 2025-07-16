'''
Pure PyTorch Face Swapping (No ONNX Dependency)
- For pre-cropped face videos only
- Bypasses InsightFace detection
'''

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

def process_video_frames(video_path, model, id_vector, output_path, target_size=224):
    # Video setup
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Failed to open video at {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Processing loop
    frame_times = []
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame.shape[:2] != (target_size, target_size):
                frame = cv2.resize(frame, (target_size, target_size))

            start_time = time.time()

            # Convert to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).cuda()

            # Face swap
            with torch.no_grad():
                swapped = model(None, frame_tensor, id_vector, None, True)[0]
                swapped = swapped.squeeze().cpu().numpy().transpose(1, 2, 0)
                swapped = (swapped * 255).clip(0, 255).astype(np.uint8)

            # Save result
            out.write(cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR))
            frame_times.append(time.time() - start_time)

            pbar.update(1)

    # Print stats
    if frame_times:
        avg_fps = 1 / np.mean(frame_times)
        print(f"\nAverage FPS: {avg_fps:.2f}")
        print(f"Avg time per frame: {1000 * np.mean(frame_times):.2f} ms")
    else:
        print("\nNo frames were processed. Please check your input video.")

    video.release()
    out.release()

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.which_epoch = 'latest'

    model = load_model(opt)
    id_vector = torch.from_numpy(np.load('ironman_embedding.npy')).cuda()
    id_vector = id_vector / id_vector.norm(dim=1, keepdim=True)

    process_video_frames(
        '/scratch/aa10947/SimSwap/test_files/aligned_faces.mp4',
        model,
        id_vector,
        'july8th_video.mp4'
    )
