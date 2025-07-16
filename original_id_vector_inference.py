import cv2
import torch
import numpy as np
import time
import os
from tqdm import tqdm
from models.models import create_model
from options.test_options import TestOptions

def verify_video(video_path):
    """Check if video file is readable"""
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def save_frames_as_images(output_dir, frames):
    """Save frames as individual images"""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:05d}.png"), frame)

def get_video_writer(output_path, fps, width, height):
    """Try multiple codecs to find one that works"""
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
        ('X264', cv2.VideoWriter_fourcc(*'X264')),  # Alternative H.264
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion-JPEG
    ]
    
    for codec_name, fourcc in codecs_to_try:
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Successfully initialized video writer with codec: {codec_name}")
                return out
            out.release()
        except:
            continue
    return None

def load_model(opt):
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    return model

def process_video_frames(video_path, model, id_vector, output_path, target_size=224):
    # Open video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Input video info - Frames: {frame_count}, Original Resolution: {orig_width}x{orig_height}, FPS: {fps}")
    print(f"Resizing frames to model's expected input size: {target_size}x{target_size}")

    # Prepare video writer
    out = get_video_writer(output_path, fps, orig_width, orig_height)
    save_as_images = out is None
    if save_as_images:
        output_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save frames as images in: {output_dir}")
    
    # Warmup GPU
    print("Warming up GPU...")
    warmup_tensor = torch.randn(1, 3, target_size, target_size).cuda()
    for _ in range(10):
        _ = model(None, warmup_tensor, id_vector, None, True)
    
    # Process frames
    print(f"Processing {frame_count} frames...")
    total_time = 0
    frame_times = []
    processed_frames = []
    
    with torch.no_grad():
        for frame_idx in tqdm(range(frame_count)):
            ret, frame = video.read()
            if not ret:
                print("Warning: Frame read failed, ending early")
                break
                
            start_time = time.time()
            
            # Convert to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_resized.transpose(2, 0, 1)).float().div(255).unsqueeze(0).cuda()
            
            # Perform face swap
            swapped_face = model(None, frame_tensor, id_vector, None, True)[0]
            
            # Convert back to numpy and resize
            swapped_face = swapped_face.squeeze().cpu().numpy().transpose(1, 2, 0)
            swapped_face = (swapped_face * 255).clip(0, 255).astype(np.uint8)
            swapped_face = cv2.resize(swapped_face, (orig_width, orig_height))
            swapped_face_bgr = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
            
            end_time = time.time()
            
            # Record timing
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            total_time += frame_time
            
            # Save output
            if save_as_images:
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:05d}.png"), swapped_face_bgr)
            else:
                out.write(swapped_face_bgr)
            processed_frames.append(swapped_face_bgr)
    
    # Calculate statistics
    if frame_times:
        avg_fps = len(frame_times) / total_time
        avg_frame_time = total_time / len(frame_times)
        min_fps = 1 / max(frame_times)
        max_fps = 1 / min(frame_times)
        
        print("\nPerformance Metrics:")
        print(f"Frames processed: {len(frame_times)}/{frame_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average frame time: {avg_frame_time*1000:.2f} ms")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
    else:
        print("Error: No frames were processed successfully")
    
    video.release()
    if not save_as_images and out is not None:
        out.release()
    
    # Verify output if we saved as video
    if not save_as_images and processed_frames:
        if not verify_video(output_path):
            print("\nVideo verification failed, saving frames as images...")
            save_frames_as_images(output_path.replace('.mp4', '_frames_fallback'), processed_frames)
            print("Frames saved as images in fallback directory")

if __name__ == '__main__':
    # Initialize options
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.which_epoch = 'latest'
    opt.crop_size = 224  # Model's expected input size
    
    # Load model
    model = load_model(opt)
    
    # Load ID embedding
    id_vector = torch.from_numpy(np.load('/scratch/aa10947/SimSwap/test_files/ironman_embedding.npy')).cuda()
    id_vector = id_vector.unsqueeze(0)  # Add batch dimension
    
    # Process video
    input_video = '/scratch/aa10947/SimSwap/test_files/aligned_faces.mp4'
    output_video = '/scratch/aa10947/SimSwap/test_files/swapped_faces.mp4'
    
    process_video_frames(input_video, model, id_vector, output_video, target_size=opt.crop_size)
    
    # Final verification
    if os.path.exists(output_video):
        if verify_video(output_video):
            print(f"\nSuccess! Output video verified: {output_video}")
        else:
            print(f"\nWarning: Output video may be corrupted: {output_video}")
    elif os.path.exists(output_video.replace('.mp4', '_frames')):
        print(f"\nFrames saved as images in: {output_video.replace('.mp4', '_frames')}")