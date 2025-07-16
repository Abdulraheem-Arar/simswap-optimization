'''
Modified videoswap.py with swapped crops video output
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, 
               temp_results_dir='./temp_results', crop_size=224, 
               no_simswaplogo=False, use_mask=False):
    
    # Initialize directories
    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir, exist_ok=True)
    
    # Create subdirectories
    swapped_crops_dir = os.path.join(temp_results_dir, 'swapped_crops')
    os.makedirs(swapped_crops_dir, exist_ok=True)
    crops_video_dir = os.path.join(temp_results_dir, 'crops_video_frames')
    os.makedirs(crops_video_dir, exist_ok=True)

    # Video initialization
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Processing {frame_count} frames at {fps:.2f} FPS")
    except Exception as e:
        raise RuntimeError(f"Video initialization failed: {str(e)}")

    # Audio handling
    try:
        video_forcheck = VideoFileClip(video_path)
        no_audio = video_forcheck.audio is None
        video_audio_clip = None if no_audio else AudioFileClip(video_path)
        del video_forcheck
    except Exception as e:
        print(f"Audio handling warning: {str(e)}")
        no_audio = True

    # Initialize models
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    spNorm = SpecificNorm()
    net = None
    if use_mask:
        try:
            net = BiSeNet(n_classes=19).cuda()
            net.load_state_dict(torch.load('./parsing_model/checkpoint/79999_iter.pth'))
            net.eval()
        except Exception as e:
            print(f"Failed to initialize parsing model: {str(e)}")
            use_mask = False

    # For tracking crops video
    crops_video_frames = []
    processed_count = 0

    for frame_index in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = video.read()
        if not ret:
            print(f"Frame {frame_index} read failed")
            continue

        try:
            # Face detection
            detect_results = detect_model.get(frame, crop_size)
            if detect_results is None:
                print(f"No faces detected in frame {frame_index}")
                # Save blank frame for crops video
                blank_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                crops_video_frames.append(blank_frame)
                cv2.imwrite(os.path.join(crops_video_dir, f'crop_{frame_index:07d}.png'), blank_frame)
                continue

            frame_align_crop_list, frame_mat_list = detect_results
            swap_result_list = []
            frame_align_crop_tenor_list = []

            # Process each face in frame
            for i, frame_align_crop in enumerate(frame_align_crop_list):
                try:
                    # Face swapping
                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    
                    # Convert swapped face to numpy
                    swapped_face = swap_result.squeeze().cpu().numpy().transpose(1, 2, 0)
                    swapped_face = (swapped_face * 255).clip(0, 255).astype(np.uint8)
                    swapped_face_bgr = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
                    
                    # Save individual crops
                    cv2.imwrite(
                        os.path.join(swapped_crops_dir, f'swapped_crop_{frame_index:07d}_{i}.png'),
                        swapped_face_bgr
                    )
                    cv2.imwrite(
                        os.path.join(swapped_crops_dir, f'original_crop_{frame_index:07d}_{i}.png'),
                        frame_align_crop
                    )

                    # Add to crops video (show first face only for simplicity)
                    if i == 0:
                        crops_video_frames.append(swapped_face_bgr)
                        cv2.imwrite(os.path.join(crops_video_dir, f'crop_{frame_index:07d}.png'), swapped_face_bgr)

                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                except Exception as e:
                    print(f"Face swap failed for frame {frame_index}: {str(e)}")
                    continue

            # Handle frames with no faces after processing
            if not frame_align_crop_list:
                blank_frame = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                crops_video_frames.append(blank_frame)
                cv2.imwrite(os.path.join(crops_video_dir, f'crop_{frame_index:07d}.png'), blank_frame)

            # Save final blended frame
            output_path = os.path.join(temp_results_dir, f'frame_{frame_index:07d}.jpg')
            reverse2wholeimage(
                frame_align_crop_tenor_list,
                swap_result_list,
                frame_mat_list,
                crop_size,
                frame,
                logoclass,
                output_path,
                no_simswaplogo,
                pasring_model=net,
                use_mask=use_mask,
                norm=spNorm
            )
            processed_count += 1

        except Exception as e:
            print(f"Frame {frame_index} processing failed: {str(e)}")
            continue

    video.release()
    print(f"Successfully processed {processed_count}/{frame_count} frames")

    # Create final blended video
    try:
        image_filenames = sorted(glob.glob(os.path.join(temp_results_dir, '*.jpg')))
        if not image_filenames:
            raise RuntimeError("No frames processed for final video")
        
        clips = ImageSequenceClip(image_filenames, fps=fps)
        if not no_audio and video_audio_clip:
            clips = clips.set_audio(video_audio_clip)
        clips.write_videofile(save_path, audio_codec='aac', verbose=False)
        print(f"Saved final video to {save_path}")
    except Exception as e:
        raise RuntimeError(f"Final video creation failed: {str(e)}")

    # Create swapped crops video
    try:
        crops_video_path = os.path.join(os.path.dirname(save_path), 
                            f"swapped_crops_{os.path.basename(save_path)}")
        
        # Create video from collected frames
        crops_clip = ImageSequenceClip(
            [os.path.join(crops_video_dir, f) for f in sorted(os.listdir(crops_video_dir))],
            fps=fps
        )
        crops_clip.write_videofile(crops_video_path, verbose=False)
        print(f"Saved swapped crops video to {crops_video_path}")
    except Exception as e:
        print(f"Warning: Could not create crops video: {str(e)}")

    return save_path, crops_video_path