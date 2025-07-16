import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def video_swap(video_path, id_vector, detect_model, save_path, temp_results_dir='./temp_results',
               crop_size=224, no_simswaplogo=False, use_mask=False, timing_info=None):

    # TIMING INITIALIZATION
    if timing_info:
        start_time = time.time()
        processed_frames = 0
        log_interval = 30

    # Check audio
    video_forcheck = VideoFileClip(video_path)
    no_audio = video_forcheck.audio is None
    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Set providers - CUDA if available
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

    onnx_session = ort.InferenceSession('/scratch/aa10947/SimSwap/simswap_224.onnx',
                                        providers=providers,
                                        sess_options=sess_options)

    output_name = onnx_session.get_outputs()[0].name

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    spNorm = SpecificNorm()

    if use_mask:
        net = BiSeNet(n_classes=19)
        net.cuda()
        net.load_state_dict(torch.load('./parsing_model/checkpoint/79999_iter.pth'))
        net.eval()
    else:
        net = None

    for frame_index in tqdm(range(frame_count)):
        if timing_info:
            frame_start = time.time()

        ret, frame = video.read()
        if not ret:
            break

        if timing_info:
            processed_frames += 1

        detect_results = detect_model.get(frame, crop_size)

        if detect_results is not None:
            if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)

            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tensor_list = []

            for frame_align_crop in frame_align_crop_list:
                # Prepare input for ONNX (no GPU tensors, just numpy)
                frame_align_crop_rgb = cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB)
                frame_align_crop_np = frame_align_crop_rgb.astype(np.float32) / 255.0
                frame_align_crop_np = np.transpose(frame_align_crop_np, (2, 0, 1))  # HWC to CHW
                frame_align_crop_np = frame_align_crop_np[np.newaxis, :]  # add batch dim

                # id_vector is torch tensor on CUDA, move to CPU numpy once
                ort_inputs = {
                    'target_image': frame_align_crop_np,
                    'latent_vector': id_vector.cpu().numpy()
                }

                # ONNX inference
                ort_outs = onnx_session.run([output_name], ort_inputs)
                swap_result_np = ort_outs[0]  # [1,3,H,W] or similar

                # Convert output to torch tensor on CUDA for further processing
                swap_result = torch.from_numpy(swap_result_np).float().cuda()
                if swap_result.dim() == 4:
                    swap_result = swap_result.squeeze(0)
                if swap_result.size(0) == 4:
                    swap_result = swap_result[:3]

                swap_result_list.append(swap_result)

                # Keep input tensor for reverse2wholeimage (convert to torch tensor on cuda)
                frame_align_crop_tensor = torch.from_numpy(frame_align_crop_np).float().cuda()
                frame_align_crop_tensor_list.append(frame_align_crop_tensor)

            # Write frame once per frame, after all faces processed
            cv2.imwrite(os.path.join(temp_results_dir, f'frame_{frame_index:0>7d}.jpg'), frame)

            # Compose the swapped faces back
            reverse2wholeimage(frame_align_crop_tensor_list, swap_result_list, frame_mat_list,
                               crop_size, frame, logoclass,
                               os.path.join(temp_results_dir, f'frame_{frame_index:0>7d}.jpg'),
                               no_simswaplogo, pasring_model=net, use_mask=use_mask, norm=spNorm)

        else:
            if not os.path.exists(temp_results_dir):
                os.mkdir(temp_results_dir)
            frame = frame.astype(np.uint8)
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            cv2.imwrite(os.path.join(temp_results_dir, f'frame_{frame_index:0>7d}.jpg'), frame)

        if timing_info and processed_frames % log_interval == 0:
            elapsed = time.time() - start_time
            curr_fps = processed_frames / elapsed
            print(f"Processed {processed_frames} frames | Current FPS: {curr_fps:.2f}")

    video.release()

    # Compile output video
    image_filenames = sorted(glob.glob(os.path.join(temp_results_dir, '*.jpg')))
    clips = ImageSequenceClip(image_filenames, fps=fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    if timing_info:
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time
        print("\nFINAL TIMING:")
        print(f"Total frames: {processed_frames}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

    clips.write_videofile(save_path, audio_codec='aac')
