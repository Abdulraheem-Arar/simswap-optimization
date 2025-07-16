'''
ONNX Version of time_test_video_swapsingle.py
Identical functionality but uses ONNX Runtime for face swapping
'''

import os
import cv2
import torch
import time
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
import onnxruntime as ort
from datetime import datetime


# ONNX Model Path
ONNX_MODEL_PATH = '/scratch/aa10947/SimSwap/simswap_224.onnx'

def create_onnx_session():
    """Create optimized ONNX session for face swapping"""
    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Configure providers - CUDA first, fallback to CPU
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'HEURISTIC'
        }),
        'CPUExecutionProvider'
    ]
    
    return ort.InferenceSession(ONNX_MODEL_PATH, 
                              providers=providers,
                              sess_options=sess_options)

# Image transformations (same as original)
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    opt = TestOptions().parse()
    crop_size = opt.crop_size
    
    # Initialize models (same as original)
    model = create_model(opt)
    model.eval()
    
    # Initialize face detection (same as original)
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))

    # Create ONNX session for face swapping
    onnx_session = create_onnx_session()
    onnx_input_name = onnx_session.get_inputs()[0].name
    
    with torch.no_grad():
        # Process source image (same as original)
        img_a_whole = cv2.imread(opt.pic_a_path)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_id = transformer_Arcface(img_a).unsqueeze(0).cuda()

        # Create latent id (same as original)
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        # Start timing
        start_time = time.time()
        frame_count = 0
        
        # Call the ONNX version of video_swap
        from util.videoswap_onnx import video_swap
        video_swap(
            video_path=opt.video_path,
            id_vector=latend_id,
            detect_model=app,
            save_path=opt.output_path,
            temp_results_dir=opt.temp_path,
            crop_size=crop_size,
            no_simswaplogo=opt.no_simswaplogo,
            use_mask=opt.use_mask,
            timing_info={
                'start_time': start_time,
                'frame_count': frame_count
            }
        )