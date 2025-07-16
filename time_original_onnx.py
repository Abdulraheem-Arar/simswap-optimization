import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import hashlib
from torchvision import transforms
import os
from models.models import create_model
from options.test_options import TestOptions
import time

# Configuration
CONFIG = {
    'onnx_path': 'simswap_224.onnx',
    'source_image': 'crop_224/6.jpg',
    'target_image': 'crop_224/ds.jpg',
    'output_dir': 'validation_results',
    'crop_size': 224,
    'arcface_path': 'arcface_model/arcface_checkpoint.tar'
}

# Define transformers
transformer = transforms.Compose([transforms.ToTensor()])
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def setup_environment():
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("ONNX Runtime GPU version detected")
        else:
            print("ONNX Runtime CPU version detected")
        return ort
    except ImportError:
        raise ImportError("Install with: pip install onnxruntime-gpu")

def initialize_model():
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = CONFIG['arcface_path']
    opt.isTrain = False
    opt.crop_size = CONFIG['crop_size']
    opt.checkpoints_dir = 'checkpoints'
    model = create_model(opt)
    model.eval()
    return model

def denormalize_tensor(tensor):
    """Inverse of ArcFace normalization"""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    tensor = tensor.clone()
    device = tensor.device
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    return tensor


def save_image(tensor, path, denormalize=False):
    if denormalize:
        tensor = denormalize_tensor(tensor if isinstance(tensor, torch.Tensor) else torch.from_numpy(tensor))
    if isinstance(tensor, torch.Tensor):
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        # If it's a numpy array, just rearrange axes if needed
        if tensor.ndim == 4:
            arr = tensor.squeeze(0).transpose(1, 2, 0)
        elif tensor.ndim == 3:
            arr = tensor.transpose(1, 2, 0)
        else:
            arr = tensor
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def validate_onnx_model():
    # Full pipeline start
    full_start = time.perf_counter()
    
    # 1. Setup
    setup_start = time.perf_counter()
    ort = setup_environment()
    model = initialize_model()
    setup_time = time.perf_counter() - setup_start
    
    # 2. Preprocessing
    preprocess_start = time.perf_counter()
    img_id = transformer_Arcface(Image.open(CONFIG['source_image']).convert('RGB')).unsqueeze(0).cuda()
    img_att = transformer(Image.open(CONFIG['target_image']).convert('RGB')).unsqueeze(0).numpy()
    preprocess_time = time.perf_counter() - preprocess_start
    
    # 3. Latent generation
    latent_start = time.perf_counter()
    with torch.no_grad():
        latent = F.interpolate(img_id, size=(112,112))
        latent = model.netArc(latent)
        latent = latent / latent.norm(dim=1, keepdim=True)
        latent = latent.cpu().numpy()
    latent_time = time.perf_counter() - latent_start
    
    # 4. Model inference
    inference_start = time.perf_counter()
    sess = ort.InferenceSession(CONFIG['onnx_path'], 
                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Warmup
    _ = sess.run(['output_image'], {
        'target_image': img_att.astype(np.float32),
        'latent_vector': latent.astype(np.float32)
    })
    
    # Timed inference
    num_runs = 10
    inference_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        onnx_output = sess.run(['output_image'], {
            'target_image': img_att.astype(np.float32),
            'latent_vector': latent.astype(np.float32)
        })[0]
        inference_times.append(time.perf_counter() - start)
    
    avg_inference = np.mean(inference_times)
    inference_time = time.perf_counter() - inference_start
    
    # 5. Postprocessing
    postprocess_start = time.perf_counter()
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    save_image(img_id, f"{CONFIG['output_dir']}/source.jpg", denormalize=True)
    save_image(torch.from_numpy(img_att), f"{CONFIG['output_dir']}/target.jpg")
    save_image(onnx_output, f"{CONFIG['output_dir']}/onnx_result.jpg")
    postprocess_time = time.perf_counter() - postprocess_start
    
    # Timing results
    full_time = time.perf_counter() - full_start
    
    print(f"\nTIMING RESULTS:")
    print(f"1. Setup: {setup_time*1000:.2f}ms")
    print(f"2. Preprocessing: {preprocess_time*1000:.2f}ms")
    print(f"3. Latent generation: {latent_time*1000:.2f}ms")
    print(f"4. Model inference (avg of {num_runs} runs): {avg_inference*1000:.2f}ms")
    print(f"5. Postprocessing: {postprocess_time*1000:.2f}ms")
    total_pipeline_time = (
    setup_time +
    preprocess_time +
    latent_time +
    avg_inference +  
    postprocess_time
    )

    total_inference_time = (
    preprocess_time +
    avg_inference +
    postprocess_time
    )

    print(f"\nTOTAL PIPELINE TIME: {total_pipeline_time:.2f}s")
    print(f"\nTOTAL INFERENCE TIME:(our concern) {total_inference_time*1000:.2f}ms")
    print(f"Output saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    validate_onnx_model()