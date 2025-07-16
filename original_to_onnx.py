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

# Hardcoded configuration - EDIT THESE
CONFIG = {
    'onnx_path': 'simswap_224.onnx',
    'source_image': 'crop_224/6.jpg',
    'target_image': 'crop_224/ds.jpg',
    'output_dir': 'validation_results',
    'crop_size': 224,  # Must match your model
    'arcface_path': 'arcface_model/arcface_checkpoint.tar'
}

def setup_environment():
    """Handle ONNX Runtime import with version checking"""
    try:
        import onnxruntime as ort
        # Check if it's the GPU version by looking for CUDA provider
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("ONNX Runtime GPU version detected")
        else:
            print("ONNX Runtime CPU version detected")
        return ort
    except ImportError:
        raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")
    except AttributeError:
        # Fallback for older versions
        import onnxruntime as ort
        return ort

def initialize_model():
    """Initialize the SimSwap model with proper configuration"""
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = CONFIG['arcface_path']
    opt.isTrain = False
    opt.crop_size = CONFIG['crop_size']  # Critical for correct model initialization
    opt.checkpoints_dir = 'checkpoints'
    
    model = create_model(opt)
    model.eval()
    return model

def validate_onnx_model():
    # 1. Setup ONNX Runtime
    ort = setup_environment()
    
    # 2. Verify ONNX model
    with open(CONFIG['onnx_path'], 'rb') as f:
        model_hash = hashlib.md5(f.read()).hexdigest()
    print(f"ONNX Model Hash: {model_hash}")

    # 3. Initialize model with correct crop_size
    model = initialize_model()

    # 4. Process images
    img_id = transformer_Arcface(Image.open(CONFIG['source_image']).convert('RGB')).unsqueeze(0).cuda()
    img_att = transformer(Image.open(CONFIG['target_image']).convert('RGB')).unsqueeze(0).numpy()

    # 5. Generate latent vector
    with torch.no_grad():
        latent = F.interpolate(img_id, size=(112, 112))
        latent = model.netArc(latent)
        latent = latent / latent.norm(dim=1, keepdim=True)
        latent = latent.cpu().numpy()

    # 6. ONNX Inference
    # 6. ONNX Inference
    try:
        sess = ort.InferenceSession(CONFIG['onnx_path'], 
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Warmup run (don't time this)
        _ = sess.run(
            ['output_image'],
            {
                'target_image': img_att.astype(np.float32),
                'latent_vector': latent.astype(np.float32)
            }
        )
        
        # Timed inference runs
        num_runs = 10  # Number of iterations for stable measurement
        start_time = time.perf_counter()  # Highest resolution timer
        
        for _ in range(num_runs):
            onnx_output = sess.run(
                ['output_image'],
                {
                    'target_image': img_att.astype(np.float32),
                    'latent_vector': latent.astype(np.float32)
                }
            )[0]
        
        elapsed = (time.perf_counter() - start_time) / num_runs
        
        print(f"\nInference Time: {elapsed*1000:.2f}ms per image (avg of {num_runs} runs)")
        print(f"Throughput: {1/elapsed:.2f} images/second")

    except Exception as e:
        raise RuntimeError(f"Failed to create ONNX session: {str(e)}")

    # 7. Save and verify results
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    def denormalize_tensor(tensor):
        """Exact inverse of ArcFace normalization"""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).to(device)  # Ensure tensor is on correct device
        tensor = tensor.clone()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = tensor + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        return tensor

    def save_image(tensor, path, apply_denorm=False):
        """Universal saving function matching test_one_image.py"""
        if apply_denorm:
            tensor = denormalize_tensor(tensor)
        
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).to(device)
        
        # Move to CPU for saving
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    # Get current device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Usage - ensure tensors are on correct device
    save_image(img_id.to(device), f"{CONFIG['output_dir']}/source.jpg", apply_denorm=True)
    save_image(torch.from_numpy(img_att).to(device), f"{CONFIG['output_dir']}/target.jpg")
    save_image(torch.from_numpy(onnx_output).to(device), f"{CONFIG['output_dir']}/onnx_result.jpg")

    
    # Add verification watermark
    result_img = cv2.imread(f"{CONFIG['output_dir']}/onnx_result.jpg")
    cv2.putText(result_img, f"ONNX Model: {model_hash[:8]}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(f"{CONFIG['output_dir']}/onnx_result_verified.jpg", result_img)

    print(f"\nValidation successful! Results saved to {CONFIG['output_dir']}")
    print(f"Output value range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")

# Define transformers after function definitions
transformer = transforms.Compose([transforms.ToTensor()])
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



if __name__ == "__main__":
    try:
        validate_onnx_model()
    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        print("Possible solutions:")
        print("1. Ensure onnxruntime-gpu is installed: pip install onnxruntime-gpu")
        print("2. Verify the crop_size matches your model architecture")
        print("3. Check all file paths in the CONFIG dictionary")