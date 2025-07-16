import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import hashlib
from torchvision import transforms
import os
from models.models import create_model
from options.test_options import TestOptions

# Hardcoded paths - EDIT THESE
ONNX_PATH = "simswap_224.onnx"
SOURCE_IMAGE = "crop_224/6.jpg"
TARGET_IMAGE = "crop_224/ds.jpg"
OUTPUT_DIR = "validation_results"

# Preprocessing transforms
transformer = transforms.Compose([transforms.ToTensor()])
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def validate_onnx_model():
    # 1. Verify CUDA availability
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError("ONNX Runtime GPU version required. Install with: pip install onnxruntime-gpu")
    
    # 2. Verify ONNX model
    with open(ONNX_PATH, 'rb') as f:
        model_hash = hashlib.md5(f.read()).hexdigest()
    print(f"ONNX Model Hash: {model_hash}")

    # 3. Initialize ArcFace model
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    model = create_model(opt)
    model.eval()

    # 4. Process images
    img_id = transformer_Arcface(Image.open(SOURCE_IMAGE).convert('RGB')).unsqueeze(0).cuda()
    img_att = transformer(Image.open(TARGET_IMAGE).convert('RGB')).unsqueeze(0).numpy()

    # 5. Generate latent vector
    with torch.no_grad():
        latent = F.interpolate(img_id, size=(112, 112))
        latent = model.netArc(latent)
        latent = latent / latent.norm(dim=1, keepdim=True)
        latent = latent.cpu().numpy()

    # 6. ONNX Inference with GPU
    so = ort.SessionOptions()
    sess = ort.InferenceSession(ONNX_PATH, so, providers=['CUDAExecutionProvider'])

    # Run inference
    onnx_output = sess.run(
        ['output_image'],
        {
            'target_image': img_att.astype(np.float32),
            'latent_vector': latent.astype(np.float32)
        }
    )[0]

    # 7. Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def save_image(arr, path):
        arr = arr[0].transpose(1, 2, 0)
        arr = (arr * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    save_image(img_id.cpu().numpy(), f"{OUTPUT_DIR}/source.jpg")
    save_image(img_att, f"{OUTPUT_DIR}/target.jpg")
    save_image(onnx_output, f"{OUTPUT_DIR}/onnx_result.jpg")

    # Add verification watermark
    result_img = cv2.imread(f"{OUTPUT_DIR}/onnx_result.jpg")
    cv2.putText(result_img, f"ONNX Model: {model_hash[:8]}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(f"{OUTPUT_DIR}/onnx_result_verified.jpg", result_img)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Output range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")

if __name__ == "__main__":
    validate_onnx_model()