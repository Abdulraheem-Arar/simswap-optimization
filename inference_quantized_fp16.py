import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


# Preprocessing (from original script)
transformer = transforms.Compose([
    transforms.ToTensor(),
])
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def tensor_to_cv2(tensor):
    """Robust tensor to OpenCV conversion"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().permute(1, 2, 0)
    return (tensor.numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8)

def compare_ssim(tensor1, tensor2):
    """
    Robust SSIM comparison that handles:
    - Small images
    - Proper channel handling
    - Dimension validation
    """
    # Convert tensors to numpy arrays
    img1 = tensor1.squeeze().permute(1, 2, 0).cpu().numpy()  # [H,W,C]
    img2 = tensor2.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Validate dimensions
    min_dim = min(img1.shape[:2])
    win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
    
    # Calculate SSIM for each channel
    ssim_values = []
    for ch in range(img1.shape[2]):
        ssim_val = ssim(
            img1[:, :, ch],
            img2[:, :, ch],
            data_range=1.0,
            win_size=win_size,
            channel_axis=None,  # Single channel
            gaussian_weights=True
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def run_quantized_inference():
    opt = TestOptions().parse()
    model = create_model(opt).eval().cuda()
    
    # Load and preprocess inputs
    img_id = transformer_Arcface(Image.open(opt.pic_a_path).convert('RGB')).unsqueeze(0).cuda()
    img_att = transformer(Image.open(opt.pic_b_path).convert('RGB')).unsqueeze(0).cuda()

    # Create latent IDs
    with autocast(enabled=False):
        latent_id = model.netArc(F.interpolate(img_id, size=(112,112)))
        latent_id = latent_id.detach() / (latent_id.norm(dim=1, keepdim=True) + 1e-6)
        latent_att = latent_id.clone()

    # FP32 Reference
    with torch.no_grad(), autocast(enabled=False):
        fp32_output = model(img_id, img_att, latent_id, latent_att, True)
        cv2.imwrite('fp32_reference.jpg', tensor_to_cv2(fp32_output))

    # AMP Quantized
    with torch.no_grad(), autocast():
        # Protect sensitive layers
        model.netG.BottleNeck.float()
        model.netG.last_layer.float()
        for name, module in model.named_modules():
            if 'style' in name or 'Arc' in name:
                module.float()
        
        amp_output = model(img_id, img_att, latent_id, latent_att, True)
        cv2.imwrite('amp_output.jpg', tensor_to_cv2(amp_output))

    # Metrics
    print(f"SSIM: {compare_ssim(fp32_output, amp_output.float()):.4f}")
    diff = (fp32_output - amp_output.float()).abs().mean(dim=1, keepdim=True)
    cv2.imwrite('difference.jpg', tensor_to_cv2(diff))

if __name__ == '__main__':
    run_quantized_inference()