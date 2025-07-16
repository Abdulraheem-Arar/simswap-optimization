import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from options.test_options import TestOptions
from models.models import create_model

if __name__ == '__main__':
    # 1. Setup with minimal required options
    opt = TestOptions().parse()
    opt.name = 'people'  # Required for model creation
    opt.isTrain = False  # Disable training-specific setup
    
    # 2. Initialize only what we need
    model = create_model(opt)
    model.eval()
    
    # 3. Preprocess input image
    img = Image.open(opt.pic_a_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0).cuda()  # Add batch dim and move to GPU
    
    # 4. Extract and save embedding
    with torch.no_grad():
        # ArcFace requires 112x112 input
        img_down = F.interpolate(img, size=(112,112))
        embedding = model.netArc(img_down)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)  # Normalize
        
        # Save as numpy array
        np.save(opt.output_path, embedding.cpu().numpy())
        print(f"Embedding saved to {opt.output_path} (shape: {embedding.shape})")