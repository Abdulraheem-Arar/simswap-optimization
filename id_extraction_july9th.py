import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.isTrain = False

    model = create_model(opt)
    model.eval()

    # Setup face detector/alignment
    crop_size = opt.crop_size if hasattr(opt, 'crop_size') else 224
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640), mode='None')

    # Load and align source image
    img_path = opt.pic_a_path
    print(f"🔍 Loading image from: {img_path}")
    img_whole = cv2.imread(img_path)

    if img_whole is None:
        raise FileNotFoundError(f"❌ Could not load image from: {img_path}")

    aligned_faces, _ = app.get(img_whole, crop_size)
    if aligned_faces is None or len(aligned_faces) == 0:
        raise ValueError("❌ No face detected in identity image.")

    # Convert aligned face to PIL for transform
    aligned_pil = Image.fromarray(cv2.cvtColor(aligned_faces[0], cv2.COLOR_BGR2RGB))

    # Normalize for ArcFace
    transformer_arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transformer_arcface(aligned_pil).unsqueeze(0).cuda()

    # Get ID vector
    with torch.no_grad():
        img_downsample = F.interpolate(img_tensor, size=(112, 112))
        embedding = model.netArc(img_downsample)
        embedding = F.normalize(embedding, p=2, dim=1)

    # Save ID vector
    output_path = opt.output_path if hasattr(opt, 'output_path') else './ironman_embedding_july9th.npy'
    np.save(output_path, embedding.detach().cpu().numpy())
    print(f"✅ ID embedding saved to: {output_path}")
