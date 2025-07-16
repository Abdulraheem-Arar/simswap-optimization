import time
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

if __name__ == '__main__':
    full_start = time.perf_counter()

    # 1. Setup
    setup_start = time.perf_counter()
    opt = TestOptions().parse()
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    setup_time = time.perf_counter() - setup_start

    with torch.no_grad():
        # 2. Preprocessing
        preprocess_start = time.perf_counter()
        pic_a = opt.pic_a_path
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path
        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        img_id = img_id.cuda()
        img_att = img_att.cuda()
        preprocess_time = time.perf_counter() - preprocess_start

        # 3. Latent generation
        latent_start = time.perf_counter()
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latent_id = model.netArc(img_id_downsample)
        latent_id = latent_id.detach().to('cpu')
        latent_id = latent_id / latent_id.norm(dim=1, keepdim=True)
        latent_id = latent_id.to('cuda')
        latent_time = time.perf_counter() - latent_start

        # 4. Model inference
        inference_start = time.perf_counter()
        img_fake = model(img_id, img_att, latent_id, latent_id, True)
        inference_time = time.perf_counter() - inference_start

        # 5. Postprocessing
        postprocess_start = time.perf_counter()
        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu').numpy()
        output = output[..., ::-1]  # RGB to BGR
        output = (output * 255).astype(np.uint8)
        cv2.imwrite(opt.output_path + 'result.jpg', output)
        postprocess_time = time.perf_counter() - postprocess_start

    full_time = time.perf_counter() - full_start

    print("\nTIMING RESULTS:")
    print(f"1. Setup: {setup_time * 1000:.2f}ms")
    print(f"2. Preprocessing: {preprocess_time * 1000:.2f}ms")
    print(f"3. Latent generation: {latent_time * 1000:.2f}ms")
    print(f"4. Model inference: {inference_time * 1000:.2f}ms")
    print(f"5. Postprocessing: {postprocess_time * 1000:.2f}ms")
    print(f"\nTOTAL PIPELINE TIME: {full_time:.2f}s")
