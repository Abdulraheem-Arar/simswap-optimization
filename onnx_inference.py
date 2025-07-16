import cv2
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms
from options.test_options import TestOptions
import os
import time

# Configuration
ONNX_MODEL_PATH = "/scratch/aa10947/SimSwap/simswap_pruned.onnx"
ARCFACE_CKPT = "/scratch/aa10947/SimSwap/arcface_model/arcface_checkpoint.tar"

class FaceSwapper:
    def __init__(self, opt):
        self.opt = opt
        self._verify_paths()
        
        # Configure ONNX runtime with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._analyze_onnx_model()
        self._init_arcface()

    def _verify_paths(self):
        """Verify all required files exist"""
        required_paths = {
            "ONNX Model": ONNX_MODEL_PATH,
            "ArcFace Checkpoint": ARCFACE_CKPT,
            "Source Image": self.opt.pic_a_path,
            "Target Image": self.opt.pic_b_path
        }
        
        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")
        print("All required files verified")

    def _analyze_onnx_model(self):
        """Print detailed ONNX model information"""
        print("\nONNX Model Analysis:")
        print(f"Inputs ({len(self.ort_session.get_inputs())}):")
        for i, inp in enumerate(self.ort_session.get_inputs()):
            print(f"  {i}: {inp.name} - Shape: {inp.shape} - Type: {inp.type}")
            
        print(f"\nOutputs ({len(self.ort_session.get_outputs())}):")
        for i, out in enumerate(self.ort_session.get_outputs()):
            print(f"  {i}: {out.name} - Shape: {out.shape} - Type: {out.type}")
        
        # Verify expected input/output shapes
        self.expected_input_shape = self.ort_session.get_inputs()[0].shape
        self.expected_latent_shape = self.ort_session.get_inputs()[1].shape
        print("\nExpected shapes:")
        print(f"  Image input: {self.expected_input_shape}")
        print(f"  Latent input: {self.expected_latent_shape}")

    def _init_arcface(self):
        """Initialize ArcFace with proper architecture"""
        from models.arcface_models import ResNet, IRBlock
        
        def conv3x3(in_planes, out_planes, stride=1):
            return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        
        IRBlock.conv3x3 = staticmethod(conv3x3)
        
        # Architecture selection
        print("\nInitializing ArcFace...")
        self.model = ResNet(IRBlock, [3, 4, 23, 3], use_se=True).cuda()
        
        # Load checkpoint
        print(f"Loading checkpoint from: {ARCFACE_CKPT}")
        checkpoint = torch.load(ARCFACE_CKPT, map_location='cpu')
        
        # State dict extraction with fallbacks
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        # Clean and filter state dict
        arcface_weights = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('netArc.', '')
            if 'fc' not in new_key:  # Skip classification layer if present
                arcface_weights[new_key] = v
        
        # Load weights with detailed reporting
        load_result = self.model.load_state_dict(arcface_weights, strict=False)
        print("\nArcFace Loading Report:")
        print(f"Missing keys: {len(load_result.missing_keys)}")
        print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
        
        if load_result.unexpected_keys:
            print("\nTop 10 unexpected keys:")
            for key in list(load_result.unexpected_keys)[:10]:
                print(f"  {key}")
        
        self.model.eval()
        print("ArcFace initialized successfully")

    def run(self):
        """Main execution pipeline with timing"""
        print("\nStarting face swap pipeline...")
        start_time = time.time()
        
        try:
            # 1. Process source image
            latent_time = time.time()
            latent = self._get_latent(self.opt.pic_a_path)
            print(f"Latent extraction time: {time.time() - latent_time:.2f}s")
            
            # 2. Process target image
            target_time = time.time()
            img_b = self._preprocess_target(self.opt.pic_b_path)
            print(f"Target processing time: {time.time() - target_time:.2f}s")
            
            # 3. Verify input shapes
            self._verify_input_shapes(img_b, latent)
            
            # 4. Run ONNX inference
            inference_time = time.time()
            ort_inputs = {
                'input': img_b.astype(np.float32),
                'latent': latent.astype(np.float32)
            }
            ort_outs = self.ort_session.run(None, ort_inputs)
            print(f"Inference time: {time.time() - inference_time:.2f}s")
            
            # 5. Process and save result
            output = ort_outs[0]
            self._save_result(output)
            
            print(f"\nTotal processing time: {time.time() - start_time:.2f}s")
            self._verify_output(output)
            
        except Exception as e:
            print(f"\nPipeline failed: {str(e)}")
            raise

    def _verify_input_shapes(self, img_b, latent):
        """Verify inputs match ONNX expectations"""
        print("\nInput Verification:")
        print(f"Image shape: {img_b.shape} (expected {self.expected_input_shape})")
        print(f"Latent shape: {latent.shape} (expected {self.expected_latent_shape})")
        
        if img_b.shape != tuple(self.expected_input_shape):
            raise ValueError(f"Image shape mismatch. Got {img_b.shape}, expected {self.expected_input_shape}")
        
        if latent.shape != tuple(self.expected_latent_shape):
            raise ValueError(f"Latent shape mismatch. Got {latent.shape}, expected {self.expected_latent_shape}")

    def _verify_output(self, output):
        """Verify output looks valid"""
        print("\nOutput Verification:")
        print(f"Shape: {output.shape}")
        print(f"Value range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"NaN values: {np.isnan(output).sum()}")
        print(f"Inf values: {np.isinf(output).sum()}")
        
        if output.min() < -1.5 or output.max() > 1.5:
            print("Warning: Output values outside expected [-1, 1] range")
        
        # Save debug output (fixed syntax)
        debug_output = (output[0].transpose(1,2,0) * 127.5 + 127.5).clip(0,255).astype(np.uint8)
        debug_output = debug_output[..., ::-1]  # Convert RGB to BGR
        cv2.imwrite('debug_output.jpg', debug_output)

    def _get_latent(self, image_path):
        """Enhanced latent extraction with validation"""
        try:
            # Load and verify image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print(f"\nSource Image: {img.size} {img.mode}")
            
            # Preprocessing pipeline
            transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            img_tensor = transform(img).unsqueeze(0).cuda()
            
            # Validate tensor
            if torch.any(torch.isnan(img_tensor)):
                raise ValueError("Source image contains NaN values")
            
            print(f"Input range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
            
            # Extract latent
            with torch.no_grad():
                latent = self.model(img_tensor)
                norm_latent = latent / latent.norm(dim=1, keepdim=True)
                print(f"Latent norm: {norm_latent.norm().item():.4f}")
                return norm_latent.cpu().numpy()
                
        except Exception as e:
            print(f"Latent extraction failed for {image_path}")
            raise

    def _preprocess_target(self, image_path):
        """Robust target image preprocessing"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print(f"\nTarget Image: {img.size} {img.mode}")
            
            # Resize to expected dimensions
            if tuple(self.expected_input_shape[2:]) != (self.opt.crop_size, self.opt.crop_size):
                print(f"Resizing to ONNX expected size: {self.expected_input_shape[2:]}")
                img = img.resize(self.expected_input_shape[2:][::-1])  # PIL uses (w,h)
            else:
                img = img.resize((self.opt.crop_size, self.opt.crop_size))
            
            # Convert to tensor and normalize
            img_tensor = transforms.ToTensor()(img)
            img_tensor = transforms.Normalize([0.5]*3, [0.5]*3)(img_tensor)
            
            # Validate
            if torch.any(torch.isnan(img_tensor)):
                raise ValueError("Target image contains NaN values")
            
            print(f"Processed range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
            return img_tensor.unsqueeze(0).numpy()
            
        except Exception as e:
            print(f"Target processing failed for {image_path}")
            raise

    def _save_result(self, output):
        """Enhanced result saving with validation"""
        try:
            print(f"Raw output shape: {output.shape}")  # Should be [1,3,224,224]
            # Convert and denormalize main output
            output = output[0]  # Remove batch dimension [3,224,224]
            output = np.transpose(output, (1, 2, 0))  # CHW to HWC [224,224,3]
            output = (output + 1) * 127.5  # [-1,1] to [0,255]
            output = np.clip(output, 0, 255).astype(np.uint8)
            
            # Save main result (convert RGB to BGR for OpenCV)
            output_path = os.path.join(self.opt.output_path, "result.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            
            print(f"\nResult saved to {output_path}")
            print(f"Final output range: [{output.min()}, {output.max()}]")
            
            # Save debug views
            debug_paths = {
                'debug_output.jpg': output,
                'debug_bgr.jpg': cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            }
            
            for path, img in debug_paths.items():
                cv2.imwrite(path, img)
                
        except Exception as e:
            print(f"Failed to save result: {str(e)}")
            raise

if __name__ == '__main__':
    opt = TestOptions().parse()
    FaceSwapper(opt).run()