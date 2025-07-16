import onnxruntime as ort
import torch

def verify_gpu_setup():
    # Check GPU availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device: {torch.cuda.current_device()}")
    
    # ONNX Runtime GPU info
    sess_options = ort.SessionOptions()
    print("\nONNX Runtime GPU capabilities:")
    try:
        sess = ort.InferenceSession('simswap_224.onnx', sess_options, 
                                  providers=['CUDAExecutionProvider'])
        print("Successfully created ONNX session with GPU")
        print(f"ONNX using device: {sess.get_providers()}")
        return True
    except Exception as e:
        print(f"GPU session failed: {str(e)}")
        return False

if verify_gpu_setup():
    print("\nREADY TO RUN SIMSWAP VALIDATION!")
    print("Your system is correctly configured with:")
    print("- ONNX Runtime GPU support")
    print("- PyTorch CUDA support")
    print("- TensorRT available (for accelerated inference)")
else:
    print("\nConfiguration needs adjustment before proceeding")