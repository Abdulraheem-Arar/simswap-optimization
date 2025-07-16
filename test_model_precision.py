import torch
from models.models import create_model
from options.test_options import TestOptions
from collections import defaultdict

def analyze_model_precision(model):
    """Comprehensive analysis of all parameters and buffers"""
    precision_stats = defaultdict(int)
    layer_types = defaultdict(int)
    
    print("\nMODEL PRECISION ANALYSIS")
    print("=" * 40)
    
    # Check parameters (weights)
    for name, param in model.named_parameters():
        dtype = str(param.dtype).split('.')[-1]
        size_mb = param.numel() * param.element_size() / 1e6
        precision_stats[dtype] += size_mb
        layer_types[f"{dtype} params"] += 1
        print(f"{name:<60} {dtype:<10} {size_mb:.2f}MB")
    
    # Check buffers (batch norm stats etc.)
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf):
            dtype = str(buf.dtype).split('.')[-1]
            size_mb = buf.numel() * buf.element_size() / 1e6
            precision_stats[dtype] += size_mb
            layer_types[f"{dtype} buffers"] += 1
            print(f"{name:<60} {dtype:<10} {size_mb:.2f}MB")
    
    # Summary
    print("\nPRECISION SUMMARY")
    print("=" * 40)
    for dtype, size in precision_stats.items():
        print(f"{dtype.upper()+':':<10} {size:.2f}MB ({size/220*100:.1f}%)")
    
    print("\nLAYER TYPE COUNT")
    print("=" * 40)
    for ltype, count in layer_types.items():
        print(f"{ltype+':':<20} {count}")

if __name__ == '__main__':
    # Load model with your exact parameters
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.crop_size = 224
    
    print("Loading model...")
    model = create_model(opt).eval()
    
    # Run analysis
    analyze_model_precision(model)
    
    # Check if already FP16
    total_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    if total_size < 150:  # FP16 models are typically <150MB
        print("\nCONCLUSION: Model appears to be already in FP16/mixed precision")
    else:
        print("\nCONCLUSION: Model is primarily FP32")