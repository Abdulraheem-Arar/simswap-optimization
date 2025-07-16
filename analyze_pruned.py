import torch
from collections import defaultdict

def analyze_pruned_weights(weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')
    
    print("=" * 80)
    print(f"Pruned Model Weights Analysis: {weights_path}")
    print("=" * 80)
    
    # Group weights by layer
    layer_shapes = defaultdict(dict)
    
    for name, param in state_dict.items():
        parts = name.split('.')
        layer_name = '.'.join(parts[:-1])  # e.g., "down3.1"
        param_type = parts[-1]             # e.g., "weight"
        
        if param_type == "weight":
            if len(param.shape) == 4:  # Conv2d
                layer_shapes[layer_name]["type"] = "Conv2d"
                layer_shapes[layer_name]["in_channels"] = param.shape[1]
                layer_shapes[layer_name]["out_channels"] = param.shape[0]
                layer_shapes[layer_name]["kernel_size"] = (param.shape[2], param.shape[3])
            elif len(param.shape) == 2:  # Linear
                layer_shapes[layer_name]["type"] = "Linear"
                layer_shapes[layer_name]["in_features"] = param.shape[1]
                layer_shapes[layer_name]["out_features"] = param.shape[0]
        elif param_type == "bias":
            layer_shapes[layer_name]["bias"] = param.shape[0]
        elif "running_mean" in param_type:
            layer_shapes[layer_name]["norm_stats"] = param.shape[0]  # For BatchNorm
    
    # Print results
    for layer_name, info in layer_shapes.items():
        print(f"Layer: {layer_name}")
        for key, val in info.items():
            print(f"  {key}: {val}")
        print("-" * 50)

# Run analysis
analyze_pruned_weights("/scratch/aa10947/SimSwap/checkpoints/people/pruned20_net_G.pth")