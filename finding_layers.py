import torch
from torchinfo import summary
from pathlib import Path
import sys

# Add the repo directory to path
repo_path = Path("/scratch/aa10947/SimSwap")  # CHANGE TO YOUR PATH
sys.path.append(str(repo_path))

# Import the original model
from models.fs_networks import Generator_Adain_Upsample

def analyze_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load original model
    model = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    
    # Model summary with torchinfo
    print("\n" + "="*80)
    print("SimSwap Generator Analysis (torchinfo)")
    print("="*80)
    
    summary(
        model,
        input_size=[(1, 3, 224, 224), (1, 512)],  # (image, latent)
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        col_width=20,
        depth=5,  # Show more layer details
        device=device
    )
    
    # Get pruning recommendations
    print("\nPruning Recommendations:")
    print("-"*40)
    print("Top Conv2d Layers by Parameter Count:")
    
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params = sum(p.numel() for p in module.parameters())
            conv_layers.append((name, module.out_channels, params, module.kernel_size))
    
    # Sort by parameter count (descending)
    conv_layers.sort(key=lambda x: x[2], reverse=True)
    
    for i, (name, out_channels, params, kernel) in enumerate(conv_layers[:10]):
        print(f"{i+1}. {name:30} {out_channels:4} channels  {params:>10,} params  kernel: {kernel}")
    
    print("\nSuggested Pruning Approach:")
    print("- Focus on high-parameter layers first (down3.0, BottleNeck layers)")
    print("- Avoid pruning first/last conv layers (kernel_size=7)")
    print("- Good candidates: down1.0, down2.0, down3.0")

if __name__ == "__main__":
    MODEL_PATH = "/scratch/aa10947/SimSwap/checkpoints/people/pruned20_net_G.pth"
    analyze_model(MODEL_PATH)