import torch
from torchinfo import summary
from pathlib import Path
import sys
from options.test_options import TestOptions
from models.models import create_model

# Configuration
repo_path = Path("/scratch/aa10947/SimSwap")
sys.path.append(str(repo_path))
ORIGINAL_PATH = "/scratch/aa10947/SimSwap/checkpoints/people/latest_net_G.pth"
PRUNED_PATH = "/scratch/aa10947/SimSwap/checkpoints/people/pruned20_net_G.pth"

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.crop_size = 224
    opt.gpu_ids = []
    opt.isTrain = False
    
    model = create_model(opt).to(device).eval()
    state_dict = torch.load(model_path, map_location='cpu')
    model.netG.load_state_dict(state_dict['netG'] if 'netG' in state_dict else state_dict)
    return model

class ModelWrapper(torch.nn.Module):
    def __init__(self, netG):
        super().__init__()
        self.netG = netG
        
    def forward(self, img):
        dlatents = torch.randn(img.size(0), 512).to(img.device)
        dlatents = dlatents / torch.norm(dlatents, p=2, dim=1, keepdim=True)
        return self.netG(img, dlatents=dlatents)

def compare_architectures():
    # Load both models
    original = load_model(ORIGINAL_PATH)
    pruned = load_model(PRUNED_PATH)
    
    print("\n" + "="*100)
    print("SIMSWAP GENERATOR ARCHITECTURE COMPARISON (ORIGINAL vs PRUNED)")
    print("="*100)
    
    # Compare all Conv2d layers
    print("\n{:<50} | {:>15} | {:>15} | {:>10}".format(
        "Layer", "Original Channels", "Pruned Channels", "Reduction"
    ))
    print("-"*100)
    
    total_original = 0
    total_pruned = 0
    
    # Create dictionaries of layers for easy comparison
    orig_layers = {name: module for name, module in original.netG.named_modules() 
                  if isinstance(module, torch.nn.Conv2d)}
    pruned_layers = {name: module for name, module in pruned.netG.named_modules() 
                    if isinstance(module, torch.nn.Conv2d)}
    
    for name, orig_module in orig_layers.items():
        if name in pruned_layers:
            pruned_module = pruned_layers[name]
            orig_channels = orig_module.out_channels
            pruned_channels = pruned_module.out_channels
            reduction = f"{(1 - pruned_channels/orig_channels)*100:.1f}%" if orig_channels != pruned_channels else "0%"
            
            # Mark pruned layers with *
            prune_marker = " *" if reduction != "0%" else ""
            print("{:<50} | {:>15} | {:>15} | {:>10}{}".format(
                name, orig_channels, pruned_channels, reduction, prune_marker
            ))
            
            total_original += sum(p.numel() for p in orig_module.parameters())
            total_pruned += sum(p.numel() for p in pruned_module.parameters())
    
    # Print totals
    print("\n{:<50} | {:>15,} | {:>15,} | {:>10.1f}%".format(
        "TOTAL PARAMETERS", 
        total_original, 
        total_pruned,
        (1 - total_pruned/total_original)*100
    ))
    
    # Print summary of pruned layers only
    print("\nPruned Layers Summary:")
    print("-"*100)
    for name, orig_module in orig_layers.items():
        if name in pruned_layers:
            pruned_module = pruned_layers[name]
            if orig_module.out_channels != pruned_module.out_channels:
                print("{:<50} {:>4} → {:<4} channels ({:>6} reduction)".format(
                    name,
                    orig_module.out_channels,
                    pruned_module.out_channels,
                    f"{int((1-pruned_module.out_channels/orig_module.out_channels)*100)}%"
                ))

if __name__ == "__main__":
    compare_architectures()