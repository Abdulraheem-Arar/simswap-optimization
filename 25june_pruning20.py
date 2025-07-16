import torch
import torch.nn as nn
import torch_pruning as tp
from torch.nn.utils import prune
from copy import deepcopy
from options.test_options import TestOptions
from models.models import create_model

##############################################
# 1. YOUR EXISTING MODEL LOADING CODE (UNCHANGED)
##############################################

class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)  # Changed from fc to linear to match your code

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

def load_pretrained_model():
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.crop_size = 224
    opt.gpu_ids = []
    opt.isTrain = False
    
    model = create_model(opt)
    model.eval()
    
    state_dict = torch.load('/scratch/aa10947/SimSwap/checkpoints/people/latest_net_G.pth', 
                          map_location='cpu')
    model.netG.load_state_dict(state_dict['netG'] if 'netG' in state_dict else state_dict, strict=False)
    return model

class GeneratorWrapper(nn.Module):
    def __init__(self, netG):
        super().__init__()
        self.netG = netG
        
    def forward(self, img):
        dlatents = torch.randn(img.size(0), 512).to(img.device)
        dlatents = dlatents / torch.norm(dlatents, p=2, dim=1, keepdim=True)
        return self.netG(img, dlatents=dlatents)

##############################################
# 2. PRUNING FUNCTIONS (ADAPTED TO YOUR SETUP)
##############################################

def find_style_layers(module, parent_name=""):
    """Recursively find all ApplyStyle layers"""
    style_layers = []
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, ApplyStyle):
            style_layers.append((full_name, child))
        else:
            style_layers.extend(find_style_layers(child, full_name))
    return style_layers

def prune_style_layers(model, pruning_idxs, original=512, target=410):
    for name, module in model.named_modules():
        if isinstance(module, ApplyStyle):
            # Original layer: nn.Linear(512, 1024)
            old_linear = module.linear
            
            # Create new layer: nn.Linear(512, 820)
            new_linear = nn.Linear(old_linear.in_features, target*2)
            
            # Copy kept weights (both scale and bias)
            new_linear.weight[:target] = old_linear.weight[pruning_idxs]  # Scale part
            new_linear.weight[target:] = old_linear.weight[original + pruning_idxs]  # Bias part
            
            new_linear.bias[:target] = old_linear.bias[pruning_idxs]
            new_linear.bias[target:] = old_linear.bias[original + pruning_idxs]
            
            module.linear = new_linear

def prune_model_hybrid(model, original_channels=512, target_channels=410):
    """Main pruning function with better error handling"""
    model = deepcopy(model)
    
    # 1. First find all style layers to verify they exist
    style_layers = find_style_layers(model.netG)
    if not style_layers:
        style_layers = [(name, module) for name, module in model.netG.named_modules() 
                       if isinstance(module, ApplyStyle)]
    
    if not style_layers:
        raise ValueError("Critical Error: No ApplyStyle layers found in model.netG. Possible causes:\n"
                       "1. Model architecture differs from expected\n"
                       "2. ApplyStyle layers are named differently\n"
                       "3. Model is not properly initialized")
    
    print(f"Found {len(style_layers)} style layers to prune")
    
    # 2. Proceed with pruning
    DG = tp.DependencyGraph()
    DG.build_dependency(GeneratorWrapper(model.netG), example_inputs=torch.randn(1, 3, 224, 224))
    
    # Get pruning indices
    example_conv = next(module for module in model.netG.modules() 
                       if isinstance(module, nn.Conv2d) and module.out_channels == original_channels)
    pruning_idxs = tp.strategy.L1Strategy()(example_conv.weight, 
                                         amount=(original_channels-target_channels)/original_channels)
    
    print(f"Pruning from {original_channels} to {target_channels} channels")
    
    # 3. Prune conv layers
    for name, module in model.netG.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.out_channels == original_channels:
                plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs)
                if plan:
                    plan.exec()
                    print(f"Pruned CONV {name} outputs")
            
            if module.in_channels == original_channels:
                plan = DG.get_pruning_plan(module, tp.prune_conv_in, idxs=pruning_idxs)
                if plan:
                    plan.exec()
                    print(f"Pruned CONV {name} inputs")
    
    # 4. Prune style layers
    prune_style_layers(model, pruning_idxs, original_channels, target_channels)
    
    return model

##############################################
# 3. VERIFICATION AND EXECUTION
##############################################

def verify_pruning(model):
    """Test if pruned model works"""
    print("\n=== VERIFICATION ===")
    
    # Count style layers
    style_count = sum(1 for _ in model.netG.modules() if isinstance(_, ApplyStyle))
    print(f"Found {style_count} style layers")
    
    # Test forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = GeneratorWrapper(model.netG)(dummy_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Loading original model...")
    original_model = load_pretrained_model()
    
    print("\nPruning model...")
    pruned_model = prune_model_hybrid(original_model)
    
    print("\nVerifying pruned model...")
    if verify_pruning(pruned_model):
        save_path = '/scratch/aa10947/SimSwap/checkpoints/people/pruned20_net_G.pth'
        torch.save({'netG': pruned_model.netG.state_dict()}, save_path)
        print(f"\nPruned model saved to {save_path}")
    else:
        print("\nPruning verification failed - model not saved")