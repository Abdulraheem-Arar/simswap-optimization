import torch
import torch.nn as nn
import torch_pruning as tp
from options.test_options import TestOptions
from models.models import create_model
from copy import deepcopy

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

def prune_model(model, target_channels=410):
    model = deepcopy(model)
    
    # 1. Initialize pruning
    DG = tp.DependencyGraph()
    DG.build_dependency(GeneratorWrapper(model.netG), example_inputs=torch.randn(1, 3, 224, 224))
    
    # 2. Find and prune conv layers (512 channels)
    conv_layers = [m for m in model.netG.modules() 
                  if isinstance(m, nn.Conv2d) and m.out_channels == 512]
    
    if not conv_layers:
        raise ValueError("No conv layers with 512 channels found")
    
    pruning_idxs = tp.strategy.L1Strategy()(conv_layers[0].weight, 
                                         amount=(512-target_channels)/512)
    
    # 3. Prune conv layers
    for name, module in model.netG.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.out_channels == 512:
                plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs)
                if plan: 
                    plan.exec()
                    print(f"Pruned CONV {name} outputs")
            if module.in_channels == 512:
                plan = DG.get_pruning_plan(module, tp.prune_conv_in, idxs=pruning_idxs)
                if plan: 
                    plan.exec()
                    print(f"Pruned CONV {name} inputs")
    
    # 4. Prune style layers (ApplyStyle modules)
    style_count = 0
    for name, module in model.netG.named_modules():
        if 'style' in name and hasattr(module, 'linear'):
            style_count += 1
            with torch.no_grad():
                # Create new smaller layer
                new_fc = nn.Linear(512, target_channels * 2)
                
                # Copy kept weights (scale and bias)
                new_fc.weight[:target_channels] = module.linear.weight[pruning_idxs]
                new_fc.weight[target_channels:] = module.linear.weight[512 + torch.tensor(pruning_idxs)]
                
                new_fc.bias[:target_channels] = module.linear.bias[pruning_idxs]
                new_fc.bias[target_channels:] = module.linear.bias[512 + torch.tensor(pruning_idxs)]
                
                # Replace layer
                module.linear = new_fc
                print(f"Pruned style layer {name}")
    
    print(f"Pruned {style_count} style layers")
    return model

if __name__ == "__main__":
    print("Loading model...")
    model = load_pretrained_model()
    
    print("\nPruning model...")
    try:
        pruned_model = prune_model(model)
        
        print("\nVerifying pruned model...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = GeneratorWrapper(pruned_model.netG)(dummy_input)
            print(f"Forward pass successful! Output shape: {output.shape}")
            
            save_path = '/scratch/aa10947/SimSwap/checkpoints/people/pruned20_net_G.pth'
            torch.save({'netG': pruned_model.netG.state_dict()}, save_path)
            print(f"Model saved to {save_path}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nModel structure summary:")
        for name, module in model.netG.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) or 'style' in name:
                if hasattr(module, 'in_features'):
                    print(f"{name}: Linear (in={module.in_features}, out={module.out_features})")
                elif hasattr(module, 'in_channels'):
                    print(f"{name}: Conv2d (in={module.in_channels}, out={module.out_channels})")
                elif hasattr(module, 'linear'):
                    print(f"{name}: ApplyStyle (linear: in={module.linear.in_features}, out={module.linear.out_features})")