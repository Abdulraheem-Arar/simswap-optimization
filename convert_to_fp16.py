import torch
from models.models import create_model
from options.test_options import TestOptions

def smart_quantize_model():
    # 1. EXACT reproduction of inference script setup
    opt = TestOptions().parse()
    opt.name = 'people'
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.crop_size = 224
    opt.isTrain = False
    
    # Critical flags from inference script
    opt.no_flip = True  
    opt.no_dropout = True
    opt.no_antialias = False
    opt.preprocess = 'none'
    
    # 2. Create model IDENTICALLY to inference script
    torch.nn.Module.dump_patches = True  # Matches inference script
    model = create_model(opt).eval()
    
    # 3. Load weights with proper key handling
    pretrained_path = '/scratch/aa10947/SimSwap/checkpoints/people/latest_net_G.pth'
    print(f"Loading weights from {pretrained_path}...")
    
    try:
        state_dict = torch.load(pretrained_path)
        
        # Handle the 'netG.' prefix mismatch
        if not any(k.startswith('netG.') for k in state_dict.keys()):
            # Add prefix if missing (consistent with create_model behavior)
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'netArc' in k:
                    new_state_dict[k] = v  # Arcface weights stay unprefixed
                else:
                    new_state_dict[f'netG.{k}'] = v
            state_dict = new_state_dict
        
        # Load with strict=False to handle any residual mismatches
        model.load_state_dict(state_dict, strict=False)
        
        # Verify Arcface loading
        if not hasattr(model, 'netArc'):
            raise RuntimeError("netArc not found in model - architecture mismatch")
            
    except Exception as e:
        print(f"❌ Error loading weights: {str(e)}")
        return

    # 4. Protected layers (updated with netG prefix)
    PROTECTED_LAYERS = {
        'netArc', 'netG.last_layer', 'netG.up', 'netG.norm', 
        'netG.adain', 'netG.BottleNeck', 'netG.first_layer',
        'netG.down1', 'netG.down2', 'netG.down3'
    }

    # 5. Apply quantization
    print("\nApplying precision adjustment:")
    for name, param in model.named_parameters():
        if any(protected in name for protected in PROTECTED_LAYERS):
            param.data = param.data.float()
            print(f"🛡️ Protected (FP32): {name}")
        else:
            param.data = param.data.half()
            print(f"⚡ Quantized (FP16): {name}")

    # 6. Save with complete state
    output_path = 'simswap_optimized.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'opt': vars(opt),  # Save all options as dict
        'protected_layers': list(PROTECTED_LAYERS)
    }, output_path)
    
    print(f"\n✅ Successfully saved optimized model to {output_path}")

if __name__ == '__main__':
    smart_quantize_model()