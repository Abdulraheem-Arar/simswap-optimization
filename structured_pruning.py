import torch
from models.fs_networks import Generator_Adain_Upsample

# --- CONFIG ---
MODEL_PATH = "/scratch/aa10947/SimSwap/checkpoints/people/latest_net_G.pth"
PRUNE_RATIO = 0.2  # Remove 20% of channels
SAVE_PATH = MODEL_PATH.replace(".pth", "_pruned_20percent(input_corrected).pth")

def manual_prune_conv(conv_layer, prune_ratio):
    """Actually prune channels from a Conv2d or BatchNorm2d layer"""
    with torch.no_grad():
        if isinstance(conv_layer, torch.nn.Conv2d):
            # Handle Conv2d layers
            # 1. Calculate channel importance
            norms = torch.norm(conv_layer.weight.data, p=1, dim=(1,2,3))
            
            # 2. Determine channels to REMOVE (not keep)
            n_prune = int(conv_layer.out_channels * prune_ratio)
            prune_indices = torch.argsort(norms)[:n_prune]  # Weakest channels
            
            # 3. Create mask to KEEP channels
            mask = torch.ones(conv_layer.out_channels, dtype=torch.bool)
            mask[prune_indices] = False
            
            # 4. Apply pruning
            conv_layer.weight.data = conv_layer.weight.data[mask]
            if conv_layer.bias is not None:
                conv_layer.bias.data = conv_layer.bias.data[mask]
            
            # 5. Update layer's output channels
            conv_layer.out_channels = mask.sum().item()
            return mask
            
        elif isinstance(conv_layer, torch.nn.BatchNorm2d):
            # Handle BatchNorm2d layers
            # Use the same mask as the corresponding conv layer
            # We'll need to modify the main function to handle this
            pass

def main():
    # --- 1. LOAD MODEL ---
    print("1. Loading model...")
    model = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # --- 2. SELECT TARGET LAYERS ---
    print("\n2. Target layers:")
    # We'll prune these layers in a coordinated way
    targets = [
        # Down3 layer (256->512 originally, pruning to 256->410)
        (model.down3[0], model.down3[1]),
        
        # First bottleneck conv1 (512->512 originally, pruning to 410->410)
        (model.BottleNeck[0].conv1[1], None),
        
        # First bottleneck conv2 (512->512 originally, pruning to 410->410)
        (model.BottleNeck[0].conv2[1], None)
    ]
    
    # --- 3. PRUNE LAYERS ---
    print("\n3. Pruning results:")
    
    # First prune down3 layer (256->410)
    down3_conv, down3_bn = targets[0]
    down3_mask = manual_prune_conv(down3_conv, PRUNE_RATIO)
    if down3_bn is not None:
        with torch.no_grad():
            down3_bn.weight.data = down3_bn.weight.data[down3_mask]
            down3_bn.bias.data = down3_bn.bias.data[down3_mask]
            down3_bn.running_mean = down3_bn.running_mean[down3_mask]
            down3_bn.running_var = down3_bn.running_var[down3_mask]
            down3_bn.num_features = down3_mask.sum().item()
    print(f"  down3: {down3_conv.in_channels}→{down3_conv.out_channels}")
    
    # Now prune bottleneck layers using THE SAME CHANNEL COUNT (410)
    bottleneck_prune_count = 512 - 410  # Matching down3's output
    
    for conv_layer, _ in targets[1:]:
        with torch.no_grad():
            # Calculate importance using L1 norm
            norms = torch.norm(conv_layer.weight.data, p=1, dim=(1,2,3))
            
            # Get prune indices (weakest channels)
            prune_indices = torch.argsort(norms)[:bottleneck_prune_count]
            mask = torch.ones(conv_layer.out_channels, dtype=torch.bool)
            mask[prune_indices] = False
            
            # Prune output channels
            conv_layer.weight.data = conv_layer.weight.data[mask]
            if conv_layer.bias is not None:
                conv_layer.bias.data = conv_layer.bias.data[mask]
            conv_layer.out_channels = mask.sum().item()
            
            # Prune input channels to match down3's output
            conv_layer.weight.data = conv_layer.weight.data[:, down3_mask]
            conv_layer.in_channels = down3_mask.sum().item()
            
        print(f"  bottleneck: {conv_layer.in_channels}→{conv_layer.out_channels}")

    # --- 4. SAVE MODEL ---
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n4. Saved pruned model to:\n  {SAVE_PATH}")

if __name__ == "__main__":
    main()