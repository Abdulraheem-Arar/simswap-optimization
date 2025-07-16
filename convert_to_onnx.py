import torch
import torch.nn as nn
import torch.nn.functional as F

PRUNED_MODEL_PATH = "/scratch/aa10947/SimSwap/checkpoints/people/latest_net_G_pruned_20percent(input_corrected).pth"

class PrunedSimSwap(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial layers (unchanged)
        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Downsample layers
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 410, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(410),
            nn.ReLU()
        )
        
        # Bottleneck with proper channel transitions
        self.bottleneck = nn.ModuleList()
        
        # First block (pruned)
        self.bottleneck.append(ADainBlock(410, 410))  # 410→410
        
        # Transition block (critical!)
        self.bottleneck.append(nn.Sequential(
            nn.Conv2d(410, 512, kernel_size=1),
            nn.InstanceNorm2d(512),
            nn.ReLU()
        ))
        
        # Remaining blocks (original 512 channels)
        for _ in range(7):  # 1 transition + 7 original = 9 total blocks
            self.bottleneck.append(ADainBlock(512, 512))
        
        # Upsample layers
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x, latent):
        x = self.first_layer(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        for block in self.bottleneck:
            if isinstance(block, ADainBlock):
                x = block(x, latent)
            else:
                x = block(x)  # For the transition layer
                
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        return self.last_layer(x)

class ADainBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels  # If out_channels not provided, use in_channels
        
        self.resnet = ResnetBlock_Adain(in_channels, out_channels)
        self.style = ApplyStyle(out_channels)

    def forward(self, x, latent):
        x = self.resnet(x)
        x = self.style(x, latent)
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Use BatchNorm2d instead of InstanceNorm for export compatibility
        self.norm1 = nn.BatchNorm2d(out_channels, affine=False) if torch.onnx.is_in_onnx_export() \
                   else nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels, affine=False) if torch.onnx.is_in_onnx_export() \
                   else nn.InstanceNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels) if torch.onnx.is_in_onnx_export() \
                else nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual

class ApplyStyle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(512, channels*2)  # For scale and shift
        
    def forward(self, x, latent):
        style = self.linear(latent)
        style = style.view(style.size(0), 2, -1, 1, 1)
        gamma, beta = style.chunk(2, 1)
        return x * (1 + gamma.squeeze(1)) + beta.squeeze(1)

if __name__ == "__main__":
    model = PrunedSimSwap()
    model.eval()  # Must be in eval mode
    
    try:
        state_dict = torch.load(PRUNED_MODEL_PATH)
        model.load_state_dict(state_dict, strict=False)
        
        # Create dummy inputs with fixed batch size for export
        dummy_img = torch.randn(1, 3, 224, 224)
        dummy_latent = torch.randn(1, 512)
        
        # Verify
        with torch.no_grad():
            output = model(dummy_img, dummy_latent)
            print("Output shape:", output.shape)
        
        # Export with fixed batch size
        torch.onnx.export(
            model,
            (dummy_img, dummy_latent),
            "simswap_pruned.onnx",
            input_names=["input", "latent"],
            output_names=["output"],
            opset_version=13,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            dynamic_axes=None  # Disable dynamic axes for export
        )
        
        print("ONNX export successful!")
        
    except Exception as e:
        print(f"Error: {str(e)}")