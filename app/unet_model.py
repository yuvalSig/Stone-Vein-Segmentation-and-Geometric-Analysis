import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Core building block: (Conv -> BatchNorm -> ReLU) * 2
    Keeps spatial dimensions the same via padding=1.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    """
    Standard UNet architecture for image segmentation.
    Includes Down-sampling (Encoder), Bottleneck, and Up-sampling (Decoder) with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        # Build Encoder: Downward path
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Latent space representation
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Build Decoder: Upward path
        rev = list(reversed(features))
        up_ch = features[-1]*2
        for f in rev:
            # Transposed convolution to double spatial size
            self.ups.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            # DoubleConv after concatenating skip connection
            self.ups.append(DoubleConv(up_ch, f))
            up_ch = f

        # Final 1x1 convolution to produce target masks
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        # Encoder: Save feature maps for skip connections
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        # Decoder: Upsample and concatenate with skip connections
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Transpose Conv
            skip = skips[i//2]
            
            # Pad if spatial dimensions don't match due to odd input sizes
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
            
            # Skip connection (concatenation along channel dim)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x) # DoubleConv

        return self.head(x)
