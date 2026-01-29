import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        rev = list(reversed(features))
        up_ch = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_ch, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(up_ch, f))
            up_ch = f

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]

            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)

        return self.head(x)
