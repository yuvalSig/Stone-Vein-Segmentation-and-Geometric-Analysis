import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplabv3(num_classes: int = 1) -> nn.Module:
    model = deeplabv3_resnet50(weights="DEFAULT")
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    return model


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return (out["out"] if isinstance(out, dict) and "out" in out else out)
