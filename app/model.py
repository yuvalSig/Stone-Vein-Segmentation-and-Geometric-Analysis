import torch

def forward_logits(model, x):
    out = model(x)
    # DeepLab / torchvision returns dict {'out': tensor, ...}
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    # UNet and many custom models return tensor directly
    return out
