import torch
import torch.nn.functional as F

@torch.no_grad()
def sliding_prob(model, img_3chw, patch=512, overlap=128, device="cpu", batch_size=3):
    """
    Performs sliding window inference to handle large images that don't fit in GPU memory.
    
    img_3chw: Input tensor [3, H, W] in range 0..1
    returns:  Probability map [H, W] in range 0..1
    """
    assert img_3chw.ndim == 3 and img_3chw.shape[0] == 3
    x = img_3chw
    H, W = x.shape[1], x.shape[2]
    stride = patch - overlap
    if stride <= 0:
        raise ValueError("overlap must be < patch")

    # 1) Context padding: Reflect edges to improve prediction accuracy at tile boundaries
    ctx = patch // 2
    x = F.pad(x.unsqueeze(0), (ctx, ctx, ctx, ctx), mode="reflect").squeeze(0)

    # 2) Additional padding to ensure the image dimensions are multiples of the patch size
    Hp, Wp = x.shape[1], x.shape[2]
    pad_h = max(0, patch - Hp)
    pad_w = max(0, patch - Wp)
    x = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").to(device)
    _, _, Hp, Wp = x.shape

    # 3) Generate grid coordinates for overlapping tiles
    ys = list(range(0, Hp - patch + 1, stride))
    if ys[-1] != Hp - patch:
        ys.append(Hp - patch)
    xs = list(range(0, Wp - patch + 1, stride))
    if xs[-1] != Wp - patch:
        xs.append(Wp - patch)

    # Accumulators for summed probabilities and overlap counts
    acc = torch.zeros((1, 1, Hp, Wp), device=device)
    cnt = torch.zeros((1, 1, Hp, Wp), device=device)
    w = torch.ones((1, 1, patch, patch), device=device)

    coords = [(y0, x0) for y0 in ys for x0 in xs]

    # 4) Batch processing of tiles
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]

        # Extract tiles from padded image
        tiles = torch.cat(
            [x[..., y0:y0+patch, x0:x0+patch] for (y0, x0) in batch_coords],
            dim=0
        )  # [B,3,patch,patch]

        # Model forward pass
        logits = model(tiles)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        prob = torch.sigmoid(logits).float()  # [B,1,patch,patch]

        # Reconstruct full probability map by adding tiles back
        for b, (y0, x0) in enumerate(batch_coords):
            acc[..., y0:y0+patch, x0:x0+patch] += prob[b:b+1] * w
            cnt[..., y0:y0+patch, x0:x0+patch] += w

    # 5) Average overlapping predictions
    prob_full = acc / cnt.clamp_min(1e-6)
    prob_full = prob_full.squeeze(0).squeeze(0)

    # 6) Crop back to original dimensions (removes initial context padding)
    prob_full = prob_full[ctx:ctx+H, ctx:ctx+W].contiguous()
    prob_full = torch.nan_to_num(prob_full, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    
    return prob_full
