import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.model import build_deeplabv3, forward_logits
from app.infer import pil_to_rgb, to_tensor, hysteresis, refine_mask

def read_mask_binary(path: Path, thr: int = 128) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m >= thr).astype(np.uint8)

def metrics_no_gt(mask01: np.ndarray) -> dict:
    crack_pixels = int(mask01.sum())
    num, _ = cv2.connectedComponents(mask01.astype(np.uint8), connectivity=8)
    components = max(0, num - 1)
    return {"crack_pixels": crack_pixels, "components": components}

def metrics_with_gt(pred01: np.ndarray, gt01: np.ndarray) -> dict:
    tp = int(((pred01==1) & (gt01==1)).sum())
    fp = int(((pred01==1) & (gt01==0)).sum())
    fn = int(((pred01==0) & (gt01==1)).sum())
    inter = tp
    union = int(((pred01==1) | (gt01==1)).sum())
    iou = inter / union if union > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "precision": prec, "recall": rec}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--masks_dir", type=str, default="")
    ap.add_argument("--hyst_low", type=float, default=0.30)
    ap.add_argument("--hyst_high", type=float, default=0.50)
    ap.add_argument("--open_k", type=int, default=3)
    ap.add_argument("--close_k", type=int, default=5)
    ap.add_argument("--min_area", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    ckpt = torch.load("app/best.pt", map_location=device)
    model = build_deeplabv3(num_classes=1).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None

    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        imgs = imgs[:args.limit]

    print("file,raw_components,ref_components,raw_pixels,ref_pixels,delta_components,delta_pixels,iou_raw,iou_ref,prec_raw,prec_ref,rec_raw,rec_ref")
    for p in imgs:
        pil = Image.open(str(p))
        rgb = pil_to_rgb(pil)
        x = to_tensor(rgb).to(device)

        with torch.no_grad():
            logits = forward_logits(model, x)
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()

        raw = hysteresis(prob, args.hyst_low, args.hyst_high).astype(np.uint8)
        ref = refine_mask(raw, open_k=args.open_k, close_k=args.close_k, min_area=args.min_area).astype(np.uint8)

        mr = metrics_no_gt(raw)
        mf = metrics_no_gt(ref)

        iou_r=iou_f=pr_r=pr_f=re_r=re_f=""
        if masks_dir is not None:
            mp = masks_dir / p.name
            if mp.exists():
                gt = read_mask_binary(mp)
                # resize gt to prediction size
                gt = cv2.resize(gt, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
                gr = metrics_with_gt(raw, gt)
                gf = metrics_with_gt(ref, gt)
                iou_r = f"{gr['iou']:.4f}"
                iou_f = f"{gf['iou']:.4f}"
                pr_r  = f"{gr['precision']:.4f}"
                pr_f  = f"{gf['precision']:.4f}"
                re_r  = f"{gr['recall']:.4f}"
                re_f  = f"{gf['recall']:.4f}"

        print(
            f"{p.name},"
            f"{mr['components']},{mf['components']},"
            f"{mr['crack_pixels']},{mf['crack_pixels']},"
            f"{mf['components']-mr['components']},{mf['crack_pixels']-mr['crack_pixels']},"
            f"{iou_r},{iou_f},{pr_r},{pr_f},{re_r},{re_f}"
        )

if __name__ == "__main__":
    main()
