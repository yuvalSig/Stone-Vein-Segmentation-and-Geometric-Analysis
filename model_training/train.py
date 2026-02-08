import torch.optim.lr_scheduler as lr_scheduler
import argparse, random, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from src.dataset import build_samples, CrackSegDataset
from src.model import build_unet, forward_logits

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1 - ((2. * inter + smooth) / (union + smooth)).mean()

def iou_from_logits(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) >= thr).float()
    inter = (p * y).sum(dim=(2,3))
    union = (p + y - p*y).sum(dim=(2,3)).clamp_min(1e-6)
    return (inter / union).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/crack-dataset")
    ap.add_argument("--img_size", "--patch_size", type=int, default=512, dest="patch_size")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--out_dir", type=str, default="outputs_unet")

    # Early stopping
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    # Scheduler (ReduceLROnPlateau)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_patience", type=int, default=1)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(Path(args.data_root))
    random.seed(args.seed)
    random.shuffle(samples)

    # 875 batches (7000/8 = 875)
    train_s = samples[:7000]
    val_s = samples[7000:]

    tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), translate_percent=(0.0, 0.03), p=0.5),
    ])

    ds_train = CrackSegDataset(train_s, transform=tf, patch_size=args.patch_size, random_crop=True)
    ds_val   = CrackSegDataset(val_s, patch_size=args.patch_size, random_crop=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    model = build_unet().to(device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        opt, mode="max",
        factor=args.sched_factor,
        patience=1,
        min_lr=args.sched_min_lr,
        verbose=True
    )

    print(f"UNet | Train Samples: {len(train_s)} | Batches: {len(dl_train)}")

    best_iou = -1.0
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(dl_train, desc=f"train {epoch}/{args.epochs}")
        for batch in pbar:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad()
            logits = forward_logits(model, x)
            loss = 0.5 * bce_criterion(logits, y) + 0.5 * dice_loss(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        ious = []
        with torch.no_grad():
            for batch in dl_val:
                x, y = batch["image"].to(device), batch["mask"].to(device)
                logits = forward_logits(model, x)
                ious.append(iou_from_logits(logits, y, thr=0.5))

        val_iou = float(np.mean(ious)) if ious else 0.0
        print(f"Epoch {epoch}: train_loss={loss_sum/len(dl_train):.4f} val_iou={val_iou:.4f}")

        # Scheduler step
        scheduler.step(val_iou)

        lr_now = opt.param_groups[0]['lr']
        print(f"  LR={lr_now:.2e}")

        # Best + early stop
        if val_iou > best_iou + args.early_stop_min_delta:
            best_iou = val_iou
            no_improve_epochs = 0
            torch.save(model.state_dict(), out_dir / "best_unet.pt")
            print(f"  >>> New Best IoU: {best_iou:.4f} - Saved!")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best_iou={best_iou:.4f})")
                break

if __name__ == "__main__":
    main()
