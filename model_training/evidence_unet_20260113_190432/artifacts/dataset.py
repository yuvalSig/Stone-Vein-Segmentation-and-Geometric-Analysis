from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Optional[Path]


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _read_mask_binary(path: Path, thr: int = 128) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m >= thr).astype(np.uint8) if m is not None else None


def build_samples(root: Path) -> List[Sample]:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    pos_img_dir = root / "Concrete" / "Concrete" / "Positive" / "Images"
    pos_msk_dir = root / "Concrete" / "Concrete" / "Positive" / "Masks"
    neg_img_dir = root / "Concrete" / "Concrete" / "Negative" / "Images"

    pos_names = sorted(set(p.name for p in pos_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS))
    samples = [Sample(pos_img_dir / n, pos_msk_dir / n) for n in pos_names if (pos_msk_dir / n).exists()]

    neg_imgs = sorted(p for p in neg_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    samples += [Sample(p, None) for p in neg_imgs]
    return samples


def _pad_to_at_least(img: np.ndarray, mask: np.ndarray, patch: int):
    """Pad image/mask so that H,W >= patch.
    Image: reflect padding (keeps texture natural)
    Mask: constant 0 padding (avoids creating fake cracks by reflection)
    """
    h, w = img.shape[:2]
    pad_h = max(0, patch - h)
    pad_w = max(0, patch - w)
    if pad_h == 0 and pad_w == 0:
        return img, mask

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_p = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    msk_p = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return img_p, msk_p


class CrackSegDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        transform=None,
        patch_size: int = 512,
        random_crop: bool = True,
        edge_prob: float = 0.35,
        edge_jitter: int = 16,
    ):
        self.samples = samples
        self.transform = transform
        self.patch_size = int(patch_size)
        self.random_crop = bool(random_crop)
        self.edge_prob = float(edge_prob)
        self.edge_jitter = int(edge_jitter)

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_top_left(self, h: int, w: int):
        """Edge-aware crop selection.
        With probability edge_prob, snap crop to an image edge (with small jitter).
        Otherwise uniform random crop.
        """
        P = self.patch_size

        if not self.random_crop:
            return (h - P) // 2, (w - P) // 2

        max_top = h - P
        max_left = w - P

        if np.random.rand() < self.edge_prob:
            # choose near-top or near-bottom
            if np.random.rand() < 0.5:
                top = 0
                top += np.random.randint(0, min(self.edge_jitter, max_top) + 1)
            else:
                top = max_top
                top -= np.random.randint(0, min(self.edge_jitter, max_top) + 1)

            # choose near-left or near-right
            if np.random.rand() < 0.5:
                left = 0
                left += np.random.randint(0, min(self.edge_jitter, max_left) + 1)
            else:
                left = max_left
                left -= np.random.randint(0, min(self.edge_jitter, max_left) + 1)

            return int(top), int(left)

        # uniform random crop
        top = np.random.randint(0, max_top + 1)
        left = np.random.randint(0, max_left + 1)
        return int(top), int(left)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img = _read_rgb(s.image_path)
        if s.mask_path:
            mask = _read_mask_binary(s.mask_path)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # pad if needed (instead of resizing up/down which distorts crack thickness)
        img, mask = _pad_to_at_least(img, mask, self.patch_size)

        h, w = img.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            top, left = self._choose_top_left(h, w)
            img = img[top:top + self.patch_size, left:left + self.patch_size]
            mask = mask[top:top + self.patch_size, left:left + self.patch_size]
        else:
            # should not happen due to padding, but keep as a safety fallback
            img = cv2.resize(img, (self.patch_size, self.patch_size))
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = mask.astype(np.float32)[None, ...]
        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask)}
