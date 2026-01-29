from __future__ import annotations
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A

from src.dataset import build_samples, CrackSegDataset
from src.model import forward_logits


# ===================== utils =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(patch_size: int = 512):
    # Patch extraction is handled inside CrackSegDataset (dataset.py).
    # For your case (thick cracks near borders), keep geometry mild.
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        # Mild geometry: small scale/rotate, no translate (helps border stability)
        A.Affine(scale=(0.95, 1.05), rotate=(-7, 7), translate_percent=(0.0, 0.0), p=0.35),
    ])
    val_tf = A.Compose([])  # dataset already returns patch_size x patch_size
    return train_tf, val_tf
