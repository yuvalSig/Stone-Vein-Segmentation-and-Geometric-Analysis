from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

# Global configuration for standard model input size
IMG_SIZE = 448  # match your training resize


def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    """Converts PIL image to RGB numpy array and resizes to model input dimensions."""
    rgb = np.array(pil_img.convert("RGB"))
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return rgb


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Normalizes image and converts numpy array to Torch tensor (BCHW)."""
    x = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    return x


def hysteresis(prob: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Performs dual-threshold filtering. 
    Keeps weak predictions only if they are connected to strong predictions.
    """
    strong = (prob >= high).astype(np.uint8)
    weak = (prob >= low).astype(np.uint8)

    num, labels = cv2.connectedComponents(weak, connectivity=8)
    if num <= 1:
        return np.zeros_like(strong)

    strong_labels = np.unique(labels[strong == 1])
    keep = np.isin(labels, strong_labels)
    keep[labels == 0] = False
    return keep.astype(np.uint8)

def refine_mask(mask: np.ndarray, open_k: int = 3, close_k: int = 5, min_area: int = 80) -> np.ndarray:
    """
    Applies morphological operations to remove noise, close gaps, 
    and filter out small components by area.
    """
    m = (mask > 0).astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    # 1) remove tiny noise
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

    # 2) connect nearby segments / close small gaps
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)

    # 3) remove small connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cleaned = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    return cleaned

def closing_only(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Applies a simple morphological closing to bridge small discontinuities 
    while preserving thin crack structures.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

def pil_to_rgb_full(pil_img: Image.Image) -> np.ndarray:
    """Converts PIL image to RGB numpy array without resizing for high-res inference."""
    return np.array(pil_img.convert("RGB"))
