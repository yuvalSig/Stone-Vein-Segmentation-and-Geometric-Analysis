import numpy as np
import cv2


def skeletonize_opencv(bin01: np.ndarray) -> np.ndarray:
    """
    Reduces binary objects to 1-pixel wide lines (skeletons).
    Tries fast Zhang-Suen thinning first, falls back to morphological erosion if needed.
    
    bin01: HxW uint8 {0,1}
    returns: HxW uint8 {0,1}
    """
    bin01 = (bin01 > 0).astype(np.uint8)
    img = (bin01 * 255).astype(np.uint8)

    # Best option: use specialized thinning algorithm from opencv-contrib
    try:
        from cv2 import ximgproc  # type: ignore
        sk = ximgproc.thinning(img, thinningType=ximgproc.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: iterative morphological skeletonization (slower but robust)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = img.copy()
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return (skel > 0).astype(np.uint8)


def crack_table_from_mask(pred01: np.ndarray, min_skel_pixels: int = 25):
    """
    Analyzes a segmentation mask to extract physical properties of each crack.
    Returns a list of dictionaries containing length, orientation, and width.
    """
    pred01 = (pred01 > 0).astype(np.uint8)

    # Calculate distance to nearest background pixel (used for width estimation)
    dist = cv2.distanceTransform(pred01, cv2.DIST_L2, 3)

    # Label individual connected components (each crack is treated separately)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred01, connectivity=8)

    rows = []
    crack_id = 0

    for comp in range(1, n):
        comp_mask = (labels == comp).astype(np.uint8)

        # Get the skeleton of the current component
        skel = skeletonize_opencv(comp_mask)
        ys, xs = np.where(skel == 1)
        
        # Filter out noise or very small artifacts
        if ys.size < min_skel_pixels:
            continue

        crack_id += 1

        # Approximation: length is proportional to the number of skeleton pixels
        length_px = int(ys.size)

        # Orientation calculation using Principal Component Analysis (PCA) on coordinates
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean = pts.mean(axis=0, keepdims=True)
        X = pts - mean
        cov = (X.T @ X) / max(1, (X.shape[0] - 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        if angle < 0:
            angle += 180.0  # normalize to [0, 180) degrees

        # Width calculation: 2x distance from skeleton to edge
        w = 2.0 * dist[ys, xs]
        mean_w = float(np.mean(w)) if w.size else 0.0
        max_w = float(np.max(w)) if w.size else 0.0

        area_px = int(stats[comp, cv2.CC_STAT_AREA])

        rows.append({
            "length": length_px,
            "orientation": angle,
            "avg_width": mean_w,
            "max_width": max_w,
            "area": area_px,
        })

    # Sort results by length (longest cracks first)
    rows.sort(key=lambda r: r["length"], reverse=True)
    return rows
