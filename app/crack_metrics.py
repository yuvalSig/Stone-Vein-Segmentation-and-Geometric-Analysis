import numpy as np
import cv2


def skeletonize_opencv(bin01: np.ndarray) -> np.ndarray:
    """
    OpenCV-only skeletonization.
    Uses ximgproc.thinning if available, otherwise morphological skeletonization.
    bin01: HxW uint8 {0,1}
    returns: HxW uint8 {0,1}
    """
    bin01 = (bin01 > 0).astype(np.uint8)
    img = (bin01 * 255).astype(np.uint8)

    # Best option (opencv-contrib)
    try:
        from cv2 import ximgproc  # type: ignore
        sk = ximgproc.thinning(img, thinningType=ximgproc.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: morphological skeleton
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
    pred01: HxW uint8 {0,1} crack mask
    Returns: list of dict rows sorted by length descending
    """
    pred01 = (pred01 > 0).astype(np.uint8)

    # distance to background -> half width in pixels
    dist = cv2.distanceTransform(pred01, cv2.DIST_L2, 3)

    # connected components = separate cracks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred01, connectivity=8)

    rows = []
    crack_id = 0

    for comp in range(1, n):
        comp_mask = (labels == comp).astype(np.uint8)

        skel = skeletonize_opencv(comp_mask)
        ys, xs = np.where(skel == 1)
        if ys.size < min_skel_pixels:
            continue

        crack_id += 1

        # length ~ number of skeleton pixels
        length_px = int(ys.size)

        # orientation via PCA on skeleton points (x,y)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean = pts.mean(axis=0, keepdims=True)
        X = pts - mean
        cov = (X.T @ X) / max(1, (X.shape[0] - 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        if angle < 0:
            angle += 180.0  # normalize to [0,180)

        # width along skeleton = 2 * dist at skeleton pixels
        w = 2.0 * dist[ys, xs]
        mean_w = float(np.mean(w)) if w.size else 0.0
        max_w = float(np.max(w)) if w.size else 0.0

        area_px = int(stats[comp, cv2.CC_STAT_AREA])

        rows.append({
            
            "length_px": length_px,
            "orientation_deg": angle,
            "mean_width_px": mean_w,
            "max_width_px": max_w,
            "area_px": area_px,
        })

    rows.sort(key=lambda r: r["length_px"], reverse=True)
    rows = sorted(rows, key=lambda r: r["length_px"], reverse=True)
    return rows
