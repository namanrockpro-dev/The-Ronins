"""Crop-query search (Colab Module-3 logic): a small crop is located in the
database — coarse SIFT-affine ranking over all products, then LoFTR ROI
verification of the top hits with overlap quad + connection lines."""
import time
import uuid

import cv2
import numpy as np

from . import features as F
from . import geom as G


def coarse_score(crop, db_img):
    """SIFT similarity-affine inliers of crop inside db image (fast, small)."""
    c8 = F._u8(crop)
    d8 = F._u8(db_img)
    s = min(1.0, 900 / max(d8.shape))
    if s < 1.0:
        d8 = cv2.resize(d8, (int(d8.shape[1] * s), int(d8.shape[0] * s)),
                        interpolation=cv2.INTER_AREA)
    try:
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
    except Exception:
        return None
    k1, des1 = sift.detectAndCompute(c8, None)
    k2, des2 = sift.detectAndCompute(d8, None)
    if des1 is None or des2 is None or len(k1) < 4 or len(k2) < 4:
        return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    try:
        m = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None
    good = [x for x, y in m if len((x, y)) == 2 and x.distance < 0.75 * y.distance]
    if len(good) < 4:
        return {"inliers": 0, "raw": len(good), "M": None, "scale": s}
    p1 = np.float32([k1[x.queryIdx].pt for x in good])
    p2 = np.float32([k2[x.trainIdx].pt for x in good]) / s
    M, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
    inl = int(mask.sum()) if mask is not None else 0
    return {"inliers": inl, "raw": len(good), "M": M, "scale": s}


def verify_roi(crop, db_img, M_affine):
    """Colab Module-4 style: warp crop, LoFTR inside ROI, LK refine, RMSE."""
    hD, wD = db_img.shape[:2]
    hC, wC = crop.shape[:2]
    if M_affine is None:
        return None
    warped = cv2.warpAffine(F._u8(crop), M_affine, (wD, hD))
    corners = cv2.transform(np.float32(
        [[[0, 0], [wC, 0], [wC, hC], [0, hC]]]), M_affine).reshape(-1, 2)
    x0 = max(0, int(corners[:, 0].min()))
    y0 = max(0, int(corners[:, 1].min()))
    x1 = min(wD, int(corners[:, 0].max()))
    y1 = min(hD, int(corners[:, 1].max()))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    pad = int(max(x1 - x0, y1 - y0) * 0.10)
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(wD, x1 + pad), min(hD, y1 + pad)
    roi = F._u8(db_img)[ry0:ry1, rx0:rx1]
    wA = warped[ry0:ry1, rx0:rx1]
    try:
        kA, kB, conf, _ = F.match_loftr(wA, roi)
    except Exception:
        return None
    if len(kA) < 8:
        return {"inliers": 0, "rmse": float("inf"), "quad": corners,
                "roi": (rx0, ry0, rx1, ry1)}
    kB_full = kB + np.float32([rx0, ry0])
    H, mask = cv2.findHomography(np.float32(kA), np.float32(kB),
                                 cv2.USAC_MAGSAC, 3.0)
    if H is None or mask is None:
        return {"inliers": 0, "rmse": float("inf"), "quad": corners,
                "roi": (rx0, ry0, rx1, ry1)}
    inl = mask.ravel().astype(bool)
    r = float(np.sqrt(np.mean(np.sum(
        (cv2.perspectiveTransform(np.float32(kA[inl]).reshape(-1, 1, 2), H)
         .reshape(-1, 2) - np.float32(kB[inl])) ** 2, axis=1)))) if inl.sum() else float("inf")
    return {"inliers": int(inl.sum()), "rmse": r, "quad": corners,
            "roi": (rx0, ry0, rx1, ry1), "kA": np.float32(kA[inl]),
            "kB": np.float32(kB_full[inl]), "H": H}


def search_db(crop, db_items, top_coarse=6):
    """db_items: list of (id, preview_gray). Returns ranked hits."""
    scored = []
    for pid, img in db_items:
        t0 = time.time()
        c = coarse_score(crop, img)
        if c is None or c["inliers"] < 4:
            continue
        scored.append({"id": pid, "coarse_inliers": c["inliers"],
                       "raw": c["raw"], "M": c["M"],
                       "ms": int((time.time() - t0) * 1000)})
    scored.sort(key=lambda x: -x["coarse_inliers"])
    return scored[:top_coarse]
