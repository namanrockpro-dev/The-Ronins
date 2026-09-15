"""PHASE 3 — Geometric verification: MAGSAC++ -> GC-RANSAC -> sub-pixel."""
import cv2
import numpy as np


def _usac(name, default):
    return getattr(cv2, name, default)


def estimate(p1, p2, model="auto", thresh=3.0):
    """Robust estimate. Returns dict(H, inliers, model, rmse_inliers)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    out = {"H": None, "inliers": np.zeros(len(p1), bool), "model": None,
           "rmse": float("inf"), "n_inliers": 0, "method": None}
    if len(p1) < 8:
        return out
    cands = ["homography", "affine"] if model == "auto" else [model]
    best = None
    for m in cands:
        H, inl = _fit(p1, p2, m, thresh, "magsac")
        if H is None:
            continue
        # GC-RANSAC local refinement
        H2, inl2 = _fit(p1[inl] if inl.sum() > 8 else p1,
                        p2[inl] if inl.sum() > 8 else p2, m,
                        max(1.5, thresh / 2), "gcransac")
        if H2 is not None:
            H, m2 = H2, "gcransac"
            inl = _inliers(p1, p2, H, m, thresh)
        else:
            m2 = "magsac"
        r = _rmse(p1[inl], p2[inl], H, m) if inl.sum() >= 4 else float("inf")
        score = (inl.sum(), -r)
        if best is None or score > best[0]:
            best = (score, {"H": H, "inliers": inl, "model": m, "rmse": r,
                            "n_inliers": int(inl.sum()), "method": m2})
    return best[1] if best else out


def _fit(p1, p2, model, thresh, kind):
    if len(p1) < (4 if model == "homography" else 3):
        return None, np.zeros(len(p1), bool)
    if kind == "magsac":
        meth = _usac("USAC_MAGSAC", cv2.RANSAC)
    elif kind == "gcransac":
        meth = _usac("USAC_GC", _usac("USAC_MAGSAC", cv2.RANSAC))
    else:
        meth = cv2.RANSAC
    try:
        if model == "homography":
            H, mask = cv2.findHomography(p1, p2, meth, thresh, maxIters=5000)
            if H is None:
                return None, np.zeros(len(p1), bool)
            return H, mask.ravel().astype(bool)
        H, mask = cv2.estimateAffine2D(p1, p2, meth, thresh, maxIters=5000)
        if H is None:
            return None, np.zeros(len(p1), bool)
        H = np.vstack([H, [0, 0, 1]])
        return H, mask.ravel().astype(bool)
    except cv2.error:
        return None, np.zeros(len(p1), bool)


def _apply(H, pts, model):
    p = np.asarray(pts, dtype=np.float64)
    if model == "affine":
        return (p @ H[:2, :2].T) + H[:2, 2]
    h = np.column_stack([p, np.ones(len(p))]) @ H.T
    return h[:, :2] / h[:, 2:3]


def _inliers(p1, p2, H, model, thresh):
    return np.linalg.norm(_apply(H, p1, model) - p2, axis=1) < thresh


def _rmse(p1, p2, H, model):
    if len(p1) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((_apply(H, p1, model) - p2) ** 2, axis=1))))


def refine_subpixel(img, pts, win=11):
    """cornerSubPix refinement on image A keypoints."""
    g = img if img.dtype == np.uint8 else _u8(img)
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    try:
        return cv2.cornerSubPix(g, p, (win, win), (-1, -1), crit).reshape(-1, 2)
    except cv2.error:
        return np.asarray(pts, dtype=np.float32)


def _u8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
