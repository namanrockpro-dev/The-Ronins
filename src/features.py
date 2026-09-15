"""PHASE 2 — Difficulty estimator, model selector, matchers (ORB/SIFT/LoFTR)."""
import cv2
import numpy as np

_loftr_model = None


def difficulty(img_a, img_b, meta_a=None, meta_b=None):
    """Score 0..1 (higher = harder) + contributing factors."""
    def feats(img):
        g = img if img.dtype == np.uint8 else _u8(img)
        blur = cv2.Laplacian(g, cv2.CV_64F).var()
        ent = _entropy(g)
        fast = cv2.FastFeatureDetector_create(threshold=25)
        n = len(fast.detect(g, None) or [])
        tex = min(1.0, n / 4000.0)
        return blur, ent, tex
    ba, ea, ta = feats(img_a)
    bb, eb, tb = feats(img_b)
    blur_s = 1.0 / (1.0 + min(ba, bb) / 120.0)
    tex_s = 1.0 - (ta + tb) / 2.0
    ent_s = abs(ea - eb) / 8.0
    sun_s = 0.0
    try:
        ia = (meta_a or {}).get("incidence")
        ib = (meta_b or {}).get("incidence")
        if ia and ib:
            sun_s = min(1.0, abs(ia - ib) / 20.0)
    except Exception:
        pass
    res_s = 0.0
    try:
        ra = max(img_a.shape) / max(img_b.shape)
        res_s = min(1.0, abs(np.log(max(ra, 1e-3))) / 2.5)
    except Exception:
        pass
    score = float(np.clip(0.25 * blur_s + 0.25 * tex_s + 0.1 * ent_s
                          + 0.3 * sun_s + 0.1 * res_s, 0, 1))
    return {"score": round(score, 3),
            "factors": {"blur": round(blur_s, 3), "texture": round(tex_s, 3),
                        "entropy_gap": round(min(ent_s, 1), 3),
                        "sun_gap": round(sun_s, 3), "res_gap": round(res_s, 3)}}


def select_model(score, preference="auto"):
    if preference in ("orb", "sift", "loftr"):
        return preference
    if score < 0.3:
        return "orb"
    if score < 0.55:
        return "sift"
    return "loftr"


def _u8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def match_orb(a, b, n=8000):
    a8, b8 = _u8(a), _u8(b)
    orb = cv2.ORB_create(nfeatures=n)
    k1, d1 = orb.detectAndCompute(a8, None)
    k2, d2 = orb.detectAndCompute(b8, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return [], [], []
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(d1, d2, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    p1 = np.float32([k1[x.queryIdx].pt for x in good])
    p2 = np.float32([k2[x.trainIdx].pt for x in good])
    return p1, p2, [x.distance for x in good]


def match_sift(a, b, n=8000):
    a8, b8 = _u8(a), _u8(b)
    try:
        sift = cv2.SIFT_create(nfeatures=n)
    except Exception:
        return match_orb(a, b, n)
    k1, d1 = sift.detectAndCompute(a8, None)
    k2, d2 = sift.detectAndCompute(b8, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return [], [], []
    m = cv2.BFMatcher(crossCheck=False).knnMatch(d1, d2, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]
    p1 = np.float32([k1[x.queryIdx].pt for x in good])
    p2 = np.float32([k2[x.trainIdx].pt for x in good])
    return p1, p2, [x.distance for x in good]


def match_loftr(a, b, max_side=880, conf_thresh=0.1):
    """Kornia LoFTR on GPU/CPU; raises on any failure (caller falls back)."""
    global _loftr_model
    import torch
    from kornia.feature import LoFTR
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a8, b8 = _u8(a), _u8(b)
    s = min(1.0, max_side / max(a8.shape))
    if s < 1.0:
        a8 = cv2.resize(a8, (int(a8.shape[1] * s), int(a8.shape[0] * s)))
    s2 = min(1.0, max_side / max(b8.shape))
    if s2 < 1.0:
        b8 = cv2.resize(b8, (int(b8.shape[1] * s2), int(b8.shape[0] * s2)))
    # pad to multiple of 8
    def pad(im):
        h, w = im.shape
        ph, pw = (-h) % 8, (-w) % 8
        return cv2.copyMakeBorder(im, 0, ph, 0, pw, cv2.BORDER_REFLECT), (h, w)
    a8p, ashp = pad(a8)
    b8p, bshp = pad(b8)
    if _loftr_model is None:
        _loftr_model = LoFTR(pretrained="outdoor").eval().to(dev)
    ta = torch.from_numpy(a8p)[None, None].float().to(dev) / 255.0
    tb = torch.from_numpy(b8p)[None, None].float().to(dev) / 255.0
    with torch.no_grad():
        out = _loftr_model({"image0": ta, "image1": tb})
    k0 = out["keypoints0"].cpu().numpy()
    k1 = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    m = conf > conf_thresh
    # map back to original (pre-resize) coords of caller images
    k0[:, 0] /= s
    k0[:, 1] /= s
    k1[:, 0] /= s2
    k1[:, 1] /= s2
    return (np.float32(k0[m]), np.float32(k1[m]),
            [float(1 - c) for c in conf[m]], {"device": dev})


def _entropy(g):
    h, _ = np.histogram(g, bins=32, range=(0, 255), density=True)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())
