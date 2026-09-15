"""PHASE 1 — Preprocessing: illumination norm, shadow-invariant, Hapke, pyramids."""
import cv2
import numpy as np


def to_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def clahe(img, clip=2.5, grid=8):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return c.apply(to_uint8(img))


def gamma_correct(img, target_mean=0.45):
    f = np.asarray(img, dtype=np.float32)
    f /= (f.max() or 1.0)
    m = f.mean()
    g = float(np.log(target_mean) / np.log(max(m, 1e-6)))
    g = float(np.clip(g, 0.4, 2.5))
    return (np.power(f, g) * 255).astype(np.uint8), g


def contrast_stretch(img, p=(2, 98)):
    f = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(f, p)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def shadow_invariant(img):
    """Gradient-coherence map: strong where structure survives shadows."""
    g = to_uint8(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 1.5)
    m = mag.max()
    return (mag / m * 255).astype(np.uint8) if m > 0 else np.zeros_like(g, np.uint8)


def hapke_correct(img, incidence_deg, strength=1.0):
    """First-order photometric normalization (Lommel-Seeliger style).

    Divides out expected brightness falloff with incidence angle so two
    sun geometries become comparable. strength=0 disables.
    """
    if not incidence_deg or strength <= 0:
        return to_uint8(img), 1.0
    inc = np.deg2rad(float(np.clip(incidence_deg, 0, 89)))
    mu0 = max(np.cos(inc), 0.05)
    mu = 1.0  # near-nadir emission assumed
    factor = (mu0 / (mu0 + mu)) / 0.5  # normalize to i=60deg reference
    f = np.asarray(img, dtype=np.float32)
    out = f / max(factor, 1e-3) ** float(strength)
    lo, hi = np.percentile(out, (1, 99))
    out = np.clip((out - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out, float(factor)


def build_pyramid(img, levels=3):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def preprocess(img, meta, level="standard"):
    """Run Phase-1 chain. level: light | standard | full (fallback-B)."""
    incidence = (meta or {}).get("incidence")
    base = to_uint8(img)
    c = clahe(base, clip=3.0 if level == "full" else 2.5)
    g, gamma = gamma_correct(c)
    s = contrast_stretch(g)
    h, hf = hapke_correct(s, incidence, strength=1.0 if level == "full" else 0.65)
    sh = shadow_invariant(h)
    # blend photometric + structural channels for matching robustness
    blend = cv2.addWeighted(h, 0.7, sh, 0.3, 0)
    info = {"gamma": round(gamma, 3), "hapke_factor": round(hf, 3),
            "level": level, "incidence": incidence}
    return {"base": base, "clahe": c, "gamma": g, "hapke": h,
            "shadow_inv": sh, "blend": blend, "info": info}
