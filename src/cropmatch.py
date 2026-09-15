"""Crop-vs-full matching: small query crop located in one full-size image.

Outputs inlier AND outlier connection lines on both images plus a warped
overlap — the Colab Module-3/4/5 flow in miniature (SIFT-affine coarse lock,
LoFTR refine attempt, MAGSAC verify).
"""
import cv2
import numpy as np

from . import features as F
from . import geom as G


def match(crop, full, loftr_side=880):
    """Returns dict with p1/p2 (crop/full coords), inlier mask, H, metrics."""
    c8, f8 = F._u8(crop), F._u8(full)
    out = {"n_raw": 0, "model": None, "H": None}
    # Stage 1: sensitive SIFT, both directions kept via Lowe ratio
    try:
        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    except Exception:
        sift = None
    p1 = p2 = np.zeros((0, 2), np.float32)
    if sift is not None:
        k1, d1 = sift.detectAndCompute(c8, None)
        k2, d2 = sift.detectAndCompute(f8, None)
        if d1 is not None and d2 is not None and len(k1) >= 4 and len(k2) >= 4:
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
            try:
                m = flann.knnMatch(d1, d2, k=2)
                good = [x for x, y in m if len((x, y)) == 2 and x.distance < 0.75 * y.distance]
                p1 = np.float32([k1[x.queryIdx].pt for x in good])
                p2 = np.float32([k2[x.trainIdx].pt for x in good])
            except cv2.error:
                pass
    out["n_raw"] = int(len(p1))
    out["stage1"] = "sift"
    # Stage 2: if SIFT starves, LoFTR on crop vs downscaled full
    if len(p1) < 8:
        s = min(1.0, loftr_side / max(f8.shape))
        f_small = cv2.resize(f8, (int(f8.shape[1] * s), int(f8.shape[0] * s)),
                             interpolation=cv2.INTER_AREA) if s < 1 else f8
        try:
            q1, q2, _, _ = F.match_loftr(c8, f_small)
            p1 = np.asarray(q1, np.float32)
            p2 = np.asarray(q2, np.float32) / s
            out["stage1"] = "loftr"
            out["n_raw"] = int(len(p1))
        except Exception as e:
            out["error"] = str(e)[:150]
    if len(p1) < 4:
        out.update({"found": False, "reason": f"only {len(p1)} raw matches"})
        return out
    est = G.estimate(p1, p2, model="auto", thresh=4.0)
    inl = est["inliers"]
    out.update({"found": bool(est["n_inliers"] >= 8),
                "p1": np.asarray(p1, np.float32), "p2": np.asarray(p2, np.float32),
                "inliers": inl, "H": est["H"], "model": est["model"],
                "rmse": est["rmse"], "n_inliers": est["n_inliers"],
                "inlier_ratio": round(float(est["n_inliers"] / max(len(p1), 1)), 3)})
    return out


def render(crop, full, m, out_dir, max_lines=120):
    """Write inliers / outliers / combined / overlap PNGs. Returns filenames."""
    from pathlib import Path
    out_dir = Path(out_dir)
    c8, f8 = F._u8(crop), F._u8(full)
    ch, cw = c8.shape
    fh, fw = f8.shape
    H = m["H"]
    p1, p2, inl = m["p1"], m["p2"], m["inliers"]
    files = {}

    def canvas():
        h = max(ch, fh)
        cv = np.zeros((h, cw + fw, 3), np.uint8)
        cv[:ch, :cw] = cv2.cvtColor(c8, cv2.COLOR_GRAY2BGR)
        cv[:fh, cw:] = cv2.cvtColor(f8, cv2.COLOR_GRAY2BGR)
        return cv

    def lines(mask, color, name, title):
        cv = canvas()
        idx = np.where(mask)[0]
        sel = idx[::max(1, len(idx) // max_lines)] if len(idx) else idx
        for i in sel:
            cv2.line(cv, (int(p1[i][0]), int(p1[i][1])),
                     (int(p2[i][0] + cw), int(p2[i][1])), color, 1)
            cv2.circle(cv, (int(p1[i][0]), int(p1[i][1])), 3, color, -1)
            cv2.circle(cv, (int(p2[i][0] + cw), int(p2[i][1])), 3, color, -1)
        cv2.putText(cv, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / name), cv)
        files[name] = title

    if m.get("found"):
        lines(inl, (0, 220, 80), "inliers.png",
              f"INLIERS {inl.sum()}/{len(p1)}  rmse {m['rmse']:.2f}px")
        lines(~inl, (60, 60, 230), "outliers.png",
              f"OUTLIERS {(~inl).sum()}/{len(p1)} (rejected by MAGSAC)")
        # combined
        cv = canvas()
        for i in np.where(~inl)[0][:max_lines]:
            cv2.line(cv, (int(p1[i][0]), int(p1[i][1])),
                     (int(p2[i][0] + cw), int(p2[i][1])), (60, 60, 230), 1)
        for i in np.where(inl)[0][:max_lines]:
            cv2.line(cv, (int(p1[i][0]), int(p1[i][1])),
                     (int(p2[i][0] + cw), int(p2[i][1])), (0, 220, 80), 1)
        cv2.putText(cv, "green=inlier  red=outlier", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / "combined.png"), cv)
        files["combined.png"] = "inliers (green) + outliers (red)"
        # overlap: warp crop into full image + quad
        ov = cv2.cvtColor(f8, cv2.COLOR_GRAY2BGR)
        try:
            warped = cv2.warpPerspective(c8, np.float64(H), (fw, fh))
            mask = (warped > 0).astype(np.uint8) * 255
            for ch_i in range(3):
                chn = ov[:, :, ch_i]
                chn[mask > 0] = (0.5 * chn[mask > 0] + 0.5 *
                                 warped[mask > 0]).astype(np.uint8)
            corners = cv2.perspectiveTransform(
                np.float32([[[0, 0], [cw, 0], [cw, ch], [0, ch]]]),
                np.float64(H)).reshape(-1, 2)
            cv2.polylines(ov, [corners.astype(int)], True, (0, 255, 0), 3)
            files["overlap.png"] = "warped crop blended onto full image"
        except Exception:
            files["overlap.png"] = "warp failed"
        cv2.imwrite(str(out_dir / "overlap.png"), ov)
    else:
        cv = canvas()
        for i in range(min(len(p1), max_lines)):
            cv2.line(cv, (int(p1[i][0]), int(p1[i][1])),
                     (int(p2[i][0] + cw), int(p2[i][1])), (60, 60, 230), 1)
        cv2.putText(cv, f"NOT FOUND ({m.get('reason', '')})", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / "outliers.png"), cv)
        files["outliers.png"] = "all candidates rejected"
    return files
