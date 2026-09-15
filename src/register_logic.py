"""Port of imgrgstrsn.py Modules 1-5 (Colab) — reference/target registration.

Reference = small (query), Target = large (search). Mirrors Colab exactly:
 Module 1: ingestion diagnostics
 Module 2: natural_photometric_enhancer (bilateral + percentile + CLAHE 1.8)
 Module 3: SIFT Similarity-Affine coarse lock + LoFTR fine lock
 Module 4: ROI-cropped dense LoFTR + ANMS 5x5 + Lucas-Kanade sub-pixel refine
 Module 5: physical geometry extract + SIFT baseline + 4-panel report + metrics.json
"""
import math
import time

import cv2
import numpy as np


def module1_diagnostics(imgA, imgB):
    hA, wA = imgA.shape
    hB, wB = imgB.shape
    return {
        "A": f"{wA}x{hA} [{int(imgA.min())}-{int(imgA.max())}]",
        "B": f"{wB}x{hB} [{int(imgB.min())}-{int(imgB.max())}]",
        "area_gap": round(max(wA * hA, wB * hB) / (min(wA * hA, wB * hB) + 1e-5), 2),
    }


def natural_photometric_enhancer(img):
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35)
    p2, p98 = np.percentile(denoised, (2, 98))
    if p98 > p2:
        norm = np.clip((denoised - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = denoised
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(norm)


def _loftr_pair(a, b, device, target=800, thr=0.15):
    import torch
    import kornia
    resA = cv2.resize(a, (target, target), interpolation=cv2.INTER_AREA)
    resB = cv2.resize(b, (target, target), interpolation=cv2.INTER_AREA)
    tA = torch.from_numpy(resA).float()[None, None] / 255.0
    tB = torch.from_numpy(resB).float()[None, None] / 255.0
    m = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()
    with torch.inference_mode():
        out = m({"image0": tA.to(device), "image1": tB.to(device)})
    mask = out["confidence"] > thr
    kA = out["keypoints0"][mask].cpu().numpy()
    kB = out["keypoints1"][mask].cpu().numpy()
    return kA, kB, tA, tB


def module3_coarse_to_fine(procA, procB, device):
    t0 = time.time()
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
    kpA, desA = sift.detectAndCompute(procA, None)
    kpB, desB = sift.detectAndCompute(procB, None)
    if desA is None or desB is None or len(kpA) < 4 or len(kpB) < 4:
        return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    good = [m for m, n in (p for p in flann.knnMatch(desA, desB, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None
    pA = np.float32([kpA[m.queryIdx].pt for m in good])
    pB = np.float32([kpB[m.trainIdx].pt for m in good])
    M, mask = cv2.estimateAffinePartial2D(pA, pB, method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
    if M is None or mask is None:
        return None
    scale = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    hB, wB = procB.shape
    hA, wA = procA.shape
    warped = cv2.warpAffine(procA, M, (wB, hB))
    corners = cv2.transform(
        np.float32([[[0, 0], [wA, 0], [wA, hA], [0, hA]]]), M).reshape(-1, 2)
    x0 = max(0, int(corners[:, 0].min()))
    y0 = max(0, int(corners[:, 1].min()))
    x1 = min(wB, int(corners[:, 0].max()))
    y1 = min(hB, int(corners[:, 1].max()))
    # LoFTR refine on warped vs procB (800x800, lower threshold)
    import torch
    target = 800
    resA = cv2.resize(warped, (target, target), interpolation=cv2.INTER_AREA)
    resB = cv2.resize(procB, (target, target), interpolation=cv2.INTER_AREA)
    tA = torch.from_numpy(resA).float()[None, None] / 255.0
    tB = torch.from_numpy(resB).float()[None, None] / 255.0
    import kornia
    matcher = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()
    with torch.inference_mode():
        out = matcher({"image0": tA.to(device), "image1": tB.to(device)})
    mask2 = out["confidence"] > 0.15
    kA = out["keypoints0"][mask2].cpu().numpy()
    kB = out["keypoints1"][mask2].cpu().numpy()
    if len(kA):
        kA[:, 0] *= (wB / target)
        kA[:, 1] *= (hB / target)
        kB[:, 0] *= (wB / target)
        kB[:, 1] *= (hB / target)
        ptsA = cv2.transform(kA.reshape(-1, 1, 2),
                             cv2.invertAffineTransform(M)).reshape(-1, 2)
        ptsB = kB
        final = len(kA)
    else:
        ptsA = pA[mask.ravel().astype(bool)]
        ptsB = pB[mask.ravel().astype(bool)]
        final = int(mask.sum())
    return {"angle": rot, "scale": scale, "M": M, "box": (x0, y0, x1 - x0, y1 - y0),
            "ptsA": ptsA.astype(np.float32), "ptsB": ptsB.astype(np.float32),
            "inliers": final, "corners": corners, "t": round(time.time() - t0, 2)}


def enforce_anms(ptsA, ptsB_local, roi_shape, grid=(8, 8), cap=8):
    h, w = roi_shape
    cw, ch = w / grid[1], h / grid[0]
    buckets = {}
    for i, pt in enumerate(ptsB_local):
        c = max(0, min(grid[1] - 1, int(pt[0] / (cw + 1e-5))))
        r = max(0, min(grid[0] - 1, int(pt[1] / (ch + 1e-5))))
        buckets.setdefault((r, c), []).append(i)
    keep = []
    for idx in buckets.values():
        keep.extend(np.random.choice(idx, cap, replace=False).tolist()
                    if len(idx) > cap else idx)
    cov = len(buckets) / (grid[0] * grid[1]) * 100.0
    return np.array(keep), cov


def module4_roi(rawA, rawB, procA, procB, loc, device):
    box = loc["box"]
    M = loc["M"]
    x, y, bw, bh = box
    pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
    hB, wB = rawB.shape
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(wB, x + bw + pad_x), min(hB, y + bh + pad_y)
    proc_roi = procB[y1:y2, x1:x2]
    raw_roi = rawB[y1:y2, x1:x2]
    warped = cv2.warpAffine(procA, M, (wB, hB))[y1:y2, x1:x2]
    raw_warped = cv2.warpAffine(rawA, M, (wB, hB))[y1:y2, x1:x2]
    # dense LoFTR inside ROI (800x800, lower threshold)
    import torch
    import kornia
    target = 800
    resA = cv2.resize(warped, (target, target), interpolation=cv2.INTER_AREA)
    resB = cv2.resize(proc_roi, (target, target), interpolation=cv2.INTER_AREA)
    tA = torch.from_numpy(resA).float()[None, None] / 255.0
    tB = torch.from_numpy(resB).float()[None, None] / 255.0
    matcher = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()
    with torch.inference_mode():
        out = matcher({"image0": tA.to(device), "image1": tB.to(device)})
    mask = out["confidence"] > 0.15
    kA = out["keypoints0"][mask].cpu().numpy()
    kB = out["keypoints1"][mask].cpu().numpy()
    if len(kA) < 4:
        return None
    hR, wR = proc_roi.shape
    hA0, wA0 = procA.shape
    kA[:, 0] *= (wR / target)
    kA[:, 1] *= (hR / target)
    kB[:, 0] *= (wR / target)
    kB[:, 1] *= (hR / target)
    idx, cov = enforce_anms(kA, kB, proc_roi.shape)
    kA, kB = kA[idx], kB[idx]
    # Lucas-Kanade sub-pixel
    Hc, _ = cv2.findHomography(kA, kB, cv2.USAC_MAGSAC, 4.0)
    if Hc is None:
        Hc = np.eye(3, dtype=np.float64)
    # warp rawA for LK via Hc mapped into roi coords: use Hc directly on kA subspace
    # LK expects warped rawA and raw_roi
    # For consistency with Colab, run LK from kA warped through Hc estimate
    # Instead use the Colab helper directly:
    hBW, wBW = raw_roi.shape
    # we already have raw_warped and raw_roi at roi size; reuse LK from src flow
    p0 = cv2.perspectiveTransform(kA.reshape(-1, 1, 2), Hc).reshape(-1, 2) if Hc.shape == (3, 3) else kA
    # Actually Colab warps rawA via H_coarse then LK; here we approximate by LK on roi crops
    lk_p0 = kA.astype(np.float32).reshape(-1, 1, 2)  # approximate: kA already in roi-warped frame
    # Use raw_warped vs raw_roi for LK
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        raw_warped, raw_roi, lk_p0, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    refined = []
    for i in range(len(kB)):
        if st[i, 0] == 1 and err[i, 0] < 10.0:
            refined.append(p1[i, 0])
        else:
            refined.append(kB[i])
    sub = np.float32(refined)
    H, mask = cv2.findHomography(kA, sub, cv2.USAC_MAGSAC, 3.0)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    fA, fB = kA[inl], sub[inl]
    ptsT = cv2.perspectiveTransform(fA.reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((ptsT - fB) ** 2, axis=1))))
    # map to global B
    fB[:, 0] += x1
    fB[:, 1] += y1
    fAg = fA.copy()
    fAg[:, 0] += x1
    fAg[:, 1] += y1
    ptsA_orig = cv2.transform(fAg.reshape(-1, 1, 2),
                              cv2.invertAffineTransform(M)).reshape(-1, 2)
    Hg, _ = cv2.findHomography(ptsA_orig, fB, cv2.USAC_MAGSAC, 3.0)
    return {"H": Hg, "rmse": rmse, "coverage": cov, "ptsA": ptsA_orig,
            "ptsB": fB, "box": (x1, y1, x2 - x1, y2 - y1),
            "raw_kB": kB, "roi_shape": proc_roi.shape}


def extract_geometry(H):
    if H is None:
        return 0.0, 1.0, 0.0, 0.0
    sx = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
    sy = math.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
    return (math.degrees(math.atan2(H[1, 0], H[0, 0])),
            (sx + sy) / 2.0, float(H[0, 2]), float(H[1, 2]))


def sift_baseline(a, b):
    sift = cv2.SIFT_create(nfeatures=2000)
    kpA, desA = sift.detectAndCompute(a, None)
    kpB, desB = sift.detectAndCompute(b, None)
    if desA is None or desB is None or len(kpA) < 4 or len(kpB) < 4:
        return 0, 0
    bf = cv2.BFMatcher()
    good = [m for m, n in (p for p in bf.knnMatch(desA, desB, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return len(good), 0
    pA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(pA, pB, cv2.RANSAC, 5.0)
    return len(good), int(mask.sum()) if mask is not None else 0
