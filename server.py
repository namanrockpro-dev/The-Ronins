"""LunarMatch server (SIH 2026): FastAPI + static console. Host: python server.py."""
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import io_pds, pre1
from src.pipeline import run as run_pipe

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(exist_ok=True)
app = FastAPI(title="LunarMatch Console")


class RunReq(BaseModel):
    a: str
    b: str
    preference: str = "auto"
    max_side: int = 1400
    max_rounds: int = 3


class SearchReq(BaseModel):
    src: str
    x: int
    y: int
    w: int
    h: int
    top_k: int = 5
    verify_k: int = 3


class CropMatchReq(BaseModel):
    crop_src: str  # library id or up/<file> (whole image used as crop)
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    full: str = ""
    full_side: int = 1600


@app.post("/api/register")
async def api_register(ref: UploadFile = File(None), target: UploadFile = File(None),
                       ref_id: str = None, target_id: str = None,
                       crop_x: int = 0, crop_y: int = 0, crop_w: int = 0, crop_h: int = 0):
    """Two-upload endpoint: Reference (crop) searched in Target (full), Colab Modules 1-5.

    Accept either uploaded files (ref/target) or library ids (ref_id/target_id).
    Optional crop_x/y/w/h crops the reference preview before matching.
    Returns metrics.json + report paths, plus the artefact filenames.
    """
    import cv2
    import torch
    from src import register_logic as RL

    def _load_from_upload(up: UploadFile):
        data = up.file.read() if hasattr(up, "file") else None
        if data is None:
            raise HTTPException(400, "empty upload")
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise HTTPException(400, f"unreadable image: {up.filename}")
        return arr

    if ref is not None and hasattr(ref, "filename") and ref.filename:
        refA = _load_from_upload(ref)
    elif ref_id:
        refA = cv2.imread(str(preview_png(ref_id)), cv2.IMREAD_GRAYSCALE)
        if refA is None:
            raise HTTPException(400, "unreadable ref_id")
        if crop_w > 8 and crop_h > 8:
            h, w = refA.shape
            x0 = max(0, min(crop_x, w - 8))
            y0 = max(0, min(crop_y, h - 8))
            x1 = max(x0 + 8, min(crop_x + crop_w, w))
            y1 = max(y0 + 8, min(crop_y + crop_h, h))
            refA = refA[y0:y1, x0:x1]
    else:
        raise HTTPException(400, "provide ref file or ref_id")

    if target is not None and hasattr(target, "filename") and target.filename:
        tgtB = _load_from_upload(target)
    elif target_id:
        # full target at working side (strided if needed)
        if target_id.startswith("up/"):
            tgtB = cv2.imread(str(_resolve(target_id)), cv2.IMREAD_GRAYSCALE)
        else:
            from src import io_pds
            tgtB, _, _ = io_pds.load_preview(
                str(ROOT / "data" / "raw" / target_id), 1600)
            tgtB = cv2.cvtColor(tgtB, cv2.COLOR_GRAY2BGR) if tgtB.ndim == 3 else tgtB
            if tgtB.dtype != np.uint8:
                tgtB = pre1.to_uint8(tgtB)
        if tgtB is None:
            raise HTTPException(400, "unreadable target_id")
    else:
        raise HTTPException(400, "provide target file or target_id")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    diag = RL.module1_diagnostics(refA, tgtB)
    rawA, rawB = refA, tgtB
    procA = RL.natural_photometric_enhancer(rawA)
    procB = RL.natural_photometric_enhancer(rawB)
    loc = RL.module3_coarse_to_fine(procA, procB, device)
    if loc is None:
        raise HTTPException(422, "Module 3 coarse lock failed — no SIFT overlap")
    roi = RL.module4_roi(rawA, rawB, procA, procB, loc, device)
    if roi is None:
        raise HTTPException(422, "Module 4 ROI dense matching failed")
    H = roi["H"]
    rot, scale, sx, sy = RL.extract_geometry(H)
    s_raw, s_inl = RL.sift_baseline(rawA, rawB)
    is_ok = (len(roi["ptsA"]) >= 15 and roi["rmse"] <= 1.0
             and roi["coverage"] >= 50.0)
    rid = f"r{int(time.time())}"
    d = RUNS / rid
    d.mkdir(exist_ok=True)
    # warped + overlay
    hB, wB = rawB.shape
    warped = cv2.warpPerspective(rawA, H, (wB, hB)) if H is not None else rawA
    overlay = cv2.addWeighted(warped, 0.5, rawB, 0.5, 0)
    cv2.imwrite(str(d / "warped_A.png"), warped)
    cv2.imwrite(str(d / "overlay.png"), overlay)
    # correspondence canvas (Module 5 helper adapted)
    x1, y1, bw, bh = roi["box"]
    pad = 15
    cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
    cx1, cy1 = min(wB, x1 + bw + pad), min(hB, y1 + bh + pad)
    cropB = rawB[cy0:cy1, cx0:cx1]
    dispA = cv2.resize(rawA, (cropB.shape[1], cropB.shape[0]),
                       interpolation=cv2.INTER_AREA)
    sx_c, sy_c = cropB.shape[1] / rawA.shape[1], cropB.shape[0] / rawA.shape[0]
    pA = roi["ptsA"].copy()
    pB = roi["ptsB"].copy()
    pA[:, 0] *= sx_c
    pA[:, 1] *= sy_c
    pB[:, 0] -= cx0
    pB[:, 1] -= cy0
    hC, wC = cropB.shape
    canvas = np.zeros((hC, wC * 2), np.uint8)
    canvas[:, :wC] = dispA
    canvas[:, wC:] = cropB
    cbgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    n_draw = min(len(pA), 80)
    idx = np.linspace(0, max(0, len(pA) - 1), n_draw, dtype=int) if len(pA) else []
    for i in idx:
        cv2.line(cbgr, (int(pA[i, 0]), int(pA[i, 1])),
                 (int(pB[i, 0] + wC), int(pB[i, 1])), (0, 255, 80), 1, cv2.LINE_AA)
        cv2.circle(cbgr, (int(pA[i, 0]), int(pA[i, 1])), 3, (0, 0, 255), -1)
        cv2.circle(cbgr, (int(pB[i, 0] + wC), int(pB[i, 1])), 3, (255, 200, 0), -1)
    cv2.imwrite(str(d / "matches.png"), cbgr)
    # 4-panel report
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ISRO CHANDRAYAAN-2 REGISTRATION — REFERENCE IN TARGET",
                 fontsize=14, fontweight="bold")
    axes[0, 0].imshow(rawA, cmap="gray")
    axes[0, 0].set_title("1. Reference (crop)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(rawB, cmap="gray")
    axes[0, 1].set_title("2. Target (full)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(cbgr, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"3. Connections ({len(roi['ptsA'])} vs SIFT {s_inl})",
                         fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    col = "darkgreen" if is_ok else "darkorange"
    axes[1, 1].imshow(overlay, cmap="gray")
    axes[1, 1].set_title(
        f"4. Overlap — RMSE {roi['rmse']:.4f}px | Rot {rot:.1f}° | Cov {roi['coverage']:.1f}%",
        fontsize=11, fontweight="bold", color=col)
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(d / "FINAL_Registration_Report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics = {
        "status": "ACCEPTABLE" if is_ok else "REJECTED",
        "our_pipeline_inliers": int(len(roi["ptsA"])),
        "native_subpixel_rmse": round(float(roi["rmse"]), 4),
        "grid_coverage_pct": round(float(roi["coverage"]), 2),
        "sift_baseline_inliers": int(s_inl),
        "sift_baseline_raw": int(s_raw),
        "extracted_rotation_deg": round(float(rot), 2),
        "extracted_zoom_scale": round(float(scale), 2),
        "shift_px": [round(float(sx), 1), round(float(sy), 1)],
        "homography_matrix": H.tolist() if H is not None else None,
        "diagnostics": diag,
        "module3": {"angle": round(loc["angle"], 2), "scale": round(loc["scale"], 3),
                    "inliers": int(loc["inliers"]), "t": loc["t"]},
        "accept": bool(is_ok), "device": str(device),
        "ms": int((time.time() - t0) * 1000),
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cv2.imwrite(str(d / "ref.png"), rawA)
    cv2.imwrite(str(d / "target.png"), rawB)
    return {"rid": rid, **metrics, "report": f"/runs/{rid}/FINAL_Registration_Report.png",
            "images": {"report": f"/runs/{rid}/FINAL_Registration_Report.png",
                        "matches": f"/runs/{rid}/matches.png",
                       "overlay": f"/runs/{rid}/overlay.png",
                       "warped": f"/runs/{rid}/warped_A.png"}}


def _resolve(fid):
    if fid.startswith("up/"):
        p = UPLOADS / fid[3:]
    else:
        p = ROOT / "data" / "raw" / fid
    if not p.exists():
        raise HTTPException(400, f"unknown file id: {fid}")
    return p


@app.post("/api/upload")
async def api_upload(f: UploadFile = File(...)):
    name = _safe(Path(f.filename or "crop.png").name)
    dest = UPLOADS / name
    dest.write_bytes(await f.read())
    return {"id": f"up/{name}", "bytes": dest.stat().st_size}


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def preview_png(fid, max_side=1200):
    """Cached grayscale preview PNG path + scale vs full product."""
    from src import io_pds
    if fid.startswith("up/"):
        return _resolve(fid)
    p = ROOT / "data" / "raw" / fid
    if not p.exists():
        raise HTTPException(400, "unknown file id")
    c = ROOT / "data" / "cache" / f"pv_{_safe(fid)}.png"
    if not c.exists():
        import cv2
        img, _, _ = io_pds.load_preview(str(p), max_side)
        cv2.imwrite(str(c), pre1.to_uint8(img))
    return c


@app.get("/api/preview")
def api_preview(id: str):
    return FileResponse(str(preview_png(id)), media_type="image/png")


def _lib():
    out = []
    for sub in ("box2", "legacy_tmc"):
        for z in sorted((ROOT / "data" / "raw" / sub).glob("*.zip")):
            try:
                m = io_pds.parse_meta(str(z))
                d = m["dims"]
                out.append({
                    "id": f"{sub}/{z.name}", "sensor": m["sensor"],
                    "obs": m.get("obs_start"),
                    "sun_az": m.get("sun_azimuth"), "sun_el": m.get("sun_elevation"),
                    "incidence": m.get("incidence"),
                    "shape": [d.get("LINE"), d.get("SAMPLE") or d.get("BAND")],
                    "bands": d.get("BAND"), "bytes": z.stat().st_size,
                })
            except Exception as e:
                out.append({"id": f"{sub}/{z.name}", "error": str(e)[:150]})
    return out


@app.get("/api/health")
def health():
    import torch, cv2
    return {"ok": True, "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "cv2": cv2.__version__,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/api/library")
def library():
    return {"files": _lib()}


@app.get("/api/runs")
def runs():
    items = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        j = d / "result.json"
        if j.exists():
            try:
                r = json.loads(j.read_text())
                items.append({k: r.get(k) for k in
                              ("id", "status", "verdict", "model_used", "metrics",
                               "ms", "a", "b", "created")})
            except Exception:
                pass
    return {"runs": items}


def _save_fig(path, draw):
    fig = plt.figure(figsize=(10, 5), dpi=110)
    try:
        draw()
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)


def _render(rid, a8, b8, res):
    from src import geom as G
    d = RUNS / rid
    cv2 = __import__("cv2")
    p1 = np.float32(res["p1"])
    p2 = np.float32(res["p2"])
    inl = np.array(res["inliers"], bool)
    H = np.float32(res["H"]) if res["H"] else None

    def draw_match():
        h = max(a8.shape[0], b8.shape[0])
        canvas = np.zeros((h, a8.shape[1] + b8.shape[1], 3), np.uint8)
        ca = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR) if a8.ndim == 2 else a8
        cb = cv2.cvtColor(b8, cv2.COLOR_GRAY2BGR) if b8.ndim == 2 else b8
        canvas[:a8.shape[0], :a8.shape[1]] = ca
        canvas[:b8.shape[0], a8.shape[1]:] = cb
        ox = a8.shape[1]
        idx = np.where(inl)[0]
        sel = idx[::max(1, len(idx) // 250)]
        rng = np.random.default_rng(7)
        cols = (rng.random((len(sel), 3)) * 255).astype(int)
        for (i, c) in zip(sel, cols):
            x1, y1 = p1[i]
            x2, y2 = p2[i]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2 + ox), int(y2)),
                     tuple(map(int, c)), 1)
        plt.imshow(canvas[..., ::-1])
        plt.axis("off")
        plt.title(f"inlier matches ({len(sel)} shown / {inl.sum()}), {res['model_used']}")
    _save_fig(d / "matches.png", draw_match)

    def draw_overlay():
        if H is None:
            plt.imshow(a8, cmap="gray"); plt.axis("off"); return
        h, w = b8.shape[:2]
        warped = cv2.warpPerspective(a8, H, (w, h))
        ov = np.zeros((h, w, 3), np.uint8)
        ov[..., 1] = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ov[..., 2] = b8 if b8.ndim == 2 else cv2.cvtColor(b8, cv2.COLOR_BGR2GRAY)
        plt.imshow(ov)
        plt.axis("off")
        plt.title("overlay: warped A (green) vs B (red)")
    _save_fig(d / "overlay.png", draw_overlay)

    def draw_heat():
        m = res["metrics"]
        plt.imshow(m["cells"], cmap="inferno")
        plt.colorbar(label="inliers / cell")
        plt.title(f"match density  coverage={m['coverage']} std={m['uniformity_std']}")
    _save_fig(d / "heatmap.png", draw_heat)

    def draw_hist():
        if H is None or not inl.sum():
            return
        Ht = np.float64(res["H"])
        mod = res["metrics"]["model"]
        r = np.linalg.norm(G._apply(Ht, p1[inl], mod) - p2[inl], axis=1)
        plt.hist(r, bins=40, color="#e8a33d", edgecolor="black")
        plt.axvline(r.mean(), color="red", label=f"mean {r.mean():.2f}px")
        plt.xlabel("reprojection error (px)")
        plt.legend()
        plt.title("error histogram (inliers)")
    _save_fig(d / "errhist.png", draw_hist)

    def draw_scatter():
        conf = np.array(res.get("conf", []), float)
        if H is None or not inl.sum() or len(conf) != len(p1):
            return
        Ht = np.float64(res["H"])
        mod = res["metrics"]["model"]
        r = np.linalg.norm(G._apply(Ht, p1, mod) - p2, axis=1)
        plt.scatter(conf, r, s=6, c=np.where(inl, "#2e7d32", "#c62828"), alpha=0.6)
        plt.xlabel("matcher cost (lower = confident)")
        plt.ylabel("reprojection error (px)")
        plt.title("confidence vs error")
    _save_fig(d / "scatter.png", draw_scatter)


def _latlon(meta, H, shape, max_side_scale):
    """Project working-res image center through H into B, map to lat/lon boxes."""
    try:
        c = meta.get("corners") or {}
        need = ("upper_left", "upper_right", "lower_left", "lower_right")
        if any(c.get(k) is None for k in need):
            return None
        (la0, lo0), (la1, lo1) = c["upper_left"], c["upper_right"]
        (la2, lo2), (la3, lo3) = c["lower_left"], c["lower_right"]
        h, w = shape[:2]
        cx, cy = w / 2, h / 2
        # bilinear in pixel space -> latlon
        fx, fy = cx / w, cy / h
        top = (la0 + (la1 - la0) * fx, lo0 + (lo1 - lo0) * fx)
        bot = (la2 + (la3 - la2) * fx, lo2 + (lo3 - lo2) * fx)
        lat_a = top[0] + (bot[0] - top[0]) * fy
        lon_a = top[1] + (bot[1] - top[1]) * fy
        return {"a_center": [round(lat_a, 4), round(lon_a, 4)]}
    except Exception:
        return None


@app.post("/api/run")
def api_run(req: RunReq):
    if not req.a or not req.b or not req.a.strip() or not req.b.strip():
        raise HTTPException(400, "Upload Reference and Target first — both are required")
    pa = _resolve(req.a) if req.a.startswith("up/") else ROOT / "data" / "raw" / req.a
    pb = _resolve(req.b) if req.b.startswith("up/") else ROOT / "data" / "raw" / req.b
    if not pa.exists() or pa.is_dir() or not pb.exists() or pb.is_dir():
        raise HTTPException(400, f"File not found: {req.a!r} / {req.b!r}")
    import cv2
    if req.a.startswith("up/"):
        fa = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        ma = {"sensor": "UPLOAD", "file": pa.name, "sun_azimuth": None,
              "sun_elevation": None, "incidence": None,
              "obs_start": None, "dims": {"LINE": fa.shape[0], "SAMPLE": fa.shape[1]}}
        sa = 1.0
        if fa is None:
            raise HTTPException(400, f"Cannot read Reference: {req.a}")
        if max(fa.shape) > req.max_side:
            s = req.max_side / max(fa.shape)
            fa = cv2.resize(fa, (int(fa.shape[1] * s), int(fa.shape[0] * s)), interpolation=cv2.INTER_AREA)
            sa = s
        fa = fa.astype(np.float32)
    else:
        fa, ma, sa = io_pds.load_preview(str(pa), req.max_side)
    if req.b.startswith("up/"):
        fb = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        mb = {"sensor": "UPLOAD", "file": pb.name, "sun_azimuth": None,
              "sun_elevation": None, "incidence": None,
              "obs_start": None, "dims": {"LINE": fb.shape[0], "SAMPLE": fb.shape[1]}}
        sb = 1.0
        if fb is None:
            raise HTTPException(400, f"Cannot read Target: {req.b}")
        if max(fb.shape) > req.max_side:
            s = req.max_side / max(fb.shape)
            fb = cv2.resize(fb, (int(fb.shape[1] * s), int(fb.shape[0] * s)), interpolation=cv2.INTER_AREA)
            sb = s
        fb = fb.astype(np.float32)
    else:
        fb, mb, sb = io_pds.load_preview(str(pb), req.max_side)
    res = run_pipe(fa, fb, ma, mb, req.preference, req.max_rounds)
    res.update({"a": req.a, "b": req.b, "scale_a": sa, "scale_b": sb,
                "meta_a": {k: ma.get(k) for k in
                           ("sensor", "file", "sun_azimuth", "sun_elevation",
                            "incidence", "obs_start", "dims")},
                "meta_b": {k: mb.get(k) for k in
                           ("sensor", "file", "sun_azimuth", "sun_elevation",
                            "incidence", "obs_start", "dims")},
                "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    res["geo"] = {"a": _latlon(ma, None, fa.shape, sa),
                  "b": _latlon(mb, None, fb.shape, sb)}
    try:  # Colab-style classical baseline benchmark (SIFT + RANSAC)
        import cv2 as _cv
        from src import features as _F
        _p1, _p2, _ = _F.match_sift(pre1.to_uint8(fa), pre1.to_uint8(fb))
        _H, _m = _cv.findHomography(np.asarray(_p1), np.asarray(_p2),
                                    _cv.RANSAC, 5.0) if len(_p1) >= 4 else (None, None)
        res["sift_baseline"] = {"raw": int(len(_p1)),
                                "inliers": int(_m.sum()) if _m is not None else 0}
    except Exception as e:
        res["sift_baseline"] = {"error": str(e)[:120]}
    d = RUNS / res["id"]
    d.mkdir(exist_ok=True)
    a8, b8 = pre1.to_uint8(fa), pre1.to_uint8(fb)
    cv2.imwrite(str(d / "a.png"), a8)
    cv2.imwrite(str(d / "b.png"), b8)
    if res["status"] != "FAILED":
        _render(res["id"], a8, b8, res)
        res["images"] = ["a.png", "b.png", "matches.png", "overlay.png",
                         "heatmap.png", "errhist.png", "scatter.png"]
    keep = {k: v for k, v in res.items()
            if k not in ("p1", "p2", "inliers", "conf", "H")}
    (d / "result.json").write_text(json.dumps(
        {**keep, "n_p1": len(res.get("p1", []))}))
    slim = dict(keep)
    slim["id"] = res["id"]
    return slim


@app.post("/api/cropmatch")
def api_cropmatch(req: CropMatchReq):
    """One crop vs one full image: inlier/outlier lines on both + overlap."""
    import cv2
    from src import cropmatch as CM
    from src import features as F
    t0 = time.time()
    if req.crop_src.startswith("up/"):
        crop = cv2.imread(str(_resolve(req.crop_src)), cv2.IMREAD_GRAYSCALE)
        if crop is None:
            raise HTTPException(400, "unreadable upload")
    else:
        sp = preview_png(req.crop_src)
        simg = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        h, w = simg.shape
        x0 = max(0, min(req.x, w - 8))
        y0 = max(0, min(req.y, h - 8))
        x1 = max(x0 + 8, min(req.x + req.w, w))
        y1 = max(y0 + 8, min(req.y + req.h, h))
        crop = simg[y0:y1, x0:x1]
    if req.full.startswith("up/"):
        full = cv2.imread(str(_resolve(req.full)), cv2.IMREAD_GRAYSCALE)
    else:
        from src import io_pds
        full, _, _ = io_pds.load_preview(
            str(ROOT / "data" / "raw" / req.full), req.full_side)
        full = F._u8(full)
    if crop is None or full is None or crop.size == 0 or full.size == 0:
        raise HTTPException(400, "empty crop or full image")
    m = CM.match(crop, full)
    cid = f"c{int(time.time())}"
    cd = RUNS / cid
    cd.mkdir(exist_ok=True)
    cv2.imwrite(str(cd / "crop.png"), F._u8(crop))
    cv2.imwrite(str(cd / "full.png"), F._u8(full))
    files = CM.render(crop, full, m, cd) if m.get("found") else CM.render(
        crop, full, {**m, "p1": m.get("p1", np.zeros((0, 2))),
                     "p2": m.get("p2", np.zeros((0, 2))),
                     "inliers": np.zeros(0, bool), "H": np.eye(3)}, cd)
    out = {"cid": cid,
           "found": bool(m.get("found", False)),
           "reason": m.get("reason", ""),
           "stage": m.get("stage1", ""),
           "n_raw": int(m.get("n_raw", 0)),
           "n_inliers": int(m.get("n_inliers", 0)),
           "n_outliers": int(m.get("n_raw", 0)) - int(m.get("n_inliers", 0)),
           "inlier_ratio": m.get("inlier_ratio", 0.0),
           "rmse": round(float(m.get("rmse", -1)), 3),
           "model": m.get("model", ""),
           "images": files,
           "ms": int((time.time() - t0) * 1000)}
    (cd / "cropmatch.json").write_text(json.dumps(out))
    return out


@app.get("/api/run/{rid}")
def api_result(rid: str):
    j = RUNS / rid / "result.json"
    if not j.exists():
        raise HTTPException(404, "no such run")
    return json.loads(j.read_text())


@app.post("/api/search")
def api_search(req: SearchReq):
    """Crop-query search: locate a crop of `src` across the product library."""
    import cv2
    from src import search as S
    from src import features as F
    t0 = time.time()
    sp = preview_png(req.src)
    src_img = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        raise HTTPException(500, "preview failed")
    h, w = src_img.shape
    x0 = max(0, min(req.x, w - 8))
    y0 = max(0, min(req.y, h - 8))
    x1 = max(x0 + 8, min(req.x + req.w, w))
    y1 = max(y0 + 8, min(req.y + req.h, h))
    crop = src_img[y0:y1, x0:x1]
    if crop.size == 0:
        raise HTTPException(400, "empty crop")
    cv2.imwrite(str(RUNS / "last_query.png"), crop)
    # coarse rank over library (excluding the source itself)
    db = []
    for f in _lib():
        if f.get("error") or f["id"] == req.src:
            continue
        try:
            im = cv2.imread(str(preview_png(f["id"])), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                db.append((f["id"], im))
        except Exception:
            pass
    ranked = S.search_db(crop, db)[: max(1, req.top_k)]
    qid = f"q{int(time.time())}"
    qd = RUNS / qid
    qd.mkdir(exist_ok=True)
    cv2.imwrite(str(qd / "query.png"), crop)
    hits = []
    for i, r in enumerate(ranked[: max(0, req.verify_k)]):
        dim = next(x[1] for x in db if x[0] == r["id"])
        v = S.verify_roi(crop, dim, r["M"])
        if v is None:
            continue
        # connection lines: crop (left) vs ROI (right)
        rx0, ry0, rx1, ry1 = v["roi"]
        roi = dim[ry0:ry1, rx0:rx1]
        sc = 320 / max(roi.shape + crop.shape)
        ch, cw = int(crop.shape[0] * sc), int(crop.shape[1] * sc)
        rh, rw = int(roi.shape[0] * sc), int(roi.shape[1] * sc)
        canvas = np.zeros((max(ch, rh), cw + rw, 3), np.uint8)
        canvas[:ch, :cw] = cv2.cvtColor(cv2.resize(crop, (cw, ch)), cv2.COLOR_GRAY2BGR)
        canvas[:rh, cw:] = cv2.cvtColor(cv2.resize(roi, (rw, rh)), cv2.COLOR_GRAY2BGR)
        if "kA" in v and len(v["kA"]):
            kk = min(len(v["kA"]), 60)
            idx = np.linspace(0, len(v["kA"]) - 1, kk, dtype=int)
            for j in idx:
                ax, ay = v["kA"][j] * sc
                # kA are in warped-crop coords ~= crop coords; kB in full-db coords
                bx, by = (v["kB"][j] - np.float32([rx0, ry0])) * sc
                cv2.line(canvas, (int(ax), int(ay)), (int(bx + cw), int(by)), (0, 255, 80), 1)
        cv2.imwrite(str(qd / f"hit{i}_connect.png"), canvas)
        # overlap quad on the found image
        ov = cv2.cvtColor(dim, cv2.COLOR_GRAY2BGR)
        cv2.polylines(ov, [v["quad"].astype(int)], True, (0, 255, 0), 3)
        x0b, y0b = int(v["quad"][:, 0].min()), int(v["quad"][:, 1].min())
        x1b, y1b = int(v["quad"][:, 0].max()), int(v["quad"][:, 1].max())
        cv2.imwrite(str(qd / f"hit{i}_overlap.png"), ov)
        # overlay chip: warped crop blended onto ROI
        try:
            H, _ = cv2.findHomography(
                (v["kA"] - np.float32([0, 0])).astype(np.float32),
                (v["kB"] - np.float32([rx0, ry0])).astype(np.float32),
                cv2.USAC_MAGSAC, 3.0) if "kA" in v else (None, None)
            chip = cv2.addWeighted(
                cv2.warpPerspective(crop, H, (rx1 - rx0, ry1 - ry0))
                if H is not None else cv2.resize(crop, (rx1 - rx0, ry1 - ry0)),
                0.5, roi, 0.5, 0)
        except Exception:
            chip = roi
        cv2.imwrite(str(qd / f"hit{i}_overlay.png"), chip)
        hits.append({"rank": i + 1, "id": r["id"],
                     "coarse_inliers": r["coarse_inliers"],
                     "verify_inliers": v["inliers"],
                     "rmse": round(v["rmse"], 3) if v["rmse"] != float("inf") else None,
                     "quad": v["quad"].round(1).tolist(), "roi": list(v["roi"]),
                     "images": [f"hit{i}_connect.png", f"hit{i}_overlap.png",
                                f"hit{i}_overlay.png"], "ms": r["ms"]})
    others = [{"id": r["id"], "coarse_inliers": r["coarse_inliers"], "raw": r["raw"]}
              for r in ranked[len(hits):]]
    out = {"qid": qid, "src": req.src, "crop": [x0, y0, x1 - x0, y1 - y0],
           "crop_url": f"/runs/{qid}/query.png",
           "hits": hits, "other_candidates": others,
           "n_searched": len(db), "ms": int((time.time() - t0) * 1000)}
    (qd / "search.json").write_text(json.dumps(out))
    return out


app.mount("/runs", StaticFiles(directory=str(RUNS)), name="runs")
app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


def main():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan = s.getsockname()[0]
    except Exception:
        lan = "?"
    finally:
        s.close()
    print(f"LunarMatch console -> http://127.0.0.1:8000  |  WLAN http://{lan}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
