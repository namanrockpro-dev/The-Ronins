# -*- coding: utf-8 -*-
"""TMC-2 full-resolution registration runner (local RTX adaptation of imgrgstrsn.py).
Faithful to original Modules 1-5, loops all 15 pairs, uses CUDA RTX 4050.
"""
import matplotlib
matplotlib.use("Agg")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os, sys, cv2, math, json, time, itertools, pathlib, traceback
import numpy as np
import matplotlib.pyplot as plt
import torch, kornia

BASE = pathlib.Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman")
CONV = BASE / "converted_fullres"
OUTROOT = BASE / "tmc_results_fullres"
OUTROOT.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if not torch.cuda.is_available():
    print("FATAL: RTX GPU required but torch.cuda.is_available()==False. Aborting (refusing CPU fallback).")
    sys.exit(1)
print(f"Active GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
assert "4050" in torch.cuda.get_device_name(0) or "NVIDIA" in torch.cuda.get_device_name(0), "Unexpected GPU"
print(f"torch {torch.__version__} kornia {kornia.__version__} cv2 {cv2.__version__}")
print("NOTE: SIFT/bilateral/warp are CPU-only by OpenCV design; LoFTR deep matching is GPU (RTX 4050). High CPU + GPU spikes is expected.")

# Cache LoFTR model (load once, GPU-enforced)
_loftr = None
def get_loftr():
    global _loftr
    if _loftr is None:
        print(f"Loading LoFTR outdoor weights to {device} ({torch.cuda.get_device_name(0)})...")
        _loftr = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()
        print(f" LoFTR on GPU: mem_allocated={torch.cuda.memory_allocated()/1e6:.0f}MB mem_reserved={torch.cuda.memory_reserved()/1e6:.0f}MB")
    return _loftr

def natural_photometric_enhancer(img):
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35)
    p2, p98 = np.percentile(denoised, (2, 98))
    if p98 > p2:
        norm = np.clip((denoised - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = denoised
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    return enhanced

def thumb(img, max_side=1600):
    h, w = img.shape[:2]
    s = max(h, w) / max_side
    if s <= 1:
        return img
    return cv2.resize(img, (int(w/s), int(h/s)), interpolation=cv2.INTER_AREA)

def run_module_3(procA, procB, pair_dir):
    t0 = time.time()
    print("  [M3] SIFT coarse + LoFTR fine...")
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015)
    kpA, desA = sift.detectAndCompute(procA, None)
    kpB, desB = sift.detectAndCompute(procB, None)
    print(f"    SIFT kpA={len(kpA) if kpA is not None else 0} kpB={len(kpB) if kpB is not None else 0}")
    if desA is None or desB is None or len(kpA) < 4 or len(kpB) < 4:
        print("    SIFT failed"); return None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(desA, desB, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    print(f"    good Lowe matches: {len(good)}")
    if len(good) < 4:
        return None
    ptsA_sift = np.float32([kpA[m.queryIdx].pt for m in good])
    ptsB_sift = np.float32([kpB[m.trainIdx].pt for m in good])
    M_affine, inlier_mask = cv2.estimateAffinePartial2D(ptsA_sift, ptsB_sift, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M_affine is None:
        print("    affine failed"); return None
    inliers_sift = int(inlier_mask.sum())
    scale_x = math.sqrt(M_affine[0,0]**2 + M_affine[1,0]**2)
    rot_deg = math.degrees(math.atan2(M_affine[1,0], M_affine[0,0]))
    print(f"    coarse: rot={rot_deg:.2f}deg scale={scale_x:.3f}x shift=({M_affine[0,2]:.1f},{M_affine[1,2]:.1f}) inliers={inliers_sift}")

    hB, wB = procB.shape
    hA, wA = procA.shape
    procA_warped = cv2.warpAffine(procA, M_affine, (wB, hB))
    cornersA = np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)
    cornersB_t = cv2.transform(cornersA, M_affine).reshape(-1,2)
    x_min = max(0, int(np.min(cornersB_t[:,0]))); y_min = max(0, int(np.min(cornersB_t[:,1])))
    x_max = min(wB, int(np.max(cornersB_t[:,0]))); y_max = min(hB, int(np.max(cornersB_t[:,1])))
    box_w, box_h = x_max-x_min, y_max-y_min

    # LoFTR refine on 640
    target = 640
    resA = cv2.resize(procA_warped, (target, target), interpolation=cv2.INTER_AREA)
    resB = cv2.resize(procB, (target, target), interpolation=cv2.INTER_AREA)
    tA = torch.from_numpy(resA).float()[None,None]/255.0
    tB = torch.from_numpy(resB).float()[None,None]/255.0
    matcher = get_loftr()
    with torch.inference_mode():
        out = matcher({"image0": tA.to(device), "image1": tB.to(device)})
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mask = out["confidence"] > 0.20
    kA = out["keypoints0"][mask].cpu().numpy()
    kB = out["keypoints1"][mask].cpu().numpy()
    if len(kA) > 0:
        kA[:,0] *= (wB/target); kA[:,1] *= (hB/target)
        kB[:,0] *= (wB/target); kB[:,1] *= (hB/target)
        M_inv = cv2.invertAffineTransform(M_affine)
        ptsA_orig = cv2.transform(kA.reshape(-1,1,2), M_inv).reshape(-1,2)
        ptsB_full = kB
        final_inliers = len(kA)
    else:
        ptsA_orig = ptsA_sift[inlier_mask.ravel().astype(bool)]
        ptsB_full = ptsB_sift[inlier_mask.ravel().astype(bool)]
        final_inliers = inliers_sift
    dt = time.time()-t0
    print(f"  [M3] done in {dt:.1f}s, final pts={final_inliers}")

    # viz (thumbnails to keep figure small)
    visB = cv2.cvtColor(procB, cv2.COLOR_GRAY2BGR)
    try:
        cv2.polylines(visB, [cornersB_t.astype(np.int32).reshape((-1,1,2))], True, (0,255,0), 8)
    except Exception:
        pass
    fig, axes = plt.subplots(1,2,figsize=(16,8))
    fig.suptitle(f"MODULE 3 COARSE-TO-FINE ({dt:.1f}s)", fontsize=14, fontweight="bold")
    axes[0].imshow(thumb(procA), cmap="gray"); axes[0].set_title("Payload A"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(thumb(visB), cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"B + quad rot={rot_deg:.1f} scale={scale_x:.2f}x", color="darkgreen", fontweight="bold"); axes[1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_3_Aerospace_Localization.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    return {"angle": rot_deg, "scale": scale_x, "box": (x_min,y_min,box_w,box_h),
            "inliers": final_inliers, "ptsA": ptsA_orig.astype(np.float32),
            "ptsB_full": ptsB_full.astype(np.float32), "M_affine": M_affine}

def enforce_anms(ptsA, ptsB_local, roi_shape, grid_size=(5,5), max_pts_per_cell=6):
    h_roi, w_roi = roi_shape
    cell_w = w_roi/grid_size[1]; cell_h = h_roi/grid_size[0]
    buckets = {}
    for i,(pa,pb) in enumerate(zip(ptsA, ptsB_local)):
        col = max(0,min(grid_size[1]-1,int(pb[0]/(cell_w+1e-5))))
        row = max(0,min(grid_size[0]-1,int(pb[1]/(cell_h+1e-5))))
        buckets.setdefault((row,col),[]).append(i)
    idx=[]
    for k,v in buckets.items():
        if len(v)>max_pts_per_cell:
            idx.extend(np.random.choice(v,size=max_pts_per_cell,replace=False))
        else:
            idx.extend(v)
    cov = len(buckets)/(grid_size[0]*grid_size[1])*100.0
    return np.array(idx), cov

def run_loftr_dense_roi(imgA, roiB, target_size=640):
    hA,wA = imgA.shape; hB,wB = roiB.shape
    resA = cv2.resize(imgA,(target_size,target_size),interpolation=cv2.INTER_AREA)
    resB = cv2.resize(roiB,(target_size,target_size),interpolation=cv2.INTER_AREA)
    tA = torch.from_numpy(resA).float()[None,None]/255.0
    tB = torch.from_numpy(resB).float()[None,None]/255.0
    matcher = get_loftr()
    with torch.inference_mode():
        out = matcher({"image0": tA.to(device), "image1": tB.to(device)})
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mask = out["confidence"]>0.20
    kA = out["keypoints0"][mask].cpu().numpy(); kB = out["keypoints1"][mask].cpu().numpy()
    if len(kA)<4:
        return np.zeros((0,2)), np.zeros((0,2))
    kA[:,0]*=(wA/target_size); kA[:,1]*=(hA/target_size)
    kB[:,0]*=(wB/target_size); kB[:,1]*=(hB/target_size)
    return kA,kB

def lk_refine(rawA, raw_roiB, ptsA, ptsB_local, H_coarse):
    hB,wB = raw_roiB.shape
    rawA_warped = cv2.warpPerspective(rawA, H_coarse, (wB,hB))
    ptsA_warped = cv2.perspectiveTransform(ptsA.reshape(-1,1,2), H_coarse).reshape(-1,2)
    p0 = ptsA_warped.astype(np.float32).reshape(-1,1,2)
    p1,status,err = cv2.calcOpticalFlowPyrLK(rawA_warped, raw_roiB, p0, None,
        winSize=(21,21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    out=[]
    for i in range(len(p0)):
        if status[i]==1 and err[i]<8.0:
            out.append(p1[i].ravel())
        else:
            out.append(ptsB_local[i])
    return np.array(out)

def run_module_4(rawA, rawB, procA, procB, loc_data, pair_dir):
    print("  [M4] ROI dense + subpixel...")
    if loc_data is None: return None
    M_affine = loc_data["M_affine"]
    x,y,bw,bh = loc_data["box"]
    pad_x, pad_y = int(bw*0.10), int(bh*0.10)
    hBf,wBf = rawB.shape
    x1,y1 = max(0,x-pad_x), max(0,y-pad_y)
    x2,y2 = min(wBf,x+bw+pad_x), min(hBf,y+bh+pad_y)
    print(f"    ROI [{x1}:{x2},{y1}:{y2}] {(x2-x1)}x{(y2-y1)}")
    procB_roi = procB[y1:y2,x1:x2]; rawB_roi = rawB[y1:y2,x1:x2]
    procA_warped = cv2.warpAffine(procA, M_affine, (wBf,hBf))[y1:y2,x1:x2]
    rawA_warped = cv2.warpAffine(rawA, M_affine, (wBf,hBf))[y1:y2,x1:x2]
    kA_roi,kB_roi = run_loftr_dense_roi(procA_warped, procB_roi, device if False else 640) if False else run_loftr_dense_roi(procA_warped, procB_roi)
    print(f"    dense ROI matches: {len(kA_roi)}")
    if len(kA_roi)<4: return None
    uidx,grid_cov = enforce_anms(kA_roi,kB_roi,roi_shape=procB_roi.shape)
    unif_kA, unif_kB = kA_roi[uidx], kB_roi[uidx]
    print(f"    ANMS {len(unif_kA)} pts cov={grid_cov:.1f}%")
    H_roi_coarse,_ = cv2.findHomography(unif_kA, unif_kB, cv2.USAC_MAGSAC, 4.0)
    if H_roi_coarse is None: return None
    subpix_kB = lk_refine(rawA_warped, rawB_roi, unif_kA, unif_kB, H_roi_coarse)
    H_roi_native,inlier_mask = cv2.findHomography(unif_kA, subpix_kB, cv2.USAC_MAGSAC, 2.5)
    if H_roi_native is None: return None
    m = inlier_mask.flatten().astype(bool)
    final_kA, final_kB_local = unif_kA[m], subpix_kB[m]
    ptsA_trans = cv2.perspectiveTransform(final_kA.reshape(-1,1,2), H_roi_native).reshape(-1,2)
    rmse = float(np.sqrt(np.mean(np.sum((ptsA_trans-final_kB_local)**2,axis=1))))
    final_kB_global = final_kB_local.copy(); final_kB_global[:,0]+=x1; final_kB_global[:,1]+=y1
    M_inv = cv2.invertAffineTransform(M_affine)
    tmp = final_kA.copy(); tmp[:,0]+=x1; tmp[:,1]+=y1
    final_ptsA_orig = cv2.transform(tmp.reshape(-1,1,2), M_inv).reshape(-1,2)
    H_native_global,_ = cv2.findHomography(final_ptsA_orig, final_kB_global, cv2.USAC_MAGSAC, 3.0)
    print(f"    RMSE={rmse:.4f}px inliers={len(final_ptsA_orig)}")
    # viz
    fig,axes = plt.subplots(1,2,figsize=(16,8))
    fig.suptitle("MODULE 4 ROI DENSE + ANMS", fontsize=14, fontweight="bold")
    vis_before = cv2.cvtColor(rawB_roi, cv2.COLOR_GRAY2BGR)
    for p in kB_roi[:2000]:
        cv2.circle(vis_before,(int(p[0]),int(p[1])),3,(0,0,255),-1)
    axes[0].imshow(cv2.cvtColor(thumb(vis_before),cv2.COLOR_BGR2RGB)); axes[0].set_title(f"BEFORE dense {len(kB_roi)}"); axes[0].axis("off")
    vis_after = cv2.cvtColor(rawB, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_after,(x1,y1),(x2,y2),(0,255,0),8)
    for p in final_kB_global:
        cv2.circle(vis_after,(int(p[0]),int(p[1])),10,(0,255,0),-1)
    axes[1].imshow(thumb(cv2.cvtColor(vis_after,cv2.COLOR_BGR2RGB))); axes[1].set_title(f"AFTER {len(final_ptsA_orig)} RMSE={rmse:.4f}", color="darkgreen" if rmse<=1 else "darkorange", fontweight="bold"); axes[1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_4_ROI_Subpixel_Verification.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    return {"H_native":H_native_global,"native_rmse":rmse,"grid_coverage":grid_cov,
            "final_ptsA":final_ptsA_orig,"final_ptsB":final_kB_global,"box":(x1,y1,x2-x1,y2-y1)}

def extract_geo(H):
    if H is None: return 0.0,1.0,0.0,0.0
    sx = math.sqrt(H[0,0]**2+H[1,0]**2); sy = math.sqrt(H[0,1]**2+H[1,1]**2)
    return math.degrees(math.atan2(H[1,0],H[0,0])), (sx+sy)/2.0, H[0,2], H[1,2]

def sift_baseline(a,b):
    sift = cv2.SIFT_create(nfeatures=2000)
    ka,da = sift.detectAndCompute(a,None); kb,db = sift.detectAndCompute(b,None)
    if da is None or db is None or len(ka)<4 or len(kb)<4: return 0,0
    bf = cv2.BFMatcher(); ms = bf.knnMatch(da,db,k=2)
    good=[m for pr in ms if len(pr)==2 for m,n in [pr] if m.distance<0.75*n.distance]
    if len(good)<4: return len(good),0
    pa=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pb=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    _,mask=cv2.findHomography(pa,pb,cv2.RANSAC,5.0)
    return len(good), int(mask.sum()) if mask is not None else 0

def draw_lines(imgA,imgB,ptsA,ptsB,box,max_lines=60):
    hA,wA=imgA.shape; hB,wB=imgB.shape
    x,y,bw,bh=[int(v) for v in box]; pad=15
    x1,y1=max(0,x-pad),max(0,y-pad); x2,y2=min(wB,x+bw+pad),min(hB,y+bh+pad)
    cropB=imgB[y1:y2,x1:x2]
    dispA=cv2.resize(imgA,(cropB.shape[1],cropB.shape[0]),interpolation=cv2.INTER_AREA)
    sx,sy=cropB.shape[1]/wA,cropB.shape[0]/hA
    pa=ptsA.copy(); pa[:,0]*=sx; pa[:,1]*=sy
    pb=ptsB.copy(); pb[:,0]-=x1; pb[:,1]-=y1
    hC,wC=cropB.shape
    canvas=np.zeros((hC,wC*2),dtype=np.uint8); canvas[:,:wC]=dispA; canvas[:,wC:]=cropB
    cbgr=cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
    n=min(len(pa),max_lines)
    idx=np.linspace(0,len(pa)-1,n,dtype=int) if len(pa)>0 else []
    for i in idx:
        p1=(int(pa[i,0]),int(pa[i,1])); p2=(int(pb[i,0]+wC),int(pb[i,1]))
        cv2.line(cbgr,p1,p2,(0,255,80),1,cv2.LINE_AA)
        cv2.circle(cbgr,p1,3,(0,0,255),-1); cv2.circle(cbgr,p2,3,(255,200,0),-1)
    return cbgr

def run_pair(pathA, pathB, pair_id):
    pair_dir = OUTROOT / pair_id
    pair_dir.mkdir(exist_ok=True)
    done_flag = pair_dir / "output" / "metrics.json"
    if done_flag.exists():
        try:
            prev = json.loads(done_flag.read_text())
            print(f"SKIP {pair_id}: already done status={prev.get('status')} rmse={prev.get('native_subpixel_rmse')}")
            return prev
        except Exception:
            pass
    print("="*70); print(f"PAIR {pair_id}: {pathA.name} vs {pathB.name}"); print("="*70)
    print(f" GPU enforced: {torch.cuda.get_device_name(0)} mem_free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")
    t_all=time.time()
    rawA=cv2.imread(str(pathA),cv2.IMREAD_GRAYSCALE); rawB=cv2.imread(str(pathB),cv2.IMREAD_GRAYSCALE)
    hA,wA=rawA.shape; hB,wB=rawB.shape
    print(f"  rawA {wA}x{hA} rawB {wB}x{hB}")
    # M1 fig
    fig,ax=plt.subplots(1,2,figsize=(16,7)); fig.suptitle(f"M1 {pair_id}",fontweight="bold")
    ax[0].imshow(thumb(rawA),cmap="gray"); ax[0].set_title(f"A {wA}x{hA}"); ax[0].axis("off")
    ax[1].imshow(thumb(rawB),cmap="gray"); ax[1].set_title(f"B {wB}x{hB}"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_1_Diagnostic.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    # M2
    print("  [M2] photometric...")
    procA=natural_photometric_enhancer(rawA); procB=natural_photometric_enhancer(rawB)
    fig,ax=plt.subplots(2,2,figsize=(16,12)); fig.suptitle("M2 RAW vs ENHANCED",fontweight="bold")
    ax[0,0].imshow(thumb(rawA),cmap="gray"); ax[0,0].set_title("BEFORE A"); ax[0,0].axis("off")
    ax[0,1].imshow(thumb(procA),cmap="gray"); ax[0,1].set_title("AFTER A",color="darkgreen"); ax[0,1].axis("off")
    ax[1,0].imshow(thumb(rawB),cmap="gray"); ax[1,0].set_title("BEFORE B"); ax[1,0].axis("off")
    ax[1,1].imshow(thumb(procB),cmap="gray"); ax[1,1].set_title("AFTER B",color="darkgreen"); ax[1,1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_2_Natural_Verification.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    # M3 M4
    loc=run_module_3(procA,procB,pair_dir)
    if loc is None:
        print("  M3 FAILED"); (pair_dir/"FAILED_M3.txt").write_text("M3 failed")
        return {"pair":pair_id,"status":"FAILED_M3"}
    m4=run_module_4(rawA,rawB,procA,procB,loc,pair_dir)
    if m4 is None or m4["H_native"] is None:
        print("  M4 FAILED"); (pair_dir/"FAILED_M4.txt").write_text("M4 failed")
        return {"pair":pair_id,"status":"FAILED_M4"}
    # M5
    H=m4["H_native"]; rmse=float(m4["native_rmse"]); cov=float(m4["grid_coverage"])
    ptsA=m4["final_ptsA"]; ptsB=m4["final_ptsB"]; box=m4["box"]
    rot,zoom,sx,sy=extract_geo(H)
    print(f"  [M5] rot={rot:.2f} zoom={zoom:.3f} shift=({sx:.1f},{sy:.1f})")
    s_raw,s_in=sift_baseline(rawA,rawB)
    print(f"  SIFT baseline raw={s_raw} inliers={s_in} vs ours={len(ptsA)}")
    hB2,wB2=rawB.shape
    warped=cv2.warpPerspective(rawA,H,(wB2,hB2))
    overlay=cv2.addWeighted(warped,0.5,rawB,0.5,0)
    match_vis=draw_lines(rawA,rawB,ptsA,ptsB,box)
    ok=(len(ptsA)>=15) and (rmse<=1.0) and (cov>=50.0)
    odir=pair_dir/"output"; odir.mkdir(exist_ok=True)
    cv2.imwrite(str(odir/"warped_A.png"),warped)
    cv2.imwrite(str(odir/"overlay.png"),overlay)
    cv2.imwrite(str(odir/"matches.png"),match_vis)
    metrics={"pair":pair_id,"imageA":pathA.name,"imageB":pathB.name,
        "imageA_px":[int(wA),int(hA)],"imageB_px":[int(wB),int(hB)],
        "status":"ACCEPTABLE" if ok else "REJECTED","our_pipeline_inliers":int(len(ptsA)),
        "native_subpixel_rmse":round(rmse,4),"grid_coverage_pct":round(cov,2),
        "sift_baseline_raw":int(s_raw),"sift_baseline_inliers":int(s_in),
        "extracted_rotation_deg":round(float(rot),2),"extracted_zoom_scale":round(float(zoom),3),
        "shift_x":round(float(sx),1),"shift_y":round(float(sy),1),
        "homography_matrix":H.tolist(),"elapsed_s":round(time.time()-t_all,1)}
    (odir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(16,12)); fig.suptitle(f"FINAL REPORT {pair_id} [{'ACCEPTABLE' if ok else 'REJECTED'}]",fontsize=16,fontweight="bold")
    ax[0,0].imshow(thumb(rawA),cmap="gray"); ax[0,0].set_title("1. A"); ax[0,0].axis("off")
    ax[0,1].imshow(thumb(rawB),cmap="gray"); ax[0,1].set_title("2. B"); ax[0,1].axis("off")
    ax[1,0].imshow(cv2.cvtColor(thumb(match_vis),cv2.COLOR_BGR2RGB)); ax[1,0].set_title(f"3. Matches ours={len(ptsA)} sift={s_in}"); ax[1,0].axis("off")
    ax[1,1].imshow(thumb(overlay),cmap="gray"); ax[1,1].set_title(f"4. Overlay RMSE={rmse:.4f} rot={rot:.1f} cov={cov:.1f}%",color="darkgreen" if ok else "darkorange"); ax[1,1].axis("off")
    plt.tight_layout(); plt.savefig(odir/"FINAL_Registration_Report.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  DONE {pair_id} status={metrics['status']} elapsed={metrics['elapsed_s']}s")
    del rawA,rawB,procA,procB,warped,overlay
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--only",default="",help="only this pair_id substring, e.g. pilot")
    ap.add_argument("--pilot",action="store_true",help="run single pilot pair only")
    args=ap.parse_args()
    imgs=sorted(CONV.glob("*_full.png"))
    print(f"Found {len(imgs)} images:")
    for p in imgs: print(" ",p.name)
    # short names: nca/ncf/ncn + date
    def short(p):
        n=p.name
        sensor="nca" if "_nca_" in n else ("ncf" if "_ncf_" in n else "ncn")
        date="0811" if "20260811" in n else "0812"
        return f"{sensor}_{date}"
    pairs=list(itertools.combinations(imgs,2))
    print(f"Total pairs: {len(pairs)}")
    # pilot = same-date stereo ncf_0811 vs ncn_0811 (most overlap expected)
    if args.pilot:
        a=[p for p in imgs if "ncf_20260811" in p.name][0]
        b=[p for p in imgs if "ncn_20260811" in p.name][0]
        pairs=[(a,b)]
    if args.only:
        pairs=[(a,b) for a,b in pairs if args.only in f"{short(a)}_vs_{short(b)}"]
    allm=[]
    for a,b in pairs:
        pid=f"{short(a)}_vs_{short(b)}"
        try:
            m=run_pair(a,b,pid)
        except Exception as e:
            print(f"EXCEPTION {pid}: {e}"); traceback.print_exc()
            m={"pair":pid,"status":f"EXCEPTION: {e}"}
        allm.append(m)
        (OUTROOT/"ALL_metrics.json").write_text(json.dumps(allm,indent=2))
    # summary csv
    import csv
    with open(OUTROOT/"SUMMARY.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["pair","imageA","imageB","status","our_pipeline_inliers","native_subpixel_rmse","grid_coverage_pct","sift_baseline_inliers","extracted_rotation_deg","extracted_zoom_scale","elapsed_s"])
        w.writeheader()
        for m in allm:
            w.writerow({k:m.get(k,"") for k in w.fieldnames})
    print("SUMMARY:")
    for m in allm: print(m)
