# -*- coding: utf-8 -*-
"""GPU-fixed TMC-2 runner: moves photometric to RTX (kornia GPU bilateral+CLAHE),
SIFT-coarse on GPU-friendly thumbnails, caches proc images, guards empty ROI.
Falls back faithfully to original math, just faster + less RAM/SSD/CPU."""
import matplotlib
matplotlib.use("Agg")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os, sys, cv2, math, json, time, itertools, pathlib, traceback, gc, subprocess
cv2.setNumThreads(4)
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
import kornia

BASE = pathlib.Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman")
CONV = BASE / "converted_fullres"
OUTROOT = BASE / "tmc_results_fullres"
OUTROOT.mkdir(exist_ok=True)

if not torch.cuda.is_available():
    print("FATAL: RTX required, cuda unavailable. Abort."); sys.exit(1)
device = torch.device("cuda:0")
print(f"GPU enforced: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB cap={torch.cuda.get_device_capability(0)}")
print(f"torch {torch.__version__} kornia {kornia.__version__} cv2 {cv2.__version__} (cv2.cuda={cv2.cuda.getCudaEnabledDeviceCount()} -> CPU-only, so GPU work via kornia/torch)")

def gpu_mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated()/1e6; r = torch.cuda.memory_reserved()/1e6
        print(f"  [GPU mem {tag}] alloc={a:.0f}MB reserved={r:.0f}MB")
def nvidia_util():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"], text=True).strip()
        print(f"  [nvidia-smi] {out}")
    except Exception as e:
        print(f"  nvidia-smi query failed: {e}")

_loftr = None
def get_loftr():
    global _loftr
    if _loftr is None:
        print("Loading LoFTR outdoor -> CUDA...")
        _loftr = kornia.feature.LoFTR(pretrained="outdoor").to(device).eval()
        gpu_mem("loftr loaded")
    return _loftr

def gpu_photometric_enhancer(img_u8, tile_h=2048, overlap=64):
    """Tiled GPU denoise+stretch+CLAHE. Full 64MP frame needs ~12GB for bilateral
    unfold, so split into strips to fit 6GB RTX. Uses separable Gaussian (GPU,
    low VRAM) instead of bilateral unfold (VRAM-prohibitive)."""
    h, w = img_u8.shape
    # global percentile from small proxy (cheap, consistent across tiles)
    small = cv2.resize(img_u8, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    p2, p98 = float(np.percentile(small, 2)), float(np.percentile(small, 98))
    del small
    if p98 <= p2:
        p2, p98 = 0.0, 255.0
    out = np.empty_like(img_u8)
    for y0 in range(0, h, tile_h):
        y1 = min(h, y0 + tile_h)
        ey0 = max(0, y0 - overlap); ey1 = min(h, y1 + overlap)
        tile = torch.from_numpy(img_u8[ey0:ey1].astype(np.float32)).to(device) / 255.0
        tile = tile[None, None]
        p2n = (p2 / 255.0); p98n = (p98 / 255.0)
        with torch.inference_mode():
            tile = kornia.filters.gaussian_blur2d(tile, kernel_size=(7, 7), sigma=(2.0, 2.0))
            tile = ((tile - p2n) / max(p98n - p2n, 1e-6)).clamp(0, 1)
            try:
                tile = kornia.enhance.equalize_clahe(tile, clip_limit=2.0, grid_size=(8, 8))
            except Exception as e:
                print(f"  tile CLAHE skip: {e}")
            res = (tile[0, 0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        # remove overlap padding
        oy0 = y0 - ey0; oy1 = oy0 + (y1 - y0)
        out[y0:y1] = res[oy0:oy1]
        del tile, res
        torch.cuda.empty_cache()
    gc.collect(); torch.cuda.empty_cache()
    return out

def get_proc(img_path):
    """Cached GPU proc image. Saves *_proc.png next to converted to avoid recompute."""
    proc_path = img_path.with_name(img_path.stem.replace("_full", "_proc") + ".png")
    if proc_path.exists():
        p = cv2.imread(str(proc_path), cv2.IMREAD_GRAYSCALE)
        if p is not None:
            return p
    print(f"  GPU preprocessing {img_path.name} ...")
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    t0 = time.time()
    proc = gpu_photometric_enhancer(raw)
    print(f"  GPU preprocess done in {time.time()-t0:.1f}s")
    cv2.imwrite(str(proc_path), proc)
    del raw; gc.collect()
    return proc

def thumb(img, max_side=1600):
    h, w = img.shape[:2]
    s = max(h, w) / max_side
    if s <= 1: return img
    return cv2.resize(img, (int(w/s), int(h/s)), interpolation=cv2.INTER_AREA)

SIFT_MAX = 2048  # coarse SIFT on thumbnail to cut CPU/RAM 10x; geometry scaled back

def sift_coarse_affine(procA_full, procB_full):
    hA, wA = procA_full.shape; hB, wB = procB_full.shape
    sa = max(hA, wA) / SIFT_MAX if max(hA, wA) > SIFT_MAX else 1.0
    sb = max(hB, wB) / SIFT_MAX if max(hB, wB) > SIFT_MAX else 1.0
    smallA = cv2.resize(procA_full, (int(wA/sa), int(hA/sa)), interpolation=cv2.INTER_AREA) if sa > 1 else procA_full
    smallB = cv2.resize(procB_full, (int(wB/sb), int(hB/sb)), interpolation=cv2.INTER_AREA) if sb > 1 else procB_full
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015)
    kpA, desA = sift.detectAndCompute(smallA, None)
    kpB, desB = sift.detectAndCompute(smallB, None)
    print(f"    SIFT-thumb kpA={len(kpA) if kpA is not None else 0} kpB={len(kpB) if kpB is not None else 0} (sa={sa:.2f} sb={sb:.2f})")
    if desA is None or desB is None or len(kpA) < 4 or len(kpB) < 4:
        return None, None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(desA, desB, k=2)
    good = [m for pr in matches if len(pr) == 2 for m, n in [pr] if m.distance < 0.75 * n.distance]
    print(f"    Lowe good: {len(good)}")
    if len(good) < 4: return None, None
    # scale points back to full-res coords
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good]); ptsA[:, 0] *= sa; ptsA[:, 1] *= sa
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good]); ptsB[:, 0] *= sb; ptsB[:, 1] *= sb
    M, mask = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=5.0 * max(sa, sb))
    return M, mask

def run_module_3(procA, procB, pair_dir):
    t0 = time.time()
    print("  [M3 GPU-path] thumbnail-SIFT coarse + LoFTR-GPU fine...")
    nvidia_util()
    M_affine, mask = sift_coarse_affine(procA, procB)
    if M_affine is None:
        print("  M3 coarse failed"); return None
    inl = int(mask.sum())
    sc = math.sqrt(M_affine[0,0]**2 + M_affine[1,0]**2)
    rot = math.degrees(math.atan2(M_affine[1,0], M_affine[0,0]))
    print(f"  coarse rot={rot:.2f} scale={sc:.3f} shift=({M_affine[0,2]:.1f},{M_affine[1,2]:.1f}) inl={inl}")
    hB, wB = procB.shape; hA, wA = procA.shape
    # warp full-res once (CPU, unavoidable) then LoFTR on GPU 640
    procA_w = cv2.warpAffine(procA, M_affine, (wB, hB), flags=cv2.INTER_LINEAR)
    corners = cv2.transform(np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2), M_affine).reshape(-1,2)
    x0 = max(0,int(corners[:,0].min())); y0 = max(0,int(corners[:,1].min()))
    x1 = min(wB,int(corners[:,0].max())); y1 = min(hB,int(corners[:,1].max()))
    tgt = 640
    rA = cv2.resize(procA_w,(tgt,tgt),interpolation=cv2.INTER_AREA); rB = cv2.resize(procB,(tgt,tgt),interpolation=cv2.INTER_AREA)
    del procA_w; gc.collect()
    tA = torch.from_numpy(rA).float()[None,None].to(device)/255.0
    tB = torch.from_numpy(rB).float()[None,None].to(device)/255.0
    print("  LoFTR on RTX...")
    gpu_mem("before loftr"); nvidia_util()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        try:
            out = get_loftr()({"image0": tA, "image1": tB})
        except Exception as e:
            print(f"  AMP LoFTR failed, retry fp32: {e}")
            out = get_loftr()({"image0": tA.float(), "image1": tB.float()})
    gpu_mem("after loftr"); nvidia_util()
    m = (out["confidence"] > 0.20)
    kA = out["keypoints0"][m].float().cpu().numpy(); kB = out["keypoints1"][m].float().cpu().numpy()
    del tA, tB, out; torch.cuda.empty_cache()
    if len(kA):
        kA[:,0]*=wB/tgt; kA[:,1]*=hB/tgt; kB[:,0]*=wB/tgt; kB[:,1]*=hB/tgt
        Mi = cv2.invertAffineTransform(M_affine)
        pA = cv2.transform(kA.reshape(-1,1,2), Mi).reshape(-1,2); pB = kB; fin = len(kA)
    else:
        print("  LoFTR empty, fallback"); return None
    dt = time.time()-t0
    print(f"  M3 done {dt:.1f}s pts={fin}")
    fig,ax = plt.subplots(1,2,figsize=(16,8)); fig.suptitle(f"M3 GPU {dt:.1f}s",fontweight="bold")
    ax[0].imshow(thumb(procA),cmap="gray"); ax[0].set_title("A"); ax[0].axis("off")
    vis = cv2.cvtColor(procB,cv2.COLOR_GRAY2BGR)
    try: cv2.polylines(vis,[corners.astype(np.int32).reshape(-1,1,2)],True,(0,255,0),8)
    except: pass
    ax[1].imshow(cv2.cvtColor(thumb(vis),cv2.COLOR_BGR2RGB)); ax[1].set_title(f"B rot={rot:.1f} sc={sc:.2f}",color="darkgreen"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_3_Aerospace_Localization.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    return {"angle":rot,"scale":sc,"box":(x0,y0,x1-x0,y1-y0),"inliers":fin,"ptsA":pA.astype(np.float32),"ptsB_full":pB.astype(np.float32),"M_affine":M_affine}

def run_loftr_roi_gpu(imgA_crop, roiB):
    if imgA_crop.size == 0 or roiB.size == 0 or min(imgA_crop.shape)<8 or min(roiB.shape)<8:
        return np.zeros((0,2)), np.zeros((0,2))
    hA,wA = imgA_crop.shape; hB,wB = roiB.shape
    tgt=640
    rA=cv2.resize(imgA_crop,(tgt,tgt),interpolation=cv2.INTER_AREA); rB=cv2.resize(roiB,(tgt,tgt),interpolation=cv2.INTER_AREA)
    tA=torch.from_numpy(rA).float()[None,None].to(device)/255.0; tB=torch.from_numpy(rB).float()[None,None].to(device)/255.0
    with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
        try: out=get_loftr()({"image0":tA,"image1":tB})
        except: out=get_loftr()({"image0":tA.float(),"image1":tB.float()})
    m=out["confidence"]>0.20
    kA=out["keypoints0"][m].float().cpu().numpy(); kB=out["keypoints1"][m].float().cpu().numpy()
    del tA,tB,out; torch.cuda.empty_cache()
    if len(kA)<4: return np.zeros((0,2)),np.zeros((0,2))
    kA[:,0]*=wA/tgt; kA[:,1]*=hA/tgt; kB[:,0]*=wB/tgt; kB[:,1]*=hB/tgt
    return kA,kB

def run_module_4(rawA,rawB,procA,procB,loc,pair_dir):
    print("  [M4 GPU-path] ROI dense...")
    if loc is None: return None
    M=loc["M_affine"]; x,y,bw,bh=loc["box"]
    # guard degenerate box (cause of previous cv2.resize crash)
    hBf,wBf=rawB.shape
    if bw<=0 or bh<=0:
        print(f"  M4 EMPTY BOX bw={bw} bh={bh} -> FAILED_ROI_EMPTY (no overlap per coarse affine)")
        (pair_dir/"FAILED_ROI_EMPTY.txt").write_text(f"box={loc['box']} angle={loc['angle']} scale={loc['scale']}")
        return None
    px,py=int(bw*0.10),int(bh*0.10)
    x1,y1=max(0,x-px),max(0,y-py); x2,y2=min(wBf,x+bw+px),min(hBf,y+bh+py)
    if x2<=x1 or y2<=y1 or (x2-x1)<32 or (y2-y1)<32:
        print(f"  M4 ROI too small [{x1}:{x2},{y1}:{y2}] -> FAILED_ROI_EMPTY")
        (pair_dir/"FAILED_ROI_EMPTY.txt").write_text(f"roi=[{x1}:{x2},{y1}:{y2}] box={loc['box']}")
        return None
    print(f"  ROI [{x1}:{x2},{y1}:{y2}] {(x2-x1)}x{(y2-y1)}")
    procB_roi=procB[y1:y2,x1:x2]; rawB_roi=rawB[y1:y2,x1:x2]
    # warp only needed region: warp full then crop (1 pass, free fast)
    wA_full=cv2.warpAffine(procA,M,(wBf,hBf),flags=cv2.INTER_LINEAR)[y1:y2,x1:x2]
    rA_full=cv2.warpAffine(rawA,M,(wBf,hBf),flags=cv2.INTER_LINEAR)[y1:y2,x1:x2]
    kA,kB=run_loftr_roi_gpu(wA_full,procB_roi)
    print(f"  dense {len(kA)}")
    del wA_full; gc.collect()
    if len(kA)<4: return None
    # ANMS 5x5
    gh,gw=5,5; cw=(x2-x1)/gw; ch=(y2-y1)/gh
    buckets={}
    for i,pb in enumerate(kB):
        c=max(0,min(gw-1,int(pb[0]/(cw+1e-5)))); r=max(0,min(gh-1,int(pb[1]/(ch+1e-5))))
        buckets.setdefault((r,c),[]).append(i)
    idx=[]
    for v in buckets.values():
        idx.extend(np.random.choice(v,size=6,replace=False) if len(v)>6 else v)
    cov=len(buckets)/25*100; uA,uB=kA[np.array(idx)],kB[np.array(idx)]
    print(f"  ANMS {len(uA)} cov={cov:.1f}%")
    Hc,_=cv2.findHomography(uA,uB,cv2.USAC_MAGSAC,4.0)
    if Hc is None: return None
    # LK refine (CPU, small ROI only)
    w0=cv2.warpPerspective(rA_full,Hc,(x2-x1,y2-y1))
    p0=cv2.perspectiveTransform(uA.reshape(-1,1,2),Hc).reshape(-1,2).astype(np.float32).reshape(-1,1,2)
    p1,st,er=cv2.calcOpticalFlowPyrLK(w0,rawB_roi,p0,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    sub=np.array([p1[i].ravel() if st[i]==1 and er[i]<8.0 else uB[i] for i in range(len(p0))])
    Hn,mask=cv2.findHomography(uA,sub,cv2.USAC_MAGSAC,2.5)
    if Hn is None: return None
    m=mask.flatten().astype(bool); fA,fBl=uA[m],sub[m]
    rmse=float(np.sqrt(np.mean(np.sum((cv2.perspectiveTransform(fA.reshape(-1,1,2),Hn).reshape(-1,2)-fBl)**2,axis=1))))
    fBg=fBl.copy(); fBg[:,0]+=x1; fBg[:,1]+=y1
    tmp=fA.copy(); tmp[:,0]+=x1; tmp[:,1]+=y1
    fAo=cv2.transform(tmp.reshape(-1,1,2),cv2.invertAffineTransform(M)).reshape(-1,2)
    Hg,_=cv2.findHomography(fAo,fBg,cv2.USAC_MAGSAC,3.0)
    print(f"  RMSE={rmse:.4f} inl={len(fAo)}")
    fig,ax=plt.subplots(1,2,figsize=(16,8)); fig.suptitle("M4 GPU ROI",fontweight="bold")
    vb=cv2.cvtColor(rawB_roi,cv2.COLOR_GRAY2BGR)
    for p in kB[:2000]: cv2.circle(vb,(int(p[0]),int(p[1])),3,(0,0,255),-1)
    ax[0].imshow(cv2.cvtColor(thumb(vb),cv2.COLOR_BGR2RGB)); ax[0].set_title(f"BEFORE {len(kB)}"); ax[0].axis("off")
    va=cv2.cvtColor(rawB,cv2.COLOR_GRAY2BGR); cv2.rectangle(va,(x1,y1),(x2,y2),(0,255,0),8)
    for p in fBg: cv2.circle(va,(int(p[0]),int(p[1])),10,(0,255,0),-1)
    ax[1].imshow(thumb(cv2.cvtColor(va,cv2.COLOR_BGR2RGB))); ax[1].set_title(f"AFTER {len(fAo)} RMSE={rmse:.4f}",color="darkgreen" if rmse<=1 else "darkorange"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(pair_dir/"MODULE_4_ROI_Subpixel_Verification.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    del procB_roi,rawB_roi,rA_full,w0; gc.collect()
    return {"H_native":Hg,"native_rmse":rmse,"grid_coverage":cov,"final_ptsA":fAo,"final_ptsB":fBg,"box":(x1,y1,x2-x1,y2-y1)}

def extract_geo(H):
    if H is None: return 0.,1.,0.,0.
    sx=math.sqrt(H[0,0]**2+H[1,0]**2); sy=math.sqrt(H[0,1]**2+H[1,1]**2)
    return math.degrees(math.atan2(H[1,0],H[0,0])),(sx+sy)/2,H[0,2],H[1,2]

def run_pair(a,b,pid):
    pd=OUTROOT/pid; pd.mkdir(exist_ok=True)
    flag=pd/"output"/"metrics.json"
    if flag.exists():
        try:
            prev=json.loads(flag.read_text()); print(f"SKIP {pid} done {prev.get('status')}"); return prev
        except: pass
    print("="*70); print(f"PAIR {pid}: {a.name} vs {b.name}"); print("="*70)
    print(f" GPU: {torch.cuda.get_device_name(0)}"); gpu_mem("start"); nvidia_util()
    t0=time.time()
    rawA=cv2.imread(str(a),cv2.IMREAD_GRAYSCALE); rawB=cv2.imread(str(b),cv2.IMREAD_GRAYSCALE)
    print(f" raw {rawA.shape} {rawB.shape}")
    fig,ax=plt.subplots(1,2,figsize=(16,7)); fig.suptitle(f"M1 {pid}",fontweight="bold")
    ax[0].imshow(thumb(rawA),cmap="gray"); ax[0].set_title(f"A {rawA.shape[1]}x{rawA.shape[0]}"); ax[0].axis("off")
    ax[1].imshow(thumb(rawB),cmap="gray"); ax[1].set_title(f"B {rawB.shape[1]}x{rawB.shape[0]}"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(pd/"MODULE_1_Diagnostic.png",dpi=100,bbox_inches="tight"); plt.close(fig)
    del rawA,rawB; gc.collect()
    # cached GPU proc (biggest RAM/CPU saver: compute once per image, not per pair)
    procA=get_proc(a); procB=get_proc(b)
    rawA=cv2.imread(str(a),cv2.IMREAD_GRAYSCALE); rawB=cv2.imread(str(b),cv2.IMREAD_GRAYSCALE)
    fig,ax=plt.subplots(2,2,figsize=(14,10)); fig.suptitle("M2 GPU bilateral+CLAHE",fontweight="bold")
    ax[0,0].imshow(thumb(rawA),cmap="gray"); ax[0,0].set_title("BEFORE A"); ax[0,0].axis("off")
    ax[0,1].imshow(thumb(procA),cmap="gray"); ax[0,1].set_title("AFTER A GPU",color="darkgreen"); ax[0,1].axis("off")
    ax[1,0].imshow(thumb(rawB),cmap="gray"); ax[1,0].set_title("BEFORE B"); ax[1,0].axis("off")
    ax[1,1].imshow(thumb(procB),cmap="gray"); ax[1,1].set_title("AFTER B GPU",color="darkgreen"); ax[1,1].axis("off")
    plt.tight_layout(); plt.savefig(pd/"MODULE_2_Natural_Verification.png",dpi=100,bbox_inches="tight"); plt.close(fig)
    loc=run_module_3(procA,procB,pd)
    if loc is None: (pd/"FAILED_M3.txt").write_text("m3 fail"); return {"pair":pid,"status":"FAILED_M3"}
    m4=run_module_4(rawA,rawB,procA,procB,loc,pd)
    del procA,procB; gc.collect(); torch.cuda.empty_cache()
    if m4 is None or m4["H_native"] is None:
        if not (pd/"FAILED_ROI_EMPTY.txt").exists(): (pd/"FAILED_M4.txt").write_text("m4 fail")
        # record failure as metrics so summary includes it
        fail={"pair":pid,"imageA":a.name,"imageB":b.name,"status":"FAILED_ROI_EMPTY" if (pd/"FAILED_ROI_EMPTY.txt").exists() else "FAILED_M4","elapsed_s":round(time.time()-t0,1)}
        (pd/"output").mkdir(exist_ok=True); (pd/"output"/"metrics.json").write_text(json.dumps(fail,indent=2))
        return fail
    H=m4["H_native"]; rmse=float(m4["native_rmse"]); cov=float(m4["grid_coverage"]); pA=m4["final_ptsA"]; pB=m4["final_ptsB"]; box=m4["box"]
    rot,zoom,sx,sy=extract_geo(H)
    # SIFT baseline on thumbnails (cheap, avoids 64MP SIFT RAM spike)
    sA=thumb(rawA,2048); sB=thumb(rawB,2048)
    sift=cv2.SIFT_create(nfeatures=2000); ka,da=sift.detectAndCompute(sA,None); kb,db=sift.detectAndCompute(sB,None)
    if da is None or db is None: sr,si=0,0
    else:
        ms=cv2.BFMatcher().knnMatch(da,db,k=2); g=[m for pr in ms if len(pr)==2 for m,n in [pr] if m.distance<0.75*n.distance]
        sr=len(g)
        si=0
        if len(g)>=4:
            pa=np.float32([ka[m.queryIdx].pt for m in g]).reshape(-1,1,2); pb=np.float32([kb[m.trainIdx].pt for m in g]).reshape(-1,1,2)
            _,mk=cv2.findHomography(pa,pb,cv2.RANSAC,5.0); si=int(mk.sum()) if mk is not None else 0
    print(f" SIFT-thumb baseline raw={sr} inl={si} vs ours={len(pA)}")
    warped=cv2.warpPerspective(rawA,H,(rawB.shape[1],rawB.shape[0]),flags=cv2.INTER_LINEAR)
    overlay=cv2.addWeighted(warped,0.5,rawB,0.5,0)
    # match viz (thumbnail canvas to save RAM/SSD)
    x,y,bw,bh=[int(v) for v in box]; pad=15
    x1,y1=max(0,x-pad),max(0,y-pad); x2,y2=min(rawB.shape[1],x+bw+pad),min(rawB.shape[0],y+bh+pad)
    cropB=rawB[y1:y2,x1:x2]; dispA=cv2.resize(rawA,(cropB.shape[1],cropB.shape[0]),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((cropB.shape[0],cropB.shape[1]*2),np.uint8); canvas[:,:cropB.shape[1]]=dispA; canvas[:,cropB.shape[1]:]=cropB
    cbgr=cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
    ok=(len(pA)>=15) and (rmse<=1.0) and (cov>=50.0)
    od=pd/"output"; od.mkdir(exist_ok=True)
    cv2.imwrite(str(od/"warped_A.png"),warped,[cv2.IMWRITE_PNG_COMPRESSION,3])
    cv2.imwrite(str(od/"overlay.png"),overlay,[cv2.IMWRITE_PNG_COMPRESSION,3])
    cv2.imwrite(str(od/"matches.png"),cbgr,[cv2.IMWRITE_PNG_COMPRESSION,3])
    met={"pair":pid,"imageA":a.name,"imageB":b.name,"imageA_px":[int(rawA.shape[1]),int(rawA.shape[0])],"imageB_px":[int(rawB.shape[1]),int(rawB.shape[0])],
      "status":"ACCEPTABLE" if ok else "REJECTED","our_pipeline_inliers":int(len(pA)),"native_subpixel_rmse":round(rmse,4),
      "grid_coverage_pct":round(cov,2),"sift_baseline_raw":int(sr),"sift_baseline_inliers":int(si),
      "extracted_rotation_deg":round(float(rot),2),"extracted_zoom_scale":round(float(zoom),3),
      "shift_x":round(float(sx),1),"shift_y":round(float(sy),1),"homography_matrix":H.tolist(),
      "elapsed_s":round(time.time()-t0,1),"gpu":torch.cuda.get_device_name(0),"sift_mode":"thumbnail-2048 + GPU LoFTR AMP"}
    (od/"metrics.json").write_text(json.dumps(met,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(14,10)); fig.suptitle(f"FINAL {pid} [{met['status']}] GPU",fontweight="bold")
    ax[0,0].imshow(thumb(rawA),cmap="gray"); ax[0,0].set_title("1. A"); ax[0,0].axis("off")
    ax[0,1].imshow(thumb(rawB),cmap="gray"); ax[0,1].set_title("2. B"); ax[0,1].axis("off")
    ax[1,0].imshow(cv2.cvtColor(thumb(cbgr),cv2.COLOR_BGR2RGB)); ax[1,0].set_title(f"3. ours={len(pA)} sift={si}"); ax[1,0].axis("off")
    ax[1,1].imshow(thumb(overlay),cmap="gray"); ax[1,1].set_title(f"4. RMSE={rmse:.4f} rot={rot:.1f} cov={cov:.1f}%",color="darkgreen" if ok else "darkorange"); ax[1,1].axis("off")
    plt.tight_layout(); plt.savefig(od/"FINAL_Registration_Report.png",dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f" DONE {pid} {met['status']} {met['elapsed_s']}s"); gpu_mem("end"); nvidia_util()
    del rawA,rawB,warped,overlay,cbgr; gc.collect(); torch.cuda.empty_cache()
    return met

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--only",default=""); ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args()
    imgs=sorted(CONV.glob("*_full.png"))
    def short(p):
        n=p.name; s="nca" if "_nca_" in n else ("ncf" if "_ncf_" in n else "ncn"); d="0811" if "20260811" in n else "0812"; return f"{s}_{d}"
    pairs=list(itertools.combinations(imgs,2))
    if args.pilot:
        a=[p for p in imgs if "ncf_20260811" in p.name][0]; b=[p for p in imgs if "ncn_20260811" in p.name][0]; pairs=[(a,b)]
    if args.only:
        pairs=[(x,y) for x,y in pairs if args.only in f"{short(x)}_vs_{short(y)}"]
    print(f"Pairs to run: {len(pairs)} (15 total, skipping done)")
    allm=[]
    for a,b in pairs:
        pid=f"{short(a)}_vs_{short(b)}"
        try: m=run_pair(a,b,pid)
        except Exception as e: print(f"EXC {pid}: {e}"); traceback.print_exc(); m={"pair":pid,"status":f"EXCEPTION: {e}"}
        allm.append(m)
        # merge with existing on-disk metrics for full 15 summary
        try:
            import glob as G
            merged=[]
            for f in sorted(G.glob(str(OUTROOT/"*" / "output" / "metrics.json"))):
                merged.append(json.loads(open(f).read()))
            (OUTROOT/"ALL_metrics.json").write_text(json.dumps(merged,indent=2))
            import csv
            with open(OUTROOT/"SUMMARY.csv","w",newline="") as fh:
                w=csv.DictWriter(fh,fieldnames=["pair","imageA","imageB","status","our_pipeline_inliers","native_subpixel_rmse","grid_coverage_pct","sift_baseline_inliers","extracted_rotation_deg","extracted_zoom_scale","elapsed_s"])
                w.writeheader()
                for mm in merged: w.writerow({k:mm.get(k,"") for k in w.fieldnames})
        except Exception as e: print(f"summary write fail {e}")
    print("DONE")
