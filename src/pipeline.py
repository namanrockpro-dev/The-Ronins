"""Pipeline orchestrator: Phase 1->2->3, confidence gate, fallback A/B/C loop."""
import time
import uuid

import cv2
import numpy as np

from . import features as F
from . import geom as G
from . import metrics as M
from . import pre1 as P1

GATE_INLIER = 0.60
GATE_RMSE = 2.0
SWAP_ORDER = ["sift", "loftr", "orb"]


def _match(model, a, b):
    t0 = time.time()
    try:
        if model == "loftr":
            p1, p2, conf, extra = F.match_loftr(a, b)
            return p1, p2, conf, {"model": "loftr", **extra,
                                  "ms": round((time.time() - t0) * 1000)}
        if model == "sift":
            p1, p2, conf = F.match_sift(a, b)
        else:
            p1, p2, conf = F.match_orb(a, b)
        return p1, p2, conf, {"model": model,
                              "ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        return [], [], [], {"model": model, "error": str(e)[:200]}


def _attempt(a_img, b_img, model, thresh=3.0):
    p1, p2, conf, minfo = _match(model, a_img, b_img)
    p1 = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(p2, dtype=np.float32).reshape(-1, 2)
    if len(p1) < 8:
        return {"ok": False, "reason": f"only {len(p1)} raw matches",
                "matcher": minfo, "model_used": model}
    p1r = G.refine_subpixel(a_img, p1)
    est = G.estimate(p1r, p2, thresh=thresh)
    summ = M.summarize(p1, p2, est, a_img.shape)
    summ["matcher_ms"] = minfo.get("ms")
    return {"ok": True, "p1": p1r, "p2": np.asarray(p2, np.float32),
            "conf": list(map(float, np.asarray(conf).ravel())) if len(conf) else [],
            "est": est, "metrics": summ, "matcher": minfo, "model_used": model}


def _gate(metrics):
    if metrics["inlier_ratio"] > GATE_INLIER and metrics["rmse"] < GATE_RMSE:
        return "YES"
    if metrics["inlier_ratio"] > 0.35 and metrics["rmse"] < 4.0:
        return "MARGINAL"
    return "NO"


def run(a_img, b_img, meta_a=None, meta_b=None, preference="auto",
        max_rounds=3, progress=None):
    """Full adaptive run. Returns run dict (JSON-serializable + arrays)."""
    t0 = time.time()
    run_id = uuid.uuid4().hex[:12]
    log = []

    def say(s):
        log.append(s)
        if progress:
            progress(s)

    pre_a = P1.preprocess(a_img, meta_a or {}, "standard")
    pre_b = P1.preprocess(b_img, meta_b or {}, "standard")
    A, B = pre_a["blend"], pre_b["blend"]
    diff = F.difficulty(A, B, meta_a, meta_b)
    first = F.select_model(diff["score"], preference)
    say(f"difficulty={diff['score']} factors={diff['factors']} -> {first}")
    order = [first] + [m for m in SWAP_ORDER if m != first]
    if preference != "auto":
        order = [preference] + [m for m in SWAP_ORDER if m != preference]

    best, verdict = None, "NO"
    rounds = []
    for rnd in range(max_rounds):
        # Strategy B on rounds>=1: full preprocessing; Strategy C on last: shadow channel
        if rnd == 1:
            pre_a = P1.preprocess(a_img, meta_a or {}, "full")
            pre_b = P1.preprocess(b_img, meta_b or {}, "full")
            A, B = pre_a["blend"], pre_b["blend"]
            say("fallback-B: full Hapke + stronger chain")
        if rnd == 2:
            A, B = pre_a["shadow_inv"], pre_b["shadow_inv"]
            say("fallback-C: shadow-invariant cross-modal channel")
        model = order[min(rnd, len(order) - 1)]  # Strategy A: model swap
        say(f"round {rnd + 1}: matcher={model}")
        att = _attempt(A, B, model)
        if not att["ok"]:
            rounds.append({"round": rnd + 1, "model": model, "ok": False,
                           "reason": att["reason"]})
            say(f"round {rnd + 1} failed: {att['reason']}")
            continue
        v = _gate(att["metrics"])
        rounds.append({"round": rnd + 1, "model": model, "ok": True,
                       "verdict": v, "metrics": att["metrics"]})
        say(f"round {rnd + 1}: inliers={att['metrics']['inlier_ratio']} "
            f"rmse={att['metrics']['rmse']} -> {v}")
        if best is None or att["metrics"]["inlier_ratio"] > best["metrics"]["inlier_ratio"]:
            best, verdict = att, v
        if v == "YES":
            break

    if best is None:
        return {"id": run_id, "status": "FAILED", "verdict": "NO",
                "difficulty": diff, "rounds": rounds, "log": log,
                "ms": int((time.time() - t0) * 1000)}
    out = {"id": run_id, "status": "OK" if verdict in ("YES", "MARGINAL") else "LOWCONF",
           "verdict": verdict, "difficulty": diff, "rounds": rounds,
           "metrics": best["metrics"], "model_used": best["model_used"],
           "matcher": best["matcher"], "pre_a": pre_a["info"],
           "pre_b": pre_b["info"], "log": log,
           "ms": int((time.time() - t0) * 1000),
           "H": np.asarray(best["est"]["H"]).tolist(),
           "inliers": best["est"]["inliers"].astype(bool).tolist(),
           "conf": best["conf"]}
    # downsampled viz coords: p1/p2 already in working resolution
    out["p1"] = np.asarray(best["p1"]).tolist()
    out["p2"] = np.asarray(best["p2"]).tolist()
    return out
