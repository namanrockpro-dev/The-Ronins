"""Evaluation metrics (roadmap Phase: quantitative + visual analytics data)."""
import numpy as np


def uniformity(p1, shape, grid=8):
    """Match spread: coverage fraction + std of per-cell counts (lower=better)."""
    h, w = shape[:2]
    cells = np.zeros((grid, grid))
    for x, y in np.asarray(p1):
        j = min(grid - 1, max(0, int(x / w * grid)))
        i = min(grid - 1, max(0, int(y / h * grid)))
        cells[i, j] += 1
    cov = float((cells > 0).mean())
    return {"coverage": round(cov, 3), "cell_std": round(float(cells.std()), 3),
            "cells": cells.astype(int).tolist()}


def repeatability_proxy(p1a, p1b, shape, tol=3.0):
    """Fraction of A keypoints re-found near B keypoints (cheap proxy)."""
    from scipy.spatial import cKDTree
    a = np.asarray(p1a, dtype=float)
    b = np.asarray(p1b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    try:
        d, _ = cKDTree(b).query(a)
        return round(float((d < tol).mean()), 3)
    except Exception:
        return 0.0


def summarize(p1_all, p2_all, est, shape):
    inl = est["inliers"]
    u = uniformity(np.asarray(p1_all)[inl] if inl.sum() else [], shape)
    return {
        "model": est["model"], "method": est["method"],
        "n_matches": int(len(p1_all)), "n_inliers": int(est["n_inliers"]),
        "inlier_ratio": round(float(est["n_inliers"] / max(len(p1_all), 1)), 3),
        "rmse": round(float(est["rmse"]), 3),
        "coverage": u["coverage"], "uniformity_std": u["cell_std"],
        "cells": u["cells"],
    }
