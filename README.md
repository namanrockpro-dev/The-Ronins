# LunarMatch Console — SIH 2026 · TestNaman

Adaptive self-correcting satellite image matching for Chandrayaan-2
(OHRC / TMC-2 / IIRS). Roadmap: `../Downloads/deepseek_html_20260906_749b3e.html`.

## Run (Python server, localhost + WLAN)
```
cd C:\Users\swati\OneDrive\Desktop\TestNaman
python server.py
```
- Local: http://127.0.0.1:8000
- WLAN: http://192.168.137.1:8000 (or http://192.168.33.199:8000 — use your current Wi-Fi IP; allow Python through Windows Firewall if another device can't reach it — needs one admin click)

Needs: `pip install -r requirements.txt` (torch CUDA build recommended; LoFTR auto-falls back to SIFT/ORB on CPU-only boxes).

## Pipeline (maps to roadmap)
| Roadmap | Code |
|---|---|
| Phase 1 preprocessing (CLAHE/gamma/contrast, shadow-invariant, Hapke, pyramids) | `src/pre1.py` |
| Phase 2 difficulty estimator → ORB/SIFT/LoFTR selector | `src/features.py` |
| Phase 3 MAGSAC++ → GC-RANSAC → sub-pixel | `src/geom.py` |
| Phase 4 gate (inlier > 60% and RMSE < 2 px) | `src/pipeline.py` |
| Phase 5B fallback (model swap → full preprocess → shadow channel) | `src/pipeline.py` |
| Phase 5A output (overlay, heatmap, geolocation, confidence) + metrics plots | `server.py` + `web/` |

PDS4 products are read straight from the PRADAN zips (`src/io_pds.py`, strided reads — 919 MB strips never fully load into RAM).

## Data (all local)
- `data/raw/box2/` — same-coordinate sextet, lat ~5.1, lon ~−125.5: OHRC×2 (2 h apart — sun-angle pair), TMC-2×2 + IIRS×2 (Feb vs Apr 2022 — opposite sun azimuth 94° vs 268°)
- `data/raw/legacy_tmc/` — 6 older TMC-2 strips (fore/nadir/aft × 2 dates)
- `data/shapefiles/` — OHRC/IIRS/TMC-2 footprint index · `tools/` — footprint + label utilities

## Verified
- OHRC sun pair (1400 px): **1108 matches, 625 inliers (56%), RMSE 1.35, coverage 88% → MARGINAL**, improving every fallback round (43.7% → 52.8% → 56.4%)
- TMC-2 opposite-sun pair: honestly reported LOWCONF (few matches) — the hard SIH case, pipeline degrades gracefully instead of hallucinating
- `archive/` holds superseded scripts/results; `runs/` holds new run artefacts
