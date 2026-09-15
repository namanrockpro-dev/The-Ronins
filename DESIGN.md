# LunarMatch Console — SIH 2026 (TestNaman)

**Reading this as: scientific mission console for technical evaluators, with a dark-tech mission-control language, leaning toward hand-built CSS + vanilla JS served by Python FastAPI (fully offline).**
Dials: VARIANCE 6 / MOTION 4 / DENSITY 7 · Theme: dark locked · Accent: lunar amber `#e8a33d` (single) · Type: system stack + mono numerals.

## Layout
- `server.py` — FastAPI + static console, `python server.py` (binds `0.0.0.0:8000`)
- `src/` — `io_pds` (PDS4-in-zip reader) · `pre1` (Phase 1) · `features` (Phase 2) · `geom` (Phase 3) · `metrics` · `pipeline` (gate + fallback loop)
- `web/` — offline console (no CDN)
- `data/raw/{box2,legacy_tmc}` — PRADAN products · `data/shapefiles` — footprint index
- `tools/` — footprint query + label utilities · `runs/` — run artefacts
- `archive/` — legacy scripts + old results (superseded, kept for reference)

## Roadmap mapping
Phase 1 (illumination norm, shadow-invariant, Hapke, pyramids) → `pre1.py` ·
Phase 2 (difficulty estimator → ORB/SIFT/LoFTR) → `features.py` ·
Phase 3 (MAGSAC++ → GC-RANSAC → sub-pixel) → `geom.py` ·
Phase 4 (inlier>60% + RMSE<2px gate) + Phase 5B (model swap → full preprocess → shadow-channel retry) → `pipeline.py` ·
Phase 5A (overlay, heatmap, geolocation, confidence) + evaluation plots → `server.py` + console.
