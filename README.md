<div align="center">

# 🌕 LunarMatch AI
**Adaptive Self-Correcting Satellite Image Registration for Chandrayaan-2**

[![SIH 2026](https://img.shields.io/badge/Smart_India_Hackathon-2026-e8a33d?style=for-the-badge)](https://sih.gov.in)
[![Problem](https://img.shields.io/badge/Problem-SIH26166-3fb96c?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()

[**Watch the 2-Minute Demo Video**](#) • [**Read the Design Docs**](DESIGN.md) • [**Team Ronins**](#team)

</div>

---

> **TL;DR:** Lunar satellite imagery suffers from extreme shadow inversion and zero atmospheric diffusion, making standard computer vision matching fail. LunarMatch AI is an **offline-first, physics-driven pipeline** that dynamically routes images between ORB, SIFT, and LoFTR Transformers. It handles **900MB+ PDS4 strips with zero RAM spikes** and achieves **sub-1.5px RMSE** precision.

---

## 🚀 Core Innovations

Unlike standard OpenCV wrappers, LunarMatch AI is built for the harsh realities of planetary orbital data:

1. 🌓 **Physics-Based Photometric Correction:** Applies Hapke reflectance models and shadow-invariant Sobel gradients to match craters regardless of extreme sun-angle deltas.
2. 🧠 **Adaptive Difficulty Engine:** Evaluates scene entropy to dynamically route processing to ORB (Easy), SIFT (Medium), or LoFTR Neural Matcher (Hard/Polar regions).
3. 💾 **Zero-RAM Spike Ingestion:** Natively parses ISRO PRADAN PDS4 XML/Binary strips using `numpy.memmap`, allowing 900MB+ TMC-2 strips to process on standard laptops without crashing.
4. 🔄 **Self-Correcting Quality Gate:** If a match fails our internal threshold (>60% inliers, <2.0px RMSE), a 3-stage fallback architecture automatically retries with escalating shadow-invariant strategies.
5. 🎛️ **Mission Control HUD:** A bespoke, zero-dependency vanilla HTML/JS/CSS dashboard designed for ISRO scientists to visually audit sub-pixel matches.

---

## 📊 Performance: Us vs. Standard Baselines

| Metric / Challenge | Standard Pipeline (SIFT/ORB) | **LunarMatch AI (Our System)** |
| :--- | :--- | :--- |
| **Extreme Sun-Angle Shifts** | ❌ Fails (< 10% inliers) | ✅ **Passed (> 65% via Hapke Blend)** |
| **Execution Strategy** | Fixed Single-Pass Algorithm | ✅ **Adaptive 3-Round Feedback Loop** |
| **Spatial Uniformity** | Matches clustered on 1 crater | ✅ **ANMS 8×8 Grid Distribution** |
| **Registration Precision** | Integer pixel (~3-5px) | ✅ **Sub-Pixel LK Flow (< 1.5px RMSE)** |
| **PDS4 900MB+ Handling** | 💥 Out of Memory / Crashes | ✅ **Zero-RAM Spike via `np.memmap`** |

---

## 🏗️ System Architecture

The project maintains a strict separation of concerns between the High-Performance AI Backend and the Mission-Control Frontend.

```text
LUNARMATCH AI
 ├── web/                   ← Frontend: Mission Control HUD
 │    ├── index.html        (Vanilla HTML5, No React bloat)
 │    ├── styles.css        (Custom CSS Variables, Amber/Dark theme)
 │    └── app.js            (Fetch API, Canvas Crop Engine, Dashboard Logic)
 │
 ├── server.py              ← Bridge: FastAPI Server (REST endpoints)
 │
 └── src/                   ← Backend: AI & CV Brain
      ├── io_pds.py         (Native PDS4 ZIP/XML parser & disk-mapper)
      ├── pre1.py           (CLAHE, Hapke, Shadow-Invariant blend)
      ├── features.py       (Difficulty Engine -> ORB / SIFT / LoFTR)
      ├── geom.py           (MAGSAC++, GC-RANSAC, Lucas-Kanade flow)
      ├── metrics.py        (Quality Gates, RMSE, Grid Coverage)
      └── pipeline.py       (The 3-stage self-correcting orchestrator)