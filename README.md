# 🌕 LunarMatch AI
**Adaptive Self-Correcting Satellite Image Registration for Chandrayaan-2**

[![SIH 2026](https://img.shields.io/badge/Smart_India_Hackathon-2026-e8a33d?style=for-the-badge)](https://sih.gov.in)
[![Problem](https://img.shields.io/badge/Problem-SIH26166-3fb96c?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Mission_Ready-009688?style=for-the-badge)]()

> **TL;DR:** Lunar satellite imagery suffers from extreme shadow inversion and zero atmospheric diffusion, making standard computer vision matching fail. LunarMatch AI is a **local-first, physics-driven pipeline** that dynamically routes images between ORB, SIFT, and LoFTR Transformers. It handles **900MB+ PDS4 strips with zero RAM spikes** and achieves **sub-1.5px RMSE** precision.

---

## 🚀 Executive Summary

Lunar satellite imagery (Chandrayaan-2 OHRC, TMC-2, IIRS) presents unique challenges. Traditional computer vision algorithms (like standard SIFT/ORB) fail completely when sun angles change, and deep learning models crash when attempting to load massive 900MB+ uncompressed orbital strips into memory.

**LunarMatch AI** is a locally hosted registration pipeline that solves this. It dynamically estimates scene difficulty, blends photometric physics (Hapke model) with deep neural transformers (LoFTR), and orchestrates a 3-stage self-correcting feedback loop to achieve verified sub-pixel accuracy on planetary data.

---

## ⚡ Core Technical Innovations

### 1. Physics-Driven Preprocessing
The Moon has no atmosphere; shadows are absolute. To match craters taken at different times of the lunar day, we extract sun-angle metadata directly from PDS4 labels. We apply a **Hapke Bidirectional Reflectance Correction** and blend it with a **Shadow-Invariant Gradient Channel**. This normalizes illumination mathematically *before* any AI processes the image.

### 2. Zero-RAM Spike PDS4 Ingestion Engine
Loading 900MB TMC-2 strips into Python crashes standard laptops (OOM errors). We bypass this entirely by parsing native ISRO PRADAN XML labels and mapping the binary files directly to disk using `numpy.memmap`. The system extracts only the required localized patches, ensuring memory usage stays flat.

### 3. Adaptive Difficulty Arbiter
Not every image requires expensive deep learning. Our pipeline computes a **Scene Difficulty Score (0.0 to 1.0)** based on texture entropy, blur, and scale delta. 
* **Score < 0.30** ➔ Routes to ORB (Ultra-fast)
* **Score 0.30 - 0.55** ➔ Routes to SIFT (Balanced)
* **Score > 0.55** ➔ Routes to LoFTR Transformers (Dense matching for polar regions)

### 4. Geometric Coarse-to-Subpixel Lock
Once matched, we prevent clustered keypoints by enforcing **ANMS (Adaptive Non-Maximal Suppression)** over an 8x8 grid. Outliers are eliminated via **MAGSAC++** and **GC-RANSAC** homography estimation. Final coordinates are refined to sub-pixel floating-point precision using **Lucas-Kanade optical flow**.

---

## 🧠 System Architecture

```text
┌────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ ISRO PRADAN    │     │ PHYSICS ENGINE       │     │ DIFFICULTY ARBITER   │
│ PDS4 Datasets  │ ──> │ • CLAHE / Bilateral  │ ──> │ • Entropy Analysis   │
│ OHRC/TMC-2     │     │ • Hapke Photometric  │     │ • Scale/Sun Delta    │
└────────────────┘     │ • Shadow-Invariant   │     └──────────┬───────────┘
                       └──────────────────────┘                │
                                                               ▼
┌────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ MISSION HUD UI │     │ QUALITY GATE         │     │ HYBRID MATCHING      │
│ • Local HTML/JS│ <── │ • Inliers > 60%      │ <── │ [Easy] ➔ ORB         │
│ • Metrics.json │     │ • RMSE < 2.0px       │     │ [Med]  ➔ SIFT        │
│ • Web Canvas   │     │ (Self-Corrects if ❌)│     │ [Hard] ➔ LoFTR / CNN │
└────────────────┘     └──────────────────────┘     └──────────────────────┘
        ▲                                                      │
        └────────────────── [ MAGSAC++ & Sub-Pixel LK ] ───────┘


        📊 Performance: LunarMatch AI vs. Standard Baselines
Metric / Constraint	Standard Pipeline (SIFT/OpenCV)	LunarMatch AI (Our System)
Extreme Sun-Angle Shifts	❌ Fails completely (< 10% inliers)	✅ Passed (> 65% via Hapke Blend)
PDS4 Massive File Limits	💥 RAM Overload / Application Crash	✅ Zero-RAM Spike (np.memmap)
Matching Distribution	⚠️ Clustered on high-contrast ridges	✅ Uniform 8x8 ANMS Spatial Grid
Registration Precision	⚠️ Integer pixel locking (~3.0 px)	✅ Sub-pixel Lucas-Kanade (< 1.5 px)
Processing Strategy	⚠️ Fixed Algorithm (Brittle)	✅ Adaptive 3-Round Fallback Loop
💻 Local Installation & Deployment

This system is built for secure, offline-first environments. It requires zero cloud APIs, zero proprietary licenses, and runs 100% locally.
1. System Requirements

    Python 3.9+
    CPU capable (CUDA/NVIDIA GPU automatically detected and utilized for LoFTR acceleration)

2. Clone & Setup

Bash

# Clone the repository
git clone https://github.com/namanrockpro-dev/The-Ronins.git
cd The-Ronins

# Initialize and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

3. Launch Mission Control Server

Bash

uvicorn server:app --reload --port 8000

Open your browser and navigate to: http://127.0.0.1:8000
🎛️ Using the Mission Control HUD

The frontend UI is built purely in Vanilla HTML5, CSS3, and JavaScript, styled like a JPL/ISRO aerospace telemetry dashboard.

    Upload Reference: Drop a small lunar crop (e.g., a target crater). Use the interactive canvas to draw a bounding box.
    Upload Target: Drop the massive full-scene orbital image.
    Launch Pipeline: Watch the 5-phase telemetry monitor animate as the backend crunches the data.
    Audit Results: The UI returns a verified Verdict, a Confidence Gauge, sub-pixel RMSE readouts, and visual tabs for Match Density Heatmaps and Warped Overlays.

📚 Core Technologies Used

    PyTorch & Kornia: Powers the LoFTR Transformer neural matching engine.
    OpenCV (cv2): Handles CLAHE, SIFT, ORB, Lucas-Kanade optical flow, and transformations.
    NumPy & SciPy: Enables disk-streaming of massive arrays and KD-Tree metric calculations.
    FastAPI: High-performance asynchronous REST API bridging the Python brain to the Web HUD.

🏆 Team Ronins

Developed for the Smart India Hackathon (SIH 2026) — Problem Statement SIH26166.

    Naman Mishra (Team Lead)
    Aditya
    Khushi
    Aayesha
    Samagra
    Navneet

Institution: Quantum University