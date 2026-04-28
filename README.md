---
title: MicroScope QC
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MicroScope QC v3 — Cellpose Edition

A microscopy image quality analysis tool for pathology labs. Built for blood smears with secondary support for general histology.

## What it does
Analyses any microscopy image for four quality metrics and gives an auditable PASS / REVIEW / REJECT verdict:

1. **Sharpness** — Laplacian variance + Tenengrad + edge density + FFT high-frequency energy
2. **Lighting** — Exposure, dynamic range, vignette detection, saturation clipping
3. **Noise** — Immerkær σ-estimator, SNR, salt-and-pepper artifact detection
4. **Density** — Cell counting via **Cellpose deep learning** (cyto3 model)

Every conclusion is backed by a measured value, a documented threshold, and a deterministic decision rule.

## Tech stack
- **Backend:** FastAPI + OpenCV + Cellpose (PyTorch)
- **Frontend:** Vanilla HTML/CSS/JS — no framework
- **Deployment:** Docker on HuggingFace Spaces

## Local development
```bash
pip install -r requirements.txt
python server.py
# open http://localhost:7860
```

## Docker
```bash
docker build -t microscope-qc .
docker run -p 7860:7860 microscope-qc
```
