"""
MicroScope QC v3 — FastAPI Backend (HuggingFace Spaces)
"""

import os
import base64
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_image, QualityReport, VERSION

app = FastAPI(
    title="MicroScope QC API",
    description="Microscopy image quality analysis (Cellpose-powered)",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def bgr_to_data_url(bgr_img: np.ndarray, fmt: str = ".jpg", quality: int = 88) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, bgr_img, encode_params)
    if not ok:
        raise RuntimeError("Image encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def report_to_dict(report: QualityReport, original_bgr: np.ndarray) -> dict:
    def metric_to_dict(m):
        return {
            "name": m.name,
            "score": m.score,
            "severity": m.severity,
            "measurements": m.measurements,
            "findings": [asdict(f) for f in m.findings],
        }
    return {
        "version": report.version,
        "timestamp": report.timestamp,
        "image_info": report.image_info,
        "overall_score": report.overall_score,
        "verdict": asdict(report.verdict),
        "metrics": {k: metric_to_dict(v) for k, v in report.metrics.items()},
        "images": {
            "original":  bgr_to_data_url(original_bgr),
            "annotated": bgr_to_data_url(report.annotated_image) if report.annotated_image is not None else None,
            "heatmap":   bgr_to_data_url(report.heatmap_image)   if report.heatmap_image   is not None else None,
        },
        "histogram": report.histogram_data,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type and not str(file.content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (>25 MB)")

    arr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Resize to keep Cellpose runtime reasonable on CPU
    h, w = img_bgr.shape[:2]
    MAX_DIM = 1024  # smaller than v2 (was 1600) — Cellpose is CPU-bound
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    try:
        report = analyze_image(img_bgr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse(report_to_dict(report, img_bgr))


if __name__ == "__main__":
    import uvicorn
    # HuggingFace Spaces sets PORT env var; default to 7860 (HF standard)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
