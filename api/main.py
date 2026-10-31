"""FastAPI inference microservice and web application server for COVID-19 CT slice classification.

Endpoints:
    GET  /                    - Custom Web UI interface
    GET  /health              - Healthcheck & model status
    GET  /api/info            - Model metadata & medical disclaimer
    GET  /api/samples         - List of available sample test slices
    GET  /api/sample/{cls}    - Retrieve a sample CT image
    POST /predict             - Predict class from uploaded slice
    POST /explain             - Predict class & generate Grad-CAM attention heatmap
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ctcovid import CLASS_DESCRIPTIONS, CLASS_NAMES, __version__
from ctcovid.explain import explain_image
from ctcovid.predict import DISCLAIMER, predict_image

app = FastAPI(
    title="AETHER·SCAN COVID-19 CT Scanner AI API",
    description="Transfer learning (MobileNetV3) classification of chest CT slices into nCT, pCT, and NiCT.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = Path("outputs/model.pt")
WEB_DIR = ROOT / "web"
SAMPLES_DIR = ROOT / "data" / "synthetic"

SAMPLE_FILES = {
    "nCT": SAMPLES_DIR / "nCT" / "nCT_0000.jpg",
    "pCT": SAMPLES_DIR / "pCT" / "pCT_0000.jpg",
    "NiCT": SAMPLES_DIR / "NiCT" / "NiCT_0000.jpg",
}


@app.get("/")
def serve_index():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI index.html not found.")
    return FileResponse(index_file)


@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_ready": CHECKPOINT_PATH.exists(),
    }


@app.get("/api/info")
def model_info():
    return {
        "service": "COVID-19 CT-Scan Classifier",
        "version": __version__,
        "model_architecture": "MobileNetV3-Large",
        "input_shape": [224, 224, 3],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "classes": CLASS_NAMES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/samples")
def list_samples():
    return {
        cls: f"/api/sample/{cls}"
        for cls, path in SAMPLE_FILES.items()
        if path.exists()
    }


@app.get("/api/sample/{class_name}")
def get_sample_image(class_name: str):
    if class_name not in SAMPLE_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown class '{class_name}'. Must be one of {list(SAMPLE_FILES.keys())}")
    file_path = SAMPLE_FILES[class_name]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file for '{class_name}' not found on disk.")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float = Query(0.50, ge=0.0, le=1.0, description="Uncertainty confidence threshold"),
    topk: int = Query(3, ge=1, le=3, description="Number of top predictions"),
):
    if not CHECKPOINT_PATH.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found. Initialize checkpoint first.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    result = predict_image(image, checkpoint=str(CHECKPOINT_PATH), threshold=threshold, topk=topk)
    result["class_description"] = CLASS_DESCRIPTIONS.get(result["predicted_class"], "")
    result["filename"] = file.filename
    return result


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),  # noqa: B008
    alpha: float = Query(0.45, ge=0.0, le=1.0, description="Grad-CAM blend opacity"),
    threshold: float = Query(0.50, ge=0.0, le=1.0),
):
    if not CHECKPOINT_PATH.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    pred = predict_image(image, checkpoint=str(CHECKPOINT_PATH), threshold=threshold)
    cam = explain_image(image, checkpoint=str(CHECKPOINT_PATH), alpha=alpha)

    # Encode overlay image to base64 data URL
    buf = io.BytesIO()
    cam["overlay"].save(buf, format="PNG")
    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Encode vivid colormapped heatmap to base64 data URL
    heat_buf = io.BytesIO()
    cam["heatmap_color"].save(heat_buf, format="PNG")
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(heat_buf.getvalue()).decode()



    return {
        "label": pred["label"],
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "is_uncertain": pred["is_uncertain"],
        "probs": pred["probs"],
        "top_k": pred["top_k"],
        "class_description": CLASS_DESCRIPTIONS.get(pred["predicted_class"], ""),
        "overlay_image": overlay_b64,
        "heatmap_image": heatmap_b64,
        "target_class": cam["class_name"],
        "disclaimer": DISCLAIMER,
    }


# Mount web static directory if it exists
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
