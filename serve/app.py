"""Scratch inspection API.

    GET  /health            -> model and threshold info
    POST /predict           -> JSON: accept/reject decision, score, scratch regions
    POST /predict/overlay   -> PNG: the image with predicted scratch outlines in red
Upload the image as multipart form field "file".

Local run (repo root, venv active):  uvicorn serve.app:app --port 8000
Docker: see serve/Dockerfile
"""
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

try:                                   # package import (uvicorn serve.app:app) or flat (Docker)
    from serve.inference import ScratchSegmenter
except ImportError:
    from inference import ScratchSegmenter

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model.onnx"))
# chosen with a cost model on out-of-fold data before the locked test (PROGRESS.md #25)
THRESHOLD = float(os.environ.get("SCRATCH_THRESHOLD", "0.59"))

app = FastAPI(title="Metal nut scratch inspection",
              description="YOLOv8n-seg trained on synthetic renders + 10 real images, ONNX Runtime")
segmenter = ScratchSegmenter(MODEL_PATH, THRESHOLD)


def read_image(upload: UploadFile) -> np.ndarray:
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode the image")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "threshold": THRESHOLD}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img = read_image(file)
    r = segmenter.predict(img)
    return {
        "decision": "reject" if r["regions"] else "accept",
        "best_score": r["best_score"],
        "threshold": THRESHOLD,
        "n_regions": len(r["regions"]),
        "scratch_area_pct": round(100 * float(r["mask"].mean()), 3),
        "regions": r["regions"],
        "image_size": [img.shape[1], img.shape[0]],
        "inference_ms": r["inference_ms"],
    }


@app.post("/predict/overlay")
def predict_overlay(file: UploadFile = File(...)):
    img = read_image(file)
    r = segmenter.predict(img)
    contours, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), max(2, img.shape[1] // 250))
    label = f"{'REJECT' if r['regions'] else 'ACCEPT'}  score {r['best_score']:.2f}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)
    ok, png = cv2.imencode(".png", img)
    return Response(content=png.tobytes(), media_type="image/png")
