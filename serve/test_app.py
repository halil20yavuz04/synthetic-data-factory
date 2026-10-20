"""API tests (run from the repo root, venv active):  python -m pytest serve/test_app.py -q
Uses a real seen_pool scratch and a real train/good image (never the locked test)."""
from pathlib import Path

from fastapi.testclient import TestClient

from serve.app import app

client = TestClient(app)
SCRATCH = Path("data/real/mvtec/metal_nut/test/scratch/021.png")   # seen_pool
GOOD = Path("data/real/mvtec/metal_nut/train/good/000.png")


def post(path, image):
    with open(image, "rb") as f:
        return client.post(path, files={"file": (image.name, f, "image/png")})


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_scratch_is_rejected():
    body = post("/predict", SCRATCH).json()
    assert body["decision"] == "reject" and body["n_regions"] >= 1
    assert body["best_score"] >= body["threshold"]


def test_good_is_accepted():
    body = post("/predict", GOOD).json()
    assert body["decision"] == "accept" and body["n_regions"] == 0


def test_overlay_returns_png():
    r = post("/predict/overlay", SCRATCH)
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_bad_upload_gives_400():
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
