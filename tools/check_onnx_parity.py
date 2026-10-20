"""Check that the ONNX service gives the same answers as the PyTorch model.

On the 10 seen_pool scratches + 30 train/good images: same accept/reject decision at the
service threshold, similar best score, and high IoU between the two union masks.
    python tools/check_onnx_parity.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, "serve")
sys.path.insert(0, "tools")
from inference import ScratchSegmenter  # noqa: E402
from eval import predict_mask           # noqa: E402

THRESHOLD = 0.59
torch_model = YOLO("output/models/final_v2_ft_replay.pt")
onnx_model = ScratchSegmenter("serve/model.onnx", THRESHOLD)
images = (sorted(Path("data/yolo/real/images/seen_pool").glob("*.png"))
          + sorted(Path("data/real/mvtec/metal_nut/train/good").glob("*.png"))[:30])

agree, ious, score_diff, ms = 0, [], [], []
for p in images:
    t_mask, t_score = predict_mask(torch_model, p, THRESHOLD)
    t_best = predict_mask(torch_model, p, 0.01)[1]
    o = onnx_model.predict(cv2.imread(str(p)))
    o_best = onnx_model.predict(cv2.imread(str(p)), conf=0.01)["best_score"]
    agree += bool(t_mask.any()) == bool(o["mask"].any())
    score_diff.append(abs(t_best - o_best))
    ms.append(o["inference_ms"])
    if t_mask.any() or o["mask"].any():
        ious.append((t_mask & o["mask"]).sum() / (t_mask | o["mask"]).sum())
print(f"{len(images)} images | same decision: {agree}/{len(images)} | "
      f"best-score difference max {max(score_diff):.4f} | "
      f"mask IoU torch vs onnx: mean {np.mean(ious):.3f}, min {np.min(ious):.3f} | "
      f"onnx CPU time median {np.median(ms):.0f} ms")
