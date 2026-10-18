"""Evaluate a YOLOv8-seg scratch model on real images (our own metrics, NumPy).

Per image: predicted mask = union of all predicted scratch masks with conf >= threshold;
the image is "rejected" (scratch found) if there is at least one such mask.
Metrics:
  recall       share of scratch images that were rejected (image level)
  false alarm  share of good images that were rejected (image level)
  precision    rejected images that really had a scratch
  mask IoU     mean IoU(pred, ground truth) over scratch images (0 if nothing found)

Used as a module (train/real_only_cv.py) or from the command line:
    python tools/eval.py --weights best.pt --images data/yolo/real/images/seen_pool
Never run it on locked_test before the final evaluation.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CONF = 0.25  # default YOLO confidence threshold; Week 4 tunes it with a cost model


def label_to_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    """YOLO-seg label file -> boolean mask (empty file -> all False)."""
    mask = np.zeros((height, width), np.uint8)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            xy = np.array(line.split()[1:], float).reshape(-1, 2) * [width, height]
            cv2.fillPoly(mask, [np.round(xy).astype(np.int32)], 1)
    return mask.astype(bool)


def predict_mask(model, image_path: Path, conf: float, imgsz: int = 512):
    """Union of predicted masks at the original image size, plus the best confidence."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    mask = np.zeros((h, w), np.uint8)
    if r.masks is not None:
        for poly in r.masks.xy:  # polygons already in original pixel coordinates
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    best = float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0
    return mask.astype(bool), best


def evaluate(model, samples, conf: float = CONF):
    """samples: list of (image_path, label_path). Returns (summary dict, per-image list)."""
    rows = []
    for image_path, label_path in samples:
        h, w = cv2.imread(str(image_path)).shape[:2]
        gt = label_to_mask(label_path, w, h)
        pred, best = predict_mask(model, image_path, conf)
        union = (gt | pred).sum()
        rows.append({
            "image": image_path.name,
            "has_scratch": bool(gt.any()),
            "rejected": bool(pred.any()),
            "best_conf": round(best, 3),
            "iou": round(float((gt & pred).sum() / union), 3) if gt.any() else None,
        })
    return summarize(rows), rows


def summarize(rows):
    pos = [r for r in rows if r["has_scratch"]]
    neg = [r for r in rows if not r["has_scratch"]]
    tp = sum(r["rejected"] for r in pos)
    fp = sum(r["rejected"] for r in neg)
    return {
        "scratch_images": len(pos), "good_images": len(neg),
        "tp": tp, "fn": len(pos) - tp, "fp": fp, "tn": len(neg) - fp,
        "recall": round(tp / len(pos), 3) if pos else None,
        "false_alarm": round(fp / len(neg), 3) if neg else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "mask_iou": round(float(np.mean([r["iou"] for r in pos])), 3) if pos else None,
    }


if __name__ == "__main__":
    from ultralytics import YOLO
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True,
                        help="YOLO images folder; labels are read from the sibling labels folder")
    parser.add_argument("--conf", type=float, default=CONF)
    args = parser.parse_args()
    assert "locked_test" not in str(args.images), "locked_test is for the final evaluation only"
    label_dir = Path(str(args.images).replace("/images/", "/labels/"))
    samples = [(p, label_dir / f"{p.stem}.txt") for p in sorted(args.images.glob("*.png"))]
    summary, _ = evaluate(YOLO(args.weights), samples, args.conf)
    print(json.dumps(summary, indent=1))
