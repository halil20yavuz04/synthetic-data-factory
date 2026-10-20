"""THE one-time evaluation on the locked test set (13 scratch + 22 good).

Model and threshold were fixed before this run (PROGRESS.md entries 25-26). Nothing may
be tuned on these results. Requires an explicit flag so it cannot run by accident.

    python tools/final_eval.py --model output/models/final_v2_ft_replay.pt --threshold 0.59 \
        --i-confirm-final-evaluation
Output: reports/final_locked_test.json, reports/final_locked_test.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import evaluate, label_to_mask, predict_mask  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--threshold", type=float, required=True)
parser.add_argument("--i-confirm-final-evaluation", action="store_true")
args = parser.parse_args()
if not args.i_confirm_final_evaluation:
    raise SystemExit("locked test: add --i-confirm-final-evaluation (one-time final run only)")

images = Path("data/yolo/real/images/locked_test")
labels = Path("data/yolo/real/labels/locked_test")
samples = [(p, labels / f"{p.stem}.txt") for p in sorted(images.glob("*.png"))]
model = YOLO(args.model)
summary, rows = evaluate(model, samples, conf=args.threshold)
print(json.dumps(summary, indent=1))
for r in rows:
    if r["has_scratch"] != r["rejected"]:
        print("error:", r["image"], "scratch" if r["has_scratch"] else "good",
              "conf", r["best_conf"])

Path("reports/final_locked_test.json").write_text(json.dumps(
    {"model": args.model, "threshold": args.threshold, "summary": summary, "images": rows},
    indent=1))

# visual record: every locked test image, green = ground truth, red = prediction
T, tiles = 180, []
for img_p, lab_p in samples:
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    gt = label_to_mask(lab_p, w, h)
    pred, conf = predict_mask(model, img_p, args.threshold)
    for m, col in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 4)
    t = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
    cv2.putText(t, f"{img_p.stem} {conf:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
while len(tiles) % 7:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)])
cv2.imwrite("reports/final_locked_test.png", grid)
print("wrote reports/final_locked_test.json and reports/final_locked_test.png")
