"""Real-only baseline: YOLOv8n-seg trained on the 10 real seen_pool scratches only,
evaluated with 5-fold cross-validation (no synthetic data, no locked_test).

Fold k: train on 8 scratch images + 40 good (train/good), evaluate on the other
2 scratch images + 50 other good images. Over 5 folds every real scratch is tested
exactly once by a model that never saw it.

Run from the repo root (venv active, after tools/mask_to_yolo.py real):
    python train/real_only_cv.py              # full run
    python train/real_only_cv.py --smoke      # 1 fold, 2 epochs: pipeline check
Output: output/real_cv/ (datasets, runs), reports/real_only_cv.json
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from eval import evaluate, summarize  # noqa: E402

SEED = 0
N_FOLDS = 5
N_GOOD_TRAIN, N_GOOD_EVAL = 40, 50
SEEN = Path("data/yolo/real")                 # from tools/mask_to_yolo.py real
GOOD_DIR = Path("data/real/mvtec/metal_nut/train/good")
OUT = Path("output/real_cv")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--eval-only", action="store_true", help="skip training, re-evaluate runs")
args = parser.parse_args()
folds = 1 if args.smoke else N_FOLDS
epochs = 2 if args.smoke else args.epochs

rng = random.Random(SEED)
scratch = sorted((SEEN / "images/seen_pool").glob("*.png"))
rng.shuffle(scratch)
goods = sorted(GOOD_DIR.glob("*.png"))
rng.shuffle(goods)
good_train, good_eval = goods[:N_GOOD_TRAIN], goods[N_GOOD_TRAIN:N_GOOD_TRAIN + N_GOOD_EVAL]


def add(images, split, root, label_for):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img in images:
        name = f"{img.parent.name}_{img.name}"
        shutil.copy(img, root / "images" / split / name)
        shutil.copy(label_for(img), root / "labels" / split / f"{Path(name).stem}.txt")


empty = OUT / "empty.txt"
OUT.mkdir(parents=True, exist_ok=True)
empty.write_text("")
scratch_label = lambda p: SEEN / "labels/seen_pool" / f"{p.stem}.txt"  # noqa: E731

all_rows, per_fold = [], []
for k in range(folds):
    eval_scratch = scratch[2 * k: 2 * k + 2]
    train_scratch = [p for p in scratch if p not in eval_scratch]
    root = OUT / f"fold{k}"
    if not args.eval_only:
        shutil.rmtree(root, ignore_errors=True)
    add(train_scratch, "train", root, scratch_label)
    add(good_train, "train", root, lambda p: empty)
    # YOLO needs a val set during training. It picks best.pt by val score, and the val set
    # contains this fold's test images -> best.pt is selected ON THE TEST SET (leakage,
    # found after the first run: 2/10 with best.pt vs 1/10 with last.pt). So we only use
    # val for monitoring and evaluate last.pt (end of training, no selection).
    add(eval_scratch, "val", root, scratch_label)
    add(good_eval[:10], "val", root, lambda p: empty)
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
                    "names:\n  0: scratch\n")

    if not args.eval_only:
        model = YOLO("yolov8n-seg.pt")  # COCO-pretrained
        model.train(data=str(yaml), epochs=epochs, imgsz=512, batch=8, device="mps", seed=SEED,
                    project=str(OUT / "runs"), name=f"fold{k}", exist_ok=True, plots=False,
                    verbose=False)
    best = YOLO(str(OUT / "runs" / f"fold{k}" / "weights" / "last.pt"))
    samples = ([(p, scratch_label(p)) for p in eval_scratch] + [(p, empty) for p in good_eval])
    summary, rows = evaluate(best, samples)
    for r in rows:
        r["fold"] = k
    all_rows += rows
    per_fold.append(summary)
    print(f"fold {k}: {summary}", flush=True)

total = summarize(all_rows)
# goods are evaluated once per fold -> false alarm = average over folds
print("\nreal-only baseline (5-fold CV over the 10 seen_pool scratches):")
print(json.dumps(total, indent=1))
Path("reports").mkdir(exist_ok=True)
out_json = Path("reports/real_only_cv_smoke.json" if args.smoke else "reports/real_only_cv.json")
out_json.write_text(json.dumps({"total": total, "per_fold": per_fold, "images": all_rows,
                                "epochs": epochs, "weights": "last.pt"}, indent=1))
print(f"wrote {out_json}")
