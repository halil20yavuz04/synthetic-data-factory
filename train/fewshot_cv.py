"""Few-shot experiments with 5-fold cross-validation over the 10 real seen_pool scratches
(no locked_test).

Fold k: train on 8 real scratch images + 40 real good (train/good) [+ N synthetic images],
starting from --init weights; evaluate on the other 2 scratch images + 50 other good
images (same folds and goods for every variant, so results are comparable). Over 5 folds
every real scratch is tested exactly once by a model that never saw it.
Hyperparameters are fixed BEFORE looking at results (no tuning on the folds).

Run from the repo root (venv active, after tools/mask_to_yolo.py real):
    python train/fewshot_cv.py --name real_only                      # COCO init, real only
    python train/fewshot_cv.py --name v2_ft --init output/models/synth_v2_best.pt \
        --epochs 30 --lr0 0.001
    python train/fewshot_cv.py --name v2_ft_replay --init output/models/synth_v2_best.pt \
        --epochs 30 --lr0 0.001 --synth data/yolo/synth_v2 --n-synth 300
    python train/fewshot_cv.py --name real_only --smoke   # 1 fold, 2 epochs: pipeline check
Output: output/cv_<name>/ (datasets, runs), reports/cv_<name>.json
(the first real-only run wrote reports/real_only_cv.json)
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
OUT = None  # set from --name below

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--eval-only", action="store_true", help="skip training, re-evaluate runs")
parser.add_argument("--name", default="real_only", help="experiment name (output folders)")
parser.add_argument("--init", default="yolov8n-seg.pt", help="start weights (COCO or ours)")
parser.add_argument("--lr0", type=float, default=None, help="learning rate (None = YOLO auto)")
parser.add_argument("--synth", type=Path, default=None, help="YOLO synthetic dataset to mix in")
parser.add_argument("--n-synth", type=int, default=0, help="synthetic images per fold (replay)")
parser.add_argument("--final", action="store_true",
                    help="train ONE model on all 10 real scratches (no held-out fold, no eval); "
                         "saved to output/models/final_<name>.pt for the locked test")
args = parser.parse_args()
OUT = Path("output/real_cv") if args.name == "real_only" else Path(f"output/cv_{args.name}")
folds = 1 if (args.smoke or args.final) else N_FOLDS
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
    eval_scratch = [] if args.final else scratch[2 * k: 2 * k + 2]
    train_scratch = [p for p in scratch if p not in eval_scratch]
    root = OUT / ("final" if args.final else f"fold{k}")
    if not args.eval_only:
        shutil.rmtree(root, ignore_errors=True)
    add(train_scratch, "train", root, scratch_label)
    add(good_train, "train", root, lambda p: empty)
    if args.n_synth:   # replay: same synthetic sample in every fold (seeded)
        synth_imgs = sorted((args.synth / "images/train").glob("*.png"))
        synth_imgs = random.Random(SEED).sample(synth_imgs, args.n_synth)
        add(synth_imgs, "train", root, lambda p: args.synth / "labels/train" / f"{p.stem}.txt")
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
        model = YOLO(args.init)  # COCO-pretrained or our synthetic model
        extra = {"lr0": args.lr0, "optimizer": "AdamW"} if args.lr0 else {}
        model.train(data=str(yaml), epochs=epochs, imgsz=512, batch=8, device="mps", seed=SEED,
                    project=str(OUT / "runs"), name=root.name, exist_ok=True, plots=False,
                    verbose=False, **extra)
    if args.final:   # no held-out real scratches left: just keep the model
        Path("output/models").mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / "runs" / "final" / "weights" / "last.pt",
                    f"output/models/final_{args.name}.pt")
        print(f"saved output/models/final_{args.name}.pt")
        sys.exit(0)
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
print(f"\n{args.name} (5-fold CV over the 10 seen_pool scratches):")
print(json.dumps(total, indent=1))
Path("reports").mkdir(exist_ok=True)
out_json = Path(f"reports/cv_{args.name}{'_smoke' if args.smoke else ''}.json")
out_json.write_text(json.dumps({"total": total, "per_fold": per_fold, "images": all_rows,
                                "epochs": epochs, "weights": "last.pt", "init": args.init,
                                "lr0": args.lr0, "n_synth": args.n_synth}, indent=1))
print(f"wrote {out_json}")
