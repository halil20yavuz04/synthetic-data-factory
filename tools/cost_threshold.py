"""Pick the decision threshold with a cost model, using out-of-fold scores only
(cross-validation models of one experiment; locked_test is never touched).

Score of an image = highest scratch confidence the model gives it (0 if nothing).
For each fold model: its 2 held-out real scratches + all 180 train/good images that
were never in its training set. Expected cost per part at threshold t:
    p * C_miss * miss_rate(t) + (1 - p) * C_fa * false_alarm_rate(t)

Usage (repo root, venv active):
    python tools/cost_threshold.py --name v2_ft_replay
Output: reports/results/threshold_<name>.json and reports/results/threshold_<name>.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import predict_mask  # noqa: E402

SEED, N_FOLDS, N_GOOD_TRAIN = 0, 5, 40          # must match train/fewshot_cv.py
SEEN = Path("data/yolo/real/images/seen_pool")
GOOD_DIR = Path("data/real/mvtec/metal_nut/train/good")
# assumptions (stated, not measured): defect rate and cost ratio scenarios
P_DEFECT = 0.05
SCENARIOS = {"miss = 5x false alarm": 5, "miss = 20x": 20, "miss = 50x": 50}
MAIN = "miss = 20x"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="v2_ft_replay")
args = parser.parse_args()
runs = Path(f"output/cv_{args.name}/runs")

# same shuffles as train/fewshot_cv.py -> same folds and good split
rng = random.Random(SEED)
scratch = sorted(SEEN.glob("*.png"))
rng.shuffle(scratch)
goods = sorted(GOOD_DIR.glob("*.png"))
rng.shuffle(goods)
held_out_goods = goods[N_GOOD_TRAIN:]           # never used for training in any fold

pos, neg = [], []
for k in range(N_FOLDS):
    model = YOLO(str(runs / f"fold{k}" / "weights" / "last.pt"))
    pos += [predict_mask(model, p, 0.01)[1] for p in scratch[2 * k: 2 * k + 2]]
    neg += [predict_mask(model, p, 0.01)[1] for p in held_out_goods]
    print(f"fold {k} scored", flush=True)
pos, neg = np.array(pos), np.array(neg)

miss = np.array([(pos < t).mean() for t in THRESHOLDS])
fa = np.array([(neg >= t).mean() for t in THRESHOLDS])
result = {"name": args.name, "p_defect": P_DEFECT, "scratch_scores": pos.round(3).tolist(),
          "n_good_checks": len(neg), "scenarios": {}}
fig, ax = plt.subplots(figsize=(7, 3.8))
for label, ratio in SCENARIOS.items():
    cost = P_DEFECT * ratio * miss + (1 - P_DEFECT) * 1 * fa
    best = cost.min()
    # among equally cheap thresholds take the middle of the cheapest range (max margin)
    cheapest = THRESHOLDS[np.isclose(cost, best)]
    t_star = float(np.round(np.median(cheapest), 2))
    i = int(np.argmin(np.abs(THRESHOLDS - t_star)))
    result["scenarios"][label] = {"cost_ratio": ratio, "threshold": t_star,
                                  "miss_rate": float(miss[i]), "false_alarm_rate": float(fa[i]),
                                  "cheapest_range": [float(cheapest.min()), float(cheapest.max())]}
    ax.plot(THRESHOLDS, cost, label=f"{label} → t = {t_star:.2f}")
ax.set_xlabel("confidence threshold")
ax.set_ylabel("expected cost per part (false alarm = 1)")
ax.set_title(f"Threshold vs expected cost ({args.name}, defect rate {P_DEFECT:.0%})", loc="left")
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(f"reports/results/threshold_{args.name}.png", dpi=150)
result["chosen"] = result["scenarios"][MAIN]
Path(f"reports/results/threshold_{args.name}.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "scratch_scores"}, indent=1))
print("scratch scores (out-of-fold):", sorted(result["scratch_scores"]))
print("good scores: max", neg.max().round(3), "| 99th pct", np.percentile(neg, 99).round(3))
