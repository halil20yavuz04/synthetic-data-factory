"""Showcase figures for the README (reports/figures/).

    python tools/make_figures.py
Needs: real MVTec data, data/synth/synth_v1 + synth_v2 renders, data/yolo/synth_v2 labels,
models in output/models/. The locked test images appear only in figure 7 (it was already
evaluated once; showing it changes nothing).
"""
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import label_to_mask, predict_mask  # noqa: E402

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
REAL = Path("data/real/mvtec/metal_nut")
INK, INK2, GRAY = "#0b0b0b", "#52514e", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREEN_BGR, RED_BGR = (40, 200, 40), (40, 40, 230)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "figure.facecolor": "white"})


# --- helpers --------------------------------------------------------------------
def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_render(version, idx):
    with h5py.File(f"data/synth/{version}/{idx}.hdf5", "r") as h:
        return cv2.cvtColor(np.array(h["colors"]), cv2.COLOR_RGB2BGR), np.array(h["category_id_segmaps"])


def outline(img, mask, color, width=3):
    out = img.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, width)
    return out


def crop_center(img, frac, cx=0.5, cy=0.5):
    h, w = img.shape[:2]
    s = int(min(h, w) * frac)
    x0 = int(np.clip(cx * w - s / 2, 0, w - s))
    y0 = int(np.clip(cy * h - s / 2, 0, h - s))
    return img[y0:y0 + s, x0:x0 + s]


def crop_mask(img, mask, pad=0.3, min_frac=0.3):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    s = max(int(max(np.ptp(xs), np.ptp(ys)) * (1 + pad)), int(min_frac * w))
    s = min(s, h, w)   # crop can't be larger than the image (multi-part masks)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0, y0 = int(np.clip(cx - s // 2, 0, w - s)), int(np.clip(cy - s // 2, 0, h - s))
    return img[y0:y0 + s, x0:x0 + s]


def show(ax, img_bgr, title=None, sub=None):
    ax.imshow(rgb(img_bgr))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, loc="left", fontweight="bold", fontsize=10.5)
    if sub:
        ax.text(0, -0.02, sub, transform=ax.transAxes, va="top", fontsize=9, color=INK2)


def row_label(ax, text):
    ax.text(-0.06, 0.5, text, transform=ax.transAxes, rotation=90, va="center", ha="right",
            fontsize=12, fontweight="bold", color=INK)


def save(fig, name, title, left=0.02):
    fig.subplots_adjust(left=left)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def seen_mask(stem):
    return cv2.imread(str(REAL / "ground_truth" / "scratch" / f"{stem}_mask.png"), 0) > 0


# --- 1. real vs synthetic --------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for ax, f in zip(axes[0], ("000.png", "066.png", "122.png", "212.png")):
    show(ax, cv2.imread(str(REAL / "train" / "good" / f)))
for ax, i in zip(axes[1], (1, 3, 13, 17)):
    show(ax, load_render("synth_v2", i)[0])
row_label(axes[0, 0], "Real photos\n(MVTec AD)")
row_label(axes[1, 0], "Synthetic renders\n(BlenderProc)")
save(fig, "01_real_vs_synthetic.png",
     "Real vs. synthetic: every synthetic nut has its own random rotation, light, texture and color",
     left=0.08)

# --- 2. free pixel-perfect labels ------------------------------------------------
idx = 84
img, seg = load_render("synth_v2", idx)
meta = [json.loads(l) for l in open("data/synth/synth_v2/meta.jsonl")][idx]
segvis = np.zeros_like(img)
segvis[seg == 1] = (150, 150, 150)
segvis[seg == 2] = RED_BGR
split = "val" if idx % 10 == 0 else "train"
label_file = Path(f"data/yolo/synth_v2/labels/{split}/{idx:05d}.txt")
poly_img = img.copy()
for line in label_file.read_text().splitlines():
    xy = (np.array(line.split()[1:], float).reshape(-1, 2) * img.shape[1]).astype(np.int32)
    cv2.polylines(poly_img, [xy], True, (0, 255, 255), 2)
    for p in xy[::max(1, len(xy) // 25)]:
        cv2.circle(poly_img, tuple(p), 3, (0, 255, 255), -1)
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
show(axes[0], img, "1. Rendered image", "what the model sees")
show(axes[1], segvis, "2. Segmentation map", "rendered with the image, for free")
show(axes[2], outline(img, seg == 2, RED_BGR, 2), "3. Scratch mask", "red = scratch pixels")
show(axes[3], poly_img, "4. YOLO training label", "mask → polygon (round-trip IoU 0.99)")
axes[1].legend(handles=[Patch(color="#000000", label="background"),
                        Patch(color="#969696", label="nut"), Patch(color="#e62828", label="scratch")],
               loc="lower right", fontsize=8, framealpha=0.9)
types = " + ".join(sorted({d["type"] for d in meta["defects"]}))
save(fig, "02_free_labels.png",
     f"Every render comes with a pixel-perfect label (this image: {types} scratch)")

# --- 3. scratch types: real vs synthetic -----------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 7))
for ax, (stem, kind) in zip(axes[0], (("006", "hairline field"), ("021", "hairline field"),
                                      ("011", "gouge band"), ("014", "gouge band"))):
    real = cv2.imread(str(REAL / "test" / "scratch" / f"{stem}.png"))
    show(ax, crop_mask(real, seen_mask(stem)), kind)
rows = [json.loads(l) for l in open("data/synth/synth_v2/meta.jsonl")]
picks = []
for kind, k in (("field", 2), ("band", 2)):
    picks += [r for r in rows if r["n_defects"] == 1 and r["defects"][0]["type"] == kind
              and r["defects"][0]["area_px"] > 5000][2:2 + k]
for ax, r in zip(axes[1], picks):
    im, sg = load_render("synth_v2", r["index"])
    show(ax, crop_mask(im, sg == 2), "hairline field" if r["defects"][0]["type"] == "field" else "gouge band")
row_label(axes[0, 0], "Real scratches")
row_label(axes[1, 0], "Synthetic scratches")
save(fig, "03_scratch_types.png",
     "Two scratch types, drawn stroke by stroke with depth (fields: many thin grooves; bands: ground-flat strip)",
     left=0.08)

# --- 4. realism iterations v1 -> v2 ----------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
real = cv2.imread(str(REAL / "train" / "good" / "000.png"))
v1 = cv2.resize(load_render("synth_v1", 8)[0], (700, 700))
v2 = cv2.resize(load_render("synth_v2", 3)[0], (700, 700))
for col, (im, t, s) in enumerate(((v1, "synth_v1", "clean, uniform surface"),
                                  (v2, "synth_v2", "measured texture, light, countersink, rim"),
                                  (real, "real", "MVTec AD train/good"))):
    show(axes[0, col], im, t, s)
    show(axes[1, col], crop_center(im, 0.42, 0.62, 0.38))
row_label(axes[0, 0], "Full view")
row_label(axes[1, 0], "Close-up")
save(fig, "04_realism_v1_v2.png",
     "Realism iterations: v1 looked like clean plastic; v2 was tuned against real images (metrics + review)",
     left=0.08)

# --- 5. domain gap: what each model marks on real images -------------------------
models = [("synth_v1 only", YOLO("output/models/synth_v1_best.pt")),
          ("synth_v2 only", YOLO("output/models/synth_v2_best.pt")),
          ("synth_v2 + 10 real (final)", YOLO("output/models/final_v2_ft_replay.pt"))]
cases = [(REAL / "train" / "good" / "217.png", None, "good nut"),
         (REAL / "test" / "scratch" / "006.png", seen_mask("006"), "scratched nut")]
fig, axes = plt.subplots(2, 3, figsize=(12, 8.6))
for r, (p, gt, what) in enumerate(cases):
    for c, (name, m) in enumerate(models):
        pred, conf = predict_mask(m, p, 0.59 if "final" in name else 0.25)
        im = cv2.imread(str(p))
        if gt is not None:
            im = outline(im, gt, GREEN_BGR, 4)
        im = outline(im, pred, RED_BGR, 4)
        if gt is None:
            verdict = "false alarm" if pred.any() else "correctly accepted"
        else:
            iou = (pred & gt).sum() / max((pred | gt).sum(), 1)
            what = ("wrong area" if iou < 0.1 else "found, but too wide / extra regions" if iou < 0.5
                    else "scratch outlined")
            verdict = f"{what} (IoU {iou:.2f})"
        show(axes[r, c], im, name if r == 0 else None, verdict)
row_label(axes[0, 0], "Good nut")
row_label(axes[1, 0], "Scratched nut")
fig.text(0.01, -0.01, "green = true scratch, red = model prediction", fontsize=10, color=INK2)
save(fig, "05_domain_gap.png",
     "Domain gap: v1 flags any texture, v2 flags the nut rim, fine-tuning on 10 real images fixes it",
     left=0.08)

# --- 6. results ------------------------------------------------------------------
variants = ["real\nonly", "synth\nv1", "synth\nv2", "v2 +\nFT", "v2 + FT\n+ replay"]
found = [1, 10, 10, 9, 10]
false_alarm = [0, 98, 78, 0, 0]
iou = [0.06, 0.09, 0.53, 0.58, 0.68]
colors = [GRAY, GRAY, GRAY, GRAY, BLUE]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
panels = [(found, "Scratches found (of 10)", "{:.0f}", 10),
          (false_alarm, "False alarms on good nuts (%)", "{:.0f}%", 100),
          (iou, "Mask IoU on scratches", "{:.2f}", 1)]
for ax, (vals, title, fmt, top) in zip(axes, panels):
    bars = ax.bar(range(5), vals, color=colors, width=0.62, edgecolor="white", linewidth=2)
    ax.bar_label(bars, labels=[fmt.format(v) for v in vals], padding=3, color=INK, fontsize=10)
    ax.set_xticks(range(5), variants, fontsize=9.5, color=INK2)
    ax.set_ylim(0, top * 1.32)
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.tick_params(axis="y", colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ecebe7")
    ax.set_axisbelow(True)
axes[0].text(1, 11.6, "v1 marked random\ntexture (IoU ≈ 0)", ha="center", fontsize=8, color=INK2)
fig.text(0.01, -0.05, "real only = YOLOv8n-seg from COCO weights on 8 real scratch images + 40 good.  "
         "FT = fine-tuned on the same real images, starting from the synth_v2 model.  "
         "replay = 300 synthetic images mixed in.\n"
         "Dev set: 5-fold cross-validation over the 10 seen real scratches + 50 good nuts.  "
         "Final model on the untouched locked test: 12/13 scratches found, 0/22 false alarms, IoU 0.63.",
         fontsize=9.5, color=INK2, linespacing=1.6)
save(fig, "06_results.png", "Results: synthetic pre-training + 10 real images beats 8 real images alone by far")

# --- 7. final model on the locked test -------------------------------------------
final = models[2][1]
L = Path("data/yolo/real/images/locked_test")
LL = Path("data/yolo/real/labels/locked_test")
picks = ["scratch_003", "scratch_013", "scratch_017", "scratch_022", "scratch_001", "good_005"]
fig, axes = plt.subplots(2, 3, figsize=(12, 8.6))
for ax, stem in zip(axes.flat, picks):
    p = L / f"{stem}.png"
    im = cv2.imread(str(p))
    gt = label_to_mask(LL / f"{stem}.txt", im.shape[1], im.shape[0])
    pred, _ = predict_mask(final, p, 0.59)
    best = predict_mask(final, p, 0.01)[1]
    im = outline(outline(im, gt, GREEN_BGR, 4), pred, RED_BGR, 4)
    if gt.any():
        sub = f"found (score {best:.2f})" if pred.any() else f"MISSED: score {best:.2f} < threshold 0.59"
    else:
        sub = "good nut: correctly accepted"
    show(ax, im, stem.replace("_", " "), sub)
fig.text(0.01, -0.01, "green = true scratch, red = model prediction. Threshold 0.59 chosen with a cost "
         "model before the test was opened.", fontsize=10, color=INK2)
save(fig, "07_locked_test.png", "Final model on the locked test (seen once): 12/13 scratches found, 0/22 false alarms")
