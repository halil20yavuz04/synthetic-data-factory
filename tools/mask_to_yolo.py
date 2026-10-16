"""Convert scratch masks to YOLOv8-seg labels (one .txt per image).

Label format, one line per scratch region:
    0 x1 y1 x2 y2 ...    (class 0 = scratch, polygon corners normalized to 0-1)
Defect-free images get an empty .txt (tells YOLO "nothing here").

Usage (repo root, venv active):
    python tools/mask_to_yolo.py synth --db db/meta.sqlite --out data/yolo/synth_v1
    python tools/mask_to_yolo.py real --out data/yolo/real
synth: reads hdf5 renders, split (train/val) from the database.
real:  MVTec seen_pool and locked_test lists. locked_test is only converted,
       never shown or measured per image (counts only), so no test leakage.

Check: every label is drawn back into a mask and compared with the source mask
(IoU). Round-trip IoU should be ~0.98+, otherwise the labels are wrong.
"""
import argparse
import sqlite3
from pathlib import Path

import cv2
import h5py
import numpy as np

MIN_AREA_PX = 30       # drop specks: regions smaller than this are not a usable label
EPSILON_PX = 1.0       # polygon simplification tolerance (max deviation in pixels)
REAL_ROOT = Path("data/real/mvtec/metal_nut")


def mask_to_polygons(mask):
    """Binary mask -> list of polygons, each an (N, 2) array of pixel coords.
    One polygon per connected region (outer boundary only: YOLO polygons have no holes)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        c = cv2.approxPolyDP(c, EPSILON_PX, closed=True)  # fewer points, same shape
        if len(c) >= 3:
            polygons.append(c.reshape(-1, 2))
    return polygons


def polygons_to_label(polygons, width, height):
    """Polygons in pixels -> YOLO label text (normalized coordinates)."""
    lines = []
    for poly in polygons:
        xy = poly.astype(np.float64) / [width, height]
        lines.append("0 " + " ".join(f"{v:.5f}" for v in xy.flatten()))
    return "\n".join(lines) + ("\n" if lines else "")


def label_to_mask(text, width, height):
    """Inverse of polygons_to_label: draw the label's polygons into a mask."""
    mask = np.zeros((height, width), np.uint8)
    for line in text.splitlines():
        xy = np.array(line.split()[1:], float).reshape(-1, 2) * [width, height]
        cv2.fillPoly(mask, [np.round(xy).astype(np.int32)], 1)
    return mask.astype(bool)


def iou(a, b):
    union = (a | b).sum()
    return 1.0 if union == 0 else (a & b).sum() / union


def write_pair(out, split, name, image_bgr, defect_mask):
    """Save image + label, return (n_regions, round-trip IoU)."""
    h, w = defect_mask.shape
    polygons = mask_to_polygons(defect_mask)
    text = polygons_to_label(polygons, w, h)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.png"), image_bgr)
    (out / "labels" / split / f"{name}.txt").write_text(text)
    return len(polygons), iou(label_to_mask(text, w, h), defect_mask > 0)


def report(split, results, per_image_ok=True):
    n_regions = [r for r, _ in results]
    ious = np.array([i for _, i in results])
    line = (f"{split:<12} {len(results):>5} images, {sum(n_regions):>5} scratch regions, "
            f"{sum(r == 0 for r in n_regions):>5} empty labels")
    if per_image_ok:  # never for locked_test: no per-image numbers before final eval
        line += f", round-trip IoU mean {ious.mean():.3f} / min {ious.min():.3f}"
    print(line)


def convert_synth(db, out):
    con = sqlite3.connect(db)
    results = {"train": [], "val": []}
    for path, split, idx in con.execute("SELECT path, split, idx FROM images ORDER BY idx"):
        with h5py.File(path, "r") as h:
            rgb = np.array(h["colors"])
            seg = np.array(h["category_id_segmaps"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results[split].append(write_pair(out, split, f"{idx:05d}", bgr, seg == 2))
    for split, res in results.items():
        report(split, res)


def convert_real(out):
    for split in ("seen_pool", "locked_test"):
        results = []
        for rel in Path(f"splits/{split}.txt").read_text().split():
            p = Path(rel)  # e.g. test/scratch/003.png
            image = cv2.imread(str(REAL_ROOT / p))
            if p.parent.name == "good":
                mask = np.zeros(image.shape[:2], bool)  # good images have no mask file
            else:
                mask = cv2.imread(str(REAL_ROOT / "ground_truth" / p.parent.name
                                      / f"{p.stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
            name = f"{p.parent.name}_{p.stem}"  # scratch_003, good_007
            results.append(write_pair(out, split, name, image, mask))
        report(split, results, per_image_ok=(split != "locked_test"))


parser = argparse.ArgumentParser()
parser.add_argument("source", choices=["synth", "real"])
parser.add_argument("--db", type=Path, default=Path("db/meta.sqlite"))
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.source == "synth":
    convert_synth(args.db, args.out)
else:
    convert_real(args.out)
