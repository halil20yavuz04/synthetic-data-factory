"""Tile rendered images into one grid PNG for a quick visual check.

Usage (repo root, venv active, after tools/hdf5_to_png.py):
    python tools/make_grid.py data/synth/week1 reports/analysis/week1_grid.png
Takes N evenly spaced *_rgb.png from the folder (default 16 -> 4x4 grid),
labels each tile with its image index.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--cols", type=int, default=4)
parser.add_argument("--tile", type=int, default=256)
parser.add_argument("--suffix", default="rgb", help="rgb or overlay (scratch outlines)")
args = parser.parse_args()

# numeric sort: 2_rgb.png before 10_rgb.png
files = sorted(args.folder.glob(f"*_{args.suffix}.png"), key=lambda p: int(re.match(r"\d+", p.name)[0]))
assert files, f"no *_{args.suffix}.png in {args.folder}, run tools/hdf5_to_png.py first"
picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(args.n, len(files)))]

tiles = []
for f in picks:
    img = cv2.resize(cv2.imread(str(f)), (args.tile, args.tile), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f.name.split("_")[0], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(img)
while len(tiles) % args.cols:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]

args.out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(args.out), np.vstack(rows))
print(f"wrote {args.out} ({len(picks)} images from {args.folder})")
