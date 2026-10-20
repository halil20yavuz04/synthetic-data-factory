"""Estimate dominant light direction from angular brightness on the nut's ring.
The ring is rotationally symmetric, so angular variation there comes from lighting.

Uses train/good only (leakage-safe). Outputs:
  reports/analysis/light_direction.png        angular brightness profiles
  output/light_direction_check.png   sanity check: ring drawn on one image
"""
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

N_IMAGES = 20
N_BINS = 36
R_IN, R_OUT = 110, 170  # ring radii in px (700x700 image), between hole chamfer and boss edge


def find_center(gray):
    """Center = centroid of the dark hole region that contains the image center."""
    h, w = gray.shape
    dark = (gray < 40).astype(np.uint8)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(dark)
    hole = labels[h // 2, w // 2]
    if hole == 0:  # image center is not dark -> fall back to image center
        return w / 2, h / 2
    return tuple(centroids[hole])


def angular_profile(gray, cx, cy, r_in, r_out, n_bins=N_BINS):
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)
    theta = np.degrees(np.arctan2(-(yy - cy), xx - cx))  # minus: image y points down
    # -> 0° = right (3 o'clock), 90° = up, 5 o'clock ≈ -60°
    ring = (r >= r_in) & (r < r_out)
    edges = np.linspace(-180, 180, n_bins + 1)
    sums, _ = np.histogram(theta[ring], bins=edges, weights=gray[ring])
    counts, _ = np.histogram(theta[ring], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, sums / counts


def clock(deg):
    """Convert our angle (0° = 3 o'clock, counter-clockwise) to a clock position."""
    hour = (3 - deg / 30) % 12
    return 12 if round(hour) == 0 else round(hour)


paths = sorted(glob.glob("data/real/mvtec/metal_nut/train/good/*.png"))[:N_IMAGES]
assert paths, "no images found, run from the repo root"

fig, ax = plt.subplots(figsize=(8, 4))
profiles = []
print(f"{'image':<10}{'center offset (px)':>20}{'peak angle':>12}{'clock':>7}")
for i, p in enumerate(paths):
    gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    cx, cy = find_center(gray)
    centers, prof = angular_profile(gray, cx, cy, R_IN, R_OUT)
    profiles.append(prof)
    ax.plot(centers, prof, lw=0.6, alpha=0.5, color="grey")

    peak = centers[np.argmax(prof)]
    offset = np.hypot(cx - gray.shape[1] / 2, cy - gray.shape[0] / 2)
    print(f"{Path(p).name:<10}{offset:>20.1f}{peak:>11.0f}°{clock(peak):>7}")

    if i == 0:  # sanity check: is the ring on the nut's flat boss, away from the hole?
        check = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for r in (R_IN, R_OUT):
            cv2.circle(check, (int(cx), int(cy)), r, (0, 0, 255), 2)
        Path("output").mkdir(exist_ok=True)
        cv2.imwrite("output/light_direction_check.png", check)

mean = np.mean(profiles, axis=0)
peak = centers[np.argmax(mean)]
ax.plot(centers, mean, lw=2.5, color="tab:red", label=f"mean (peak {peak:.0f}°)")
ax.axvline(peak, ls="--", color="tab:red", lw=1)
ax.set_xticks(range(-180, 181, 30))
ax.set_xlabel("angle (°)   0 = right/3 o'clock, 90 = up/12, -90 = down/6")
ax.set_ylabel("mean ring brightness")
ax.set_title(f"Ring brightness vs angle, {len(paths)} train/good images")
ax.legend()
fig.tight_layout()
Path("reports/analysis").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/analysis/light_direction.png", dpi=150)

spread = np.ptp(mean) / mean.mean() * 100
print(f"\nmean peak: {peak:.0f}° (~{clock(peak)} o'clock), "
      f"brightness range across angles: {spread:.1f}% of mean")
print("wrote reports/analysis/light_direction.png and output/light_direction_check.png")
