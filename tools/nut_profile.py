"""Measure the real nut outline as a polar profile r(theta) on train/good images.

For each angle, r = distance from the hole center to the outermost nut pixel.
Profiles of 20 images are rotation-aligned and folded into one 90° period
(4 identical tabs), giving a measured tab shape for the renderer.

Outputs:
  reports/nut_profile.png   per-image aligned profiles + mean tab profile
  render/nut_profile.json   mean profile of one tab period, r / D (D = 2 * max r)
Angles: 0° = right, counter-clockwise positive (same as tools/light_direction.py).
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

N_IMAGES = 20
N_BINS = 360   # 1° per bin
PERIOD = N_BINS // 4


def find_center(gray):
    h, w = gray.shape
    _, labels, _, centroids = cv2.connectedComponentsWithStats((gray < 40).astype(np.uint8))
    return centroids[labels[h // 2, w // 2]]


def polar_profile(gray):
    cx, cy = find_center(gray)
    fg = (gray >= 40).astype(np.uint8)
    # keep only the largest bright blob (the nut), drop specks
    _, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    nut = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(nut)
    r = np.hypot(xs - cx, ys - cy)
    theta = np.degrees(np.arctan2(-(ys - cy), xs - cx)) % 360
    bins = theta.astype(int) % N_BINS
    prof = np.zeros(N_BINS)
    np.maximum.at(prof, bins, r)  # outermost nut pixel per 1° bin
    return prof


def align(prof, ref):
    """Circularly shift prof so it best matches ref (cross-correlation)."""
    scores = [np.dot(np.roll(prof, s), ref) for s in range(PERIOD)]
    return np.roll(prof, int(np.argmax(scores)))


paths = sorted(glob.glob("data/real/mvtec/metal_nut/train/good/*.png"))[:N_IMAGES]
assert paths, "no images found, run from the repo root"
profiles = [polar_profile(cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            for p in paths]

# fold each profile into one 90° period first, so the reference is a clean tab
ref = profiles[0].reshape(4, PERIOD).mean(0)
ref = np.tile(ref, 4)
aligned = [align(p, ref) for p in profiles]
tab = np.mean([a.reshape(4, PERIOD) for a in aligned], axis=(0, 1))

# start the period at the minimum (gap between tabs) for a readable plot
tab = np.roll(tab, -int(np.argmin(tab)))
D = 2 * tab.max()
tab_norm = tab / D

# slope of the two tab flanks: degrees of angle needed to rise 80% of the height
lo, hi = tab.min(), tab.max()
thr_lo, thr_hi = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
peak = int(np.argmax(tab))
rise = np.argmax(tab[:peak + 1] >= thr_hi) - np.argmax(tab[:peak + 1] >= thr_lo)
fall_part = tab[peak:]
fall = np.argmax(fall_part <= thr_lo) - np.argmax(fall_part <= thr_hi)

print(f"r_min (gap between tabs): {lo:.0f} px = {lo / D:.3f} D")
print(f"r_max (tab tip):          {hi:.0f} px = {hi / D:.3f} D  (D = {D:.0f} px)")
print(f"flank going counter-clockwise into the tab (rising): {rise}°")
print(f"flank going counter-clockwise out of the tab (falling): {fall}°")
print("-> the flank with fewer degrees is the steep, radial edge")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(PERIOD)
for a in aligned:
    folded = np.roll(a.reshape(4, PERIOD), -int(np.argmin(a.reshape(4, PERIOD).mean(0))), axis=1)
    for row in folded:
        ax.plot(x, row / D, lw=0.4, alpha=0.3, color="grey")
ax.plot(x, tab_norm, lw=2.5, color="tab:red", label="mean tab profile")
ax.set_xlabel("angle within one tab period (°), counter-clockwise →")
ax.set_ylabel("r / D")
ax.set_title(f"Nut outline, polar profile ({len(paths)} train/good images x 4 tabs)")
ax.legend()
fig.tight_layout()
fig.savefig("reports/nut_profile.png", dpi=150)

Path("render/nut_profile.json").write_text(json.dumps({
    "source": f"tools/nut_profile.py, {len(paths)} train/good images",
    "units": "r / D, D = tab-tip diameter",
    "angle_step_deg": 360 / N_BINS,
    "r_over_D": [round(float(v), 4) for v in tab_norm],
}, indent=1))
print("wrote reports/nut_profile.png and render/nut_profile.json")
