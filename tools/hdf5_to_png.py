"""Convert BlenderProc .hdf5 renders to PNG, plus a side-by-side with a real image.

Usage (repo root, venv active):
    python tools/hdf5_to_png.py output/render_test
Writes next to each N.hdf5: N_rgb.png, N_mask.png, N_compare.png
(compare = real train/good/000.png | render | mask). train/good only: leakage-safe.
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

REAL_REF = Path("data/real/mvtec/metal_nut/train/good/000.png")
# real train/good targets (20 images, resized to 512 px), for quick comparison
REAL_NUT_RGB = (89, 101, 96)
REAL_LOCAL_CONTRAST = 9.2
REAL_NUT_WIDTH = 89.1  # tab-tip diameter, % of image width
REAL_NUT_GRAY = "98.5 ± 10.0 [83.5, 121.7] (220 train/good)"
# added after a visual check showed synthetic nuts look flat (fine grain matched, but
# larger light/dark variation did not): 20 train/good images at 512 px
REAL_MID_SHADING = 20.4    # std of (blur 3 px - blur 20 px): blotches, shading
REAL_SPREAD = 87.4         # gray p95 - p5 inside the nut: overall light/dark range


def local_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    """Std of fine detail (pixel minus its blurred surroundings) inside the nut.
    Measures surface grain; the eroded mask keeps the outline edges out."""
    g = gray.astype(np.float32)
    detail = g - cv2.GaussianBlur(g, (0, 0), 3)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return float(detail[inner].std())


def mid_shading(gray: np.ndarray, mask: np.ndarray) -> float:
    """Std of mid-scale variation (3-20 px): shading blotches, bright rims, dark areas."""
    g = gray.astype(np.float32)
    band = cv2.GaussianBlur(g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 20)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return float(band[inner].std())


def spread(gray: np.ndarray, mask: np.ndarray) -> float:
    """Light/dark range inside the nut: 95th minus 5th percentile of gray."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    p5, p95 = np.percentile(gray[inner], [5, 95])
    return float(p95 - p5)


def main(folder: Path) -> None:
    files = sorted(folder.glob("*.hdf5"))
    assert files, f"no .hdf5 files in {folder}"
    many = len(files) > 10  # many files: print one summary instead of one line each
    stats = {"width": [], "gray": [], "contrast": [], "mid": [], "spread": []}
    for f in files:
        with h5py.File(f, "r") as h:
            rgb = np.array(h["colors"])                 # H x W x 3, uint8, RGB
            seg = np.array(h["category_id_segmaps"])    # H x W, category ids
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = (seg > 0).astype(np.uint8) * 255      # whole nut (incl. scratches)
        defect = (seg == 2).astype(np.uint8) * 255   # scratches only

        cv2.imwrite(str(f.with_name(f"{f.stem}_rgb.png")), bgr)
        cv2.imwrite(str(f.with_name(f"{f.stem}_mask.png")), mask)
        cv2.imwrite(str(f.with_name(f"{f.stem}_defect.png")), defect)
        # visual check: scratch mask outline in red on the render
        overlay = bgr.copy()
        contours, _ = cv2.findContours(defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
        cv2.imwrite(str(f.with_name(f"{f.stem}_overlay.png")), overlay)

        if not many:
            real = cv2.resize(cv2.imread(str(REAL_REF)), bgr.shape[1::-1],
                              interpolation=cv2.INTER_AREA)
            compare = np.hstack([real, bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.imwrite(str(f.with_name(f"{f.stem}_compare.png")), compare)

        nut = mask > 0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        contrast = local_contrast(gray, nut)
        ys, xs = np.nonzero(nut)
        h, w = nut.shape
        width = 200 * np.hypot(xs - (w - 1) / 2, ys - (h - 1) / 2).max() / w
        # same definition as the real-data measurement: nut pixels with gray >= 40
        stats["width"].append(width)
        stats["gray"].append(gray[nut & (gray >= 40)].mean())
        stats["contrast"].append(contrast)
        stats["mid"].append(mid_shading(gray, nut))
        stats["spread"].append(spread(gray, nut))
        if not many:
            print(f"{f.name}: nut width {width:.1f}% of image (real ~{REAL_NUT_WIDTH}), "
                  f"nut mean RGB {rgb[nut].mean(0).round(0)} (real ~{REAL_NUT_RGB}), "
                  f"background mean RGB {rgb[~nut].mean(0).round(0)}, "
                  f"local contrast {contrast:.1f} (real ~{REAL_LOCAL_CONTRAST}), "
                  f"mid shading {stats['mid'][-1]:.1f} (real ~{REAL_MID_SHADING}), "
                  f"spread {stats['spread'][-1]:.0f} (real ~{REAL_SPREAD})")

    if many:
        print(f"{len(files)} images converted. Summary (mean ± std, [min, max]):")
        for label, key, real in (("nut width %", "width", f"{REAL_NUT_WIDTH} ± 0.4"),
                                 ("nut mean gray", "gray", REAL_NUT_GRAY),
                                 ("local contrast", "contrast", f"~{REAL_LOCAL_CONTRAST}"),
                                 ("mid shading", "mid", f"~{REAL_MID_SHADING}"),
                                 ("spread p95-p5", "spread", f"~{REAL_SPREAD}")):
            v = np.array(stats[key])
            print(f"  {label:<15} {v.mean():6.1f} ± {v.std():4.1f}  "
                  f"[{v.min():.1f}, {v.max():.1f}]   real: {real}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/render_test"))
