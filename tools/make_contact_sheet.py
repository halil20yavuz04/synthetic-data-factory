"""Make contact sheets of real metal_nut images for Day 0 observation notes.

Shows only leakage-safe images:
  - N_GOOD random images from train/good
  - the 10 seen_pool scratch images (raw + mask contour)
Refuses to run if any selected image is in splits/locked_test.txt.
Output goes to output/contact_sheets/ (gitignored, MVTec license).
"""
import random
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("data/real/mvtec/metal_nut")
SPLIT_DIR = Path("splits")
OUT_DIR = Path("output/contact_sheets")
SEED = 0
N_GOOD = 20
TILE = 256  # each image is resized to TILE x TILE pixels
COLS = 5


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_tile(rel_path: str) -> np.ndarray:
    img = cv2.imread(str(DATA_ROOT / rel_path))
    assert img is not None, f"could not read {rel_path}"
    return cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)


def mask_path_for(rel_path: str) -> Path:
    # test/scratch/003.png -> ground_truth/scratch/003_mask.png
    p = Path(rel_path)
    return DATA_ROOT / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"


def label(tile: np.ndarray, text: str) -> np.ndarray:
    tile = tile.copy()
    cv2.putText(tile, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def make_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    # pad with black tiles so the last row is full
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> None:
    locked = set(read_list(SPLIT_DIR / "locked_test.txt"))
    seen = read_list(SPLIT_DIR / "seen_pool.txt")

    all_good = sorted(str(p.relative_to(DATA_ROOT))
                      for p in (DATA_ROOT / "train" / "good").glob("*.png"))
    good = sorted(random.Random(SEED).sample(all_good, N_GOOD))

    # leakage guard: nothing we show may come from the locked test set
    leaked = locked & set(good + seen)
    assert not leaked, f"refusing to show locked test images: {sorted(leaked)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) good sheet
    good_tiles = [label(load_tile(p), Path(p).stem) for p in good]
    cv2.imwrite(str(OUT_DIR / "good_sheet.png"), make_grid(good_tiles, COLS))

    # 2) scratch sheet: raw tile, then same tile with mask contour in red
    scratch_tiles = []
    print(f"{'image':<22}{'area_px':>9}{'area_%':>8}{'len_px':>8}{'width_px':>10}")
    for p in seen:
        img = cv2.imread(str(DATA_ROOT / p))
        mask = cv2.imread(str(mask_path_for(p)), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"missing mask for {p}"
        mask = (mask > 0).astype(np.uint8)

        # size stats on the original 700x700 image
        area = int(mask.sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # minAreaRect fits a rotated box: its long side ~ scratch length
        (_, _), (w, h), _ = cv2.minAreaRect(np.vstack(contours))
        length, width = max(w, h), min(w, h)
        print(f"{p:<22}{area:>9}{100 * area / mask.size:>8.2f}"
              f"{length:>8.0f}{width:>10.0f}")

        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)
        raw = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (TILE, TILE), interpolation=cv2.INTER_AREA)
        scratch_tiles += [label(raw, Path(p).stem), label(overlay, "mask")]

    # 2 tiles per image -> 4 columns = 2 images per row
    cv2.imwrite(str(OUT_DIR / "scratch_sheet.png"), make_grid(scratch_tiles, 4))
    print(f"\nwrote {OUT_DIR}/good_sheet.png and {OUT_DIR}/scratch_sheet.png")


if __name__ == "__main__":
    main()
