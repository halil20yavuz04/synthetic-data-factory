"""Split MVTec metal_nut test images into seen_pool and locked_test.

Run once from the repo root. Never re-run with a different seed
(that would be test leakage). Only file names are read, no images are opened.
"""
import argparse
import random
from pathlib import Path

DATA_ROOT = Path("data/real/mvtec/metal_nut")
SPLIT_DIR = Path("splits")
SEED = 0
N_SEEN = 10


def list_images(folder: Path) -> list[str]:
    """Return .png paths relative to DATA_ROOT, sorted for determinism."""
    return sorted(str(p.relative_to(DATA_ROOT)) for p in folder.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing split files (avoid!)")
    args = parser.parse_args()

    seen_path = SPLIT_DIR / "seen_pool.txt"
    locked_path = SPLIT_DIR / "locked_test.txt"

    if (seen_path.exists() or locked_path.exists()) and not args.force:
        raise SystemExit(f"Split files already exist in {SPLIT_DIR}/. "
                         "Refusing to overwrite (use --force only if you know why).")

    scratch = list_images(DATA_ROOT / "test" / "scratch")
    good = list_images(DATA_ROOT / "test" / "good")

    rng = random.Random(SEED)
    rng.shuffle(scratch)
    seen = sorted(scratch[:N_SEEN])
    locked_scratch = sorted(scratch[N_SEEN:])
    locked = locked_scratch + good

    assert len(seen) == 10, f"expected 10 seen, got {len(seen)}"
    assert len(locked_scratch) == 13, f"expected 13 locked scratch, got {len(locked_scratch)}"
    assert len(good) == 22, f"expected 22 good, got {len(good)}"
    assert not set(seen) & set(locked), "seen_pool and locked_test overlap"

    SPLIT_DIR.mkdir(exist_ok=True)
    seen_path.write_text("\n".join(seen) + "\n")
    locked_path.write_text("\n".join(locked) + "\n")

    print(f"seen_pool:   {len(seen)} scratch -> {seen_path}")
    print(f"locked_test: {len(locked_scratch)} scratch + {len(good)} good "
          f"= {len(locked)} -> {locked_path}")


if __name__ == "__main__":
    main()
