# Progress log

Chronological log of what was done, why, and where it lives.
Newest day at the bottom. Plan: 4 weeks (Day 0 → Week 4), see roadmap PDF.

---

## 2026-10-16 (Day 0: setup, data split, real-data observations)

### 1. Environment and repo
- Created the repo with the planned folder layout: `render/`, `data/`, `db/`, `tools/`,
  `train/`, `serve/`, `reports/`. `data/` and `output/` are gitignored (MVTec license:
  no data in the repo).
- Python 3.10 venv at `.venv`. BlenderProc 2.8 / Blender 4.2.1 with Metal GPU.
  Quickstart render works: 512 px at 1024 samples takes ~82 s/image.
- Downloaded MVTec AD `metal_nut` to `data/real/mvtec/metal_nut` from the HF mirror
  `foersben/mvtec-ad`, because the official links were broken.
- Counts: train/good 220; test: good 22, scratch 23, bent 25, color 22, flip 23.

### 2. Leakage-safe test split
- `tools/make_split.py` (seed 0) splits `test/scratch` into:
  - `splits/seen_pool.txt`: 10 scratch images for few-shot fine-tuning and threshold work.
  - `splits/locked_test.txt`: 13 scratch + 22 good. Not opened until the final evaluation.
- File names are sorted before shuffling, so the split is identical on Mac and Colab.
  The script refuses to overwrite existing split files.
- All 22 test/good images go to the locked test, because train/good (220) already
  covers good images, and the test needs good images to measure false alarms.
- Committed in `48851ba`.

### 3. Contact sheets for observation
- `tools/make_contact_sheet.py` builds two image grids in `output/contact_sheets/`:
  20 random train/good images, and the 10 seen_pool scratch images (raw + mask outline).
  It refuses to run if any selected image is in the locked test.
- Scratch size stats: mask area is 1.9–11.9% of the image (median ~4%).

### 4. Light direction measurement
- `tools/light_direction.py` measures brightness around the flat center ring vs angle
  on 20 train/good images. Plot: `reports/light_direction.png`.
- Result: light comes from ~10 o'clock (upper-left) and is fixed relative to the camera.
  The effect is weak (16% brightness range), so the lighting is mostly soft and even.
  Overall brightness varies a lot between images (~50–115) → randomize light power.
- Also measured: nut is centered within 2.5 px, tab-tip diameter 623 ± 2.5 px (fixed
  scale), hole/outer ratio 0.246, background #17171d.

### 5. Real image observations
- `reports/real_observations.md`: camera, background, lighting, nut shape and scratch
  observations, each labeled [measured] or [visual]. The summary table at the top lists
  the render targets for Week 1.
- Key findings: 4 hooked tabs with fixed handedness (a mirrored nut = "flip" defect);
  scratches are straight, either hairline fields or wide gouge bands; masks are coarse
  region outlines.
- Committed in `95e56bc`.

### Next
- Week 1: first single nut render in BlenderProc, then camera/light matching the
  observation table, then `render/generate.py --n 200 --seed 0`.
- Render budget: 82 s/image is too slow for 5,000 images (~114 h). Lower samples +
  denoiser.
