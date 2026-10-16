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

### 6. First single nut render (Week 1 start)
- `render/single_nut.py`: one nut, top-down camera, dark ground plane, soft world light
  + weak key light from ~10 o'clock, metallic material with the measured greenish tint.
  Outputs RGB + segmentation mask (nut = category 1) to `output/render_test/0.hdf5`.
- `tools/hdf5_to_png.py`: converts .hdf5 to `_rgb.png`, `_mask.png` and a
  `_compare.png` (real train/good image | render | mask), and prints mean colors.
- Speed: 512 px, 64 samples + Intel denoiser = ~8 s/image (vs 82 s at 1024 samples).
  5,000 images ≈ 11 h, fits in one night.

### 7. Measured nut outline
- First render used guessed tab shapes (thin saw teeth). Visual check of the real nut:
  going clockwise, the long slanted tab edge leads and the short radial edge trails.
- `tools/nut_profile.py`: measures r(θ) (center → outer edge per 1°) on 20 train/good
  images, aligns them and folds into one 90° tab period. Plot: `reports/nut_profile.png`,
  data: `render/nut_profile.json`.
- Result: gap radius 0.36 D, tip radius 0.50 D; long edge rises over ~35°, flat tip
  ~47°, short edge drops within ~3° (radial). All 80 tabs (20 images × 4) agree,
  confirming the visual observation and one fixed handedness.
- `render/single_nut.py` now builds the outline from this profile, so the synthetic
  nut has the real shape and the correct handedness (a mirrored nut = "flip" defect).

### 8. Matching brightness, color, texture and lighting
Each change was checked with numbers printed by `tools/hdf5_to_png.py`
(nut mean RGB, background RGB, local contrast), against real train/good targets.
- Color management: switched BlenderProc's default "Filmic" view transform to
  "Standard", so rendered colors are not bent by a film curve.
- Background: emissive plane with the exact measured color #17171d. It ignores
  lights, so no cast shadow (real images show none). Result: (23, 23, 29) exact.
- Material (`make_metal_material`): procedural noise for fine grain (bump), patchy
  roughness 0.3–0.5 and sparse dark pits. Nut color fixed: the measured sRGB ratios
  had been used as linear values; converted to linear (0.77, 1.0, 0.89).
- Hole chamfer: shallow cone ring (`make_chamfer`) around the hole.
- New metric "local contrast" = std of (pixel − blurred surroundings) inside the nut,
  a number for surface grain. Real: 9.2.
- Lighting finding: with uniform world light, the bump grain was invisible (contrast
  stuck at 7.6 whatever the grain settings), because every surface direction sees the
  same light. A light with direction is needed. A top square light gave grain but made
  slanted edges dark. A ring light (emissive ring 27°–51° from vertical, hidden from
  the camera) gives grain on flat areas and bright slanted edges, like the real images.
  Industrial cameras commonly use ring lights, so this is a plausible real setup.
- Result: nut mean RGB (90, 102, 97) vs real (89, 101, 96); local contrast 10.5 vs 9.2.

### 9. Close-up detail matching (hole bead, boss edge, tab edges, texture)
Compared 350 px close-ups of real vs render (`output/closeup_pair.png`).
- Lathe geometry (`make_lathe`): the center is built by rotating an (r, z)
  cross-section 360°. Hole: convex quarter-circle bead (0.05 D) with its own smoother
  material, matching the wide grey rounded rim around the real hole.
- Boss edge: a quarter-ellipse ramp ended vertically and rendered as a dark band.
  Replaced by a low (0.015 D) S-shaped cosine slope (`smooth_step`, max ~19°), which
  reflects the ring light → bright ridge like the real nut. Rule learned: with a ring
  light at 27°–51°, surfaces tilted ~13°–25° glow, steeper ones go dark.
- Tab edges: bevel 0.02 D (~10 px bright rounded rim). The bevel grew the outline,
  so the curve is offset inward by 0.7 × bevel (measured; nut width 89.7% vs 89.1%).
- Texture: real texture is mostly fine dark speckles (color), not bumps. Reduced bump
  grain, made speckles ~2 px and denser. Too-small speckles (~1 px) vanished in the
  render. Added faint concentric machining marks (1D noise along the radius).
- New metric in `tools/hdf5_to_png.py`: nut width (% of image).
- Result: width 89.7% (real 89.1), nut RGB (89, 102, 96) vs (89, 101, 96),
  background exact, local contrast 10.7 vs 9.2.

### 10. Randomized generator: 200 images (Week 1 deliverable)
- `render/single_nut.py` became `render/generate.py`
  (`blenderproc run render/generate.py --n 200 --seed 0`). One scene is built once,
  then each image only changes parameters → 1.7 s/image (200 images in ~6 min).
- Per image (`sample_params`): rotation 0–360°, camera tilt 0–3° (aimed at the nut),
  distance ±0.5%, ring light ×0.75–1.3, key light angle/elevation/power, roughness,
  grain, speckle density/darkness, color tint, background ±3 levels, and a texture
  offset so every nut has a different pit pattern. Ranges follow the real-data spread
  where measured. `--no-random` renders the base values (used to check that the
  refactor changed nothing: identical numbers to the old script).
- Reproducibility: image i uses RNG seed `[seed, i]`. All parameters go to
  `data/synth/week1/meta.jsonl` (one JSON line per image), input for the Week 2 SQL table.
- Safety: refuses to write into a folder that already has renders.
- `tools/hdf5_to_png.py` prints a summary for many files; `tools/make_grid.py` builds
  `reports/week1_grid.png` (16 evenly spaced images).
- Distribution check, synthetic (200) vs real train/good (220):

  | | synthetic | real |
  |---|---|---|
  | nut width % | 89.9 ± 0.3 | 89.1 ± 0.4 |
  | nut mean gray | 97.4 ± 8.8 [78.8, 120.1] | 98.5 ± 10.0 [83.5, 121.7] |
  | local contrast | 10.3 ± 1.4 | ~9.2 |

- Known small offset: synthetic nuts are ~0.8% wider on average. Fix with FILL
  before the Week 2 dataset render.

## 2026-10-16 (Week 2: defect synthesis and data pipeline)

### 11. Scratch synthesis
- Each scratch is a rotated rectangle on the nut (center, angle, length, width in D),
  following `reports/real_observations.md`: two types, "field" (many fine parallel
  hairlines, 70%) and "band" (wide uniform matte strip, 30%), straight, anywhere on
  the nut. 35% of images have no scratch; otherwise 1/2/3 scratches with p = 0.7/0.2/0.1.
- Appearance (shader, `add_scratch_nodes`): inside the rectangle the color is
  multiplied by a darken factor and roughness goes up (matte). Field lines = noise
  stretched along the scratch direction (scale 4 along, 500 across) and thresholded.
  The first try had too faint lines vs real seen_pool scratches; made them darker/thinner.
- Mask (category 2): the same rectangle's corners are projected into the image with
  the camera (`bproc.camera.project_points`), filled (`fill_convex_polygon`) and
  intersected with the nut mask, so parts over the hole or background are dropped.
  Appearance and mask come from the same numbers, so they line up exactly (checked
  with red outlines in `*_overlay.png`).
- Region-level masks on purpose: real MVTec masks cover the whole damaged area, not
  single lines, so the synthetic labels follow the same convention.
- `meta.jsonl` now stores per scratch: type, geometry, `area_px`, `bbox`; plus
  `n_defects` and `dataset_version` (`--version`, default `synth_v1`).
- Fixed the Week 1 size offset: FILL 0.89 → 0.882 (nut width now ~89.2%, real 89.1%).
- `tools/hdf5_to_png.py` writes `_defect.png` (scratch mask) and `_overlay.png`;
  `tools/make_grid.py --suffix overlay` shows scratch outlines in the grid.

### 12. Metadata database and data checks
- `db/schema.sql`: `images` (one row per render: path, seed, idx, split, n_defects,
  all render parameters) and `defects` (one row per scratch: type, geometry, visible
  `area_px`, bbox). Two tables because an image can have 0-3 scratches. Columns follow
  the parameters we actually randomize (adapted from the roadmap schema).
- `tools/build_db.py`: rebuilds `db/meta.sqlite` from `meta.jsonl` (the source of truth).
  Split: idx % 10 == 0 → val, else train (parameters are i.i.d. per image, so this is
  as good as random, and stable).
- `db/checks.sql`: 5 queries: split balance, defect-free share, scratch area by type,
  empty/tiny masks, parameter coverage.
- `tools/data_stats.py`: runs the queries → `reports/data_stats.md` + 2 histograms:
  scratch area per defective image (synthetic vs the 10 real seen_pool masks) and
  scratches per image. Area is compared per image, because a real mask holds all
  scratches of an image in one file.

### 13. YOLO labels
- `tools/mask_to_yolo.py`: mask → contours (OpenCV `findContours`, outer boundary) →
  simplified polygons (`approxPolyDP`, 1 px tolerance) → normalized YOLOv8-seg lines
  `0 x1 y1 x2 y2 ...`. Defect-free images get an empty label file.
- Self-check: each label is drawn back into a mask and compared with the source mask.
  Round-trip IoU: synthetic test 0.985 (min 0.971), real seen_pool 0.989 (min 0.977).
- Real data converted to `data/yolo/real/{seen_pool,locked_test}`: seen_pool 10 images
  / 18 regions (real masks often have several parts). locked_test 35 images, 22 empty
  labels (good). For locked_test only totals are printed, nothing per image.
- Known limitation: YOLO polygons have no holes. A scratch that fully surrounds the
  hole would get a label that also covers the hole. Rare; round-trip IoU would show it.

### 14. Ablation switches (Week 3 prep)
- `render/generate.py --rand light,material,camera,background` (default: all on).
  Groups: light (ring/key light), material (roughness, grain, pits, tint, texture
  offset), camera (part rotation, tilt, distance), background (color shift).
  Scratches stay random in every run.
- Controlled experiment by design: all random values are always drawn in the same
  order, then switched-off groups are reset to base values. Image i therefore has the
  same scratch geometry in A0…A3; only the tested group differs. Verified: with all
  groups on, parameters are identical to the running synth_v1 render; with none on,
  scratch geometry is identical (only image-space bbox/area change with rotation).
- meta.jsonl records `randomized` per image.

### 15. Colab training setup (Week 3 prep)
- `train/synth_v1.yaml` (dataset config) and `train/train_colab.ipynb`: mount Drive,
  pinned installs (ultralytics 8.3.40, mlflow 2.17.2), unzip data to local disk,
  1-epoch smoke test on 5% of the data, then YOLOv8n-seg (COCO-pretrained) for 60
  epochs at 512 px with early stopping. Weights, plots and MLflow logs go to Drive.

### 16. Realism gap found by eye (→ synth_v2)
- Visual side-by-side (real train/good vs synthetic) showed the synthetic nuts look
  flat: real images have much stronger light/dark variation ("x-ray look"), bright
  bent tab rims, directional speckle texture; the synthetic boss edge is a thick
  bright band that the real nut does not have.
- The Week 1 metric missed this: "local contrast" only measures fine grain (≤3 px),
  and that matched. New metrics in `tools/hdf5_to_png.py` (real targets from 20
  train/good images): mid-scale shading (std of blur 3 px − blur 20 px): real 20.4 vs
  synthetic 15.4; brightness spread (p95 − p5 inside the nut): real 87.4 vs 56.6.
- Lesson: a metric that passes is not proof of realism; check by eye too, and when the
  eye disagrees, find the metric's blind spot.
- Decision: let synth_v1 finish and train it as the baseline; improve realism in
  synth_v2 (tuned on train/good + seen_pool only), then compare v1 vs v2.

### 17. synth_v1 render (5,000 images)
- Started `blenderproc run render/generate.py --n 5000 --seed 0 --out data/synth/synth_v1`.
  Results (DB, stats, YOLO conversion) to be added when it finishes.
