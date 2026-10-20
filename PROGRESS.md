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
  on 20 train/good images. Plot: `reports/analysis/light_direction.png`.
- Result: light comes from ~10 o'clock (upper-left) and is fixed relative to the camera.
  The effect is weak (16% brightness range), so the lighting is mostly soft and even.
  Overall brightness varies a lot between images (~50–115) → randomize light power.
- Also measured: nut is centered within 2.5 px, tab-tip diameter 623 ± 2.5 px (fixed
  scale), hole/outer ratio 0.246, background #17171d.

### 5. Real image observations
- `reports/analysis/real_observations.md`: camera, background, lighting, nut shape and scratch
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
  images, aligns them and folds into one 90° tab period. Plot: `reports/analysis/nut_profile.png`,
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
  `reports/analysis/week1_grid.png` (16 evenly spaced images).
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
  following `reports/analysis/real_observations.md`: two types, "field" (many fine parallel
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
- `tools/data_stats.py`: runs the queries → `reports/analysis/data_stats.md` + 2 histograms:
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
  v1 code is tagged in git as `synth_v1`.
- synth_v2 changes (`render/generate.py`, tuned with `--no-defects` renders):
  - boss step 0.015 → 0.006 high, ramp 0.07 → 0.025 wide: thin line, not a thick band;
  - tabs bend down towards the rim (shader bump only, outline/masks unchanged), tip
    slope ~24° so the outer tab areas catch the ring light and glow, like real rims;
  - glints: tiny spots with 20% roughness that sparkle; pits stretched 3:1 (streaky);
  - roughness 0.3–0.5 → 0.22–0.42 (sharper highlights); wider hole bead (0.065 D).
  - A first try added mid-size color blotches: the metric went up but the nut looked
    like marble. Removed (kept very subtle). Same lesson again: improve the cause
    (geometry, glints), not the number.

- Second round, from more visual feedback (real vs v2 side by side):
  - countersink: straight cone (35°) sloping into the hole, gray-blue, turned rings
    (v1 had a convex green bead; the real hole edge is a gray conical seat);
  - micro-scratches on every nut (3 random directions, short thin lines, slightly
    darker, tiny grooves). Real GOOD nuts are full of them; they are normal surface,
    never in the mask, so the model learns they are not defects. Two failed tries:
    lines 700/D across were sub-pixel → gray camouflage blobs; 5/D along → curly worms;
  - green-yellow base color with neutral gray pits (bright = green, dark = gray);
  - broad waviness bump → large bright patches; ring light 0.35 → 0.40.

  | (mean of 8 good-only renders) | v1 | v2 | real |
  |---|---|---|---|
  | fine grain | 10.0 | 12.0 | 9.2 |
  | mid-scale shading | 15.4 | 20.4 | 20.4 |
  | brightness spread p95−p5 | 56.6 | 92.1 | 87.4 |

- Status: numbers match, but close-ups still look cleaner and glossier than the real,
  dense, "dirty" texture. Procedural noise shows diminishing returns here.
- Option considered: hybrid (real train/good photo as surface texture + synthetic
  scratches). Decision: stay fully procedural (cleaner "everything from Blender"
  story) and keep tuning.
- Third round:
  - second, larger pit layer (scale 110) on top of the fine pits → multi-size "dirty"
    texture; pits denser (threshold 0.48–0.6);
  - broad hue shifts (bluish / neutral / yellowish), since the real nuts look like
    yellow zinc coating whose tint varies across the part;
  - micro-scratches darker and denser;
  - lens softness: Gaussian blur σ 0.3–1.1 px on the RGB after rendering (masks stay
    crisp like real ground truth). Without it, sharpness alone would tell synthetic
    from real, and a model can learn such shortcuts;
  - ring light 0.40 → 0.48 (pits made the nut darker).
  - Intermediate try had too strong yellow (olive look) and was too dark; softened.

  | (mean of 8 good-only renders) | v1 | v2 round 2 | v2 round 3 | real |
  |---|---|---|---|---|
  | fine grain | 10.0 | 12.0 | 10.0 | 9.2 |
  | mid-scale shading | 15.4 | 20.4 | 19.2 | 20.4 |
  | brightness spread | 56.6 | 92.1 | 85.2 | 87.4 |
  | nut mean RGB | — | 78/94/85 | 79/90/83 | 89/101/96 |
- Fourth round, from annotated user feedback (red circles on real highlights):
  - New measurement: radial brightness profile (mean gray per radius, 40 train/good
    images). Real: dark cone 0.13–0.16 D, bright rim ~0.17, patchy bright zone
    0.29–0.36, tabs ~75. Used to place the geometry instead of guessing.
  - Micro-scratches redefined: the user meant dense, BRIGHT, short lines covering the
    whole surface like static, on a darker green base (not dark lines). Now 6
    directions, ~30 px long, ~2 px apart, almost everywhere, brightness from color.
    Failed try: polished (mirror) lines reflected the dark room and showed as long
    dark scratches.
  - Tab bend removed (real tabs do not look bent).
  - Lighting: the uniform ring gave uniform bright bands; real highlights are local
    and random. Ring is now a weak fill (0.12) plus 3 small lamps at random azimuth
    (35–65° elevation, random power) → patchy, very bright highlights that move from
    image to image.
  - Countersink 0.123–0.173 D at 45° (reflects sideways → dark like real), rounded rim,
    matte dark gray-blue. Bug found: the flat body top covered most of the cone (and
    earlier the bead), only a thin ring was visible; the body hole now has the
    countersink width.
  - Boss edge slope moved to the measured bright zone (0.29–0.36 D).

  | radial profile (mean gray) | cone 0.12–0.14 | rim 0.16–0.18 | flat 0.20–0.26 | bright 0.30–0.32 | tabs 0.38–0.46 |
  |---|---|---|---|---|---|
  | real | 67–72 | 99 → 72 | 70–79 | 104–111 | 73–81 |
  | v2 round 4 | 49–54 | 85–100 | 102–104 | 125 | 74–87 |

  Mean RGB 95/106/99 (real 89/101/96), fine grain 12.7, mid 18.2, spread 120.
- Fifth round (annotated close-ups from the user):
  - Texture is "iris"-like: thin LIGHT lines running radially from the center plus
    light dots, about 50/50, on an almost background-dark base. Ours looked like a
    rough surface (bump made light/dark side shading). Replaced the random-direction
    micro-scratches by radial lines: noise sampled at (direction × K, radius × L), so it
    varies fast around the circle and slowly outward; its 0.5-contours are radial
    streaks, denser near the center. Direction vector instead of angle → no seam.
    Bump grain lowered (0.4 → 0.15); grooves removed.
  - Countersink: two bright lines (lip at the hole edge + rim), color from near
    background at the hole to gray-white outwards (radius-driven color ramp), matte.
  - Tab tips: depth-of-field blur, blended in beyond 0.75 × tip radius (σ 1.5–3.5 px).
  - Result: radial profile shape close to real; mean RGB 76/86/79 (a bit dark), fine
    9.0, mid 20.7, spread 111. Countersink still looks too smooth/plastic.

### 17. synth_v1 render (5,000 images)
- Started `blenderproc run render/generate.py --n 5000 --seed 0 --out data/synth/synth_v1`.
  Results (DB, stats, YOLO conversion) to be added when it finishes.

## 2026-10-17 (first trainings)

### 18. Real-only baseline (5-fold cross-validation)
- Question: how well does YOLOv8n-seg do with only the 10 real seen_pool scratches?
  (roadmap row "real only, 10 images"). locked_test is not used.
- `tools/eval.py`: our own metrics. Per image the predicted masks (conf ≥ 0.25) are
  merged; any mask → image rejected. Recall (scratch images rejected), false alarm
  (good images rejected), precision, mask IoU. Refuses to run on locked_test.
- `train/real_only_cv.py`: 5 folds; each trains on 8 scratch + 40 train/good images
  (COCO-pretrained, 60 epochs, 512 px, Apple MPS) and tests on the 2 held-out scratches
  + 50 other train/good images. Every real scratch is tested once by a model that never
  saw it. Ultralytics installed in the venv (8.3.40, same pin as Colab).
- Bug found in my own setup: YOLO picks `best.pt` by its val score, and val contained
  the fold's test images → model selection on the test set (leakage). best.pt: 2/10;
  last.pt (no selection): 1/10. Script now evaluates last.pt.
- Result (last.pt): recall 1/10 (10%), false alarms 0/250, mean mask IoU 0.06.
  The model learned to say "no scratch" almost always (40 good vs 8 scratch images).
  Folds differ a lot (high variance with 8 training images).
- This is the bar for synthetic data: it has to find more than 1 of 10 real scratches.

### 19. synth_v1 model on real images (first sim-to-real result)
- YOLOv8n-seg trained on synth_v1 in Colab (T4). The session dropped at epoch 28/60
  (laptop lid closed); used `best.pt` from epoch ≤27 (synthetic val mask mAP50 0.95,
  selected on synthetic val only, so no leakage). Saved as `output/models/synth_v1_best.pt`.
- Colab fixes: mlflow==2.17.2 had no wheel for Colab's Python 3.13 → install unpinned
  mlflow, keep ultralytics==8.3.40; new mlflow refuses the folder backend → set
  `MLFLOW_ALLOW_FILE_STORE=true`.
- Dev evaluation (same set as the real-only baseline: 10 seen_pool scratches + 50
  train/good), `reports/results/synth_v1_dev_eval.json`:

  | model | scratches found | false alarms (50 good) | mean mask IoU |
  |---|---|---|---|
  | real only (8 real images, 5-fold CV) | 1/10 | 0/50 | 0.06 |
  | synthetic v1 (4,500 renders) | 10/10 | 49/50 | 0.09 |

  On all 220 train/good images: 215 false alarms (97.7%).
- What happens (`output/synth_v1_on_real.png`): the model marks large textured/bright
  areas (tabs, rims) as scratches, on good and scratched nuts alike; the real scratch
  is mostly missed (IoU ≈ 0), so "10/10 found" is luck. v1 surfaces were clean, so the
  model learned "any texture = scratch". Real nuts are full of micro-scratches and
  iris-like texture → domain gap, exactly the realism issue spotted by eye earlier.
- A threshold cannot fix it: false alarms on good nuts have higher confidence (up to
  0.87) than the real scratches (0.27–0.76).
- Next: train synth_v2 (iris texture, micro-scratches, patchy highlights, visible
  scratches) and compare on the same dev set; expect far fewer false alarms.

### 20. Good-nut realism rewrite (from an annotated 10-point review)
- Stopped the first synth_v2 render at 1,757/5,000 (user review: still clearly unreal).
  Fixed the defect-free nut first; scratches come next.
- `render/generate.py` good-nut part rewritten (old version kept as
  `output/generate_before_goodnut_rewrite.py`); scratch shapes/masks unchanged.
  1. Texture in COLOR, not relief: dark mottled base (noise × Voronoi), light wavy,
     broken radial "iris" streaks with varying opacity, light dots, sparse gold specks.
     Bump nearly zero (0.03). First try had streaks so dense that the dark gaps read
     as dark lines on a light base (inverted); thinner streaks fixed it.
  2. Countersink: same zinc material and texture, darker towards the hole, concentric
     machining marks; lip + rim fillets give two thin bright rings; 25° cone (45° was
     far too dark: 22-36 gray vs real 67-72).
  3. Wide boss slope replaced by a thin shallow groove; the center disk fades into the
     body over ~3° (a short step showed as a ring line).
  4. Fragmented highlights: two broad waviness layers, 4 small random lamps, roughness
     0.32-0.55, narrower lamp power range (no saturated bands, item 7).
  5. Real prong bend: body is voxel-remeshed (dense even mesh), prong tips bent down
     (0.035 D, from r = 0.40); real camera depth of field (f-stop 0.3-0.7, focus on the
     flange). The fake 2D edge blur is gone. Outline/mask unchanged (bend is z only).
  6. Outline corners rounded (2° circular smoothing) + tiny fixed edge irregularity.
  8. Warm zinc: yellow-green base, gold specks; per-image white balance.
  9. Sensor post-processing: slight blur, chromatic aberration (R/B radial scale),
     read + shot noise per channel (color noise). Masks untouched.
- Good-only test (8 renders): mean RGB 90/101/88 (real 89/101/96), fine 9.9 (9.2),
  mid 17.8 (20.4), spread 111 (87); radial profile now follows the real one
  (countersink 61-91 vs 67-72, rim ~100, flange ~100-110 vs 70-79, tabs 81-84 vs 74-81).
- Second review round (annotated): texture = SHORT strokes and dots whose orientation
  is only radial from afar, hard x-ray-like light/dark transitions; 3 subtle
  slide-like concentric steps on the prongs; a gray-white slope down at the short
  radial prong edge; a thin sloped rim along the whole outline.
  - Strokes: noise at (direction × K, radius × L) with K ≈ L (short blobs), both tails
    thresholded → bright and dark strokes; continuous contour lines removed.
  - Geometry by vertex displacement on the dense body mesh: prong steps at
    r = 0.37 / 0.415 / 0.455 D; rim slope via distance to the outline (kd-tree),
    wider (0.035 D) where the outline runs radially (short prong edge).
  - Bugs fixed on the way: distance to the 360 sparse outline points gave a beaded,
    serrated rim (now a densified outline, 8 points per segment); first steps were
    too deep (bright terraces), halved and widened.
  - Result (8 good renders): mean RGB 88/99/86 (real 89/101/96), fine 10.9 (9.2),
    mid 17.6 (20.4), spread 118 (87).
- Third review round (annotated): countersink must NOT carry the flange texture but a
  smooth ring sequence (rim inwards: white → darker → white → short dark → hole);
  light comes from several directions and varies much more; the outline and the short
  prong-edge slopes look bright, whitish and soft in real images (ours were dark).
  - Countersink: own smooth semi-metallic material, radius-driven gray ramp with the
    annotated ring order (first try too bright: a glowing donut; values halved).
  - Rim: `make_body` stores a per-vertex "rim" attribute (1 at the outline, wider at
    the short prong edges); the material reads it and makes the rim whitish and rough,
    independent of the light angle. Plus a low "grazing" ring light (~6° elevation)
    that catches outward slopes.
  - 6 smaller lamps with a wider power range, stronger waviness, larger dark spots.
  - Result (8 good renders): mean RGB 99/112/98 (real 89/101/96, real range of mean gray
    84-122), fine 11.7, mid 17.3, spread 105; countersink 64-121 (real 67-99).
- Fourth review round: real looks a bit bluer (ours too dark-green), the countersink has
  dark blue and more color variation, the short prong-edge slope is much wider and
  softer, and details overall are too crisp.
  - Base color bluer (0.74, 1.0, 0.86); dark strokes/spots now MIX to a bluish dark
    gray instead of a darker green.
  - Countersink ramp in RGB with dark-blue bands + a low-frequency hue variation.
  - Short prong-edge slope 0.035 → 0.07 D wide, softer rim glow falloff.
  - Lens blur σ 0.2-0.8 → 0.7-1.4 px.
  - Result: mean RGB 101/112/105, fine grain 9.0 (real 9.2), mid 17.5, spread 100.
- Fifth review round: countersink too smooth and a perfect circle; its blue much bluer,
  outer white part slightly turquoise; more iris texture, like many PAINTED scratches on
  a smoother glossy surface (no effect on lighting); short prong-edge slope even bigger.
  - Countersink: lathe "wobble" (low-frequency radius irregularity on the hole side,
    fading to 0 where it meets the body), ring positions warped by noise, deeper blue
    and turquoise ramp, stronger fine grain.
  - Strokes: a third, finer layer and a lower threshold (denser); micro-normal 0
    (texture purely in color); roughness 0.26-0.46 (glossier base).
  - Short prong-edge slope 0.11 D wide, softer rim falloff.
  - Result: mean RGB 98/109/104, fine 12.4 (9.2), mid 22.2 (20.4), spread 125 (87).
- Sixth review round: "blue" in the countersink meant the dark navy-black of the
  background (misread as saturated blue); bright outer countersink edge with a softer
  transition into the flange (that zone always shines); highlights too bright and too
  widespread, they should be sparse and mostly on protruding tips, not whole slopes.
  - Countersink ramp: background-dark bands, turquoise-gray light bands, bright outer
    edge; wider rounded rim (0.022 D); the flange brightens near the countersink edge
    and fades out over 0.025 D.
  - Highlights: rim glow 0.75 → 0.45 and squared falloff (only the very edge), grazing
    light 0.35 → 0.15, lamps 4.2 → 3.2 W, roughness 0.3-0.5, ring fill 0.2.
  - Result: mean RGB 93/102/97 (real 89/101/96, closest so far), fine 11.0, mid 18.6,
    spread 105 (87).
- Range tuning from a user rating of 16 random renders (good: 1, 7; bad: 0, 2, 5 "too
  gray, exaggerated highlights, too wavy and scattered texture"). Compared all logged
  parameters of good vs bad (meta.jsonl): biggest differences were waviness (bad
  1.26-1.37 vs good 0.84-0.96, +2.5 sd), lamp elevation (52-57° vs ~47°), denser
  strokes, grayish/bluish white balance, glossier surface. These match the stated
  reasons, so their ranges were narrowed towards the good ones (wave 0.7-1.05, lamp
  elevation 35-52°, stroke threshold 0.6-0.65, WB ±0.015 with slight green bias,
  roughness shift -0.02..+0.06). Differences without a plausible visual effect (camera
  tilt, chromatic aberration, background shift) were treated as chance and not changed
  (16 samples, ~30 parameters: correlation ≠ cause).
- Same seed re-rendered: the silver, blown-out, scattered look is mostly gone.

## 2026-10-18 (scratch rewrite)

### 21. Scratches drawn stroke by stroke
- Review (user + me, same list): real field scratches are separate, distinct, long thin
  strokes (dark brown, some bright), slightly curved, different lengths, crossing; ours
  were a uniform dark patch inside a rectangle. Real regions are irregular, ours were
  sharp rectangles; real bands are opaque flat gray-brown, ours translucent/pinkish;
  real masks follow the scratch shape, ours were rectangles.
- New system (`render/generate.py`, old version in `output/generate_before_stroke_scratches.py`):
  - Each image draws its scratches in NumPy into a 1024² texture in nut-local
    coordinates (R = amount, G = tone): fields = 15-60 anti-aliased strokes around a
    common direction (±6° jitter, slight curvature, 30-100% length, 25% bright);
    bands = one opaque strip with ±18% width noise (rough edges), curved axis, a few
    inner streaks. The shader mixes the surface to a tone-ramp color (dark brown /
    gray-brown / bright) by the amount.
  - Mask from the same strokes: per scratch, threshold → morphological closing (fills
    the gaps between strokes, like the coarse real masks) → small dilation, mapped into
    the image with the plane homography of the projected texture square. Countersink
    pixels (category 3 during rendering, merged into 1) are excluded: no texture there.
  - Bug found: Cycles caches an image texture after the first render and ignores later
    pixel edits (minimal test: 0.8 → 0.3 rendered unchanged). Since image 0 had no
    scratch, every scratch was invisible. Fix: a new Blender image every frame.
- Review: scratches still too dark and look painted ("drawn with a dark gray brush");
  real ones are lighter gray and look deep.
  - Colors moved from dark brown to light matte steel grays.
  - Depth: the scratch texture gets a height channel (B = amount blurred by 1.5 px for
    fields, 3 px for bands) driving an inverted bump after the micro-normal; groove
    walls now catch the light on one side and shade the other (depth randomized
    0.6-1.4). On the good surface the texture stays color-only; only scratches change
    the shading.
- Review: fields should sometimes show light-gray damage haze around and between the
  strokes; bands must look like material removed (the flange rings still showed through,
  "like a black-and-white filter"), lighter, with sharper edges and pointed ends.
  - Fields: per-scratch haze level none/mild/strong; haze = light-gray low-opacity layer
    inside the closed stroke region (noise-broken) + up to 150 faint shallow
    micro-scratches.
  - Bands: texture alpha = "flat" mask; the shader blends the surface normal towards
    straight up there, so rings, steps and waviness vanish inside the band (ground
    flat). Matte interior (roughness 0.8: the flat glossy band mirrored the dark
    surroundings and looked black), lighter gray with fine sanding grain. Width tapers
    to a point at both ends (4-15% of length), jagged ±25% width noise, sharper walls.
- Review: fields very good; bands too flat and single-colored. User measured real band
  pixels: darker parts (often near one edge) sRGB 70-85, lighter parts 85-105.
  - Band tone now varies per pixel (`band_tone`): one edge darker (random side), fine
    streaks along the band (grinding marks), broad blotches; mapped to tone 0.35-0.65.
  - Ramp stops calibrated with `output/band_stats.py` (sRGB inside band masks): before
    dark half 91-96 / light half 118-121; after dark 75-79 / light 97-101 (targets
    70-85 / 85-105).

### Planned: robustness test (after the synth_v2 model is trained)
- Goal: measure where the model breaks outside its training distribution (it saw camera
  tilt 0-3°, zinc-colored nuts, scratches only).
- Plan: render small out-of-distribution sets with `render/generate.py` by widening
  one range at a time (e.g. camera tilt 10° / 20° / 30°; nut color yellow / black-oxide;
  stronger/weaker light), ~100 images each, same seed so scratches are identical, and
  plot recall / false alarms against the shifted parameter.
- Story: the model works inside the factory camera's measured conditions; outside, the
  fix is cheap with synthetic data (change a range, re-render, labels stay automatic).

## 2026-10-19 (synth_v2 dataset ready)

### 22. synth_v2 render + pipeline
- 5,000 images rendered with the frozen v2 code (tag `synth_v2`, commit 7a70ff0), no
  errors. The Mac slept on battery overnight (lid closed): ~17:46-00:35 almost no
  progress; at normal speed ~2.3 s/image. caffeinate does not stop lid-close sleep.
- `tools/build_db.py` made version-tolerant: v1-only columns are NULL for v2, and every
  image's full parameter set is stored in `params_json`. DB: `db/meta_v2.sqlite`
  (5,000 images, 4,600 scratches). Stats: `reports/analysis/data_stats_v2.md` (34.5% defect-free,
  no empty/tiny masks).
- YOLO dataset `data/yolo/synth_v2` (4,500 train / 500 val), round-trip IoU val mean
  0.988 (min 0.903). Zip 2.1 GB for Colab.
- Colab notebook parameterized (`DATASET` at the top), MLflow file-store fix inside,
  resume cell added.

## 2026-10-20 (synth_v2 model on real images)

### 23. synth_v2 vs synth_v1 vs real-only
- YOLOv8n-seg trained on synth_v2 in Colab, full 60 epochs (same settings as v1).
  Synthetic val mask mAP50 0.900 (v1: 0.95; v2 images are harder). Weights
  `output/models/synth_v2_best.pt`, curve `reports/results/synth_v2_training_curve.csv`,
  eval `reports/results/synth_v2_dev_eval.json`.
- Dev set (10 seen_pool scratches + 50 train/good), conf 0.25:

  | model | scratches found | false alarms (50 good) | mean mask IoU on scratches |
  |---|---|---|---|
  | real only (5-fold CV) | 1/10 | 0/50 | 0.06 |
  | synth_v1 | 10/10 (luck, IoU ≈ 0) | 49/50 | 0.09 |
  | synth_v2 | 10/10 | 39/50 | 0.53 |

  The real scratches are now actually found and outlined (IoU 0.22-0.87, confidences
  0.67-0.88); v1 marked random textured areas instead.
- False alarms on all 220 train/good: 162 (74%) at conf 0.25; 42 (19%) at 0.70 with 9/10
  scratches still found; 16 (7%) at 0.80 with 5/10. Unlike v1, a threshold now helps
  (scratches score higher than most false alarms), but not enough.
- Diagnosis: at conf ≥ 0.5, the median false alarm lies 100% in the outer ring
  (r > 0.40 D), 81% of them almost entirely: thin bright arcs along the nut's outer
  edge / short prong edges (`output/synth_v2_on_real.png`). The real rim looks unlike
  our synthetic rim, so the model treats it as a defect. Next realism fix should target
  the rim/edges (data-driven, not by eye), not the flange texture.

### 24. Few-shot fine-tuning (5-fold CV, same folds as the real-only baseline)
- `train/real_only_cv.py` → `train/fewshot_cv.py` (git mv) with `--init`, `--lr0`,
  `--synth/--n-synth` (replay), `--name`. Folds, real train images (8 scratch + 40 good)
  and test images (2 held-out scratches + 50 good) identical for every variant;
  evaluated with last.pt; hyperparameters fixed before running (30 epochs, AdamW,
  lr0 0.001), nothing tuned on the folds.
- Results (`reports/results/cv_*.json`, conf 0.25):

  | variant | init | train data | found | false alarms | mean IoU |
  |---|---|---|---|---|---|
  | real only | COCO | 8 real + 40 good | 1/10 | 0/250 | 0.06 |
  | synthetic only | synth_v2 model, no fine-tune | 4,500 synthetic | 10/10 | 39/50 | 0.53 |
  | A: fine-tune | synth_v2 | 8 real + 40 good | 9/10 | 0/250 | 0.58 |
  | B: fine-tune + replay | synth_v2 | 8 real + 40 good + 300 synth_v2 | 10/10 | 0/250 | 0.68 |

  (false alarms: the same 50 good images, once per fold model → 250 checks.)
- Main finding: the same 8 real images give 1/10 from COCO but 9-10/10 with no false
  alarms when the model is first trained on synthetic data. Fine-tuning on real good
  nuts removed the rim false alarms of the synthetic-only model. Replay (mixing 300
  synthetic images) avoids forgetting: all 10 found and better masks (IoU 0.58 → 0.68).
- Caveat: 10 scratches and 50 distinct good images; 1 scratch = 10% recall. The
  locked test (13 scratch + 22 good) gives the final, independent number.

### 25. Cost-based threshold (chosen before the locked test)
- `tools/cost_threshold.py`: out-of-fold scores of the 5 CV models of variant B
  (v2 + fine-tune + replay): each model scores its 2 held-out real scratches and the
  180 train/good images never used for training (900 good checks). Score = highest
  scratch confidence in the image. Expected cost per part =
  p·C_miss·miss_rate + (1−p)·C_fa·false_alarm_rate, with stated assumptions: defect rate
  p = 5%, C_miss = 5x / 20x / 50x C_fa. Rule fixed in advance: middle of the cheapest
  threshold range.
- Out-of-fold scratch scores 0.64-0.90; good scores max 0.53 (99th percentile 0.10):
  perfectly separated on the dev data, zero cost for t in 0.54-0.64 in all scenarios.
  Chosen t = 0.59 (`reports/results/threshold_v2_ft_replay.json/.png`).
- Caveat: with only 10 scratches the cost ratio cannot change the choice here (no errors
  to trade off); the margin to the weakest scratch (0.64) is small.

### 26. Final model
- `train/fewshot_cv.py --final`: same recipe as variant B (synth_v2 init, 30 epochs,
  AdamW lr0 0.001, 300 synthetic replay) but trained once on ALL 10 real scratches +
  40 good; saved as `output/models/final_v2_ft_replay.pt`. Next: one evaluation on
  the locked test at t = 0.59.

### 27. FINAL: locked test (one-time evaluation)
- `tools/final_eval.py` (needs `--i-confirm-final-evaluation`), model
  `output/models/final_v2_ft_replay.pt`, threshold 0.59, both fixed beforehand.
  First and only look at `splits/locked_test.txt` (13 scratch + 22 good).
- Result (`reports/results/final_locked_test.json`, `reports/results/final_locked_test.png`):
  **12/13 scratches found (recall 92.3%), 0/22 false alarms (precision 100%),
  mean mask IoU 0.63.**
- Error analysis (after the fact, no changes made): the missed scratch_001 scores 0.565,
  just below the threshold (inside the 0.54-0.64 zero-cost range on dev). Some multi-part
  scratches are only partly outlined (e.g. scratch_007: 1 of 3 regions).
- Reported as is. Threshold and model stay unchanged; the locked test is now "seen".

### 28. Serving: ONNX + FastAPI + Docker
- `tools/export_onnx.py`: final model → ONNX (opset 17, fixed 1×3×512×512, 13 MB),
  copied to `serve/model.onnx`. torch 2.14's new dynamo exporter needed extra packages
  (onnxscript, onnx_ir); routed Ultralytics' call to the classic exporter (`dynamo=False`).
- `serve/inference.py`: inference with ONNX Runtime + NumPy only (letterbox, confidence
  filter, NMS, mask = sigmoid(coefs @ prototypes) cropped to the box, back to original
  size). No PyTorch/Ultralytics in the container.
- `tools/check_onnx_parity.py` vs the PyTorch model on 10 seen_pool + 30 good images:
  same decision 40/40, best-score difference ≤ 0.0005, mask IoU mean 0.97 (min 0.95),
  ~42 ms per image on the Mac CPU.
- `serve/app.py` (FastAPI): `GET /health`, `POST /predict` (accept/reject, best score,
  regions with boxes and areas, scratch area %), `POST /predict/overlay` (PNG with red
  outlines). Threshold 0.59 via `SCRATCH_THRESHOLD`. `serve/test_app.py`: 5 pytest tests
  pass (scratch rejected, good accepted, PNG overlay, bad upload → 400).
- `serve/Dockerfile` (python:3.10-slim, pinned `serve/requirements.txt`): image 583 MB;
  container tested with curl: seen_pool 021 → reject (score 0.91), good 000 → accept.

## 2026-10-21 (GitHub showcase)

### 29. Repo cleanup, figures, README
- `tools/make_figures.py` → `reports/figures/01-07` (real vs synthetic, free labels,
  scratch types, realism v1→v2, domain gap per model, results bars, locked test examples).
- `reports/` split into `figures/`, `analysis/` (measurements, data stats) and `results/`
  (CV, dev evals, threshold, locked test); code paths updated. Removed smoke-test JSONs,
  `.gitkeep`s, `train/synth_v1.yaml`, tracked SQLite DB (now gitignored with `*.pt`).
- `output/` cleaned from 1.9 GB to 263 MB: kept models, logs and CV fold weights (needed by
  `tools/cost_threshold.py`); deleted tuning renders, comparison images and temp scripts.
  Deleted the Colab zips (3.6 GB, regenerable).
- `README.md` (results, pipeline diagram, method, lessons, reproduce commands, license) and
  a pinned root `requirements.txt`.
- MIT `LICENSE` added; `learn/` removed from the repo (kept locally, gitignored).
