# Real image observations (MVTec metal_nut)

Source: train/good images and the 10 seen_pool scratch images only. No locked_test
image was opened. Tools:
- `tools/make_contact_sheet.py`: 20 random train/good (seed 0) + 10 seen_pool scratch
  with mask contours, plus scratch size stats.
- `tools/light_direction.py`: ring-brightness vs angle on the first 20 train/good images
  (`reports/light_direction.png`).

Labels: **[measured]** = computed from pixels, **[visual]** = eyeballed on contact sheets.

## Summary for rendering

| Parameter | Real data | Render setting |
|---|---|---|
| Camera | Straight top-down, fixed distance, nut centered | Fixed distance, tilt ≤ ±5°, no x/y jitter |
| Nut size | Tab-tip diameter 623 ± 2.5 px of 700 (89%) | Match framing to ~0.89 of image width |
| Rotation | Random in-plane | Random z-rotation 0–360° |
| Background | Flat #17171d | Flat dark plane, small jitter around #17171d |
| Lighting | Soft and even; weak directional component from ~10 o'clock, fixed in image | Large soft area light + weak key light upper-left; randomize power |
| Exposure | Overall brightness varies a lot between images | Randomize light power / exposure |
| Material | Metallic, slightly yellow-green silver, grainy texture | metallic=1, roughness ~0.3–0.5, tinted base color, noise bump |
| Scratches | Region-level masks, straight, 2–12% of image area | Straight strips/polygons as texture changes, generous masks |

## Camera
- Viewing angle **[visual]**: straight top-down (near-orthographic). Side walls are never
  visible; only the stamped tab edges show a thin layered rim.
- Centering **[measured]**: hole center is within 2.5 px of the image center in all
  20 images.
- Scale **[measured]**: hole diameter 153.5 ± 1.5 px, tab-tip diameter 623.5 ± 2.5 px.
  The tiny spread confirms a fixed camera distance.
- Rotation **[visual]**: random in-plane rotation (0–360°). No visible tilt.
- Focus **[visual]**: center ring is sharp; outer tab edges look slightly soft. Likely
  shallow depth of field (tabs sit at a different height), not motion blur.
- Resolution **[measured]**: 700 × 700.
- → Render: fixed camera distance, looking straight down, tilt ≤ ±5°. Randomize only
  the part's z-rotation. Optional: mild DoF.

## Background
- **[measured]** Mean corner color RGB (23, 23, 30) = #17171d: near-black with a slight
  blue tint. **[visual]** Almost no noise or texture, no visible cast shadow.
- → Render: flat dark plane, small brightness/hue jitter around #17171d.

## Lighting
- **[visual]** Mostly soft and even, no hard cast shadows → large diffuse source (or
  ring light) plus ambient.
- **[measured]** Directional component: on the flat boss ring (r = 110–170 px), mean
  brightness peaks at ~155° (≈ 10 o'clock, upper-left). All 20 per-image peaks fall in
  the left half (between ~6 and 11 o'clock) even though the nut rotation is random.
  → The light is fixed relative to the camera, not to the part.
- **[measured]** The effect is weak: brightness range across angles is 16% of the mean.
  Single-image profiles are noisy because of the grainy surface texture.
- **[visual]** Specular highlights on the tab edges: first impression was ~5 o'clock;
  on re-check they appear at varying positions, more often in the lower half. Tab edges
  are sloped, so their highlights depend on edge orientation and do not directly give
  the light direction. The ring measurement is the more reliable estimate.
- **[measured]** Overall brightness differs strongly between images (mean ring
  brightness ranges ~50–115 on 0–255).
- **[visual]** Moderate exposure; edges are bright but not blown out.
- → Render: 1 large soft area light / world ambient + 1 weaker key light at upper-left
  (~10 o'clock), low elevation. Randomize light power so overall brightness varies.

## Nut shape
- **[visual]** 4 equal hook-shaped tabs, 90° apart. Tabs are asymmetric (pinwheel-like)
  and all hook in the same rotational direction → handedness (chirality) matters.
  A mirrored nut is a separate MVTec defect class ("flip"), so the synthetic nut must
  have the correct handedness.
- **[visual]** Center: raised circular ring (boss) with a step at its outer edge; tabs
  start from it.
- **[visual]** Hole: perfectly round, black inside (through-hole), surrounded by a bright
  beveled chamfer ring.
- Proportions: hole / tab-tip diameter = 0.246 **[measured]**; boss ring ≈ 0.6–0.65 ×
  outer diameter **[visual]**.
- **[visual]** Surface: metallic with a grainy/brushed micro-texture, faint concentric
  marks on the ring, layered lines on stamped tab edges. Color: slightly greenish-yellow
  silver, not neutral grey.
- → Render: metallic=1, roughness ~0.3–0.5, base color with slight yellow-green tint,
  noise-based roughness + bump texture.

## Scratches (10 seen_pool images)
Two visual types **[visual]**:
1. Scratch fields (000, 002, 004, 006, 010, 020, 021): clusters of many fine, parallel,
   straight hairlines. The mask covers the whole region, not individual lines.
2. Gouge bands (011, 014, 018): one long, straight, wide strip with uniform matte grey
   and fairly straight edges, like a ground/sanded flat.

Size:
- **[measured]** Mask area is 1.9–11.9% of the image (9k–58k px of 490k px).
  Median ~4%.
- **[measured, rough]** Rotated bounding-box length 150–613 px (0.24–0.98 × D, with
  D = 623 px). For masks with several parts (e.g. 018) the box spans all parts, so
  length and width are overestimated there.
- **[visual]** Bands ≈ 0.6–1.0 D long, ≈ 0.06–0.1 D wide; fields ≈ 0.2–0.8 D long,
  0.1–0.4 D wide; small cases (000, 020) ≈ 0.15–0.3 D.

Other properties **[visual]**:
- Shape: straight, not curved. Random orientation.
- Count: usually 1 region per image, sometimes 2–3 components (004, 006, 011).
- Location: anywhere. They cross ring and tabs freely, can run over the hole
  (018: mask split into 2 parts) and off the nut edge (011: extra region at tab edge).
- Appearance: darker and more matte (less specular) than the surface; hairlines are
  slightly darker than the base metal. Some fields are hard to see without the mask
  overlay → low-contrast defects are part of the task.
- Annotation style: masks are coarse, generous outlines around the whole damaged area.
- → Render: implement as texture changes (roughness ↑, albedo slightly darker, line
  noise for hatched fields) inside a randomly placed straight strip/polygon, not as a
  thin curved groove. Synthetic masks must follow the same region-level convention.

## Open questions
- Light elevation: not measured. Could be estimated later by comparing renders at
  different key-light heights against `reports/light_direction.png`.
- Seen_pool has only 10 scratch images, so the size ranges above are rough. Randomize
  slightly beyond them rather than matching them exactly.
