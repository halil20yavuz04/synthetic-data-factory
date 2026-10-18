import blenderproc as bproc  # must be the very first line (BlenderProc checks this)
# Render N randomized metal nuts with synthetic scratches, top-down,
# RGB + segmentation mask (0 = background, 1 = nut, 2 = scratch).
#
# Base scene values come from reports/real_observations.md and side-by-side checks
# against real train/good images (PROGRESS.md); random ranges are listed in
# sample_params(). Every image's parameters go to meta.jsonl.
#
# Run from the repo root:
#     blenderproc run render/generate.py --n 200 --seed 0
#     blenderproc run render/generate.py --n 8 --no-defects --out output/v2_tune --version dev
#     blenderproc run render/generate.py --n 2000 --rand light --out data/synth/abl_A1 --version abl_A1
# Output: <out>/N.hdf5 + <out>/meta.jsonl  (convert with tools/hdf5_to_png.py)
import argparse
import json
import math
import time
from pathlib import Path

import bpy
import mathutils.kdtree
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="number of images")
parser.add_argument("--seed", type=int, default=0, help="master seed")
parser.add_argument("--out", default="data/synth/week1", help="output folder")
parser.add_argument("--no-random", action="store_true",
                    help="render base values only (no randomization, no scratches)")
parser.add_argument("--rand", default="light,material,camera,background",
                    help="randomization groups to switch on (ablation); '' = none. "
                         "Scratches stay random unless --no-random.")
parser.add_argument("--version", default="synth_v2", help="dataset_version label in meta")
parser.add_argument("--no-defects", action="store_true",
                    help="no scratches (for tuning realism against real good images)")
args = parser.parse_args()

IMG_SIZE = 512
SAMPLES = 64  # low + denoiser

# --- nut geometry (units: tab-tip diameter D = 1.0) ---------------------------
# Outline = measured polar profile of one tab period (tools/nut_profile.py),
# repeated 4 times. This also fixes the tab handedness to match the real nut.
PROFILE = json.load(open("render/nut_profile.json"))["r_over_D"]
N_TABS = 4
R_TIP = max(PROFILE)    # = 0.5
R_HOLE = 0.246 / 2      # hole / D = 0.246 [measured]
BODY_T = 0.06           # sheet thickness
BODY_BEVEL = 0.02       # rounded outer edges: the sheet thickness shows as a thin bright rim
OUTLINE_SMOOTH_DEG = 2.0   # rounds the sharp notch corners of the measured outline
EDGE_JITTER = 0.002     # slightly irregular stamped edge (fixed pattern, in D)
REMESH_VOXEL = 0.003    # dense, even mesh: bends, steps and rim slopes need small faces
# Concentric slide-like steps on the prongs, from the annotated real image (3 = biggest,
# at the flange edge; 2 and 1 smaller, further out). (radius, drop, width) in D.
PRONG_STEPS = [(0.37, 0.004, 0.02), (0.415, 0.0018, 0.016), (0.455, 0.0012, 0.014)]
# (first try 0.006/0.003/0.002 over 0.012/0.008: sharp bright terraces; real ones are subtle)
# Rim slope: the top surface slides down near the outline (thin sloped outline), much
# wider at the short radial prong edge (a gray-white slope going down in real images).
RIM_W, RIM_W_EDGE = 0.012, 0.11    # short prong edge slope: wide (0.035, 0.07 too narrow)
RIM_DROP = 0.01
# Prongs bend DOWN towards their tips (real tee nut). Real geometry now, not a shader
# trick: the bent tips leave the focal plane and blur through real depth of field.
R_BEND = 0.40           # bending starts here
BEND = 0.035            # drop at the tab tip (D); tip slope ~35°
# Center, built by rotating an (r, z) cross-section ("lathe"):
CSINK_W = 0.05          # countersink from the hole (0.123) out to 0.173 D
CSINK_ANGLE = 25        # from horizontal (45 reflected sideways: cone 22-36 gray vs real 67-72)
CSINK_RIM = 0.022       # rounded rim (top edge): wider -> softer transition to the flange
CSINK_LIP = 0.008       # rounded lip at the hole edge -> second thin bright ring
FLANGE_EPS = 0.0015     # center disk sits this much above the body top (no coincident faces)
R_FLAT = 0.355          # center disk radius (body takes over beyond)
R_GROOVE = 0.31         # thin shallow circular groove (real: a line, not a wide slope)
GROOVE_W = 0.012
GROOVE_DEPTH = 0.004

# --- appearance (base values; randomized per image in sample_params) ----------
# The real texture is a COLOR pattern (no relief): dark mottled base, light wavy radial
# "iris" streaks and light dots, sparse golden specks, warm yellow-green zinc.
NUT_COLOR = [0.74, 1.0, 0.86]      # zinc, a bit bluer (user: real looks bluer, ours too dark-green)
BASE_MID = 0.5                     # base between strokes = BASE_MID x color
DARK_COLOR = (0.07, 0.085, 0.12)   # dark strokes/spots: bluish dark gray, not dark green
DARK_EXTRA = 0.04                  # dark strokes need noise < 1 - t - this (fewer than bright)
GOLD = (1.0, 0.72, 0.28)           # sparse golden specks
GOLD_SCALE = 380
GOLD_AMOUNT = 0.74                 # higher = fewer golden specks
MOTTLE_SCALE = 10                  # irregular dark/light patches (noise) ...
CELL_SCALE = 22                    # ... plus cellular patches (Voronoi)
MOTTLE_LOW = 0.5                   # darkest mottle (randomized; lower = harder transitions)
# "Iris" = SHORT strokes and dots whose orientation is radial when seen from afar (no
# continuous center-to-edge lines). Noise sampled at (direction x K, radius x L) with
# L > 0 gives blobs elongated radially; thresholding both tails gives bright AND dark
# strokes -> hard, x-ray-like light/dark transitions. (K around, L along) per layer.
STROKE_LAYERS = [(110, 90), (70, 70), (40, 40)]   # three sizes: many paint-like strokes
IRIS_WAVE = 4.0                    # irregular, slightly bent orientation
STROKE_THRESHOLD = 0.6             # bright strokes where noise > t, dark where < 1 - t
DOT_SCALE = 300                    # light dots between the streaks
DOT_AMOUNT = 0.68                  # higher = fewer dots
HUE_SCALE = 4                      # broad hue drift of the zinc coating
HUE_SHIFT = [(0.85, 0.97, 1.0), (1.0, 1.0, 1.0), (1.0, 0.97, 0.82)]  # bluish / none / yellow
ROUGHNESS_RANGE = (0.3, 0.5)       # glossy base, but highlights must stay sparse
MICRO_NORMAL = 0.0                 # no relief at all: texture must not change the shading
WAVE_SCALES = (3, 8)               # low-frequency stamping waviness -> fragmented highlights
WAVE_STRENGTH = 1.1                # stronger: light varies more across the surface
# Countersink: smooth (NOT the flange texture). From the rim inwards (annotated real
# crop): white ring -> darker -> white ring -> short dark -> hole. Ramp over t = 0 (hole
# edge) .. 1 (rim), gray values with a slight cyan tint.
# "blue" in the review = the dark navy-black of the background, not a saturated blue
CSINK_RAMP = [(0.0, (0.008, 0.009, 0.013)), (0.1, (0.012, 0.013, 0.02)),  # background-dark
              (0.2, (0.3, 0.36, 0.35)), (0.3, (0.015, 0.017, 0.026)),       # light ring, dark
              (0.6, (0.06, 0.07, 0.085)), (0.85, (0.32, 0.4, 0.38)),         # -> turquoise-gray
              (1.0, (0.55, 0.64, 0.6))]                                     # bright outer edge
CSINK_HUE = [(0.3, (0.92, 0.97, 1.05)), (0.7, (1.03, 1.04, 0.96))]  # mild color variation
CSINK_WOBBLE = 0.006    # countersink is not a perfect circle (inner edge, D)
CSINK_RING_WARP = 0.12  # rings wobble: ring position shifted by noise (fraction of width)
# Rim: real outline and short prong edges look bright, whitish and soft (blurry). The body
# mesh stores "rim" (1 at the outline -> 0 inwards); the material brightens and roughens it.
RIM_COLOR = (0.8, 0.86, 0.8)
RIM_AMOUNT = 0.45                  # rim glow (0.75: whole slopes glowed, unrealistic)
CSINK_GLOW = 0.5                   # flange brightens near the countersink edge (soft transition)
CSINK_GLOW_W = 0.025               # ... fading out over this distance (D)
DARK_SPOT_SCALE = 120              # larger dark spots (x-ray look)
DARK_SPOT_AMOUNT = 0.25            # dark where noise < this
BG_COLOR_SRGB = [23, 23, 29]       # #17171d [measured]
WORLD_STRENGTH = 0.02              # faint ambient fill from all directions
RING_HEIGHT = 2.0                  # soft ring fill light above the nut
RING_RADII = (1.0, 2.5)
RING_STRENGTH = 0.2                # soft fill (lamps were dimmed)
N_LAMPS = 6                        # several small lamps at random positions -> local highlights
LAMP_SIZE = 0.2
LAMP_DIST = 2.2
LAMP_ENERGY = 3.2                  # W, randomized per lamp (4.2: too many saturated areas)
GRAZE_HEIGHT, GRAZE_RADII = 0.25, (2.0, 2.6)   # low ring light (~6° elevation): lights the
GRAZE_STRENGTH = 0.15                          # outward rim slopes (0.35 lit whole slopes)
KEY_ANGLE = 150                    # deg, ~10 o'clock in the image [measured]
KEY_ELEVATION = 30
KEY_ENERGY = 12.0

# --- scratches (reports/real_observations.md, 10 seen_pool images) -----------
# Each scratch = a rotated rectangle on the nut surface. Inside it the surface color is
# replaced by scratch color.
# v2 rewrite: scratches are drawn stroke by stroke into a texture in nut-local
# coordinates (no more rectangles). The same texture gives the mask (category 2).
MAX_SCRATCHES = 3
P_NO_DEFECT = 0.35                 # roadmap: 30-40% defect-free images
P_N_SCRATCHES = [0.7, 0.2, 0.1]    # 1, 2 or 3 scratches; real: usually 1, sometimes 2-3
P_FIELD = 0.7                      # real: 7 of 10 are hairline fields, 3 gouge bands
FIELD_LENGTH, FIELD_WIDTH = (0.15, 0.8), (0.08, 0.35)   # in D (tab-tip diameter)
BAND_LENGTH, BAND_WIDTH = (0.5, 1.0), (0.05, 0.1)
SCRATCH_CENTER_R = (0.15, 0.45)    # center lies on the nut, anywhere from hole to tabs
SCR_TEX = 1024                     # scratch texture size, covers the nut square [-0.5, 0.5] D
FIELD_STROKES = (15, 60)           # strokes per field (real: dozens of separate hairlines)
FIELD_STROKE_PX = (2.5, 6.0)       # stroke width in texture px (~1-2.5 image px)
FIELD_JITTER_DEG = 6               # each stroke's angle varies around the field direction
P_BRIGHT_STROKE = 0.25             # some real hairlines are bright, most dark brown
BAND_EDGE_NOISE = 0.25             # band width varies ±25% along its length (jagged edges)
BAND_TAPER = (0.04, 0.15)          # band ends taper to a point over this fraction of its length
HAZE_LEVELS = [0.0, 0.5, 1.0]      # field damage haze: none / mild / strong (one per scratch)
HAZE_AMOUNT = 0.3                  # max haze opacity (light gray, shallow)
HAZE_STROKES = 150                 # faint shallow micro-scratches at full haze
# texture tone (G channel) -> scratch color: 0 dark brown, 0.5 band gray-brown, 1 bright
# light matte steel grays (dark-brown version looked like paint, user review)
# tone 0 = dark field stroke, 0.35-0.65 = band (dark edge .. light), 0.8 haze, 1 bright stroke.
# Band values calibrated to the user's real band pixels: dark parts sRGB 70-85, light 85-105
# (flat 0.55 rendered 91-119 and looked like a single flat color).
SCR_COLORS = [(0.0, (0.13, 0.13, 0.125)), (0.35, (0.29, 0.295, 0.285)),
              (0.65, (0.45, 0.46, 0.445)), (1.0, (0.62, 0.64, 0.6))]
BAND_ROUGHNESS = 0.8               # ground-off metal is matte: a flat glossy band mirrored the
                                   # dark surroundings and looked almost black
# depth: scratches are grooves, not paint. Height = blurred amount (B channel), used as a
# bump: groove walls catch the light on one side and shade the other.
SCR_HEIGHT_BLUR = {"field": 1.5, "band": 1.5}   # texture px: groove walls (band 3: too round)
SCR_DEPTH = 1.0                    # bump strength (randomized)
MASK_CLOSE_PX = {"field": 14, "band": 4}   # mask = strokes with gaps closed (real masks
MASK_DILATE_PX = 3                         # cover the whole region, a bit generous)

# --- camera and sensor --------------------------------------------------------
CAM_HEIGHT = 10.0       # far away -> almost orthographic
FILL = 0.882            # nut width / image width (Week 1 check)
SENSOR_MM = 36.0        # Blender default sensor width, for the depth-of-field math


# Randomization groups for the ablation study (--rand). Scratches are always random.
RAND_GROUPS = {
    "light": ["ring_strength", "graze_strength", "key_angle_deg", "key_elevation_deg", "key_energy",
              "lamp_azimuth_deg", "lamp_elevation_deg", "lamp_energy"],
    "material": ["roughness_min", "roughness_max", "tint", "texture_offset", "mottle_low",
                 "stroke_threshold", "wave_strength", "hue_amount", "gold_amount"],
    "camera": ["rotation_deg", "cam_tilt_deg", "cam_tilt_dir_deg", "cam_dist_scale",
               "fstop", "blur_sigma", "wb_gain", "chroma_ab", "read_noise", "shot_noise"],
    "background": ["bg_shift"],
}


def sample_params(rng, groups, defects=True):
    """All per-image scene parameters. Ranges: real data spread where measured,
    otherwise jitter around the base value. groups: randomization groups that are on;
    the others keep their base values. All random values are always drawn (same order),
    disabled groups are overwritten afterwards: image i gets the same scratches in every
    ablation run, only the switched-off group differs (a controlled experiment)."""
    base = {
        "rotation_deg": 0.0, "cam_tilt_deg": 0.0, "cam_tilt_dir_deg": 0.0,
        "cam_dist_scale": 1.0,
        "fstop": 0.45,                 # depth of field: only the bent tab tips soften
        "blur_sigma": 1.0,             # lens softness (px): real details are not this crisp
        "wb_gain": [1.0, 1.0, 1.0],    # white balance per channel
        "chroma_ab": 0.001,            # chromatic aberration: R/B radial scale difference
        "read_noise": 1.5,             # sensor noise, gray levels (std)
        "shot_noise": 0.15,            # brightness-dependent noise (x sqrt(value))
        "ring_strength": RING_STRENGTH,
        "key_angle_deg": float(KEY_ANGLE), "key_elevation_deg": float(KEY_ELEVATION),
        "key_energy": KEY_ENERGY,
        "lamp_azimuth_deg": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        "graze_strength": GRAZE_STRENGTH,
        "lamp_elevation_deg": [50.0] * N_LAMPS,
        "lamp_energy": [LAMP_ENERGY] * N_LAMPS,
        "roughness_min": ROUGHNESS_RANGE[0], "roughness_max": ROUGHNESS_RANGE[1],
        "tint": [1.0, 1.0, 1.0],
        "bg_shift": 0,
        "texture_offset": [0.0, 0.0, 0.0],
        "mottle_low": MOTTLE_LOW,
        "stroke_threshold": STROKE_THRESHOLD,
        "wave_strength": WAVE_STRENGTH,
        "hue_amount": 0.6,
        "gold_amount": GOLD_AMOUNT,
        "scratch_tone": 1.0,           # brightness multiplier of the scratch colors
        "scratch_depth": SCR_DEPTH,
        "scratch_roughness": 0.45,     # inside scratches (close to the surface)
        "defects": [],
    }
    if not groups and not defects:  # --no-random: pure base scene
        return base
    p = dict(base)
    shift = rng.uniform(-0.02, 0.06)                        # was ±0.06 (glossy = blown highlights)
    p.update({
        "rotation_deg": rng.uniform(0, 360),           # real: random in-plane rotation
        "cam_tilt_deg": rng.uniform(0, 3),             # real: no visible tilt
        "cam_tilt_dir_deg": rng.uniform(0, 360),
        "cam_dist_scale": rng.uniform(0.995, 1.005),   # real: scale spread ±0.4%
        "fstop": rng.uniform(0.3, 0.7),   # 0.12-0.3 blurred whole tabs
        "blur_sigma": rng.uniform(0.7, 1.4),
        "wb_gain": (rng.normal(1.0, 0.015, 3) + [0, 0.015, 0]).round(4).tolist(),  # was ±0.025, no green bias
        "chroma_ab": rng.uniform(0.0, 0.002),
        "read_noise": rng.uniform(0.8, 2.5),
        "shot_noise": rng.uniform(0.05, 0.25),
        # real nut mean gray 98.5 ± 10 (84-122) over 220 train/good images
        "ring_strength": RING_STRENGTH * rng.uniform(0.75, 1.3),
        "graze_strength": GRAZE_STRENGTH * rng.uniform(0.5, 1.6),
        "key_angle_deg": KEY_ANGLE + rng.uniform(-30, 30),
        "key_elevation_deg": rng.uniform(20, 45),
        "key_energy": rng.uniform(0, 24),
        "lamp_azimuth_deg": rng.uniform(0, 360, N_LAMPS).round(1).tolist(),
        "lamp_elevation_deg": rng.uniform(35, 52, N_LAMPS).round(1).tolist(),   # was 35-65
        "lamp_energy": (LAMP_ENERGY * rng.uniform(0.3, 1.6, N_LAMPS)).round(2).tolist(),
        "roughness_min": ROUGHNESS_RANGE[0] + shift,
        "roughness_max": ROUGHNESS_RANGE[1] + shift,
        "tint": rng.normal(1.0, 0.03, 3).round(4).tolist(),
        "bg_shift": int(rng.integers(-3, 4)),
        # move the noise patterns, otherwise every nut has the exact same texture
        "texture_offset": rng.uniform(-100, 100, 3).round(3).tolist(),
        "mottle_low": rng.uniform(0.35, 0.65),
        "stroke_threshold": rng.uniform(0.6, 0.65),        # was 0.57-0.63
        # ranges below narrowed after a user rating of 16 renders (good 1, 7 / bad 0, 2, 5):
        # bad ones had strong waviness, high lamps, dense strokes, grayish WB, glossy surface
        "wave_strength": rng.uniform(0.7, 1.05),           # was 0.8-1.4
        "hue_amount": rng.uniform(0.3, 1.0),
        "gold_amount": rng.uniform(0.7, 0.8),
        # flat scratch color value (MIX); calibrated: real scratches mostly darker (-12)
        "scratch_tone": rng.uniform(0.8, 1.2),
        "scratch_depth": rng.uniform(0.6, 1.4),
        # close to the surface roughness: a matte scratch scatters the lamps and turns
        # LIGHTER, cancelling its darker color
        "scratch_roughness": rng.uniform(0.35, 0.55),
        "defects": sample_scratches(rng),
    })
    for group, keys in RAND_GROUPS.items():
        if group not in groups:
            p.update({k: base[k] for k in keys})
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}


def sample_scratches(rng):
    """0-3 scratches in nut-local coordinates (units of D): region center, direction,
    length, width; strokes are drawn from stroke_seed (see draw_scratch)."""
    if rng.random() < P_NO_DEFECT:
        return []
    scratches = []
    for _ in range(rng.choice([1, 2, 3], p=P_N_SCRATCHES)):
        kind = "field" if rng.random() < P_FIELD else "band"
        length = rng.uniform(*(FIELD_LENGTH if kind == "field" else BAND_LENGTH))
        width = rng.uniform(*(FIELD_WIDTH if kind == "field" else BAND_WIDTH))
        width = min(width, 0.7 * length)  # keep it an elongated strip, not a square
        r = rng.uniform(*SCRATCH_CENTER_R)
        a = rng.uniform(0, 2 * math.pi)
        scratches.append({
            "type": kind,
            "cx": round(r * math.cos(a), 4), "cy": round(r * math.sin(a), 4),
            "angle_deg": round(rng.uniform(0, 180), 2),
            "length": round(length, 4), "width": round(width, 4),
            "n_strokes": int(rng.integers(*FIELD_STROKES)) if kind == "field" else 0,
            "bend": round(float(rng.uniform(-0.08, 0.08)), 4),     # centerline curvature
            "haze": float(rng.choice(HAZE_LEVELS)) if kind == "field" else 0.0,
            "stroke_seed": int(rng.integers(0, 2**31 - 1)),
        })
    return scratches


def srgb_to_linear(c):
    """Blender colors are linear; convert an 8-bit sRGB value."""
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# --- geometry -----------------------------------------------------------------
def outline_points():
    """2D outline: measured profile x4, corners rounded by a small circular smoothing,
    plus a tiny fixed edge irregularity (stamped edge, not a perfect cut)."""
    radii = np.array(PROFILE * N_TABS)
    n = len(radii)
    step_deg = 360 / n
    sigma = OUTLINE_SMOOTH_DEG / step_deg
    k = np.exp(-0.5 * (np.arange(-3 * int(sigma + 1), 3 * int(sigma + 1) + 1) / sigma) ** 2)
    k /= k.sum()
    pad = len(k) // 2
    radii = np.convolve(np.r_[radii[-pad:], radii, radii[:pad]], k, mode="valid")
    jitter = np.random.default_rng(1234).normal(0, 1, n)
    jitter = np.convolve(np.r_[jitter[-5:], jitter, jitter[:5]], np.ones(11) / 11, mode="valid")
    radii = radii + EDGE_JITTER * jitter / (jitter.std() + 1e-9)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)], radii


def smooth01(t):
    """0 -> 1 smoothly (cosine), clamped."""
    t = min(max(t, 0.0), 1.0)
    return (1 - math.cos(math.pi * t)) / 2


def circle_points(r, n=96):
    return [(r * math.cos(a), r * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def evaluated_mesh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))


def make_body(name):
    """Flange with prongs: fill the outline, give it thickness with rounded edges
    (curve bevel), remesh into an even dense mesh, then bend the prong tips down.
    The bend changes only z, so seen from the top camera the outline (and the mask)
    stays the measured one."""
    (outer, radii), inner = outline_points(), circle_points(R_HOLE + CSINK_W)
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = max(BODY_T / 2 - BODY_BEVEL, 0)
    curve.bevel_depth = BODY_BEVEL
    curve.bevel_resolution = 4
    curve.offset = -0.7 * BODY_BEVEL  # 0.7: measured, keeps nut width at 89%
    for outline in (outer, inner):
        spline = curve.splines.new("POLY")
        spline.points.add(len(outline) - 1)
        for p, (x, y) in zip(spline.points, outline):
            p.co = (x, y, 0, 1)
        spline.use_cyclic_u = True
    curve_obj = bpy.data.objects.new(name + "_curve", curve)
    bpy.context.scene.collection.objects.link(curve_obj)
    mesh_obj = bpy.data.objects.new(name + "_tmp", evaluated_mesh(curve_obj))
    bpy.context.scene.collection.objects.link(mesh_obj)
    bpy.data.objects.remove(curve_obj)
    remesh = mesh_obj.modifiers.new("remesh", "REMESH")  # dense, even triangles/quads
    remesh.mode = "VOXEL"
    remesh.voxel_size = REMESH_VOXEL
    mesh = evaluated_mesh(mesh_obj)
    bpy.data.objects.remove(mesh_obj)

    # rim slope width per outline point: wide where the outline runs radially (the short
    # prong edge: |dr| / (r dθ) large), narrow elsewhere
    n = len(radii)
    slope = np.abs(np.roll(radii, -1) - np.roll(radii, 1)) / (radii * 2 * (2 * math.pi / n))
    widths = RIM_W + (RIM_W_EDGE - RIM_W) * np.clip((slope - 1) / 3, 0, 1)
    # distances are taken to a densified outline (8 points per segment): with only the
    # 360 profile points the distance rippled between points -> beaded, serrated rim
    dense = 8
    tree = mathutils.kdtree.KDTree(n * dense)
    for i in range(n):
        (x0, y0), (x1, y1) = outer[i], outer[(i + 1) % n]
        for j in range(dense):
            t = j / dense
            tree.insert((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 0), i * dense + j)
    tree.balance()
    widths = np.repeat(widths, dense)

    rim = np.zeros(len(mesh.vertices), np.float32)
    for v in mesh.vertices:
        r = math.hypot(v.co.x, v.co.y)
        top = min(max(v.co.z / (BODY_T / 2), 0.0), 1.0)   # 1 on the top face, 0 below
        _, idx, d = tree.find((v.co.x, v.co.y, 0))
        near_outline = r > R_HOLE + CSINK_W + 0.02
        if near_outline:
            rim[v.index] = smooth01(1 - d / (2.0 * widths[idx]))   # soft, wide glow
        if top > 0:
            # prong steps: smooth slide-like drops at fixed radii
            for rc, drop, w in PRONG_STEPS:
                v.co.z -= top * drop * smooth01((r - (rc - w / 2)) / w)
            # rim slope: slide down near the outline
            if d < widths[idx] and near_outline:
                v.co.z -= top * RIM_DROP * smooth01(1 - d / widths[idx])
        # bend: z -= BEND * t^2 with t = (r - R_BEND) / (R_TIP - R_BEND), 0 inside R_BEND
        if r > R_BEND:
            t = min((r - R_BEND) / (R_TIP - R_BEND), 1.2)
            v.co.z -= BEND * t * t
    attr = mesh.attributes.new("rim", "FLOAT", "POINT")   # read by the material
    attr.data.foreach_set("value", rim)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return bproc.object.create_from_blender_mesh(mesh, name)


def make_lathe(name, section, n=192, wobble=0.0):
    """Surface made by rotating a cross-section [(r, z), ...] 360° around the Z axis.
    wobble: low-frequency radius irregularity (imperfect circle), full at the first
    section point, fading to 0 at the last (so the outer edge still meets the body)."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rng = np.random.default_rng(7)
    shape = sum(rng.normal(0, 1) * np.cos(k * angles + rng.uniform(0, 6.3)) for k in (2, 3, 5))
    shape = shape / (np.abs(shape).max() + 1e-9)
    fade = np.linspace(1, 0, len(section))
    verts = [((r + wobble * f * dr) * math.cos(a), (r + wobble * f * dr) * math.sin(a), z)
             for (r, z), f in zip(section, fade) for a, dr in zip(angles, shape)]
    faces = []
    for i in range(len(section) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return bproc.object.create_from_blender_mesh(mesh, name)


def rounded_corner(p0, corner, p2, radius, steps=8):
    """Polyline p0 -> corner -> (towards p2), corner rounded over `radius` (quadratic
    Bezier). Returns points from p0 to the end of the rounding."""
    p0, c, p2 = map(np.array, (p0, corner, p2))
    a = c + (p0 - c) / np.linalg.norm(p0 - c) * radius
    b = c + (p2 - c) / np.linalg.norm(p2 - c) * radius
    t = np.linspace(0, 1, steps)[:, None]
    curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b
    return [tuple(p0)] + [tuple(q) for q in curve]


def smooth_step(r0, r1, z0, z1, steps=8):
    """Cosine slope from (r0, z0) to (r1, z1): flat at both ends, never vertical."""
    s = np.linspace(0, 1, steps)
    return [(r0 + (r1 - r0) * t, z1 + (z0 - z1) * (1 + math.cos(math.pi * t)) / 2) for t in s]


def groove_points(r_center, width, depth, z, steps=9):
    """Shallow smooth circular groove (cosine dip) in the (r, z) cross-section."""
    s = np.linspace(-1, 1, steps)
    return [(r_center + width / 2 * t, z - depth * (1 + math.cos(math.pi * t)) / 2) for t in s]


# --- materials ----------------------------------------------------------------
class Nodes:
    """Small helper around a material node tree."""

    def __init__(self, mat):
        self.nodes, self.links = mat.nodes, mat.links

    def new(self, kind, **inputs):
        n = self.nodes.new(kind)
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    def link(self, a, b):
        self.links.new(a, b)

    def math(self, op, *values):
        m = self.nodes.new("ShaderNodeMath")
        m.operation = op
        for socket, value in zip(m.inputs, values):
            if isinstance(value, (int, float)):
                socket.default_value = float(value)
            else:
                self.links.new(value, socket)
        return m.outputs["Value"]

    def noise(self, vector, scale, detail=2.0, dims="3D"):
        n = self.nodes.new("ShaderNodeTexNoise")
        n.noise_dimensions = dims
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        self.links.new(vector, n.inputs["Vector"])
        return n

    def ramp(self, fac, stops):
        """Color ramp with stops [(position, (r, g, b)), ...]."""
        r = self.nodes.new("ShaderNodeValToRGB")
        els = r.color_ramp.elements
        while len(els) < len(stops):
            els.new(0.5)
        for el, (pos, col) in zip(els, stops):
            el.position, el.color = pos, (*col, 1)
        self.links.new(fac, r.inputs["Fac"])
        return r

    def mix(self, blend, fac, c1, c2):
        m = self.nodes.new("ShaderNodeMixRGB")
        m.blend_type = blend
        for socket, value in zip((m.inputs["Fac"], m.inputs["Color1"], m.inputs["Color2"]),
                                 (fac, c1, c2)):
            if isinstance(value, (int, float)):
                socket.default_value = float(value)
            elif isinstance(value, (tuple, list)):
                socket.default_value = (*value, 1) if len(value) == 3 else value
            else:
                self.links.new(value, socket)
        return m


def make_metal_material(name):
    """Zinc-plated steel. Texture is in COLOR and ROUGHNESS (real surface has no visible
    relief): dark mottled base + light wavy radial streaks and dots + sparse gold specks.
    Shape-wise only broad stamping waviness (fragmented highlights).
    Returns the material and the handles randomize_material() sets per image."""
    mat = bproc.material.create(name)
    nt = Nodes(mat)
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 1.0
    h = {}

    coords = nt.new("ShaderNodeTexCoord").outputs["Object"]
    mapping = nt.new("ShaderNodeMapping")              # per-image texture offset
    nt.link(coords, mapping.inputs["Vector"])
    vec = mapping.outputs["Vector"]
    h["mapping"] = mapping
    flat = nt.new("ShaderNodeVectorMath")
    flat.operation = "MULTIPLY"
    flat.inputs[1].default_value = (1, 1, 0)
    nt.link(coords, flat.inputs[0])
    radius_n = nt.new("ShaderNodeVectorMath")
    radius_n.operation = "LENGTH"
    nt.link(flat.outputs["Vector"], radius_n.inputs[0])
    radius = radius_n.outputs["Value"]

    # 1) base color: zinc color with broad hue drift
    color = nt.new("ShaderNodeRGB")
    h["color"] = color
    hue = nt.ramp(nt.noise(vec, HUE_SCALE).outputs["Fac"],
                  [(0.3, HUE_SHIFT[0]), (0.5, HUE_SHIFT[1]), (0.7, HUE_SHIFT[2])])
    hued = nt.mix("MULTIPLY", 0.6, color.outputs["Color"], hue.outputs["Color"])
    h["hue"] = hued

    # 2) mottled tone: irregular noise patches x cellular (Voronoi) patches
    mottle = nt.ramp(nt.noise(vec, MOTTLE_SCALE, 4.0).outputs["Fac"],
                     [(0.35, (MOTTLE_LOW,) * 3), (0.65, (1.0, 1.0, 1.0))])
    h["mottle"] = mottle
    vor = nt.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = CELL_SCALE
    nt.link(vec, vor.inputs["Vector"])
    cells = nt.ramp(vor.outputs["Distance"], [(0.0, (0.85, 0.85, 0.85)), (0.8, (1.0, 1.0, 1.0))])
    toned = nt.mix("MULTIPLY", 1.0, hued.outputs["Color"], mottle.outputs["Color"])
    toned = nt.mix("MULTIPLY", 1.0, toned.outputs["Color"], cells.outputs["Color"])

    # 3) iris: short radial-oriented strokes and dots, bright and dark (hard contrast)
    light, dark, h["iris"] = add_strokes(nt, coords, vec, radius)
    lit = nt.mix("MULTIPLY", nt.math("SUBTRACT", 1.0, light), toned.outputs["Color"],
                 (BASE_MID, BASE_MID, BASE_MID))        # mid base, full color on bright strokes
    lit = nt.mix("MIX", dark, lit.outputs["Color"], DARK_COLOR)

    # 4) sparse golden specks
    gold_mask = nt.math("GREATER_THAN", nt.noise(vec, GOLD_SCALE, 0.0).outputs["Fac"], GOLD_AMOUNT)
    h["gold"] = gold_mask.node.inputs[1]
    surface = nt.mix("MIX", nt.math("MULTIPLY", gold_mask, 0.7), lit.outputs["Color"],
                     tuple(c * 0.8 for c in GOLD)).outputs["Color"]

    # rim (outline + short prong edges): bright whitish and rough -> soft glow
    rim_attr = nt.new("ShaderNodeAttribute")
    rim_attr.attribute_name = "rim"
    rim = rim_attr.outputs["Fac"]                       # 0 on parts without the attribute
    rim_sharp = nt.math("POWER", rim, 2.0)               # glow concentrated at the very edge
    surface = nt.mix("MIX", nt.math("MULTIPLY", rim_sharp, RIM_AMOUNT), surface,
                     RIM_COLOR).outputs["Color"]
    # soft bright transition from the countersink edge into the flange
    glow = nt.new("ShaderNodeMapRange")
    glow.inputs["From Min"].default_value = R_HOLE + CSINK_W + CSINK_GLOW_W
    glow.inputs["From Max"].default_value = R_HOLE + CSINK_W
    nt.link(radius, glow.inputs["Value"])
    surface = nt.mix("MIX", nt.math("MULTIPLY", nt.math("POWER", glow.outputs["Result"], 2.0),
                                    CSINK_GLOW), surface, RIM_COLOR).outputs["Color"]

    # 5) scratches: texture drawn per image in Python (R = amount, G = tone)
    tex_map = nt.new("ShaderNodeMapping")              # nut-local [-0.5, 0.5] -> [0, 1]
    tex_map.inputs["Location"].default_value = (0.5, 0.5, 0)
    nt.link(coords, tex_map.inputs["Vector"])
    tex = nt.new("ShaderNodeTexImage")
    SCRATCH_TEX_NODES.append(tex)          # image assigned per frame (update_scratch_texture)
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    nt.link(tex_map.outputs["Vector"], tex.inputs["Vector"])
    sep = nt.new("ShaderNodeSeparateColor")
    nt.link(tex.outputs["Color"], sep.inputs["Color"])
    total = sep.outputs["Red"]
    tone = nt.ramp(sep.outputs["Green"], SCR_COLORS)
    tone_mult = nt.mix("MULTIPLY", 1.0, tone.outputs["Color"], (1.0, 1.0, 1.0))
    h["scratch_tone"] = tone_mult
    grind = nt.new("ShaderNodeMapRange")          # fine sanding grain of the exposed metal
    grind.inputs["To Min"].default_value = 0.85
    grind.inputs["To Max"].default_value = 1.05
    nt.link(nt.noise(vec, 220, 2.0).outputs["Fac"], grind.inputs["Value"])
    scr_color = nt.mix("MULTIPLY", 1.0, tone_mult.outputs["Color"], grind.outputs["Result"])
    darken = nt.mix("MIX", total, surface, scr_color.outputs["Color"])
    nt.link(darken.outputs["Color"], bsdf.inputs["Base Color"])

    # roughness: broad patches in the range, scratches pulled to scratch_roughness
    rough = nt.new("ShaderNodeMapRange")
    nt.link(nt.noise(vec, 8).outputs["Fac"], rough.inputs["Value"])
    h["rough"] = rough
    diff = nt.math("SUBTRACT", 0.45, rough.outputs["Result"])
    h["scratch_roughness"] = diff.node.inputs[0]
    final_rough = nt.math("ADD", nt.math("MULTIPLY", diff, total), rough.outputs["Result"])
    band_rough = nt.math("SUBTRACT", BAND_ROUGHNESS, final_rough)
    final_rough = nt.math("ADD", nt.math("MULTIPLY", band_rough, tex.outputs["Alpha"]), final_rough)
    to_matte = nt.math("SUBTRACT", 0.85, final_rough)          # rim: rough (soft, blurry)
    final_rough = nt.math("ADD", nt.math("MULTIPLY", to_matte, rim), final_rough)
    nt.link(final_rough, bsdf.inputs["Roughness"])

    # normal: two broad waviness layers (stamping), plus an almost-zero micro relief
    normal = None
    h["waves"] = []
    for scale in WAVE_SCALES:
        b = nt.new("ShaderNodeBump", Distance=0.02, Strength=WAVE_STRENGTH)
        nt.link(nt.noise(vec, scale, 2.0).outputs["Fac"], b.inputs["Height"])
        if normal is not None:
            nt.link(normal, b.inputs["Normal"])
        normal = b.outputs["Normal"]
        h["waves"].append(b)
    micro = nt.new("ShaderNodeBump", Distance=0.002, Strength=MICRO_NORMAL)
    nt.link(nt.noise(vec, 150, 2.0).outputs["Fac"], micro.inputs["Height"])
    nt.link(normal, micro.inputs["Normal"])
    groove = nt.new("ShaderNodeBump", Distance=0.004, Strength=SCR_DEPTH)
    groove.invert = True                      # higher amount = deeper
    nt.link(sep.outputs["Blue"], groove.inputs["Height"])
    # band: material removed -> flatten the surface normal (hides rings, steps, waviness)
    flat = tex.outputs["Alpha"]
    keep = nt.new("ShaderNodeVectorMath")
    keep.operation = "SCALE"
    nt.link(micro.outputs["Normal"], keep.inputs[0])
    nt.link(nt.math("SUBTRACT", 1.0, flat), keep.inputs["Scale"])
    up = nt.new("ShaderNodeVectorMath")
    up.operation = "SCALE"
    up.inputs[0].default_value = (0, 0, 1)     # nut faces the camera; rotation only about z
    nt.link(flat, up.inputs["Scale"])
    summed = nt.new("ShaderNodeVectorMath")
    summed.operation = "ADD"
    nt.link(keep.outputs["Vector"], summed.inputs[0])
    nt.link(up.outputs["Vector"], summed.inputs[1])
    flat_n = nt.new("ShaderNodeVectorMath")
    flat_n.operation = "NORMALIZE"
    nt.link(summed.outputs["Vector"], flat_n.inputs[0])
    nt.link(flat_n.outputs["Vector"], groove.inputs["Normal"])
    nt.link(groove.outputs["Normal"], bsdf.inputs["Normal"])
    h["groove"] = groove
    return mat, h


def make_countersink_material():
    """Smooth countersink: ring sequence from CSINK_RAMP over the radius, a faint fine
    texture so it is not plastic, semi-metallic."""
    mat = bproc.material.create("nut_countersink")
    nt = Nodes(mat)
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.4
    bsdf.inputs["Roughness"].default_value = 0.45
    coords = nt.new("ShaderNodeTexCoord").outputs["Object"]
    flat = nt.new("ShaderNodeVectorMath")
    flat.operation = "MULTIPLY"
    flat.inputs[1].default_value = (1, 1, 0)
    nt.link(coords, flat.inputs[0])
    radius = nt.new("ShaderNodeVectorMath")
    radius.operation = "LENGTH"
    nt.link(flat.outputs["Vector"], radius.inputs[0])
    t = nt.new("ShaderNodeMapRange")
    t.inputs["From Min"].default_value = R_HOLE
    t.inputs["From Max"].default_value = R_HOLE + CSINK_W
    nt.link(radius.outputs["Value"], t.inputs["Value"])
    warp = nt.new("ShaderNodeMapRange")       # noise -> ring position shift (wobbly rings)
    warp.inputs["To Min"].default_value = -CSINK_RING_WARP
    warp.inputs["To Max"].default_value = CSINK_RING_WARP
    nt.link(nt.noise(coords, 9, 2.0).outputs["Fac"], warp.inputs["Value"])
    t_warped = nt.math("ADD", t.outputs["Result"], warp.outputs["Result"])
    ramp = nt.ramp(t_warped, CSINK_RAMP)
    hue = nt.ramp(nt.noise(coords, 12, 2.0).outputs["Fac"], CSINK_HUE)
    ramp = nt.mix("MULTIPLY", 1.0, ramp.outputs["Color"], hue.outputs["Color"])
    grain = nt.new("ShaderNodeMapRange")
    grain.inputs["To Min"].default_value = 0.65   # not smooth: visible fine grain
    nt.link(nt.noise(coords, 200, 1.0).outputs["Fac"], grain.inputs["Value"])
    color = nt.mix("MULTIPLY", 1.0, ramp.outputs["Color"], grain.outputs["Result"])
    nt.link(color.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_strokes(nt, coords, vec, radius):
    """Short strokes + dots, oriented radially only when seen from afar.
    Noise sampled at (direction x K + waviness, radius x L): with K around and L along
    of similar size the noise blobs are short and slightly elongated outward; bright
    strokes where noise > t, dark strokes where noise < 1 - t (both tails -> hard
    light/dark transitions). Direction vector instead of angle: no seam at ±180°.
    Returns (light 0..1, dark 0..1, handles)."""
    flat = nt.new("ShaderNodeVectorMath")
    flat.operation = "MULTIPLY"
    flat.inputs[1].default_value = (1, 1, 0)
    nt.link(coords, flat.inputs[0])
    direction = nt.new("ShaderNodeVectorMath")
    direction.operation = "NORMALIZE"
    nt.link(flat.outputs["Vector"], direction.inputs[0])
    wave = nt.new("ShaderNodeVectorMath")                 # irregular orientation
    wave.operation = "SCALE"
    wave.inputs["Scale"].default_value = IRIS_WAVE
    nt.link(nt.noise(vec, 6, 1.0).outputs["Color"], wave.inputs[0])
    light = dark = None
    layers = []
    for k, l in STROKE_LAYERS:
        around = nt.new("ShaderNodeVectorMath")
        around.operation = "SCALE"
        around.inputs["Scale"].default_value = float(k)
        nt.link(direction.outputs["Vector"], around.inputs[0])
        along = nt.new("ShaderNodeCombineXYZ")
        nt.link(nt.math("MULTIPLY", radius, float(l)), along.inputs["Z"])
        v = nt.new("ShaderNodeVectorMath")
        v.operation = "ADD"
        nt.link(around.outputs["Vector"], v.inputs[0])
        nt.link(along.outputs["Vector"], v.inputs[1])
        v2 = nt.new("ShaderNodeVectorMath")
        v2.operation = "ADD"
        nt.link(v.outputs["Vector"], v2.inputs[0])
        nt.link(wave.outputs["Vector"], v2.inputs[1])
        n = nt.noise(v2.outputs["Vector"], 1.0, 1.0, dims="4D")   # W = random phase
        bright = nt.math("GREATER_THAN", n.outputs["Fac"], STROKE_THRESHOLD)
        low = nt.math("SUBTRACT", 1.0 - DARK_EXTRA, STROKE_THRESHOLD)
        darkm = nt.math("LESS_THAN", n.outputs["Fac"], low)
        light = nt.math("MAXIMUM", light, bright) if light else bright
        dark = nt.math("MAXIMUM", dark, darkm) if dark else darkm
        layers.append({"w": n.inputs["W"], "bright": bright.node.inputs[1],
                       "dark": low.node.inputs[1]})
    dots = nt.math("GREATER_THAN", nt.noise(vec, DOT_SCALE, 0.0).outputs["Fac"], DOT_AMOUNT)
    light = nt.math("MAXIMUM", light, dots)
    spots = nt.math("LESS_THAN", nt.noise(vec, DARK_SPOT_SCALE, 1.0).outputs["Fac"], DARK_SPOT_AMOUNT)
    dark = nt.math("MAXIMUM", dark, spots)
    return light, dark, layers


def randomize_material(h, p):
    """Apply per-image parameters p to the material handles h."""
    h["mapping"].inputs["Location"].default_value = p["texture_offset"]
    h["color"].outputs["Color"].default_value = (
        [c * t for c, t in zip(NUT_COLOR, p["tint"])] + [1])
    h["hue"].inputs["Fac"].default_value = p["hue_amount"]
    low = p["mottle_low"]
    h["mottle"].color_ramp.elements[0].color = (low, low, low, 1)
    for k, layer in enumerate(h["iris"]):
        layer["w"].default_value = p["texture_offset"][k % 3]
        layer["bright"].default_value = p["stroke_threshold"]
        layer["dark"].default_value = p["stroke_threshold"]   # dark below 1 - t
    h["gold"].default_value = p["gold_amount"]
    h["rough"].inputs["To Min"].default_value = p["roughness_min"]
    h["rough"].inputs["To Max"].default_value = p["roughness_max"]
    for wave in h["waves"]:
        wave.inputs["Strength"].default_value = p["wave_strength"]

    h["groove"].inputs["Strength"].default_value = p["scratch_depth"]
    t = p["scratch_tone"]
    h["scratch_tone"].inputs["Color2"].default_value = (t, t, t, 1)
    h["scratch_roughness"].default_value = p["scratch_roughness"]


# --- scratch texture (strokes drawn in NumPy, nut-local coordinates) -----------
def to_px(xy):
    """Nut-local (x, y) in D -> texture pixel (col, row); row 0 = y -0.5 (Blender order)."""
    return (np.asarray(xy) + 0.5) * SCR_TEX


def draw_polyline(amount, tone, pts, widths, alpha, tone_value):
    """Anti-aliased thick polyline into the amount/tone arrays (max-blend).
    pts: (K, 2) texture px, widths: (K-1,) per segment in px."""
    for (x0, y0), (x1, y1), w in zip(pts[:-1], pts[1:], widths):
        pad = w / 2 + 2
        c0, c1 = int(max(min(x0, x1) - pad, 0)), int(min(max(x0, x1) + pad, SCR_TEX - 1)) + 1
        r0, r1 = int(max(min(y0, y1) - pad, 0)), int(min(max(y0, y1) + pad, SCR_TEX - 1)) + 1
        if c1 <= c0 or r1 <= r0:
            continue
        ys, xs = np.mgrid[r0:r1, c0:c1].astype(np.float32)
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
        d = np.hypot(xs - (x0 + t * dx), ys - (y0 + t * dy))
        a = alpha * np.clip(w / 2 + 0.5 - d, 0, 1)      # 1 inside, 1 px soft edge
        win_a, win_t = amount[r0:r1, c0:c1], tone[r0:r1, c0:c1]
        stronger = a > win_a
        win_t[stronger] = tone_value
        np.maximum(win_a, a, out=win_a)


def stroke_points(center, direction, length, bend, wiggle, rng, n=12):
    """Slightly curved, slightly wiggly stroke centerline in nut-local units."""
    t = np.linspace(-0.5, 0.5, n)
    normal = np.array([-direction[1], direction[0]])
    offset = bend * length * (1 - 4 * t * t) + wiggle * length * rng.normal(0, 1, n).cumsum() / n
    return center + np.outer(t * length, direction) + np.outer(offset, normal)


def draw_scratch(sc):
    """One scratch -> (amount, tone) texture arrays. Field: many separate thin strokes
    around a common direction (different lengths, crossing, mostly dark, some bright).
    Band: one opaque gray-brown strip with rough edges and a slightly curved axis."""
    rng = np.random.default_rng(sc["stroke_seed"])
    amount = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    tone = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    c = np.array([sc["cx"], sc["cy"]])
    th = math.radians(sc["angle_deg"])
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-u[1], u[0]])
    L, W = sc["length"], sc["width"]
    if sc["type"] == "field":
        for _ in range(sc["n_strokes"]):
            l = L * rng.uniform(0.3, 1.0)
            along = rng.uniform(-(L - l) / 2, (L - l) / 2)
            across = rng.uniform(-W / 2, W / 2)
            a = th + math.radians(rng.normal(0, FIELD_JITTER_DEG))
            d = np.array([math.cos(a), math.sin(a)])
            pts = to_px(stroke_points(c + along * u + across * v, d, l,
                                      sc["bend"] + rng.normal(0, 0.03), 0.01, rng))
            w = rng.uniform(*FIELD_STROKE_PX)
            bright = rng.random() < P_BRIGHT_STROKE
            draw_polyline(amount, tone, pts, np.full(len(pts) - 1, w),
                          rng.uniform(0.55, 1.0), 1.0 if bright else 0.0)
    else:
        pts = to_px(stroke_points(c, u, L, sc["bend"], 0.004, rng, n=48))
        noise = np.convolve(rng.normal(0, 1, len(pts) + 1), [0.5, 0.5], mode="same")[:len(pts) - 1]
        widths = W * SCR_TEX * (1 + BAND_EDGE_NOISE * noise / (np.abs(noise).max() + 1e-9))
        t = (np.arange(len(pts) - 1) + 0.5) / (len(pts) - 1)          # position along band
        e0, e1 = rng.uniform(*BAND_TAPER, 2)
        widths *= np.clip(np.minimum(t / e0, (1 - t) / e1), 0.05, 1)   # pointed ends
        draw_polyline(amount, tone, pts, widths, rng.uniform(0.9, 1.0), 0.5)
        flat = amount.copy()                          # band = material removed: flatten it
        inside = flat > 0.05
        tone[inside] = band_tone(c, u, v, W, rng)[inside]
        for _ in range(int(rng.integers(3, 8))):           # a few streaks inside the band
            off = rng.uniform(-W / 3, W / 3)
            sp = to_px(stroke_points(c + off * v, u, L * rng.uniform(0.3, 0.9), sc["bend"], 0.002, rng))
            draw_polyline(amount, tone, sp, np.full(len(sp) - 1, rng.uniform(2, 4)),
                          0.9, float(rng.choice([0.0, 1.0])))
        return amount, tone, flat
    if sc["haze"] > 0:
        add_haze(amount, tone, sc, c, u, v, rng)
    return amount, tone, None


def band_tone(c, u, v, W, rng):
    """Tone (0.35 dark .. 0.65 light) inside a band: one edge darker than the other
    (random side), fine streaks along the band (grinding marks), broad blotches."""
    ys, xs = np.mgrid[0:SCR_TEX, 0:SCR_TEX].astype(np.float32)
    cx, cy = to_px(c)
    across = ((xs - cx) * v[0] + (ys - cy) * v[1]) / (W * SCR_TEX / 2)   # -1 .. 1 edge to edge
    along = (xs - cx) * u[0] + (ys - cy) * u[1]
    edge = 0.5 - 0.5 * np.clip(across, -1, 1) * rng.choice([-1, 1])     # 0 dark edge, 1 light
    k = 400                                                               # streaks across width
    streak_1d = np.convolve(rng.normal(0, 1, k), np.ones(3) / 3, mode="same")
    idx = np.clip(((across + 1.2) / 2.4 * (k - 1)).astype(int), 0, k - 1)
    streak = streak_1d[idx] / (np.abs(streak_1d).max() + 1e-9)
    streak *= 0.6 + 0.4 * np.sin(along / rng.uniform(15, 40) + rng.uniform(0, 6))  # fade along
    blot = gaussian_blur(rng.normal(0, 1, (SCR_TEX // 8, SCR_TEX // 8, 1)).astype(np.float32), 1.5)[..., 0]
    blot = np.kron(blot / (np.abs(blot).max() + 1e-9), np.ones((8, 8)))
    t = 0.55 * edge + 0.25 * (0.5 + 0.5 * streak) + 0.2 * (0.5 + 0.5 * blot)
    return (0.35 + 0.3 * np.clip(t, 0, 1)).astype(np.float32)


def add_haze(amount, tone, sc, c, u, v, rng):
    """Damage around a field: low light-gray haze inside the (closed) stroke region,
    broken up by noise, plus many faint shallow micro-scratches."""
    region = max_filter(max_filter(amount > 0.2, 14), 14, np.minimum)
    region = gaussian_blur(region[..., None].astype(np.float32), 8)[..., 0]
    blotch = gaussian_blur(rng.normal(0, 1, (SCR_TEX // 4, SCR_TEX // 4, 1)).astype(np.float32), 2)[..., 0]
    blotch = np.kron((blotch - blotch.min()) / (np.ptp(blotch) + 1e-9), np.ones((4, 4)))
    haze = HAZE_AMOUNT * sc["haze"] * region * np.clip(blotch * 1.6 - 0.3, 0, 1)
    stronger = haze > amount
    tone[stronger] = 0.8
    np.maximum(amount, haze.astype(np.float32), out=amount)
    L, W = sc["length"], sc["width"]
    th = math.radians(sc["angle_deg"])
    for _ in range(int(HAZE_STROKES * sc["haze"])):
        l = L * rng.uniform(0.1, 0.5)
        a = th + math.radians(rng.normal(0, 15))
        d = np.array([math.cos(a), math.sin(a)])
        ctr = c + rng.uniform(-L / 2, L / 2) * u + rng.uniform(-W / 2, W / 2) * 1.2 * v
        pts = to_px(stroke_points(ctr, d, l, rng.normal(0, 0.04), 0.01, rng, n=6))
        draw_polyline(amount, tone, pts, np.full(len(pts) - 1, rng.uniform(1.2, 2.2)),
                      rng.uniform(0.12, 0.3), 0.8)


def max_filter(mask, r, op=np.maximum):
    """Separable square max (or min) filter of radius r on a boolean array."""
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for k in range(1, r + 1):
            op(acc, np.roll(out, k, axis), out=acc)
            op(acc, np.roll(out, -k, axis), out=acc)
        out = acc
    return out


def update_scratch_texture(defects):
    """Draw all scratches into SCRATCH_IMAGE; return one texture-space mask per scratch
    (strokes with gaps closed and slightly dilated, like the coarse real masks)."""
    amount = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    tone = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    height = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    flat_all = np.zeros((SCR_TEX, SCR_TEX), np.float32)
    masks = []
    for sc in defects:
        a, t, flat = draw_scratch(sc)
        if flat is not None:
            np.maximum(flat_all, flat, out=flat_all)
        np.maximum(height, gaussian_blur(a[..., None], SCR_HEIGHT_BLUR[sc["type"]])[..., 0],
                   out=height)
        stronger = a > amount
        tone[stronger] = t[stronger]
        np.maximum(amount, a, out=amount)
        m = a > 0.2
        r = MASK_CLOSE_PX[sc["type"]]
        m = max_filter(max_filter(m, r), r, np.minimum)          # closing: fill gaps
        masks.append(max_filter(m, MASK_DILATE_PX))
    rgba = np.zeros((SCR_TEX, SCR_TEX, 4), np.float32)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = amount, tone, height, flat_all
    # Cycles caches an image after the first render and ignores later pixel edits
    # (tested: 0.8 -> 0.3 rendered unchanged), so every frame gets a NEW image
    global SCRATCH_IMAGE
    img = bpy.data.images.new("scratches", SCR_TEX, SCR_TEX, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"   # data texture: no sRGB curve
    img.alpha_mode = "CHANNEL_PACKED"             # alpha = independent data channel
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    for node in SCRATCH_TEX_NODES:
        node.image = img
    if SCRATCH_IMAGE is not None:
        bpy.data.images.remove(SCRATCH_IMAGE)
    SCRATCH_IMAGE = img
    return masks


def texture_to_image_homography(rotation_deg):
    """3x3 homography: texture px -> image px, from the 4 projected corners of the
    texture square on the nut top plane (rotated with the nut). A camera maps a plane
    by a homography; the bent prong tips (≤0.035 D lower) shift < 1 px."""
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    local = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    world = np.column_stack([local @ rot.T, np.full(4, BODY_T)])
    dst = bproc.camera.project_points(world)
    src = to_px(local)
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -X * x, -X * y, -X], [0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y]]
    return np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)


def warp_mask_to_image(tex_mask, H, shape):
    """Sample a texture-space mask at every image pixel (inverse homography, nearest)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.linalg.inv(H) @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    cols = np.round(pts[0] / pts[2]).astype(int)
    rows = np.round(pts[1] / pts[2]).astype(int)
    ok = (cols >= 0) & (cols < SCR_TEX) & (rows >= 0) & (rows < SCR_TEX)
    out = np.zeros(h * w, bool)
    out[ok] = tex_mask[rows[ok], cols[ok]]
    return out.reshape(h, w)


# --- post-processing (NumPy only: no OpenCV inside Blender) --------------------
def gaussian_blur(img, sigma):
    """Separable Gaussian blur (float image H x W x C)."""
    if sigma <= 0:
        return img
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    out = sum(w * pad[:, i:i + img.shape[1]] for i, w in enumerate(k))
    return sum(w * out[i:i + img.shape[0], :] for i, w in enumerate(k))


def radial_scale(channel, center, factor):
    """Resample one channel scaled by `factor` around `center` (bilinear)."""
    h, w = channel.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = center[0] + (xs - center[0]) / factor
    sy = center[1] + (ys - center[1]) / factor
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
    fx, fy = np.clip(sx - x0, 0, 1), np.clip(sy - y0, 0, 1)
    top = channel[y0, x0] * (1 - fx) + channel[y0, x0 + 1] * fx
    bot = channel[y0 + 1, x0] * (1 - fx) + channel[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def sensor_postprocess(img, p, center, rng):
    """Camera/sensor realism: white balance, slight softness, chromatic aberration
    (R and B magnified slightly differently around the center), read + shot noise
    (independent per channel -> color noise). The mask is not touched."""
    x = img.astype(np.float32) * np.array(p["wb_gain"], np.float32)
    x = gaussian_blur(x, p["blur_sigma"])
    x[..., 0] = radial_scale(x[..., 0], center, 1 + p["chroma_ab"])   # R
    x[..., 2] = radial_scale(x[..., 2], center, 1 - p["chroma_ab"])   # B
    noise = rng.normal(0, 1, x.shape).astype(np.float32)
    x = x + noise * (p["read_noise"] + p["shot_noise"] * np.sqrt(np.clip(x, 0, None)))
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


# --- scratch masks --------------------------------------------------------------
def emission_node(mat):
    return next(n for n in mat.nodes if n.type == "EMISSION")


# --- setup --------------------------------------------------------------------
rand_groups = set() if args.no_random else {g for g in args.rand.split(",") if g}
assert rand_groups <= RAND_GROUPS.keys(), f"unknown --rand group in {args.rand}"
out_dir = Path(args.out)
if list(out_dir.glob("*.hdf5")):
    raise SystemExit(f"{out_dir}/ already has renders. Use a new --out folder "
                     "(mixing runs would break meta.jsonl and reproducibility).")
out_dir.mkdir(parents=True, exist_ok=True)

bproc.init()
bproc.renderer.set_output_format(view_transform="Standard")  # colors as written, no film curve
SCRATCH_IMAGE = None          # current scratch texture (new image every frame)
SCRATCH_TEX_NODES = []        # image texture nodes that show it

# --- nut ----------------------------------------------------------------------
top_z = BODY_T + FLANGE_EPS
body = make_body("nut_body")
body.set_location([0, 0, BODY_T / 2])

# center cross-section, from the hole outward: lip -> 45° countersink cone -> rounded rim
# -> flat flange with one thin shallow groove -> tiny step down onto the body at R_FLAT
csink_depth = CSINK_W * math.tan(math.radians(CSINK_ANGLE))
inner = (R_HOLE, top_z - csink_depth)
outer = (R_HOLE + CSINK_W, top_z)
csink_section = (rounded_corner((R_HOLE, inner[1] - 0.02), inner, outer, CSINK_LIP)
                 + rounded_corner(inner, outer, (outer[0] + 1.0, top_z), CSINK_RIM)[1:])
flange_section = ([csink_section[-1]]
                  + groove_points(R_GROOVE, GROOVE_W, GROOVE_DEPTH, top_z)
                  + smooth_step(R_FLAT - 0.03, R_FLAT, top_z, BODY_T - 0.0005))
# the disk fades into the body over a ~3° slope: a short steep step showed as a ring line
countersink = make_lathe("nut_countersink", csink_section, wobble=CSINK_WOBBLE)
flange = make_lathe("nut_flange", flange_section)

metal, metal_h = make_metal_material("nut_metal")
csink_mat = make_countersink_material()
nut_parts = (body, flange, countersink)
for part, mat, cat in zip(nut_parts, (metal, metal, csink_mat), (1, 1, 3)):
    part.replace_materials(mat)
    part.set_cp("category_id", cat)  # 1 = nut; 3 = countersink (merged into 1 when saving)

# --- background ---------------------------------------------------------------
# Emissive plane: shows exactly the measured color and ignores lights (no cast shadow).
ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
ground.set_location([0, 0, -0.1])  # below the bent prong tips
ground_mat = bproc.material.create("background")
ground_mat.make_emissive(1.0, replace=True, emission_color=[0, 0, 0, 1])
ground.replace_materials(ground_mat)
ground.set_cp("category_id", 0)
ground_emission = emission_node(ground_mat)

# --- lights -------------------------------------------------------------------
bproc.renderer.set_world_background([1, 1, 1], strength=WORLD_STRENGTH)
ring = make_lathe("ring_light", [(RING_RADII[0], RING_HEIGHT), (RING_RADII[1], RING_HEIGHT)])
ring_mat = bproc.material.create("ring_light")
ring_mat.make_emissive(RING_STRENGTH, replace=True, emission_color=[1, 1, 1, 1])
ring.replace_materials(ring_mat)
ring.blender_obj.visible_camera = False  # light the scene, but don't appear in the image
ring_emission = emission_node(ring_mat)
graze = make_lathe("graze_light", [(GRAZE_RADII[0], GRAZE_HEIGHT), (GRAZE_RADII[1], GRAZE_HEIGHT)])
graze_mat = bproc.material.create("graze_light")
graze_mat.make_emissive(GRAZE_STRENGTH, replace=True, emission_color=[1, 1, 1, 1])
graze.replace_materials(graze_mat)
graze.blender_obj.visible_camera = False
graze_emission = emission_node(graze_mat)

key = bproc.types.Light()
key.set_type("AREA")
key.blender_obj.data.size = 1.0
lamps = []
for _ in range(N_LAMPS):
    lamp = bproc.types.Light()
    lamp.set_type("AREA")
    lamp.blender_obj.data.size = LAMP_SIZE
    lamps.append(lamp)

# --- camera: intrinsics fixed; real depth of field ------------------------------
frame_width = 2 * R_TIP / FILL
fov = 2 * math.atan(frame_width / 2 / (CAM_HEIGHT - top_z))
bproc.camera.set_intrinsics_from_blender_params(
    lens=fov, lens_unit="FOV", image_width=IMG_SIZE, image_height=IMG_SIZE,
    clip_start=0.1, clip_end=50)
cam_data = bpy.context.scene.camera.data
cam_data.sensor_width = SENSOR_MM
cam_data.dof.use_dof = True

bproc.renderer.set_max_amount_of_samples(SAMPLES)
bproc.renderer.set_denoiser("INTEL")
bproc.renderer.enable_segmentation_output(map_by=["category_id"],
                                          default_values={"category_id": 0})

# --- render loop --------------------------------------------------------------
t_start = time.time()
with open(out_dir / "meta.jsonl", "a") as meta_file:
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])  # image i is reproducible on its own
        p = sample_params(rng, groups=rand_groups, defects=not args.no_random)
        if args.no_defects:
            p["defects"] = []

        for part in nut_parts:
            part.set_rotation_euler([0, 0, math.radians(p["rotation_deg"])])
        randomize_material(metal_h, p)
        scratch_masks = update_scratch_texture(p["defects"])
        ground_emission.inputs["Color"].default_value = (
            [srgb_to_linear(c + p["bg_shift"]) for c in BG_COLOR_SRGB] + [1])
        ring_emission.inputs["Strength"].default_value = p["ring_strength"]
        graze_emission.inputs["Strength"].default_value = p["graze_strength"]

        ka, ke = math.radians(p["key_angle_deg"]), math.radians(p["key_elevation_deg"])
        key_loc = 3.0 * np.array([math.cos(ke) * math.cos(ka),
                                  math.cos(ke) * math.sin(ka), math.sin(ke)])
        key.set_location(key_loc)
        key.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-key_loc))
        key.set_energy(p["key_energy"])
        for lamp, az, el, e in zip(lamps, p["lamp_azimuth_deg"], p["lamp_elevation_deg"],
                                   p["lamp_energy"]):
            az, el = math.radians(az), math.radians(el)
            loc = LAMP_DIST * np.array([math.cos(el) * math.cos(az),
                                        math.cos(el) * math.sin(az), math.sin(el)])
            lamp.set_location(loc)
            lamp.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-loc))
            lamp.set_energy(e)

        # camera on a small cone around the vertical, aimed at the nut center, focused
        # on the flange top: the bent prong tips fall out of focus by themselves
        bproc.utility.reset_keyframes()
        tilt, tdir = math.radians(p["cam_tilt_deg"]), math.radians(p["cam_tilt_dir_deg"])
        dist = (CAM_HEIGHT - top_z) * p["cam_dist_scale"]
        poi = np.array([0, 0, top_z])
        cam_loc = poi + dist * np.array([math.sin(tilt) * math.cos(tdir),
                                         math.sin(tilt) * math.sin(tdir), math.cos(tilt)])
        cam_rot = bproc.camera.rotation_from_forward_vec(poi - cam_loc)
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, cam_rot))
        cam_data.dof.focus_distance = dist
        cam_data.dof.aperture_fstop = p["fstop"]

        data = bproc.renderer.render()
        center_px = bproc.camera.project_points(np.array([[0.0, 0.0, top_z]]))[0]
        data["colors"][0] = sensor_postprocess(
            data["colors"][0], p, center_px, np.random.default_rng([args.seed, i, 1]))

        # scratch mask: each scratch's texture mask, mapped into the image with the
        # plane homography; only on the textured nut parts (not the countersink)
        seg = data["category_id_segmaps"][0]
        nut_px = seg == 1
        H = texture_to_image_homography(p["rotation_deg"])
        for sc, tex_mask in zip(p["defects"], scratch_masks):
            region = warp_mask_to_image(tex_mask, H, seg.shape) & nut_px
            seg[region] = 2
            ys, xs = np.nonzero(region)
            sc["area_px"] = int(region.sum())
            sc["bbox"] = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                          if len(xs) else None)  # x_min, y_min, x_max, y_max
        seg[seg == 3] = 1
        data["category_id_segmaps"][0] = seg

        bproc.writer.write_hdf5(str(out_dir), data, append_to_existing_output=True)
        meta_file.write(json.dumps({"index": i, "file": f"{i}.hdf5", "seed": args.seed,
                                    "dataset_version": args.version,
                                    "randomized": sorted(rand_groups),
                                    "n_defects": len(p["defects"]), **p}) + "\n")
        meta_file.flush()
        elapsed = time.time() - t_start
        print(f"[generate] {i + 1}/{args.n} done, {elapsed / (i + 1):.1f} s/image", flush=True)
