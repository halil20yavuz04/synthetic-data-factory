import blenderproc as bproc  # must be the very first line (BlenderProc checks this)
# Render N randomized metal nuts with synthetic scratches, top-down,
# RGB + segmentation mask (0 = background, 1 = nut, 2 = scratch).
#
# Base scene values come from reports/real_observations.md; random ranges are
# listed in sample_params(). Every image's parameters go to meta.jsonl.
#
# Run from the repo root:
#     blenderproc run render/generate.py --n 200 --seed 0
#     blenderproc run render/generate.py --n 1 --no-random --out output/render_test
# Output: <out>/N.hdf5 + <out>/meta.jsonl  (convert with tools/hdf5_to_png.py)
import argparse
import json
import math
import time
from pathlib import Path

import bpy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="number of images")
parser.add_argument("--seed", type=int, default=0, help="master seed")
parser.add_argument("--out", default="data/synth/week1", help="output folder")
parser.add_argument("--no-random", action="store_true", help="render base values only")
parser.add_argument("--version", default="synth_v1", help="dataset_version label in meta")
args = parser.parse_args()

IMG_SIZE = 512
SAMPLES = 64  # low + denoiser: ~8 s/image

# --- nut geometry (units: tab-tip diameter D = 1.0) ---------------------------
# Outline = measured polar profile of one tab period (tools/nut_profile.py),
# repeated 4 times. This also fixes the tab handedness to match the real nut.
PROFILE = json.load(open("render/nut_profile.json"))["r_over_D"]
N_TABS = 4
R_TIP = max(PROFILE)    # = 0.5
R_HOLE = 0.246 / 2      # hole / D = 0.246 [measured]
BODY_T = 0.06           # body thickness
BODY_BEVEL = 0.02       # rounded outer edges -> ~10 px bright rim like the real tabs
# The center is built by rotating a cross-section (r, z) around the axis ("lathe"):
HOLE_FILLET = 0.05      # rounded, smooth bead around the hole (quarter circle)
R_BOSS = 0.31           # raised center ring, ~0.62 D [visual]
BOSS_T = 0.015          # height of the center ring above the body (subtle)
BOSS_RAMP = 0.07        # S-shaped slope at the boss edge, max ~19° so it catches the ring light

# --- appearance (base values; randomized per image in sample_params) ----------
NUT_COLOR = [0.77, 1.0, 0.89]      # mean nut color [measured], sRGB -> linear, normalized
ROUGHNESS_RANGE = (0.3, 0.5)       # varies over the surface in large patches
GRAIN_SCALE = 120                  # surface grain (~4 px at 512 px)
GRAIN_STRENGTH = 0.4               # bump strength of the grain
SPECKLE_SCALE = 250                # small dark pits (~2 px), main source of real texture
SPECKLE_AMOUNT = 0.58              # higher = fewer pits (noise threshold)
SPECKLE_DARK = 0.55                # pit color = SPECKLE_DARK * nut color
RINGS_SCALE = 80                   # concentric machining marks: irregular radial bands
RINGS_WEIGHT = 0.25                # their share of the bump height (vs grain)
HOLE_ROUGHNESS = 0.35              # hole bead is smoother and shinier than the rest
BG_COLOR_SRGB = [23, 23, 29]       # #17171d [measured]
WORLD_STRENGTH = 0.02              # faint ambient fill from all directions
RING_HEIGHT = 2.0                  # ring light above the nut (main light)
RING_RADII = (1.0, 2.5)            # inner/outer radius -> light arrives 27°-51° from vertical
RING_STRENGTH = 0.35               # emission strength; tuned to real mean RGB + contrast
KEY_ANGLE = 150                    # deg, ~10 o'clock in the image [measured]
KEY_ELEVATION = 30                 # deg
KEY_ENERGY = 15.0                  # weak directional light (W)

# --- scratches (reports/real_observations.md, 10 seen_pool images) -----------
# Each scratch = a rotated rectangle on the nut surface. Inside it the surface is
# darker and more matte; "field" scratches also get fine parallel lines.
# The same rectangle is projected into the image to build the mask (category 2).
MAX_SCRATCHES = 3
P_NO_DEFECT = 0.35                 # roadmap: 30-40% defect-free images
P_N_SCRATCHES = [0.7, 0.2, 0.1]    # 1, 2 or 3 scratches; real: usually 1, sometimes 2-3
P_FIELD = 0.7                      # real: 7 of 10 are hairline fields, 3 gouge bands
FIELD_LENGTH, FIELD_WIDTH = (0.15, 0.8), (0.08, 0.35)   # in D (tab-tip diameter)
BAND_LENGTH, BAND_WIDTH = (0.5, 1.0), (0.05, 0.1)
SCRATCH_CENTER_R = (0.15, 0.45)    # center lies on the nut, anywhere from hole to tabs
LINE_STRETCH = (4, 500)            # noise scale along/across a field -> long thin lines

# --- camera -------------------------------------------------------------------
CAM_HEIGHT = 10.0       # far away -> almost orthographic
FILL = 0.882            # 0.89 gave 89.9% width vs real 89.1% (Week 1 check) -> scaled


def sample_params(rng, randomize):
    """All per-image scene parameters. Base values if randomize is False.
    Ranges: real data spread where measured (reports/real_observations.md),
    otherwise small jitter around the tuned base value."""
    p = {
        "rotation_deg": 0.0,
        "cam_tilt_deg": 0.0,
        "cam_tilt_dir_deg": 0.0,
        "cam_dist_scale": 1.0,
        "ring_strength": RING_STRENGTH,
        "key_angle_deg": float(KEY_ANGLE),
        "key_elevation_deg": float(KEY_ELEVATION),
        "key_energy": KEY_ENERGY,
        "roughness_min": ROUGHNESS_RANGE[0],
        "roughness_max": ROUGHNESS_RANGE[1],
        "grain_strength": GRAIN_STRENGTH,
        "speckle_amount": SPECKLE_AMOUNT,
        "speckle_dark": SPECKLE_DARK,
        "tint": [1.0, 1.0, 1.0],
        "bg_shift": 0,
        "texture_offset": [0.0, 0.0, 0.0],
        "scratch_darken": 0.65,        # scratch color = darken * surface color
        "scratch_roughness": 0.7,      # matte inside scratches
        "defects": [],
    }
    if not randomize:
        return p
    shift = rng.uniform(-0.08, 0.08)
    p.update({
        "rotation_deg": rng.uniform(0, 360),           # real: random in-plane rotation
        "cam_tilt_deg": rng.uniform(0, 3),             # real: no visible tilt
        "cam_tilt_dir_deg": rng.uniform(0, 360),
        "cam_dist_scale": rng.uniform(0.995, 1.005),   # real: scale spread ±0.4%
        # real nut mean gray 98.5 ± 10 (84-122) over 220 train/good images
        "ring_strength": RING_STRENGTH * rng.uniform(0.75, 1.3),
        "key_angle_deg": KEY_ANGLE + rng.uniform(-30, 30),
        "key_elevation_deg": rng.uniform(20, 45),
        "key_energy": rng.uniform(0, 30),
        "roughness_min": ROUGHNESS_RANGE[0] + shift,
        "roughness_max": ROUGHNESS_RANGE[1] + shift,
        "grain_strength": rng.uniform(0.2, 0.6),
        "speckle_amount": rng.uniform(0.52, 0.65),
        "speckle_dark": rng.uniform(0.45, 0.7),
        "tint": rng.normal(1.0, 0.03, 3).round(4).tolist(),
        "bg_shift": int(rng.integers(-3, 4)),
        # move the noise patterns, otherwise every nut has the exact same pits
        "texture_offset": rng.uniform(-100, 100, 3).round(3).tolist(),
        "scratch_darken": rng.uniform(0.35, 0.7),
        "scratch_roughness": rng.uniform(0.55, 0.85),
        "defects": sample_scratches(rng),
    })
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}


def sample_scratches(rng):
    """0-3 scratches as rotated rectangles in nut-local coordinates (units of D)."""
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
            # fields: how many lines (higher threshold = fewer lines)
            "line_threshold": round(rng.uniform(0.45, 0.6), 3),
        })
    return scratches


def srgb_to_linear(c):
    """Blender colors are linear; convert an 8-bit sRGB value."""
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def outline_points():
    """2D outline of the nut: one point per profile sample, going counter-clockwise."""
    radii = PROFILE * N_TABS  # list repeated 4x -> full 360°
    angles = np.linspace(0, 2 * math.pi, len(radii), endpoint=False)
    return [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]


def circle_points(r, n=96):
    return [(r * math.cos(a), r * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def make_extruded(name, outer, inner, thickness, bevel):
    """Fill the area between two closed 2D outlines and give it thickness.
    Uses a Blender curve (2D fill with a hole), then converts it to a mesh.
    The bevel rounds the edges; the inward offset keeps the outline at its true size
    (otherwise the bevel would grow the nut outward)."""
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = max(thickness / 2 - bevel, 0)
    curve.bevel_depth = bevel
    curve.bevel_resolution = 4
    curve.offset = -0.7 * bevel  # 0.7: measured, keeps nut width at 89% (tools/hdf5_to_png.py)
    for outline in (outer, inner):
        spline = curve.splines.new("POLY")
        spline.points.add(len(outline) - 1)
        for p, (x, y) in zip(spline.points, outline):
            p.co = (x, y, 0, 1)
        spline.use_cyclic_u = True

    curve_obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(curve_obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(curve_obj.evaluated_get(depsgraph))
    bpy.data.objects.remove(curve_obj)
    return bproc.object.create_from_blender_mesh(mesh, name)


def make_lathe(name, section, n=96):
    """Surface made by rotating a cross-section [(r, z), ...] 360° around the Z axis,
    like a part on a lathe or a pot on a potter's wheel."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    verts = [(r * math.cos(a), r * math.sin(a), z) for r, z in section for a in angles]
    faces = []
    for i in range(len(section) - 1):          # ring i -> ring i+1
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    for poly in mesh.polygons:
        poly.use_smooth = True                 # smooth shading on curved surfaces
    return bproc.object.create_from_blender_mesh(mesh, name)


def arc(center_r, center_z, radius_r, radius_z, t0, t1, steps=8):
    """Points on a (elliptical) arc in the (r, z) cross-section, angles in degrees."""
    ts = np.radians(np.linspace(t0, t1, steps))
    return [(center_r + radius_r * math.cos(t), center_z + radius_z * math.sin(t)) for t in ts]


def smooth_step(r0, r1, z0, z1, steps=10):
    """S-shaped (cosine) slope from (r0, z0) to (r1, z1): flat at both ends,
    steepest in the middle, never vertical."""
    s = np.linspace(0, 1, steps)
    return [(r0 + (r1 - r0) * t, z1 + (z0 - z1) * (1 + math.cos(math.pi * t)) / 2) for t in s]


def make_metal_material(name, roughness_range, grain, rings):
    """Metal with optional fine grain + concentric marks (bump), patchy roughness
    and sparse dark pits. All from procedural noise, no image files needed.
    Returns the material and the nodes that randomize_material() changes."""
    mat = bproc.material.create(name)
    nodes, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 1.0

    coords = nodes.new("ShaderNodeTexCoord")
    # shifts all noise patterns (not the concentric rings) per image
    mapping = nodes.new("ShaderNodeMapping")
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])

    def noise(scale, detail):
        n = nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    def math_node(op, *values):
        m = nodes.new("ShaderNodeMath")
        m.operation = op
        for socket, value in zip(m.inputs, values):
            if isinstance(value, float):
                socket.default_value = value
            else:
                links.new(value, socket)
        return m

    handles = {"mapping": mapping, "bump": None, "ring_phase": None}
    # bump height = grain + RINGS_WEIGHT * concentric marks
    height = None
    if grain:
        height = noise(GRAIN_SCALE, 2).outputs["Fac"]
    if rings:
        # distance from the center axis: length of (x, y, 0), unshifted coordinates
        flat = nodes.new("ShaderNodeVectorMath")
        flat.operation = "MULTIPLY"
        flat.inputs[1].default_value = (1, 1, 0)
        links.new(coords.outputs["Object"], flat.inputs[0])
        radius = nodes.new("ShaderNodeVectorMath")
        radius.operation = "LENGTH"
        links.new(flat.outputs["Vector"], radius.inputs[0])
        # random phase -> different ring positions per image
        phase = math_node("ADD", radius.outputs["Value"], 0.0)
        handles["ring_phase"] = phase.inputs[1]
        # 1D noise along the radius -> irregular concentric bands
        band = nodes.new("ShaderNodeTexNoise")
        band.noise_dimensions = "1D"
        band.inputs["Scale"].default_value = RINGS_SCALE
        band.inputs["Detail"].default_value = 3
        links.new(phase.outputs["Value"], band.inputs["W"])
        ring_h = math_node("MULTIPLY", band.outputs["Fac"], RINGS_WEIGHT).outputs["Value"]
        height = math_node("ADD", height, ring_h).outputs["Value"] if height else ring_h
    if height is not None:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = GRAIN_STRENGTH
        bump.inputs["Distance"].default_value = 0.002
        links.new(height, bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        handles["bump"] = bump

    # large patches -> roughness within roughness_range
    patches = noise(8, 2)
    rough = nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = roughness_range[0]
    rough.inputs["To Max"].default_value = roughness_range[1]
    links.new(patches.outputs["Fac"], rough.inputs["Value"])
    handles["rough"] = rough

    # speckles: where noise > threshold, darken the base color
    speckle = noise(SPECKLE_SCALE, 0)
    ramp = nodes.new("ShaderNodeValToRGB")
    links.new(speckle.outputs["Fac"], ramp.inputs["Fac"])
    mix = nodes.new("ShaderNodeMixRGB")
    links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    handles.update({"ramp": ramp, "mix": mix})

    # scratches: total = how strongly this surface point is scratched (0..1)
    total, handles["scratches"] = add_scratch_nodes(nodes, links, coords, math_node)
    # color: multiply by "darken" where scratched
    darken = nodes.new("ShaderNodeMixRGB")
    darken.blend_type = "MULTIPLY"
    links.new(total, darken.inputs["Fac"])
    links.new(mix.outputs["Color"], darken.inputs["Color1"])
    links.new(darken.outputs["Color"], bsdf.inputs["Base Color"])
    handles["darken"] = darken
    # roughness: rough + total * (scratch_roughness - rough)  -> matte where scratched
    diff = math_node("SUBTRACT", 0.7, rough.outputs["Result"])
    handles["scratch_roughness"] = diff.inputs[0]
    scaled = math_node("MULTIPLY", diff.outputs["Value"], total)
    final_rough = math_node("ADD", scaled.outputs["Value"], rough.outputs["Result"])
    links.new(final_rough.outputs["Value"], bsdf.inputs["Roughness"])
    return mat, handles


def add_scratch_nodes(nodes, links, coords, math_node):
    """Shader nodes for MAX_SCRATCHES rotated rectangles. Returns the output socket
    "total" (0 = clean, up to 1 = fully scratched) and per-scratch handles."""
    total, strips = None, []
    for _ in range(MAX_SCRATCHES):
        # nut-local (x, y) -> strip-local (u along the length, v across the width)
        to_strip = nodes.new("ShaderNodeMapping")
        links.new(coords.outputs["Object"], to_strip.inputs["Vector"])
        uv = nodes.new("ShaderNodeSeparateXYZ")
        links.new(to_strip.outputs["Vector"], uv.inputs["Vector"])
        # inside = (|u| < length/2) * (|v| < width/2)
        abs_u = math_node("ABSOLUTE", uv.outputs["X"])
        abs_v = math_node("ABSOLUTE", uv.outputs["Y"])
        in_u = math_node("LESS_THAN", abs_u.outputs["Value"], 0.0)
        in_v = math_node("LESS_THAN", abs_v.outputs["Value"], 0.0)
        inside = math_node("MULTIPLY", in_u.outputs["Value"], in_v.outputs["Value"])
        # fine parallel lines: noise stretched along u -> long thin streaks
        stretch = nodes.new("ShaderNodeMapping")
        stretch.inputs["Scale"].default_value = (LINE_STRETCH[0], LINE_STRETCH[1], 1)
        links.new(to_strip.outputs["Vector"], stretch.inputs["Vector"])
        line_noise = nodes.new("ShaderNodeTexNoise")
        line_noise.inputs["Scale"].default_value = 1.0
        line_noise.inputs["Detail"].default_value = 2
        links.new(stretch.outputs["Vector"], line_noise.inputs["Vector"])
        line_ramp = nodes.new("ShaderNodeValToRGB")
        links.new(line_noise.outputs["Fac"], line_ramp.inputs["Fac"])
        # strength = base + line_amount * lines   (band: mostly base, field: mostly lines)
        strength = math_node("MULTIPLY_ADD", line_ramp.outputs["Color"], 0.0, 0.0)
        effect = math_node("MULTIPLY", inside.outputs["Value"], strength.outputs["Value"])
        total = (math_node("MAXIMUM", total, effect.outputs["Value"]).outputs["Value"]
                 if total else effect.outputs["Value"])
        strips.append({"to_strip": to_strip, "stretch": stretch, "line_ramp": line_ramp,
                       "half_length": in_u.inputs[1], "half_width": in_v.inputs[1],
                       "line_amount": strength.inputs[1], "base": strength.inputs[2]})
    return total, strips


def randomize_material(h, p, smooth=False):
    """Apply per-image parameters p to the material nodes h."""
    h["mapping"].inputs["Location"].default_value = p["texture_offset"]
    if h["ring_phase"] is not None:
        h["ring_phase"].default_value = p["texture_offset"][0]
    if h["bump"] is not None:
        h["bump"].inputs["Strength"].default_value = p["grain_strength"]
    if not smooth:
        h["rough"].inputs["To Min"].default_value = p["roughness_min"]
        h["rough"].inputs["To Max"].default_value = p["roughness_max"]
    h["ramp"].color_ramp.elements[0].position = p["speckle_amount"]
    h["ramp"].color_ramp.elements[1].position = p["speckle_amount"] + 0.05
    color = [c * t for c, t in zip(NUT_COLOR, p["tint"])]
    h["mix"].inputs["Color1"].default_value = color + [1]
    h["mix"].inputs["Color2"].default_value = [c * p["speckle_dark"] for c in color] + [1]

    d = p["scratch_darken"]
    h["darken"].inputs["Color2"].default_value = [d, d, d, 1]
    h["scratch_roughness"].default_value = p["scratch_roughness"]
    for k, s in enumerate(h["scratches"]):
        if k >= len(p["defects"]):            # unused slot: zero-size rectangle
            s["half_length"].default_value = 0.0
            s["half_width"].default_value = 0.0
            continue
        sc = p["defects"][k]
        t = math.radians(sc["angle_deg"])
        # Mapping node computes R(rotation) * v + location; we want R(-t) * (v - center)
        s["to_strip"].inputs["Rotation"].default_value = (0, 0, -t)
        s["to_strip"].inputs["Location"].default_value = (
            -(sc["cx"] * math.cos(t) + sc["cy"] * math.sin(t)),
            -(-sc["cx"] * math.sin(t) + sc["cy"] * math.cos(t)), 0)
        s["half_length"].default_value = sc["length"] / 2
        s["half_width"].default_value = sc["width"] / 2
        s["stretch"].inputs["Location"].default_value = (
            p["texture_offset"][1] + 10 * k, p["texture_offset"][2], 0)
        s["line_ramp"].color_ramp.elements[0].position = sc["line_threshold"]
        s["line_ramp"].color_ramp.elements[1].position = sc["line_threshold"] + 0.05
        # field: faint base + strong lines; band: strong uniform matte strip
        base, lines = (0.15, 0.85) if sc["type"] == "field" else (0.85, 0.15)
        s["base"].default_value = base
        s["line_amount"].default_value = lines


def scratch_corners(sc, rotation_deg, z):
    """World-space corners of a scratch rectangle (nut-local -> rotated with the nut)."""
    t = math.radians(sc["angle_deg"])
    u = np.array([math.cos(t), math.sin(t)]) * sc["length"] / 2   # along the length
    v = np.array([-math.sin(t), math.cos(t)]) * sc["width"] / 2   # across the width
    c = np.array([sc["cx"], sc["cy"]])
    local = np.array([c + u + v, c - u + v, c - u - v, c + u - v])
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    world = local @ rot.T
    return np.column_stack([world, np.full(4, z)])


def fill_convex_polygon(corners_px, shape):
    """Boolean image mask of a convex polygon given in pixel coords [(x, y), ...].
    A pixel is inside if it lies on the same side of every edge."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    signs = []
    for (x0, y0), (x1, y1) in zip(corners_px, np.roll(corners_px, -1, axis=0)):
        signs.append((x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0) >= 0)
    signs = np.array(signs)
    return signs.all(axis=0) | (~signs).all(axis=0)  # works for both winding orders


def emission_node(mat):
    return next(n for n in mat.nodes if n.type == "EMISSION")


# --- setup --------------------------------------------------------------------
out_dir = Path(args.out)
if list(out_dir.glob("*.hdf5")):
    raise SystemExit(f"{out_dir}/ already has renders. Use a new --out folder "
                     "(mixing runs would break meta.jsonl and reproducibility).")
out_dir.mkdir(parents=True, exist_ok=True)

bproc.init()
bproc.renderer.set_output_format(view_transform="Standard")  # colors as written, no film curve

# --- nut ----------------------------------------------------------------------
top_z = BODY_T + BOSS_T
body = make_extruded("nut_body", outline_points(), circle_points(R_HOLE), BODY_T, BODY_BEVEL)
body.set_location([0, 0, BODY_T / 2])

# center cross-section, from the hole outward:
#   hole bead (convex quarter circle) -> flat boss top -> S-shaped slope down to the body
hole_bead = arc(R_HOLE + HOLE_FILLET, top_z - HOLE_FILLET, HOLE_FILLET, HOLE_FILLET, 180, 90)
boss_section = ([(R_HOLE + HOLE_FILLET, top_z), (R_BOSS - BOSS_RAMP, top_z)]
                + smooth_step(R_BOSS - BOSS_RAMP, R_BOSS, top_z, BODY_T)[1:])
hole = make_lathe("nut_hole_bead", hole_bead)
boss = make_lathe("nut_boss", boss_section)

metal, metal_h = make_metal_material("nut_metal", ROUGHNESS_RANGE, grain=True, rings=True)
smooth, smooth_h = make_metal_material("nut_hole_metal", (HOLE_ROUGHNESS, HOLE_ROUGHNESS),
                                       grain=False, rings=False)
nut_parts = (body, boss, hole)
for part, mat in zip(nut_parts, (metal, metal, smooth)):
    part.replace_materials(mat)
    part.set_cp("category_id", 1)  # 1 = nut in the segmentation mask

# --- background ---------------------------------------------------------------
# Emissive plane: shows exactly the measured color and ignores lights,
# so no cast shadow (real images show none).
ground = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
ground_mat = bproc.material.create("background")
ground_mat.make_emissive(1.0, replace=True, emission_color=[0, 0, 0, 1])
ground.replace_materials(ground_mat)
ground.set_cp("category_id", 0)
ground_emission = emission_node(ground_mat)

# --- lights -------------------------------------------------------------------
bproc.renderer.set_world_background([1, 1, 1], strength=WORLD_STRENGTH)

# Main light: a glowing flat ring above the nut, like an industrial ring light.
# Unlike uniform world light it has directions, so surface grain shows up as
# bright/dark variation, and slanted edges (bead, bevels) catch it and glow.
ring = make_lathe("ring_light", [(RING_RADII[0], RING_HEIGHT), (RING_RADII[1], RING_HEIGHT)])
ring_mat = bproc.material.create("ring_light")
ring_mat.make_emissive(RING_STRENGTH, replace=True, emission_color=[1, 1, 1, 1])
ring.replace_materials(ring_mat)
ring.blender_obj.visible_camera = False  # light the scene, but don't appear in the image
ring_emission = emission_node(ring_mat)

key = bproc.types.Light()
key.set_type("AREA")
key.blender_obj.data.size = 1.0

# --- camera intrinsics (fixed; distance jitter changes the apparent size) -----
frame_width = 2 * R_TIP / FILL
fov = 2 * math.atan(frame_width / 2 / (CAM_HEIGHT - top_z))
bproc.camera.set_intrinsics_from_blender_params(
    lens=fov, lens_unit="FOV", image_width=IMG_SIZE, image_height=IMG_SIZE,
    clip_start=0.1, clip_end=50)

bproc.renderer.set_max_amount_of_samples(SAMPLES)
bproc.renderer.set_denoiser("INTEL")
bproc.renderer.enable_segmentation_output(map_by=["category_id"],
                                          default_values={"category_id": 0})

# --- render loop --------------------------------------------------------------
t_start = time.time()
with open(out_dir / "meta.jsonl", "a") as meta_file:
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])  # image i is reproducible on its own
        p = sample_params(rng, randomize=not args.no_random)

        for part in nut_parts:
            part.set_rotation_euler([0, 0, math.radians(p["rotation_deg"])])
        randomize_material(metal_h, p)
        randomize_material(smooth_h, p, smooth=True)
        ground_emission.inputs["Color"].default_value = (
            [srgb_to_linear(c + p["bg_shift"]) for c in BG_COLOR_SRGB] + [1])
        ring_emission.inputs["Strength"].default_value = p["ring_strength"]

        ka, ke = math.radians(p["key_angle_deg"]), math.radians(p["key_elevation_deg"])
        key_loc = 3.0 * np.array([math.cos(ke) * math.cos(ka),
                                  math.cos(ke) * math.sin(ka), math.sin(ke)])
        key.set_location(key_loc)
        key.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-key_loc))
        key.set_energy(p["key_energy"])

        # camera on a small cone around the vertical, always aimed at the nut center
        bproc.utility.reset_keyframes()
        tilt, tdir = math.radians(p["cam_tilt_deg"]), math.radians(p["cam_tilt_dir_deg"])
        dist = (CAM_HEIGHT - top_z) * p["cam_dist_scale"]
        poi = np.array([0, 0, top_z])
        cam_loc = poi + dist * np.array([math.sin(tilt) * math.cos(tdir),
                                         math.sin(tilt) * math.sin(tdir), math.cos(tilt)])
        cam_rot = bproc.camera.rotation_from_forward_vec(poi - cam_loc)
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, cam_rot))

        data = bproc.renderer.render()

        # scratch mask: project each rectangle into the image, keep only nut pixels
        seg = data["category_id_segmaps"][0]
        nut_px = seg == 1
        for sc in p["defects"]:
            corners_px = bproc.camera.project_points(
                scratch_corners(sc, p["rotation_deg"], z=BODY_T))
            region = fill_convex_polygon(corners_px, seg.shape) & nut_px
            seg[region] = 2
            ys, xs = np.nonzero(region)
            sc["area_px"] = int(region.sum())
            sc["bbox"] = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                          if len(xs) else None)  # x_min, y_min, x_max, y_max
        data["category_id_segmaps"][0] = seg

        bproc.writer.write_hdf5(str(out_dir), data, append_to_existing_output=True)
        meta_file.write(json.dumps({"index": i, "file": f"{i}.hdf5", "seed": args.seed,
                                    "dataset_version": args.version,
                                    "n_defects": len(p["defects"]), **p}) + "\n")
        meta_file.flush()
        elapsed = time.time() - t_start
        print(f"[generate] {i + 1}/{args.n} done, {elapsed / (i + 1):.1f} s/image", flush=True)
