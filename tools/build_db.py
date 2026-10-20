"""Build db/meta.sqlite from the meta.jsonl files written by render/generate.py.

Usage (repo root, venv active):
    python tools/build_db.py data/synth/synth_v1
    python tools/build_db.py output/gen_test2 --db output/test.sqlite
The database is rebuilt from scratch every time (meta.jsonl is the source of truth).
Split: every 10th image (idx % 10 == 0) -> val, the rest -> train. Parameters are
drawn independently per image, so this is as good as a random split, and stable.
"""
import argparse
import json
import sqlite3
from pathlib import Path

IMAGE_PARAMS = ["rotation_deg", "cam_tilt_deg", "cam_dist_scale", "ring_strength",
                "key_angle_deg", "key_elevation_deg", "key_energy", "roughness_min",
                "roughness_max", "grain_strength", "speckle_amount", "speckle_dark",
                "bg_shift", "scratch_darken", "scratch_roughness"]
VAL_EVERY = 10

parser = argparse.ArgumentParser()
parser.add_argument("folders", nargs="+", type=Path, help="render output folders")
parser.add_argument("--db", type=Path, default=Path("db/meta.sqlite"))
args = parser.parse_args()

args.db.unlink(missing_ok=True)
con = sqlite3.connect(args.db)
con.executescript(Path("db/schema.sql").read_text())

n_images = n_defects = 0
for folder in args.folders:
    for line in (folder / "meta.jsonl").read_text().splitlines():
        m = json.loads(line)
        split = "val" if m["index"] % VAL_EVERY == 0 else "train"
        cols = (["dataset_version", "path", "seed", "idx", "split", "n_defects"] + IMAGE_PARAMS
                + ["params_json"])
        params = {k: v for k, v in m.items() if k != "defects"}
        values = ([m["dataset_version"], str(folder / m["file"]), m["seed"], m["index"],
                   split, m["n_defects"]] + [m.get(k) for k in IMAGE_PARAMS]   # NULL if absent
                  + [json.dumps(params)])
        cur = con.execute(f"INSERT INTO images ({', '.join(cols)}) "
                          f"VALUES ({', '.join('?' * len(cols))})", values)
        image_id = cur.lastrowid
        for d in m["defects"]:
            bbox = d["bbox"] or [None] * 4
            con.execute("INSERT INTO defects (image_id, type, cx, cy, angle_deg, length, "
                        "width, area_px, bbox_x0, bbox_y0, bbox_x1, bbox_y1) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [image_id, d["type"], d["cx"], d["cy"], d["angle_deg"],
                         d["length"], d["width"], d["area_px"], *bbox])
            n_defects += 1
        n_images += 1
con.commit()
print(f"wrote {args.db}: {n_images} images, {n_defects} defects")
