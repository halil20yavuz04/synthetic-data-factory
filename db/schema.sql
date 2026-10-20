-- Metadata for rendered synthetic images (filled by tools/build_db.py from meta.jsonl).
-- One row per image in `images`, one row per scratch in `defects`.

CREATE TABLE images (
    id                INTEGER PRIMARY KEY,
    dataset_version   TEXT NOT NULL,     -- e.g. synth_v1
    path              TEXT NOT NULL,     -- hdf5 file, relative to the repo root
    seed              INTEGER NOT NULL,  -- master seed of the run
    idx               INTEGER NOT NULL,  -- image index in the run (RNG seed = [seed, idx])
    split             TEXT NOT NULL,     -- train / val
    n_defects         INTEGER NOT NULL,  -- number of scratches drawn (0-3)
    -- render parameters (render/generate.py, sample_params)
    rotation_deg      REAL, cam_tilt_deg REAL, cam_dist_scale REAL,
    ring_strength     REAL, key_angle_deg REAL, key_elevation_deg REAL, key_energy REAL,
    roughness_min     REAL, roughness_max REAL, grain_strength REAL,
    speckle_amount    REAL, speckle_dark REAL, bg_shift INTEGER,
    scratch_darken    REAL, scratch_roughness REAL,
    -- every parameter of the image as JSON (versions randomize different ones;
    -- columns above are NULL when a version does not have them)
    params_json       TEXT,
    UNIQUE (dataset_version, idx)
);

CREATE TABLE defects (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id),
    type        TEXT NOT NULL,           -- field (hairlines) / band (wide strip)
    cx REAL, cy REAL, angle_deg REAL,    -- nut-local geometry, units of D
    length REAL, width REAL,
    area_px     INTEGER NOT NULL,        -- visible pixels in the mask (on the nut only)
    bbox_x0 INTEGER, bbox_y0 INTEGER, bbox_x1 INTEGER, bbox_y1 INTEGER  -- NULL if area 0
);

CREATE INDEX idx_defects_image ON defects(image_id);
