-- Sanity-check queries on db/meta.sqlite. tools/data_stats.py runs each block
-- (split on "-- name:") and writes the results to reports/analysis/data_stats.md.

-- name: 1. Split balance
-- Are train and val similar? Same defect-free share and same scratches per image.
SELECT split,
       COUNT(*)                                            AS images,
       ROUND(100.0 * AVG(n_defects = 0), 1)                AS pct_defect_free,
       ROUND(AVG(n_defects), 2)                            AS avg_scratches
FROM images
GROUP BY split;

-- name: 2. Defective vs defect-free
-- Target: 30-40% defect-free (roadmap); generator draws 35%.
SELECT n_defects,
       COUNT(*)                                            AS images,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM images), 1) AS pct
FROM images
GROUP BY n_defects
ORDER BY n_defects;

-- name: 3. Scratch area distribution by type
-- Visible mask area as % of the 512x512 image. Real seen_pool: 1.9-11.9%, median ~4%.
SELECT type,
       COUNT(*)                                            AS scratches,
       ROUND(100.0 * MIN(area_px) / (512 * 512), 2)        AS min_pct,
       ROUND(100.0 * AVG(area_px) / (512 * 512), 2)        AS avg_pct,
       ROUND(100.0 * MAX(area_px) / (512 * 512), 2)        AS max_pct
FROM defects
GROUP BY type;

-- name: 4. Empty or tiny masks
-- A scratch drawn mostly over the hole/background leaves few or no mask pixels.
-- Such labels are noise: the image "has a scratch" but almost nothing to see.
SELECT CASE WHEN area_px = 0 THEN 'empty (0 px)'
            WHEN area_px < 200 THEN 'tiny (<200 px)'
            ELSE 'ok (>=200 px)' END                       AS mask_size,
       COUNT(*)                                            AS scratches
FROM defects
GROUP BY mask_size
ORDER BY mask_size;

-- name: 5. Parameter coverage
-- Do the random parameters cover their full intended ranges?
SELECT 'rotation_deg'      AS parameter, ROUND(MIN(rotation_deg), 2) AS min, ROUND(MAX(rotation_deg), 2) AS max, ROUND(AVG(rotation_deg), 2) AS avg FROM images
UNION ALL SELECT 'ring_strength',     ROUND(MIN(ring_strength), 3),     ROUND(MAX(ring_strength), 3),     ROUND(AVG(ring_strength), 3)     FROM images
UNION ALL SELECT 'key_energy',        ROUND(MIN(key_energy), 2),        ROUND(MAX(key_energy), 2),        ROUND(AVG(key_energy), 2)        FROM images
UNION ALL SELECT 'cam_tilt_deg',      ROUND(MIN(cam_tilt_deg), 2),      ROUND(MAX(cam_tilt_deg), 2),      ROUND(AVG(cam_tilt_deg), 2)      FROM images
UNION ALL SELECT 'scratch_darken',    ROUND(MIN(scratch_darken), 3),    ROUND(MAX(scratch_darken), 3),    ROUND(AVG(scratch_darken), 3)    FROM images
UNION ALL SELECT 'scratch_roughness', ROUND(MIN(scratch_roughness), 3), ROUND(MAX(scratch_roughness), 3), ROUND(AVG(scratch_roughness), 3) FROM images;
