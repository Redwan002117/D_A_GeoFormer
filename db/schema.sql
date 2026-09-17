-- Dual-Axis GeoFormer dashboard schema (Neon Postgres)

CREATE TABLE IF NOT EXISTS training_runs (
    id            SERIAL PRIMARY KEY,
    run_name      TEXT UNIQUE NOT NULL,       -- e.g. "training_log_geoformer_801.csv"
    model_type    TEXT NOT NULL,              -- 'geoformer' | 'baseline'
    data_source   TEXT,                       -- dataset dir or 'synthetic'
    config_json   JSONB,                      -- GeoFormerConfig, null for baseline
    notes         TEXT,
    -- False for runs logged before ConfusionAccumulator (see MANUAL.md
    -- S12.3): their F1 numbers are per-batch-averaged and can read as
    -- deceptively high on rare classes even when the model never predicts
    -- them anywhere. The dashboard's "best F1 ever" stats must exclude
    -- these -- surfacing a metric-averaging artifact as the project's best
    -- real result would be exactly the kind of false positive this
    -- project exists to catch. Default true (a normal, trustworthy run).
    metric_trustworthy BOOLEAN NOT NULL DEFAULT true,
    -- The run this one --resume'd from, if any -- e.g. v13 resumed from
    -- v12's checkpoint, so v13.parent_run_id = v12's id. NULL for a
    -- from-scratch run. Lets the dashboard show a resumed run's FULL
    -- lineage (epoch 1 through its own latest epoch), not just its own
    -- epoch_logs rows, which only start wherever it resumed from --
    -- otherwise "Latest run" looked like it had no history before that
    -- point, even though it's a real continuation of earlier training.
    -- A resume chain can branch (e.g. v11 and v12 both resumed from v10),
    -- so this is a tree, not a line -- walking parent_run_id from any one
    -- run gives THAT run's own real ancestry, not siblings' branches.
    parent_run_id INTEGER REFERENCES training_runs(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS epoch_logs (
    id                  SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch               INTEGER NOT NULL,
    train_loss          DOUBLE PRECISION,
    val_loss            DOUBLE PRECISION,
    f1_background       DOUBLE PRECISION,
    f1_building         DOUBLE PRECISION,
    f1_road             DOUBLE PRECISION,
    f1_flooded          DOUBLE PRECISION,
    pred_images_background INTEGER,
    pred_images_building INTEGER,
    pred_images_road      INTEGER,
    pred_images_flooded  INTEGER,
    gt_images_building    INTEGER,
    gt_images_flooded     INTEGER,
    lr                  DOUBLE PRECISION,
    epoch_seconds        DOUBLE PRECISION,
    logged_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_epoch_logs_run_id ON epoch_logs(run_id);

CREATE TABLE IF NOT EXISTS samples (
    id              SERIAL PRIMARY KEY,
    pre_filename    TEXT NOT NULL,
    post_filename   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processed | failed
    submitted_by    TEXT,
    checkpoint_used TEXT,
    result_json     JSONB,           -- per-class prediction coverage / summary once processed
    result_image    TEXT,            -- path to saved prediction panel, once processed
    error_message   TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dataset_tiles (
    id          SERIAL PRIMARY KEY,
    aoi         TEXT NOT NULL,
    tile_id     TEXT NOT NULL,
    split       TEXT,             -- 'train' | 'val', from SpaceNet8Dataset.split()
    has_building BOOLEAN,
    has_road     BOOLEAN,
    has_flooded  BOOLEAN,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (aoi, tile_id)
);
