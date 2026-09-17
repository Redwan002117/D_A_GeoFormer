"""One-time backfill: load every existing training_log_*.csv into Neon Postgres.

Run once (safe to re-run -- upserts on (run_id, epoch)):
    python db/migrate_csv_logs.py
"""
import csv
import os
import sys

import psycopg2
from psycopg2.extras import Json

# (csv filename, model_type, data_source, notes)
# notes flag runs logged before ConfusionAccumulator (see MANUAL.md S12.3) --
# their val_f1_building / val_f1_flooded numbers are per-batch-averaged and can
# mask total class collapse. Only training_log_geoformer_801.csv used the
# corrected metric.
RUNS = [
    ("training_log.csv", "geoformer", "synthetic",
     "Pre-ConfusionAccumulator metric (per-batch F1 averaging) -- superseded, see MANUAL.md S12.3."),
    ("training_log_real.csv", "geoformer", "real_sn8_dataset (20 tiles)",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_combined.csv", "geoformer", "real_sn8_dataset (Germany+Louisiana-East, higher beta)",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_real_full.csv", "geoformer", "real_sn8_dataset_full (801 tiles, first pass)",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_real_full_test.csv", "geoformer", "real_sn8_dataset_full (801 tiles, test)",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_finetune256.csv", "geoformer", "real_sn8_dataset_full (256px finetune)",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_baseline_real.csv", "baseline", "real_sn8_dataset_full",
     "Pre-ConfusionAccumulator metric -- superseded."),
    ("training_log_geoformer_801.csv", "geoformer", "real_sn8_dataset_full (801 tiles, stable split)",
     "Corrected metric: ConfusionAccumulator (global TP/FP/FN, not per-batch averaged) + "
     "SpaceNet8Dataset.split() stable train/val split. Epochs 1-104 real; 105-106 the "
     "--oversample-rare-classes regression that destroyed the epoch-104 checkpoint, see MANUAL.md S12.7."),
    ("training_log_geoformer_801_v2.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "Restarted from epoch 0 with gentler oversampling (2x/3x) after v1's regression. "
     "Building/flooded collapsed by epoch 4 anyway -- see MANUAL.md S12.7."),
    ("training_log_geoformer_801_v3.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "Oversampling + aggressive class-weighted loss (1,3,1,8) -- traded building/flooded's "
     "collapse for road's instead. See MANUAL.md S12.9."),
    ("training_log_geoformer_801_v4.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "Rebalanced weights (1,2,2,4). road never collapsed across 12 epochs; building collapsed "
     "epoch 4 on. See MANUAL.md S12.10."),
    ("training_log_geoformer_801_v5.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "First run with an ImageNet-pretrained efficientnet_b0 backbone. Epoch 1-2 produced this "
     "project's first-ever simultaneous real predictions for all 4 classes (building F1 0.49, "
     "flooded F1 0.14) -- lost to a val_loss-based best.pt bug, see MANUAL.md S12.11-S12.12."),
    ("training_log_geoformer_801_v6.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "Same as v5 plus --checkpoint-metric min_f1 (the fix for the v5 best.pt bug). Stopped "
     "after 1 epoch once the fix was confirmed working, in favor of also adding backbone "
     "freezing (v7). See MANUAL.md S12.12."),
    ("training_log_geoformer_801_v7.csv", "geoformer", "real_sn8_dataset_full (801 tiles)",
     "v6 plus --freeze-backbone-epochs 3 -- verified ~1.6x faster per step while frozen. "
     "Current run."),
]


def get_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("DATABASE_URL"):
                    url = line.strip().split("=", 1)[1]
    if not url:
        sys.exit("DATABASE_URL not set and no .env found")
    return psycopg2.connect(url)


def migrate():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    conn = get_conn()
    cur = conn.cursor()

    for filename, model_type, data_source, notes in RUNS:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"skip (not found): {filename}")
            continue

        cur.execute(
            """
            INSERT INTO training_runs (run_name, model_type, data_source, notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (run_name) DO UPDATE SET
                model_type = EXCLUDED.model_type,
                data_source = EXCLUDED.data_source,
                notes = EXCLUDED.notes
            RETURNING id
            """,
            (filename, model_type, data_source, notes),
        )
        run_id = cur.fetchone()[0]

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            n = 0
            for row in reader:
                cur.execute(
                    """
                    INSERT INTO epoch_logs (
                        run_id, epoch, train_loss, val_loss,
                        f1_background, f1_building, f1_road, f1_flooded,
                        pred_images_background, pred_images_building, pred_images_road, pred_images_flooded,
                        lr, epoch_seconds
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (run_id, epoch) DO UPDATE SET
                        train_loss = EXCLUDED.train_loss,
                        val_loss = EXCLUDED.val_loss,
                        f1_background = EXCLUDED.f1_background,
                        f1_building = EXCLUDED.f1_building,
                        f1_road = EXCLUDED.f1_road,
                        f1_flooded = EXCLUDED.f1_flooded,
                        pred_images_background = EXCLUDED.pred_images_background,
                        pred_images_building = EXCLUDED.pred_images_building,
                        pred_images_road = EXCLUDED.pred_images_road,
                        pred_images_flooded = EXCLUDED.pred_images_flooded,
                        lr = EXCLUDED.lr,
                        epoch_seconds = EXCLUDED.epoch_seconds
                    """,
                    (
                        run_id,
                        int(row["epoch"]),
                        float(row["train_loss"]),
                        float(row["val_loss"]),
                        float(row["val_f1_background"]),
                        float(row["val_f1_building"]),
                        float(row["val_f1_road"]),
                        float(row["val_f1_flooded"]),
                        int(row["val_coverage_background_pred_images"]) if "val_coverage_background_pred_images" in row else None,
                        int(row["val_coverage_building_pred_images"]) if "val_coverage_building_pred_images" in row else None,
                        int(row["val_coverage_road_pred_images"]) if "val_coverage_road_pred_images" in row else None,
                        int(row["val_coverage_flooded_pred_images"]) if "val_coverage_flooded_pred_images" in row else None,
                        float(row["lr"]),
                        float(row["seconds"]),
                    ),
                )
                n += 1
        conn.commit()
        print(f"migrated {filename}: {n} epoch rows (run_id={run_id})")

    cur.close()
    conn.close()


if __name__ == "__main__":
    migrate()
