"""Optional Postgres (Neon) logging for train.py.

Entirely best-effort: if DATABASE_URL isn't set, psycopg2 isn't installed, or
any write fails (network blip, DB asleep), training itself must never stop
because of it -- the CSV log (train.py's own incremental write) is always the
source of truth for a run in progress. This module only ever adds a second,
queryable copy for the dashboard.
"""
from __future__ import annotations

import json
import os

try:
    import psycopg2
except ImportError:  # pragma: no cover - optional dependency
    psycopg2 = None


def _load_database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("DATABASE_URL"):
                return line.strip().split("=", 1)[1]
    return None


class DBLogger:
    """None-safe: construct it unconditionally, call .log_epoch() every
    epoch, and it silently no-ops if there's no DB configured or reachable."""

    def __init__(self, run_name: str, model_type: str, data_source: str, config_dict: dict | None,
                 parent_run_name: str | None = None):
        self.enabled = False
        self.run_id = None
        if psycopg2 is None:
            print("[db_logger] psycopg2 not installed -- skipping Postgres logging (CSV log is unaffected)")
            return
        url = _load_database_url()
        if not url:
            return  # silent: DB logging is opt-in, not required
        try:
            self.conn = psycopg2.connect(url, connect_timeout=5)
            cur = self.conn.cursor()
            # parent_run_name (the run_name recorded in the --resume'd
            # checkpoint, if any -- see train.py) is looked up to an id
            # here rather than passed as one directly, since the caller
            # only ever has the checkpoint's own recorded name, not
            # Postgres ids. A name that isn't found (e.g. logged only to
            # CSV, never to Postgres) leaves parent_run_id NULL -- the
            # dashboard's lineage chain just starts from this run instead
            # of erroring.
            parent_run_id = None
            if parent_run_name:
                cur.execute("SELECT id FROM training_runs WHERE run_name = %s", (parent_run_name,))
                row = cur.fetchone()
                if row:
                    parent_run_id = row[0]
                else:
                    print(f"[db_logger] parent run '{parent_run_name}' not found in Postgres -- "
                          f"lineage chain will start from this run instead")
            cur.execute(
                """
                INSERT INTO training_runs (run_name, model_type, data_source, config_json, parent_run_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_name) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    data_source = EXCLUDED.data_source,
                    config_json = EXCLUDED.config_json,
                    parent_run_id = COALESCE(training_runs.parent_run_id, EXCLUDED.parent_run_id)
                RETURNING id
                """,
                (run_name, model_type, data_source, json.dumps(config_dict) if config_dict else None, parent_run_id),
            )
            self.run_id = cur.fetchone()[0]
            self.conn.commit()
            self.enabled = True
            print(f"[db_logger] logging run '{run_name}' to Neon Postgres (run_id={self.run_id})")
        except Exception as e:  # noqa: BLE001 - genuinely must never crash training
            print(f"[db_logger] could not connect to Neon Postgres, continuing without it: {e}")
            self.enabled = False

    def _write_epoch(self, row: dict) -> None:
        cur = self.conn.cursor()
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
                self.run_id, row["epoch"], row["train_loss"], row["val_loss"],
                row["val_f1_background"], row["val_f1_building"], row["val_f1_road"], row["val_f1_flooded"],
                row["val_coverage_background_pred_images"], row["val_coverage_building_pred_images"],
                row["val_coverage_road_pred_images"], row["val_coverage_flooded_pred_images"],
                row["lr"], row["seconds"],
            ),
        )
        self.conn.commit()

    def log_epoch(self, row: dict) -> None:
        if not self.enabled:
            return
        try:
            self._write_epoch(row)
            return
        except Exception as e:  # noqa: BLE001
            # BUG THIS FIXES (part 1, already landed): a failed query leaves
            # the connection's transaction ABORTED until an explicit
            # ROLLBACK. rollback() alone isn't enough though -- confirmed on
            # this exact project's own v7 run: epoch 1 logged successfully,
            # then every one of epochs 2-6 silently failed, because Neon's
            # serverless compute suspends an idle connection outright (this
            # project's epochs take 400-650s each, comfortably past Neon's
            # idle-suspend window) -- rollback() on an already-DEAD
            # connection just raises again and is swallowed, so the ONE
            # long-lived self.conn opened at __init__ never worked again for
            # the rest of the run. A fresh connect() (not just rollback) is
            # the actual fix; retry the write once on the new connection
            # before giving up on this epoch.
            print(f"[db_logger] epoch {row.get('epoch')} write failed ({e}), reconnecting and retrying once")
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.conn = psycopg2.connect(_load_database_url(), connect_timeout=5)
                self._write_epoch(row)
                print(f"[db_logger] epoch {row.get('epoch')} logged after reconnecting")
            except Exception as e2:  # noqa: BLE001
                print(f"[db_logger] epoch {row.get('epoch')} still failed after reconnecting, continuing without it: {e2}")
