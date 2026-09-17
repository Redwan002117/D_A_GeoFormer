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

    def __init__(self, run_name: str, model_type: str, data_source: str, config_dict: dict | None):
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
            cur.execute(
                """
                INSERT INTO training_runs (run_name, model_type, data_source, config_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_name) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    data_source = EXCLUDED.data_source,
                    config_json = EXCLUDED.config_json
                RETURNING id
                """,
                (run_name, model_type, data_source, json.dumps(config_dict) if config_dict else None),
            )
            self.run_id = cur.fetchone()[0]
            self.conn.commit()
            self.enabled = True
            print(f"[db_logger] logging run '{run_name}' to Neon Postgres (run_id={self.run_id})")
        except Exception as e:  # noqa: BLE001 - genuinely must never crash training
            print(f"[db_logger] could not connect to Neon Postgres, continuing without it: {e}")
            self.enabled = False

    def log_epoch(self, row: dict) -> None:
        if not self.enabled:
            return
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO epoch_logs (
                    run_id, epoch, train_loss, val_loss,
                    f1_background, f1_building, f1_road, f1_flooded,
                    pred_images_building, pred_images_flooded, lr, epoch_seconds
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, epoch) DO UPDATE SET
                    train_loss = EXCLUDED.train_loss,
                    val_loss = EXCLUDED.val_loss,
                    f1_background = EXCLUDED.f1_background,
                    f1_building = EXCLUDED.f1_building,
                    f1_road = EXCLUDED.f1_road,
                    f1_flooded = EXCLUDED.f1_flooded,
                    pred_images_building = EXCLUDED.pred_images_building,
                    pred_images_flooded = EXCLUDED.pred_images_flooded,
                    lr = EXCLUDED.lr,
                    epoch_seconds = EXCLUDED.epoch_seconds
                """,
                (
                    self.run_id, row["epoch"], row["train_loss"], row["val_loss"],
                    row["val_f1_background"], row["val_f1_building"], row["val_f1_road"], row["val_f1_flooded"],
                    row["val_coverage_building_pred_images"], row["val_coverage_flooded_pred_images"],
                    row["lr"], row["seconds"],
                ),
            )
            self.conn.commit()
        except Exception as e:  # noqa: BLE001
            # BUG THIS FIXES: a failed query leaves the connection's
            # transaction in an ABORTED state until an explicit ROLLBACK --
            # Postgres refuses every subsequent command on that connection
            # ("current transaction is aborted") until then. Without this,
            # a single transient failure (a network blip, Neon's compute
            # briefly suspended/waking) would silently break DB logging for
            # every remaining epoch of the run, not just the one that hit
            # the blip -- confirmed by reproducing it directly against the
            # live DB: one failed query, then a plain `SELECT 1` on the
            # same connection also failed with InFailedSqlTransaction.
            try:
                self.conn.rollback()
            except Exception:  # noqa: BLE001 - the connection may be fully dead; give up quietly
                pass
            print(f"[db_logger] epoch {row.get('epoch')} write failed, continuing without it: {e}")
