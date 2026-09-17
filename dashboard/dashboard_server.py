"""Dual-Axis GeoFormer dashboard: training-run history + new-sample intake.

Run:
    uvicorn dashboard.dashboard_server:app --reload --port 8080

Reads/writes the same Neon Postgres DB that train.py optionally logs to
(db/db_logger.py) -- see db/schema.sql for the tables. Requires DATABASE_URL
(picked up from .env in the project root, same as train.py).

Two jobs:
1. Serve every training run + its per-epoch history for charting, with the
   class-coverage collapse flag surfaced explicitly (a bare F1 number can
   hide total class failure -- see docs/MANUAL.md S12.3).
2. Accept new pre/post image-pair uploads ("enter new samples"), store them,
   and run inference against the current best checkpoint -- either
   immediately (if a checkpoint is available and the pair is small) or left
   'pending' for a background pass.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "dashboard" / "sample_uploads"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Dual-Axis GeoFormer Dashboard")


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL"):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DATABASE_URL not set and no .env found -- see README for setup")


def get_conn():
    return psycopg2.connect(_database_url())


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/runs")
def list_runs():
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT r.id, r.run_name, r.model_type, r.data_source, r.notes, r.created_at,
                   COUNT(e.id) AS n_epochs,
                   MAX(e.epoch) AS last_epoch,
                   MIN(e.val_loss) AS best_val_loss
            FROM training_runs r
            LEFT JOIN epoch_logs e ON e.run_id = r.id
            GROUP BY r.id
            ORDER BY r.id
            """
        )
        rows = []
        for r in cur.fetchall():
            row = dict(r)
            if row.get("created_at") is not None:
                row["created_at"] = row["created_at"].isoformat()
            rows.append(row)
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/epochs")
def run_epochs(run_id: int):
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT epoch, train_loss, val_loss,
                   f1_background, f1_building, f1_road, f1_flooded,
                   pred_images_building, pred_images_flooded, lr, epoch_seconds
            FROM epoch_logs WHERE run_id = %s ORDER BY epoch
            """,
            (run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            raise HTTPException(404, f"No epoch data for run_id={run_id}")
        return JSONResponse([dict(r) for r in rows])
    finally:
        conn.close()


@app.get("/api/dataset")
def dataset_stats():
    """Live tile counts straight from real_sn8_dataset_full/index.json --
    not the DB (which only tracks tiles a migration explicitly registered)."""
    index_path = BASE_DIR / "real_sn8_dataset_full" / "index.json"
    if not index_path.exists():
        return JSONResponse({"error": "real_sn8_dataset_full/index.json not found"}, status_code=404)
    data = json.loads(index_path.read_text())
    by_aoi: dict[str, int] = {}
    for entry in data:
        # BUG THIS WORKS AROUND: the earliest 202 tiles (Germany_Training_Public)
        # were downloaded before prepare_real_data.py started recording an
        # "aoi" field per entry, so they have no "aoi" key at all -- not an
        # empty string, absent. Since this dataset only ever has two AOIs and
        # every entry that DOES carry "aoi" says Louisiana-East, the missing
        # ones are Germany by elimination; label them as such rather than
        # lumping them into an opaque "unknown" bucket.
        aoi = entry.get("aoi", "Germany_Training_Public (legacy entry, predates the aoi field)")
        by_aoi[aoi] = by_aoi.get(aoi, 0) + 1
    return JSONResponse({"total_tiles": len(data), "by_aoi": by_aoi})


@app.post("/api/samples")
async def submit_sample(pre: UploadFile = File(...), post: UploadFile = File(...)):
    sample_uuid = uuid.uuid4().hex[:12]
    pre_path = SAMPLES_DIR / f"{sample_uuid}_pre_{pre.filename}"
    post_path = SAMPLES_DIR / f"{sample_uuid}_post_{post.filename}"
    with pre_path.open("wb") as f:
        shutil.copyfileobj(pre.file, f)
    with post_path.open("wb") as f:
        shutil.copyfileobj(post.file, f)

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO samples (pre_filename, post_filename, status)
            VALUES (%s, %s, 'pending') RETURNING id
            """,
            (str(pre_path.relative_to(BASE_DIR)), str(post_path.relative_to(BASE_DIR))),
        )
        sample_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    return JSONResponse({"sample_id": sample_id, "status": "pending",
                          "note": "Saved. Run `python dashboard/process_samples.py` to run inference on pending samples."})


@app.get("/api/samples")
def list_samples():
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM samples ORDER BY id DESC LIMIT 100")
        rows = []
        for r in cur.fetchall():
            row = dict(r)
            for key in ("submitted_at", "processed_at"):
                if row.get(key) is not None:
                    row[key] = row[key].isoformat()
            rows.append(row)
        return JSONResponse(rows)
    finally:
        conn.close()


app.mount("/uploads", StaticFiles(directory=str(SAMPLES_DIR)), name="uploads")
