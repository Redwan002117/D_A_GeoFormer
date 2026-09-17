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

Also deployable read-only to Vercel (see docs/VERCEL_DEPLOYMENT.md) for the
run-history/analytics half of this file -- Vercel's filesystem is read-only
outside /tmp, so upload storage doesn't work there and is disabled rather
than silently broken; inference (dashboard/process_samples.py) needs a real
GPU/CPU + the checkpoint file anyway and was never going to run in a
serverless function. IS_VERCEL detects this automatically (Vercel sets the
VERCEL env var on every deployment) -- nothing to configure by hand.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.environ.get("VERCEL"))
SAMPLES_DIR = Path("/tmp/sample_uploads") if IS_VERCEL else BASE_DIR / "dashboard" / "sample_uploads"
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


def _safe_upload_filename(filename: str | None) -> str:
    """Strip an uploaded file's name down to a safe basename before it ever
    touches a filesystem path.

    BUG THIS FIXES (real, exploitable): submit_sample() used to build
    `SAMPLES_DIR / f"{uuid}_pre_{pre.filename}"` directly from the
    client-supplied multipart filename with no sanitization. pathlib's `/`
    operator treats any "/" or "\\" in that string as real path separators,
    so a crafted filename like "../../../../evil.txt" resolves OUTSIDE
    SAMPLES_DIR entirely (confirmed: it lands in the project root, one
    level up from where uploads are supposed to live) -- a path-traversal
    arbitrary-file-write via the public upload form. Keep only the
    basename (Path(...).name already discards any directory components,
    "..” included) and further restrict it to a safe character set so no
    other path-meaningful character can sneak through either.
    """
    name = Path(filename or "upload").name  # discards any directory components, "." and ".." included
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload"


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
                   MIN(e.val_loss) AS best_val_loss,
                   -- Latest epoch's own F1/coverage, via a LATERAL join on
                   -- the same indexed (run_id, epoch) pair -- lets the
                   -- frontend's "quick view" render straight from this one
                   -- query instead of fetching every run's full epoch
                   -- history (an N+1 pattern the dashboard used to do).
                   latest.f1_building, latest.f1_road, latest.f1_flooded,
                   latest.pred_images_building, latest.pred_images_road, latest.pred_images_flooded
            FROM training_runs r
            LEFT JOIN epoch_logs e ON e.run_id = r.id
            LEFT JOIN LATERAL (
                SELECT f1_building, f1_road, f1_flooded,
                       pred_images_building, pred_images_road, pred_images_flooded
                FROM epoch_logs
                WHERE run_id = r.id
                ORDER BY epoch DESC LIMIT 1
            ) latest ON true
            GROUP BY r.id, latest.f1_building, latest.f1_road, latest.f1_flooded,
                     latest.pred_images_building, latest.pred_images_road, latest.pred_images_flooded
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
                   pred_images_background, pred_images_building, pred_images_road, pred_images_flooded,
                   lr, epoch_seconds
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


@app.get("/api/stats/best")
def best_stats():
    """The best F1 any epoch of any run ever reached, per class -- one
    small indexed query per class instead of the dashboard's old approach
    (fetch every run's full epoch history to compute this client-side,
    an N+1 pattern that got slower every time a new run was added).

    BUG THIS FIXES (found by actually reading this endpoint's own output,
    not assumed correct because the query looked right): the first version
    had no WHERE on training_runs.metric_trustworthy, so it surfaced
    "building F1 0.999" from training_log_finetune256.csv -- a run logged
    BEFORE ConfusionAccumulator (see MANUAL.md S12.3), whose F1 numbers
    are a per-batch-averaging artifact, not real detection. Presenting
    that as this project's best real result would be exactly the kind of
    false positive the whole project exists to catch.
    """
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        result = {}
        for cls in ("building", "road", "flooded"):
            cur.execute(
                f"""
                SELECT e.epoch, e.f1_{cls} AS f1, r.run_name
                FROM epoch_logs e JOIN training_runs r ON r.id = e.run_id
                WHERE e.f1_{cls} IS NOT NULL AND r.metric_trustworthy
                ORDER BY e.f1_{cls} DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            result[cls] = dict(row) if row else None
        return JSONResponse(result)
    finally:
        conn.close()


@app.get("/api/config")
def config():
    """Tells the frontend what this deployment can actually do, so it can
    hide/disable the upload form instead of offering something that will
    fail -- see IS_VERCEL above."""
    return JSONResponse({"uploads_enabled": not IS_VERCEL})


@app.get("/api/dataset")
def dataset_stats():
    """Tile counts by AOI. Tries Postgres first (db/sync_dataset_tiles.py --
    works from anywhere, including a deployment with no access to the local
    dataset directory, e.g. Vercel) and falls back to reading
    real_sn8_dataset_full/index.json directly if that table is empty (so a
    fresh local checkout works before anyone's run the sync script)."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT aoi, COUNT(*) FROM dataset_tiles GROUP BY aoi ORDER BY 2 DESC")
        rows = cur.fetchall()
        if rows:
            by_aoi = {aoi: n for aoi, n in rows}
            return JSONResponse({"total_tiles": sum(by_aoi.values()), "by_aoi": by_aoi, "source": "postgres"})
    finally:
        conn.close()

    index_path = BASE_DIR / "real_sn8_dataset_full" / "index.json"
    if not index_path.exists():
        return JSONResponse(
            {"error": "No dataset_tiles rows in Postgres, and real_sn8_dataset_full/index.json not "
                      "found locally either. Run prepare_real_data.py, then db/sync_dataset_tiles.py."},
            status_code=404,
        )
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
    return JSONResponse({"total_tiles": len(data), "by_aoi": by_aoi, "source": "local_index_json"})


@app.post("/api/samples")
async def submit_sample(pre: UploadFile = File(...), post: UploadFile = File(...)):
    if IS_VERCEL:
        # /tmp exists on Vercel (so the app itself doesn't crash on import --
        # see SAMPLES_DIR above) but it's per-invocation and NOT shared
        # across requests or instances -- a file saved here can vanish
        # before dashboard/process_samples.py (which needs a real
        # filesystem + the checkpoint + PyTorch anyway) ever gets to read
        # it. Refuse cleanly here rather than accept an upload that quietly
        # disappears -- see docs/VERCEL_DEPLOYMENT.md for the real path
        # (run the dashboard locally, or point uploads at real object
        # storage -- not built here, named honestly as future work).
        raise HTTPException(
            501,
            "Sample uploads aren't supported on this deployment (no persistent storage here). "
            "Run the dashboard locally (see docs/VERCEL_DEPLOYMENT.md) to submit new samples.",
        )
    sample_uuid = uuid.uuid4().hex[:12]
    pre_path = SAMPLES_DIR / f"{sample_uuid}_pre_{_safe_upload_filename(pre.filename)}"
    post_path = SAMPLES_DIR / f"{sample_uuid}_post_{_safe_upload_filename(post.filename)}"
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
