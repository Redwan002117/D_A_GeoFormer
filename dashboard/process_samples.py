"""Run inference against every 'pending' sample submitted through the
dashboard, using the current checkpoints/best.pt (or --checkpoint).

Since dashboard_server.py runs inference live on submission (see
dashboard/inference.py), a sample only ends up 'pending' here if live
inference wasn't available at submit time (e.g. no checkpoint existed
yet) -- this is now the manual backfill/retry path, not the primary one.

Usage:
    python dashboard/process_samples.py
    python dashboard/process_samples.py --checkpoint checkpoints_baseline/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dashboard.inference import run_inference  # noqa: E402
from db.db_logger import _load_database_url  # noqa: E402

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()

    url = _load_database_url()
    if not url:
        sys.exit("DATABASE_URL not set -- see README / .env")
    conn = psycopg2.connect(url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM samples WHERE status = 'pending' ORDER BY id")
    pending = cur.fetchall()
    print(f"{len(pending)} pending sample(s)")

    for row in pending:
        sample_id = row["id"]
        try:
            pre_path = BASE_DIR / row["pre_filename"]
            post_path = BASE_DIR / row["post_filename"]
            result, png_bytes = run_inference(pre_path, post_path, args.checkpoint, args.image_size)

            sample_uuid = Path(row["pre_filename"]).stem.split("_pre_")[0]
            overlay_path = pre_path.parent / f"{sample_uuid}_overlay.png"
            overlay_path.write_bytes(png_bytes)

            write_cur = conn.cursor()
            write_cur.execute(
                """
                UPDATE samples
                SET status = 'processed', checkpoint_used = %s, result_json = %s,
                    result_image = %s, processed_at = now()
                WHERE id = %s
                """,
                (str(args.checkpoint), json.dumps(result), str(overlay_path.relative_to(BASE_DIR)), sample_id),
            )
            conn.commit()
            print(f"sample {sample_id}: {result['predicted_classes_present']}")
        except Exception as e:  # noqa: BLE001
            # BUG THIS FIXES: if the exception came from the 'processed'
            # UPDATE above (a DB-side error, not an image/inference one),
            # the connection's transaction is left ABORTED and this
            # recovery UPDATE would hit the same "current transaction is
            # aborted" failure -- uncaught, crashing the whole script and
            # abandoning every remaining pending sample in the loop as
            # permanently 'pending'. Same bug class just fixed in
            # db/db_logger.py's log_epoch; rollback() first so the
            # recovery write (and every sample after it) can still land.
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001 - connection may be fully dead
                pass
            write_cur = conn.cursor()
            write_cur.execute(
                "UPDATE samples SET status = 'failed', error_message = %s WHERE id = %s",
                (str(e), sample_id),
            )
            conn.commit()
            print(f"sample {sample_id}: FAILED -- {e}")

    conn.close()


if __name__ == "__main__":
    main()
