"""Run inference against every 'pending' sample submitted through the
dashboard, using the current checkpoints/best.pt (or --checkpoint).

Usage:
    python dashboard/process_samples.py
    python dashboard/process_samples.py --checkpoint checkpoints_baseline/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from checkpoint_utils import load_checkpoint_model  # noqa: E402
from db.db_logger import _load_database_url  # noqa: E402

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(BASE_DIR / args.checkpoint, device)
    model.eval()
    class_names = ["background", "building", "road", "flooded"]

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
            pre = _load_image_tensor(BASE_DIR / row["pre_filename"], args.image_size).to(device)
            post = _load_image_tensor(BASE_DIR / row["post_filename"], args.image_size).to(device)
            with torch.no_grad():
                out = model(pre, post)
            pred = out["logits"].argmax(dim=1)[0].cpu().numpy()
            coverage = {name: int((pred == c).sum()) for c, name in enumerate(class_names)}
            result = {
                "pixel_counts": coverage,
                "predicted_classes_present": [name for name, n in coverage.items() if n > 0],
            }
            write_cur = conn.cursor()
            write_cur.execute(
                """
                UPDATE samples
                SET status = 'processed', checkpoint_used = %s, result_json = %s, processed_at = now()
                WHERE id = %s
                """,
                (str(args.checkpoint), json.dumps(result), sample_id),
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
