"""Populate the dataset_tiles table from a local prepare_real_data.py output
(real_sn8_dataset_full/index.json), so the dashboard's dataset stats can be
served entirely from Postgres -- no local disk needed at read time.

This is what makes the dashboard deployable somewhere with no access to your
local filesystem (e.g. Vercel, see docs/VERCEL_DEPLOYMENT.md): run this once
from a machine that DOES have the dataset downloaded, and the numbers become
available to any dashboard instance reading the same database from then on.

Usage:
    python db/sync_dataset_tiles.py
    python db/sync_dataset_tiles.py --data-dir real_sn8_dataset_full
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

BASE_DIR = Path(__file__).resolve().parent.parent


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL"):
                return line.split("=", 1)[1].strip()
    raise SystemExit("DATABASE_URL not set and no .env found -- see README / docs/USER_GUIDE.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="real_sn8_dataset_full")
    args = p.parse_args()

    index_path = BASE_DIR / args.data_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found -- run prepare_real_data.py first")
    entries = json.loads(index_path.read_text())

    conn = psycopg2.connect(_database_url())
    cur = conn.cursor()

    writes = []
    for entry in entries:
        # Same elimination rule dashboard_server.py's own dataset endpoint
        # used before this migration -- entries this project downloaded
        # before prepare_real_data.py started recording "aoi" per entry
        # have no "aoi" key at all, and every entry that DOES have one so
        # far says Louisiana-East, so an absent aoi means Germany here.
        aoi = entry.get("aoi", "Germany_Training_Public")
        tile_id = entry.get("tile_id", entry["pre"])
        counts = entry.get("class_pixel_counts", {})
        has_building = counts.get("1", 0) > 0
        has_road = counts.get("2", 0) > 0
        has_flooded = counts.get("3", 0) > 0
        writes.append((aoi, tile_id, has_building, has_road, has_flooded))

    for aoi, tile_id, has_building, has_road, has_flooded in writes:
        cur.execute(
            """
            INSERT INTO dataset_tiles (aoi, tile_id, has_building, has_road, has_flooded)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (aoi, tile_id) DO UPDATE SET
                has_building = EXCLUDED.has_building,
                has_road = EXCLUDED.has_road,
                has_flooded = EXCLUDED.has_flooded
            """,
            (aoi, tile_id, has_building, has_road, has_flooded),
        )
    conn.commit()

    cur.execute("SELECT aoi, COUNT(*) FROM dataset_tiles GROUP BY aoi ORDER BY 2 DESC")
    print(f"Synced {len(writes)} tiles from {index_path}:")
    for aoi, n in cur.fetchall():
        print(f"  {aoi}: {n}")
    conn.close()


if __name__ == "__main__":
    main()
