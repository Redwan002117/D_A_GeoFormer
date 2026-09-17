"""One-time: create the dashboard's tables from db/schema.sql.

Usage:
    python db/init_db.py
"""
import os
from pathlib import Path

import psycopg2

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
    raise SystemExit("DATABASE_URL not set and no .env found -- copy .env.example to .env first")


def main():
    conn = psycopg2.connect(_database_url())
    cur = conn.cursor()
    cur.execute((BASE_DIR / "db" / "schema.sql").read_text())
    conn.commit()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1;")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    print(f"Tables ready: {', '.join(tables)}")


if __name__ == "__main__":
    main()
