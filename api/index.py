"""Vercel's Python runtime looks for an ASGI/WSGI app under api/ -- this
file just re-exports the real app so dashboard/dashboard_server.py stays
the single source of truth (importable and runnable the normal way too:
`uvicorn dashboard.dashboard_server:app`) rather than duplicating it here.
See docs/VERCEL_DEPLOYMENT.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.dashboard_server import app  # noqa: E402,F401
