# Deploying the Dashboard to Vercel

The dashboard splits cleanly into two halves, and only one of them can run
on Vercel. This doc says exactly which, why, and how to deploy it.

## What works on Vercel, and what doesn't

| Feature | On Vercel? | Why |
|---|---|---|
| Training run history, charts, analytics | **Yes** | Reads from Neon Postgres, which is designed for serverless/Vercel deployments (there's an official integration) |
| Dataset tile counts | **Yes** | Reads from the `dataset_tiles` table (see below), not local disk |
| Submitting new sample uploads | **No** | Vercel's filesystem is read-only outside `/tmp`, and `/tmp` isn't shared across requests or instances — an uploaded file would vanish before it could ever be processed |
| Running inference on a sample | **No** | Needs PyTorch + the trained checkpoint (~130MB+) — the wrong shape for a serverless function regardless of storage; keep this running locally or on `serve.py`'s existing Docker path |

The dashboard detects a Vercel deployment automatically (Vercel sets a
`VERCEL` environment variable on every deployment) and disables the
upload form itself, with a message explaining why — nothing to configure
by hand, and no user-facing dead end.

## One-time setup: get the dataset stats into Postgres

The dashboard's dataset tab normally reads `real_sn8_dataset_full/index.json`
straight off local disk — which doesn't exist on Vercel. Run this once,
from a machine that **does** have the dataset downloaded:

```bash
python db/sync_dataset_tiles.py
```

This populates the `dataset_tiles` table. `/api/dataset` tries Postgres
first and only falls back to the local file if that table is empty, so
this is safe to run (or skip) locally too.

## Deploying

1. **Push this repo to GitHub** (if not already) and import it in the
   [Vercel dashboard](https://vercel.com/new), or use the CLI:
   ```bash
   npm i -g vercel
   vercel
   ```
2. **Set the `DATABASE_URL` environment variable** in the Vercel project
   settings (Settings → Environment Variables) to your Neon connection
   string — the same one in your local `.env`. If you connect Neon
   through Vercel's own [Neon integration](https://vercel.com/integrations/neon),
   this is set for you automatically.
3. **Deploy.** `vercel.json` routes every request to `api/index.py`,
   which just re-exports the real FastAPI app from
   `dashboard/dashboard_server.py` — there's no separate copy to keep in
   sync.

That's it — no build step, no framework config. The whole deployment is
`vercel.json` + `api/index.py` + `api/requirements.txt` (a deliberately
lightweight dependency list — just FastAPI and psycopg2, **not** the
project's main `requirements.txt`, which includes PyTorch and would blow
past Vercel's function size limits for no reason, since
`dashboard_server.py` itself never imports torch).

## Why this split, not a different one

The alternative would be wiring uploads to real object storage (S3,
Vercel Blob) so they *do* work on Vercel. That's a legitimate next step,
just not one built here — it needs credentials this deployment doesn't
have set up, and even with that fixed, actually *running inference* on
an uploaded sample still needs PyTorch and the checkpoint file somewhere,
which was never going to be a Vercel serverless function regardless of
where the upload itself lands. The honest split is: Vercel for viewing
your results from anywhere, your own machine (or `serve.py`'s Docker
path, see `docs/DEPLOYMENT.md`) for anything that touches the model.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dataset tab shows an error | Run `python db/sync_dataset_tiles.py` locally, pointed at the same `DATABASE_URL` |
| Upload form doesn't appear | Expected on Vercel — check `/api/config` returns `{"uploads_enabled": false}` |
| 500 error on any `/api/*` route | Check `DATABASE_URL` is set in Vercel's project settings and that your Neon project allows connections from Vercel's IP ranges (Neon's default settings allow this) |
| Deployment fails to build | Make sure `api/requirements.txt` exists and doesn't accidentally include the project root's full `requirements.txt` (with PyTorch) |
