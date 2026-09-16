# Deployment Guide

**Status: the `Dockerfile` and `serve.py` are written and `serve.py` is
verified working locally (started with `uvicorn`, `/health` and `/predict`
both tested end to end against a running instance). The Docker image itself
has NOT been build-tested in this environment** — the Docker CLI is present
but the daemon isn't running here (Docker Desktop not started), so `docker
build` couldn't be executed. Run `docker build -t geoformer-proto .` yourself
before a real deployment and fix anything it surfaces; don't assume it's
clean because it reads clean.

## What's deployable

`serve.py` — a FastAPI app with two endpoints:

- `GET /health` — returns `{"status", "model_status", "device"}`. Always
  check `model_status` before trusting a prediction — it says outright
  whether the loaded checkpoint was trained on synthetic data, real data, or
  is untrained (see `docs/MANUAL.md` §2).
- `POST /predict` — multipart form fields `pre` and `post` (images); returns
  per-class pixel fractions and a base64-encoded PNG overlay.

It loads a checkpoint once at startup (`$GEOFORMER_CHECKPOINT`, default
`checkpoints/best.pt`) and serves requests from memory — no database, no
external service dependency, nothing cloud-specific.

## Run it locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn serve:app --host 0.0.0.0 --port 8000
curl -F "pre=@your_pre.jpg" -F "post=@your_post.jpg" http://localhost:8000/predict
```

## Run it in Docker, anywhere

```bash
docker build -t geoformer-proto .
docker run -p 8000:8000 geoformer-proto
```

That's the entire contract — any host that can run a Docker image can run
this. What follows is how that translates to specific platforms; all of
them boil down to "build this image, push it, point the platform at it or
its Dockerfile."

## Cloud platforms (generic instructions — none of these were executed; no
credentials or accounts are available in this environment to test against)

| Platform | Shape of the deploy |
|---|---|
| **Render** | New "Web Service" → connect the repo → Render detects the `Dockerfile` automatically → set `PORT` is handled for you. |
| **Railway** | New project → "Deploy from repo" → Railway builds the `Dockerfile` → exposes a public URL automatically. |
| **Fly.io** | `fly launch` in this directory (detects the `Dockerfile`), then `fly deploy`. |
| **Google Cloud Run** | `gcloud run deploy geoformer-proto --source .` — Cloud Run builds the `Dockerfile` via Cloud Build and serves it on `$PORT` (already wired in the `CMD`). |
| **AWS App Runner** | Push the image to ECR (`docker build`, `docker tag`, `docker push`), then create an App Runner service from that ECR image. |
| **Azure Container Apps** | `az containerapp up --source .` — builds and deploys the `Dockerfile` directly. |
| **Any VPS** (DigitalOcean, Lightsail, a bare Linux box) | `docker build -t geoformer-proto . && docker run -d -p 8000:80 geoformer-proto`, put a reverse proxy (nginx/Caddy) in front for TLS. |

All of these read `$PORT` from the environment (the `Dockerfile`'s `CMD`
already does `--port ${PORT}`), which is what Cloud Run/Render/Railway/App
Runner all inject automatically — no code change needed to move between
them.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GEOFORMER_CHECKPOINT` | `checkpoints/best.pt` | Which checkpoint `serve.py` loads at startup. |
| `GEOFORMER_IMAGE_SIZE` | `128` | Must match the checkpoint's training resolution. |
| `PORT` | `8000` | What port the server binds. Most PaaS platforms set this for you. |

## Before you deploy this for anything beyond a demo

Say this alongside any deployment: **the shipped checkpoint is trained on
synthetic data only** (`docs/MANUAL.md` §2). The API, the container, and the
platform wiring are all real and reusable; the specific weights behind
`/predict` are not a validated flood detector yet. Swapping in a checkpoint
trained on real SpaceNet-8 data (`docs/MANUAL.md` §6) is a
`GEOFORMER_CHECKPOINT` environment variable change, not a redeploy of any
different kind — that's the point of separating the serving code from the
weights.
