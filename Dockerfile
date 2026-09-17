# Dual-Axis GeoFormer inference server -- runs anywhere a container runs.
# CPU by default; nothing here depends on a specific cloud provider.

FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-image / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BUG THIS FIXES: serve.py imports checkpoint_utils, which imports
# baseline.py -- neither was in this COPY list, so `uvicorn serve:app`
# would fail immediately with ModuleNotFoundError the moment a container
# actually started (never caught earlier because the image itself was
# never build-tested in this environment -- no Docker daemon available;
# see docs/DEPLOYMENT.md's own disclosure of that gap). List every module
# serve.py's import chain actually reaches.
COPY model.py losses.py postprocess.py dataset.py baseline.py checkpoint_utils.py \
     train.py evaluate.py serve.py ./
COPY checkpoints/ ./checkpoints/

ENV GEOFORMER_CHECKPOINT=checkpoints/best.pt
ENV GEOFORMER_IMAGE_SIZE=128
# Most platforms (Cloud Run, Render, Railway, App Runner) inject $PORT;
# 8000 is the local-run default.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT}"]
