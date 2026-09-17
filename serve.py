"""
Deployable inference API for Dual-Axis GeoFormer.

Runs anywhere a container can run (local machine, any cloud VM, Render,
Railway, Fly.io, AWS App Runner / ECS, GCP Cloud Run, Azure Container Apps)
-- see docs/DEPLOYMENT.md for platform-specific one-liners. This process
itself has no cloud dependency: it's a plain FastAPI app that loads a
checkpoint once at startup and serves HTTP requests.

IMPORTANT -- same honesty note as everywhere else in this repo: whatever
checkpoint GEOFORMER_CHECKPOINT points at (synthetic-trained, real-data-
trained, or untrained), this API makes the *pipeline* deployable and
demoable -- it does not, by itself, make the *predictions* validated
flood-detection accuracy (see docs/MANUAL.md "What 'trained' means here").
The response JSON's `model_status` field reports the loaded checkpoint's
own recorded data_source for exactly this reason; surface it wherever
predictions are shown, rather than assuming what trained it.

Endpoints:
  GET  /health            -> {"status": "ok", "model_status": "...", "device": "..."}
  POST /predict            -> multipart form fields `pre` and `post` (images);
                               returns per-class pixel fractions + a base64 PNG
                               overlay visualization.

Run locally:
    uvicorn serve:app --host 0.0.0.0 --port 8000

Then, e.g.:
    curl -F "pre=@pre.jpg" -F "post=@post.jpg" http://localhost:8000/predict
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch.nn as nn

from checkpoint_utils import load_checkpoint_model
from model import DualAxisGeoFormer, GeoFormerConfig

CLASS_NAMES = ["background", "building", "road", "flooded"]
CLASS_COLORS = ["#F4F6F1", "#B7C2B9", "#0B4A5C", "#1E7FA0"]

CHECKPOINT_PATH = os.environ.get("GEOFORMER_CHECKPOINT", "checkpoints/best.pt")
IMAGE_SIZE = int(os.environ.get("GEOFORMER_IMAGE_SIZE", "128"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Dual-Axis GeoFormer Inference API",
    description="Bi-temporal multiclass flood segmentation prototype. "
                 "See /health for model_status before trusting any prediction.",
    version="0.1.0",
)

_model: nn.Module | None = None
_model_status = "not loaded"


def load_model() -> None:
    global _model, _model_status
    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists():
        model, ckpt = load_checkpoint_model(ckpt_path, device=DEVICE)
        model_type = ckpt.get("model_type", "geoformer")
        # BUG THIS FIXES: this message used to hardcode "trained on SYNTHETIC
        # data only" no matter what checkpoint GEOFORMER_CHECKPOINT actually
        # pointed at -- the exact same bug already found and fixed in
        # real_image_demo.py's caption, missed here. Report the checkpoint's
        # own recorded data_source instead of assuming.
        data_source = ckpt.get("data_source", "unknown (checkpoint predates data_source tracking)")
        _model_status = (
            f"loaded {ckpt_path.name} (model_type={model_type}, trained on: {data_source}) -- "
            "see docs/MANUAL.md for what 'trained' means for this data source before trusting a prediction"
        )
    else:
        model = DualAxisGeoFormer(GeoFormerConfig())
        _model_status = f"no checkpoint at {ckpt_path} -- random-init (untrained) weights"
    model.to(DEVICE).eval()
    _model = model
    print(f"[serve] {_model_status}")


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_status": _model_status, "device": str(DEVICE)}


def _load_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img).astype(np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return ((t - 0.5) / 0.5).to(DEVICE)


@app.post("/predict")
async def predict(pre: UploadFile = File(...), post: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        pre_img = _load_image(pre)
        post_img = _load_image(post)
    except Exception as e:
        raise HTTPException(400, f"Could not read one of the uploaded images: {e}")

    pre_t, post_t = _to_tensor(pre_img), _to_tensor(post_img)
    with torch.no_grad():
        out = _model(pre_t, post_t)
    pred = out["logits"][0].argmax(dim=0).cpu().numpy()

    total = pred.size
    class_fractions = {CLASS_NAMES[c]: float((pred == c).sum()) / total for c in range(len(CLASS_NAMES))}

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(pre_img); axes[0].set_title("pre-event", fontsize=10)
    axes[1].imshow(post_img); axes[1].set_title("post-event", fontsize=10)
    axes[2].imshow(pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3)
    axes[2].set_title("prediction", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse({
        "model_status": _model_status,
        "class_pixel_fractions": class_fractions,
        "overlay_png_base64": overlay_b64,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
