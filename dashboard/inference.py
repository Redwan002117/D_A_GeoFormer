"""Shared live-inference + overlay-visualization logic for the dashboard.

Used by both dashboard_server.py (synchronous inference right on sample
submission -- the "live" path) and process_samples.py (a manual batch
fallback for any sample that couldn't be processed live, e.g. no
checkpoint was available yet). One implementation, not two copies that
could silently drift apart -- see checkpoint_utils.py's own docstring
for the exact same rationale applied to checkpoint loading.

The overlay panel (pre / post / prediction) reuses demo.py/serve.py's
own CLASS_COLORS so a prediction looks the same whether it came from
the CLI demo, the deployable API, or the dashboard.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # checkpoint_utils.py, model.py live at the project root

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from checkpoint_utils import load_checkpoint_model  # noqa: E402

CLASS_NAMES = ["background", "building", "road", "flooded"]
CLASS_COLORS = ["#F4F6F1", "#B7C2B9", "#0B4A5C", "#1E7FA0"]

# Same env-var override serve.py uses, so a deployment that already sets
# GEOFORMER_CHECKPOINT for the standalone API gets consistent behavior here.
CHECKPOINT_PATH = os.environ.get("GEOFORMER_CHECKPOINT", str(BASE_DIR / "checkpoints" / "best.pt"))

# Cached across calls within one process (dashboard_server.py serves many
# requests from one long-lived process; re-loading a ~150MB checkpoint
# from disk on every single sample submission would make "live" anything
# but). Keyed by the resolved checkpoint path so switching --checkpoint
# (dashboard_server.py's default vs an explicit override) still works
# correctly instead of silently serving a stale model.
_model_cache: dict[str, tuple[torch.nn.Module, dict]] = {}


def get_model(checkpoint_path: str | Path):
    """Returns (model, checkpoint_dict), loading once and caching by path.
    Raises FileNotFoundError if checkpoint_path doesn't exist -- callers
    decide what "no checkpoint yet" should mean for them (dashboard_server
    leaves the sample 'pending' rather than fail the whole upload)."""
    key = str(Path(checkpoint_path).resolve())
    if key not in _model_cache:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)
        _model_cache[key] = load_checkpoint_model(checkpoint_path, device="cpu")
    return _model_cache[key]


def _load_image_array(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def run_inference(pre_path: Path, post_path: Path, checkpoint_path: str | Path,
                   image_size: int = 256) -> tuple[dict, bytes]:
    """Runs a real forward pass on a real pre/post image pair and renders
    a 3-panel overlay (pre-event / post-event / prediction, same color
    scheme as demo.py and serve.py). Returns (result_json, png_bytes) --
    callers decide where to persist the PNG (disk, for the dashboard's
    /uploads mount) or how to transport it (serve.py base64-encodes the
    same bytes for its HTTP response instead)."""
    model, _ckpt = get_model(checkpoint_path)

    pre_arr = _load_image_array(pre_path, image_size)
    post_arr = _load_image_array(post_path, image_size)
    pre_t = torch.from_numpy(pre_arr).permute(2, 0, 1).unsqueeze(0)
    post_t = torch.from_numpy(post_arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = model(pre_t, post_t)
    pred = out["logits"][0].argmax(dim=0).cpu().numpy()

    pixel_counts = {name: int((pred == c).sum()) for c, name in enumerate(CLASS_NAMES)}
    result = {
        "pixel_counts": pixel_counts,
        "predicted_classes_present": [name for name, n in pixel_counts.items() if n > 0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    axes[0].imshow(pre_arr); axes[0].set_title("pre-event", fontsize=10)
    axes[1].imshow(post_arr); axes[1].set_title("post-event", fontsize=10)
    axes[2].imshow(pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3)
    axes[2].set_title("prediction", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c], ec="#999", lw=0.5)
                      for c in range(len(CLASS_NAMES))]
    fig.legend(legend_handles, CLASS_NAMES, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)

    return result, buf.getvalue()
