"""
Runs the trained pipeline on a REAL photograph, not synthetic data.

`real_flood_sample.jpg` is a genuine, public-domain aerial photo of a
flooded New Orleans neighborhood after Hurricane Katrina (NIST image
15001, https://www.nist.gov/image/neighborhood-new-orleans-flooded-hurricane-katrina --
a U.S. government work). There is no matching real pre-event photo of this
exact site available here, so this script pairs it with a generated dry
base tile as a stand-in "pre" image, purely so the model has two inputs to
run its diff module against.

Be explicit about this when you show it: this is a ROBUSTNESS CHECK, not a
valid bi-temporal prediction -- it answers "does the real architecture run,
without crashing or producing garbage shapes, on a real photograph full of
texture, shadows, and reflections the synthetic training data never had,"
not "does it correctly detect this real flood." The model was trained only
on clean synthetic shapes (see docs/MANUAL.md); real photographic texture is
out of its training distribution by design, and the prediction should be
read as exploratory, not as evidence of accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from model import DualAxisGeoFormer, GeoFormerConfig

IMG_SIZE = 128
CLASS_NAMES = ["background", "building", "road", "flooded"]
CLASS_COLORS = ["#F4F6F1", "#B7C2B9", "#0B4A5C", "#1E7FA0"]


def load_real_post(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype(np.float32) / 255.0


def make_dry_stand_in(size: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return (t - 0.5) / 0.5


def main():
    ckpt_path = Path("checkpoints/best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GeoFormerConfig(**ckpt["config_dict"])
    model = DualAxisGeoFormer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path} ({model.num_parameters():,} params)")

    post_img = load_real_post("real_flood_sample.jpg")
    pre_img = make_dry_stand_in(IMG_SIZE)
    pre_t, post_t = to_tensor(pre_img), to_tensor(post_img)

    print("Running forward pass on a REAL photograph...")
    with torch.no_grad():
        out = model(pre_t, post_t)
    pred = out["logits"][0].argmax(dim=0).numpy()

    fractions = {CLASS_NAMES[c]: float((pred == c).sum()) / pred.size for c in range(4)}
    print("Predicted class pixel fractions:", {k: round(v, 3) for k, v in fractions.items()})

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Real-photograph robustness check -- NOT a validated flood prediction\n"
        "(trained on synthetic data only; real photo is out-of-distribution by design; see docs/MANUAL.md)",
        fontsize=10.5,
    )
    axes[0].imshow(pre_img); axes[0].set_title("'pre' input\n(generated dry stand-in, not a real photo)", fontsize=9)
    axes[1].imshow(post_img); axes[1].set_title("'post' input\nREAL photo: Katrina flooding, NIST #15001", fontsize=9)
    axes[2].imshow(pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3)
    axes[2].set_title("raw model output (argmax)\nexploratory -- not trained on real imagery", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig("real_image_demo_output.png", dpi=150)
    print("Saved real_image_demo_output.png")


if __name__ == "__main__":
    main()
