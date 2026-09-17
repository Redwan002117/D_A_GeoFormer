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
texture, shadows, and reflections the checkpoint's own training data may
not resemble," not "does it correctly detect this real flood." This photo
was never part of ANY checkpoint's training data either way -- what varies
is whether the checkpoint itself was trained on synthetic shapes or real
SpaceNet-8 tiles (see the printed/plotted data_source, sourced from the
checkpoint itself, not assumed here). Either way, a single real, unlabeled
photo run through the model once is exploratory, not evidence of accuracy.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from checkpoint_utils import load_checkpoint_model


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
    parser = argparse.ArgumentParser(description="Run the pipeline on a real photograph")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Which checkpoint to load. NOTE: 'best' is picked by lowest val_loss, and "
             "Tversky-loss values are NOT comparable across differently-distributed data -- "
             "a checkpoint resumed onto real data with a real, higher loss than a prior "
             "synthetic run's loss will never beat it, so checkpoints/best.pt can get stuck "
             "on a stale, older run while checkpoints/last.pt keeps moving. Check which one "
             "you actually want; do not assume 'best' means 'most recently trained.'",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    model, ckpt = load_checkpoint_model(ckpt_path, device="cpu")
    print(f"Loaded {ckpt_path} (model_type={ckpt.get('model_type', 'geoformer')}, "
          f"epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', float('nan')):.4f}, "
          f"{model.num_parameters():,} params)")

    post_img = load_real_post("real_flood_sample.jpg")
    pre_img = make_dry_stand_in(IMG_SIZE)
    pre_t, post_t = to_tensor(pre_img), to_tensor(post_img)

    print("Running forward pass on a REAL photograph...")
    with torch.no_grad():
        out = model(pre_t, post_t)
    pred = out["logits"][0].argmax(dim=0).numpy()

    fractions = {CLASS_NAMES[c]: float((pred == c).sum()) / pred.size for c in range(4)}
    print("Predicted class pixel fractions:", {k: round(v, 3) for k, v in fractions.items()})

    data_source = ckpt.get("data_source", "unknown (checkpoint predates data_source tracking)")
    trained_on_real = data_source not in ("synthetic", "unknown (checkpoint predates data_source tracking)")
    training_note = (
        f"trained on {data_source}, epoch {ckpt.get('epoch', '?')}, "
        f"val_loss {ckpt.get('val_loss', float('nan')):.4f}"
    )
    subtitle = (
        f"({training_note}; this NIST photo itself was never part of training data either way "
        "-- see docs/MANUAL.md)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(
        "\n".join(textwrap.wrap("Real-photograph robustness check -- NOT a validated flood prediction", 90))
        + "\n" + "\n".join(textwrap.wrap(subtitle, 100)),
        fontsize=10,
    )
    axes[0].imshow(pre_img); axes[0].set_title("'pre' input\n(generated dry stand-in, not a real photo)", fontsize=9)
    axes[1].imshow(post_img); axes[1].set_title("'post' input\nREAL photo: Katrina flooding, NIST #15001", fontsize=9)
    axes[2].imshow(pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=3)
    panel3_title = (
        "raw model output (argmax)\ntrained on real SpaceNet-8 data -- still exploratory on this photo"
        if trained_on_real else
        "raw model output (argmax)\nexploratory -- not trained on real imagery"
    )
    axes[2].set_title(panel3_title, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig("real_image_demo_output.png", dpi=150)
    print("Saved real_image_demo_output.png")


if __name__ == "__main__":
    main()
