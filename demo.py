"""
End-to-end prototype demo.

This script's own INPUT is always a generated SYNTHETIC pre/post tile pair
(a schematic road + a few buildings, with a "flood" region overlaid only
in the post-event tile and a gap cut into the road under the water) --
regardless of whether real SpaceNet-8 data has been downloaded elsewhere
in this repo or what the loaded checkpoint was itself trained on (that's
reported separately, from the checkpoint's own recorded data_source, not
assumed here). It runs that synthetic scene through the real Dual-Axis
GeoFormer network end to end:

    pre, post tiles
      -> DualAxisGeoFormer forward pass (Siamese MaxViT + Diff + U-decoder)
      -> 4-channel logits (background / building / road / flooded)
      -> grid-attention saliency map (Phase 1 byproduct)
      -> Phase 4 skeleton + attention-guided road-gap bridging

and saves one figure, `demo_output.png`, showing every stage.

By default this loads `checkpoints/best.pt` -- whatever that currently is
(check the printed/plotted data_source; "best" means lowest val_loss seen,
which is NOT the same as "most recently trained" -- see --checkpoint's own
help text and docs/MANUAL.md "What 'trained' means here"). Since this
script's own scene is always synthetic, a near-perfect panel-3 prediction
is honest evidence the pipeline and the loaded checkpoint's weights are
wired correctly together -- it is never, on its own, a SpaceNet-8 accuracy
number, regardless of what data trained the checkpoint. Pass --checkpoint
"" (empty) to fall back to random-init weights and reproduce the original
"expected to look like noise" behaviour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from checkpoint_utils import load_checkpoint_model
from model import DualAxisGeoFormer, GeoFormerConfig
from postprocess import bridge_road_gaps

IMG_SIZE = 128  # matches train.py's default --image-size
CLASS_NAMES = ["background", "building", "road", "flooded"]
CLASS_COLORS = ["#F4F6F1", "#B7C2B9", "#0B4A5C", "#1E7FA0"]


def make_synthetic_pair(seed: int = 7):
    """A schematic tile: a horizontal road, a handful of buildings, and a
    flood region that appears only post-event and cuts a gap in the road."""
    rng = np.random.default_rng(seed)
    s = IMG_SIZE / 256.0  # every hand-picked coordinate below was tuned at 256px;
                          # scale it so the same composition holds at any IMG_SIZE
                          # instead of silently drawing off-canvas (a real bug this
                          # file shipped with at IMG_SIZE=128 -- caught by actually
                          # looking at demo_output.png, not by assuming it still fit).

    def base_tile():
        img = np.full((IMG_SIZE, IMG_SIZE, 3), 0.55, dtype=np.float32)
        img += rng.normal(0, 0.03, img.shape).astype(np.float32)  # terrain texture
        return img

    pre = base_tile()
    post = base_tile()

    # road: a slightly curved horizontal band
    ys = np.arange(IMG_SIZE)
    road_center = 140 * s + 10 * s * np.sin(ys / (40.0 * s))
    road_mask = np.abs(np.arange(IMG_SIZE)[None, :] - road_center[:, None]) < max(2, 6 * s)
    for img in (pre, post):
        img[road_mask] = [0.25, 0.25, 0.28]

    # buildings: small rectangles scattered above/below the road
    building_boxes = [(40, 60, 30, 55), (90, 115, 190, 215), (170, 195, 45, 68),
                       (200, 222, 170, 195), (60, 82, 150, 172)]
    for (y0, y1, x0, x1) in building_boxes:
        y0, y1, x0, x1 = (int(round(v * s)) for v in (y0, y1, x0, x1))
        if y1 <= y0 or x1 <= x0:
            continue
        color = [0.75, 0.55, 0.35]
        pre[y0:y1, x0:x1] = color
        post[y0:y1, x0:x1] = color

    # flood: an ellipse-shaped water region, POST-EVENT ONLY, submerging the
    # middle of the road and one building
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    fcx, fcy, frx, fry = 128 * s, 150 * s, 55 * s, 30 * s
    flood_region = ((xx - fcx) ** 2) / (frx ** 2) + ((yy - fcy) ** 2) / (fry ** 2) <= 1
    post[flood_region] = [0.15, 0.35, 0.55]

    # cut a visible gap in the road under the water (the "disconnected road"
    # failure mode the thesis targets) -- post-event only
    gap_mask = flood_region & road_mask
    post[gap_mask] = [0.15, 0.4, 0.6]

    return np.clip(pre, 0, 1), np.clip(post, 0, 1), road_mask, flood_region


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return (t - 0.5) / 0.5  # normalize to roughly [-1, 1]


def main():
    parser = argparse.ArgumentParser(description="Run the Dual-Axis GeoFormer prototype demo")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Checkpoint to load. Pass an empty string to use random-init weights. NOTE: "
             "'best' means lowest val_loss seen -- Tversky loss is NOT comparable across "
             "differently-distributed data, so a real-data run resumed from a synthetic "
             "checkpoint can never beat that checkpoint's loss, and best.pt can stay stuck "
             "on an old run while checkpoints/last.pt keeps moving. Check which one you want.",
    )
    parser.add_argument("--skip-bridging", action="store_true",
                         help="Ablation (Table 2's 'GeoFormer - skeleton bridging' row): skip "
                              "Phase 4 entirely and show the raw, un-bridged road gap instead, "
                              "for a direct visual before/after comparison.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    is_trained = ckpt_path is not None and ckpt_path.exists()

    if is_trained:
        print(f"Loading trained checkpoint: {ckpt_path} "
              "(pipeline-validation run on SYNTHETIC data -- see docs/MANUAL.md)")
        model, ckpt = load_checkpoint_model(ckpt_path, device="cpu")
        if ckpt.get("model_type", "geoformer") != "geoformer":
            raise SystemExit(
                f"{ckpt_path} is a '{ckpt.get('model_type')}' checkpoint -- demo.py demonstrates "
                "Dual-Axis GeoFormer specifically (it reads grid_saliency for Phase 4, which a "
                "baseline checkpoint doesn't produce). Use evaluate.py for a baseline checkpoint."
            )
    else:
        print("Building Dual-Axis GeoFormer (random init, untrained -- "
              f"no checkpoint at {ckpt_path or '(none requested)'})...")
        model = DualAxisGeoFormer(GeoFormerConfig())
    model.eval()
    print(f"  parameters: {model.num_parameters():,}")

    pre_img, post_img, gt_road, gt_flood = make_synthetic_pair()
    pre_t, post_t = to_tensor(pre_img), to_tensor(post_img)

    print("Running forward pass...")
    with torch.no_grad():
        out = model(pre_t, post_t)
    logits = out["logits"][0]  # (4, H, W)
    saliency = out["grid_saliency"][0].numpy()  # small resolution, e.g. 16x16
    print(f"  logits shape: {tuple(out['logits'].shape)}  (batch, classes, H, W)")
    print(f"  grid saliency shape: {saliency.shape}  (full-image attention, one layer)")

    pred_classes = logits.argmax(dim=0).numpy()

    saliency_full = np.array(
        torch.nn.functional.interpolate(
            torch.from_numpy(saliency)[None, None].float(),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False,
        )[0, 0]
    )

    gap_road = gt_road & ~(gt_flood & gt_road)  # the visibly-broken road, matching post_img
    if args.skip_bridging:
        print("Ablation: Phase 4 SKIPPED (--skip-bridging) -- showing the raw, un-bridged "
              "road gap for direct comparison against the bridged version.")
        from skimage.morphology import skeletonize
        bridged, bridges = skeletonize(gap_road), []
    else:
        print("Running Phase 4 post-processing (skeleton + attention-guided bridging) "
              "on the GROUND-TRUTH road mask, to demonstrate that step in isolation "
              "from the raw model prediction...")
        bridged, bridges = bridge_road_gaps(
            gap_road, saliency_full, max_gap_px=max(10, int(40 * IMG_SIZE / 256)), saliency_threshold=0.0
        )
    print(f"  gaps bridged: {len(bridges)}")
    for b in bridges:
        print(f"    {b}")

    # ---------------- figure ----------------
    # BUG THIS FIXES: this used to hardcode "trained on SYNTHETIC data"
    # whenever is_trained was true, regardless of what checkpoint was
    # actually loaded -- the same bug already fixed in real_image_demo.py's
    # caption and serve.py's /health status, missed here. Report the
    # checkpoint's own recorded data_source instead.
    if is_trained:
        data_source = ckpt.get("data_source", "unknown (checkpoint predates data_source tracking)")
        weight_state = f"trained on {data_source}"
    else:
        weight_state = "random-init weights (untrained)"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Dual-Axis GeoFormer -- architecture prototype ({weight_state})\n"
        "synthetic pre/post tile, real forward pass, real Phase-4 post-processing",
        fontsize=12,
    )

    axes[0, 0].imshow(pre_img); axes[0, 0].set_title("Input: pre-event tile")
    axes[0, 1].imshow(post_img); axes[0, 1].set_title("Input: post-event tile\n(flood + road gap)")

    cmap = ListedColormap(CLASS_COLORS)
    axes[0, 2].imshow(pred_classes, cmap=cmap, vmin=0, vmax=3)
    pred_title = (
        f"Raw model output (argmax)\n{weight_state} -- see docs/MANUAL.md"
        if is_trained else "Raw model output (argmax)\nUNTRAINED -- expected to look like noise"
    )
    axes[0, 2].set_title(pred_title)

    axes[1, 0].imshow(saliency_full, cmap="viridis")
    axes[1, 0].set_title("Grid-attention saliency\n(real byproduct of the encoder)")

    axes[1, 1].imshow(post_img)
    axes[1, 1].imshow(gap_road, cmap=ListedColormap(["none", "#B24B1E"]), alpha=0.6)
    axes[1, 1].set_title("Ground-truth road mask\n(with the flood-cut gap, for Phase-4 demo)")

    axes[1, 2].imshow(post_img)
    axes[1, 2].imshow(bridged, cmap=ListedColormap(["none", "#5FC2E0"]), alpha=0.75)
    for b in bridges:
        y0, x0 = b["from"]; y1, x1 = b["to"]
        axes[1, 2].plot([x0, x1], [y0, y1], color="#F4F6F1", lw=1, ls="--")
    panel6_title = (
        "Ablation: Phase 4 SKIPPED\n(raw skeleton, gap left open)"
        if args.skip_bridging else
        f"Phase 4: skeleton + attention-guided\nbridging ({len(bridges)} gap closed)"
    )
    axes[1, 2].set_title(panel6_title)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("demo_output.png", dpi=150)
    print("\nSaved demo_output.png")


if __name__ == "__main__":
    main()
