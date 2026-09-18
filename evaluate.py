"""
Evaluate a trained checkpoint and print a per-class report.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --data-dir path/to/spacenet8
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from checkpoint_utils import load_checkpoint_model
from dataset import SyntheticFloodDataset, SpaceNet8Dataset, NUM_CLASSES
from losses import TverskyLoss
from postprocess import suppress_isolated_flood_predictions
from train import ConfusionAccumulator

CLASS_NAMES = ["background", "building", "road", "flooded"]


def main():
    p = argparse.ArgumentParser(description="Evaluate a Dual-Axis GeoFormer checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=None,
                    help="Real SpaceNet-8 directory. Omit to re-generate the synthetic "
                         "validation split (same base seed as train.py's default).")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--synthetic-val-size", type=int, default=24)
    p.add_argument(
        "--held-out-only", action="store_true",
        help="With --data-dir, evaluate only the same held-out validation split train.py "
             "uses (SpaceNet8Dataset.split's stable, tile_id-hash-based 10%%) instead of the "
             "full dataset. Omitting this evaluates train+val combined -- an optimistic "
             "number, since it includes tiles the model was trained on.",
    )
    p.add_argument(
        "--suppress-isolated-flood", action="store_true",
        help="Apply postprocess.py's suppress_isolated_flood_predictions before scoring "
             "(docs/RESEARCH_NOTES.md item 7, the SpaceNet-8 1st-place team's actual "
             "false-positive-suppression heuristic) -- a pure inference-time step, no "
             "retraining, only affects checkpoints with a separate flood head. Only "
             "meaningful to compare against the same checkpoint WITHOUT this flag.",
    )
    p.add_argument(
        "--suppress-min-fraction", type=float, default=0.0005,
        help="--suppress-isolated-flood only: a connected blob of predicted-flooded pixels "
             "smaller than this fraction of a tile's total pixels gets suppressed.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint_model(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} (model_type={ckpt.get('model_type', 'geoformer')}, "
          f"epoch {ckpt.get('epoch', '?')}, "
          f"val_loss at save time {ckpt.get('val_loss', float('nan')):.4f})")

    if args.data_dir:
        full = SpaceNet8Dataset(args.data_dir, image_size=args.image_size)
        if args.held_out_only:
            _, val_idx = full.split(val_fraction=0.1)
            ds = torch.utils.data.Subset(full, val_idx)
            print(f"[HELD-OUT split only -- {len(ds)}/{len(full)} tiles, same split train.py uses]")
        else:
            ds = full
            print(f"[FULL dataset -- {len(ds)} tiles, includes training data, optimistic]")
    else:
        ds = SyntheticFloodDataset(length=args.synthetic_val_size, image_size=args.image_size, base_seed=100_000)
        print("[SYNTHETIC evaluation set -- pipeline-validation numbers, not SpaceNet-8 accuracy]")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    loss_fn = TverskyLoss(num_classes=NUM_CLASSES)

    if args.suppress_isolated_flood:
        print(f"[postprocess: suppress_isolated_flood_predictions, "
              f"min_component_fraction={args.suppress_min_fraction}]")

    loss_sum, n_batches = 0.0, 0
    acc = ConfusionAccumulator(NUM_CLASSES)
    with torch.no_grad():
        for pre, post, mask in loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            out = model(pre, post)
            loss_sum += loss_fn(out["logits"], mask).item()
            n_batches += 1
            if args.suppress_isolated_flood and "flood_logit" in out:
                pred = suppress_isolated_flood_predictions(out, min_component_fraction=args.suppress_min_fraction)
                acc.update_pred(pred, mask)
            else:
                acc.update(out["logits"], mask)

    f1_final = acc.f1()
    coverage = acc.coverage_report()

    print(f"\n{'Class':<12}{'F1':>8}  {'GT imgs':>8}  {'Pred imgs':>10}")
    print("-" * 42)
    for c in range(NUM_CLASSES):
        gt_n, pred_n, total_n = coverage[c]
        f1_str = "n/a" if f1_final[c] is None else f"{f1_final[c]:.4f}"
        flag = "  <- COLLAPSED (present in GT, never predicted)" if (gt_n > 0 and pred_n == 0) else ""
        print(f"{CLASS_NAMES[c]:<12}{f1_str:>8}  {gt_n:>8}  {pred_n:>10}{flag}")
    print("-" * 42)
    print(f"{'mean loss':<12}{loss_sum / max(1, n_batches):>8.4f}")
    print(f"(GT imgs / Pred imgs out of {coverage[0][2]} total images -- a class present in many GT "
          f"images but predicted in zero is a real failure a bare F1 number can hide.)")


if __name__ == "__main__":
    main()
