"""
Training loop for Dual-Axis GeoFormer.

Runs out of the box on the built-in SyntheticFloodDataset (no download
needed) -- that run's job is to PROVE THE PIPELINE WORKS: loss decreases,
checkpoints save/load, per-class F1 computes correctly, a scheduler steps.
It is a pipeline validation run, not a trained flood detector -- see
docs/MANUAL.md "What 'trained' means here" before presenting results from it
as if they were SpaceNet-8 accuracy numbers.

Point --data-dir at a real, preprocessed SpaceNet-8 directory (see
docs/MANUAL.md "Preparing real SpaceNet-8 data") to actually train on the
real benchmark; everything else about this script is unchanged.

Usage:
    python train.py                                   # synthetic pipeline-check run
    python train.py --epochs 30 --batch-size 8 --lr 3e-4
    python train.py --data-dir path/to/spacenet8 --image-size 256 --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticFloodDataset, SpaceNet8Dataset, NUM_CLASSES
from losses import TverskyLoss
from model import DualAxisGeoFormer, GeoFormerConfig
from baseline import SN8Baseline


def per_class_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-7):
    """logits: (B,C,H,W), target: (B,H,W). Returns a dict {class_idx: f1}.
    Correct for a SINGLE call on one batch/image: if a class is genuinely
    absent from both prediction and target there, that one judgment (F1=1.0,
    "nothing to find, nothing wrongly found") is right.

    BUG THIS DOCSTRING NOW WARNS ABOUT: that per-call correctness does NOT
    make it safe to call this once per batch across a validation loop and
    AVERAGE the resulting per-batch F1s together, which is what train.py and
    evaluate.py used to do. Averaged that way, a rare class that the model
    never predicts ANYWHERE gets scored 1.0 on every batch where it happens
    to be absent from ground truth too, diluting real, total failure on the
    batches where it IS present into a misleadingly high aggregate (verified
    concretely: a model predicting zero "flooded"/"building" pixels on every
    one of 40 random real tiles -- including the ~2/3 of tiles that actually
    contain them -- still averaged to F1 0.66 / 0.30 this way, closely
    matching pure class-prevalence arithmetic, not genuine recall). Use
    `ConfusionAccumulator` below for real validation/evaluation reporting;
    this function stays as a correct single-call primitive and this session's
    unit tests exercise it as exactly that.
    """
    pred = logits.argmax(dim=1)
    f1s = {}
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()
        if tp == 0 and fp == 0 and fn == 0:
            f1s[c] = 1.0
            continue
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s[c] = 2 * precision * recall / (precision + recall + eps)
    return f1s


class ConfusionAccumulator:
    """Accumulates raw TP/FP/FN counts (and per-image class coverage) across
    an entire validation/evaluation pass, so F1 is computed ONCE from the
    global totals -- not averaged from many small per-batch F1 values. This
    is the fix for the real bug documented in per_class_f1's docstring
    above: a corpus-level F1 doesn't get diluted by absent-class batches the
    way a mean-of-per-batch-F1s does, and it makes total failure on a rare
    class visible instead of masked."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.tp = [0] * num_classes
        self.fp = [0] * num_classes
        self.fn = [0] * num_classes
        self.gt_images = [0] * num_classes    # images where this class appears in ground truth
        self.pred_images = [0] * num_classes  # images where the model predicted this class at all
        self.n_images = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        self.n_images += target.shape[0]
        for c in range(self.num_classes):
            pred_c, target_c = pred == c, target == c
            self.tp[c] += (pred_c & target_c).sum().item()
            self.fp[c] += (pred_c & ~target_c).sum().item()
            self.fn[c] += (~pred_c & target_c).sum().item()
            for b in range(target.shape[0]):
                if target_c[b].any():
                    self.gt_images[c] += 1
                if pred_c[b].any():
                    self.pred_images[c] += 1

    def f1(self, eps: float = 1e-7) -> dict:
        """Returns {class_idx: f1_or_None}. None means the class never
        appeared in prediction OR target across the whole pass -- genuinely
        undefined, not a trivial 1.0 (that per-image judgment is what caused
        the original bug when aggregated this way)."""
        result = {}
        for c in range(self.num_classes):
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            if tp == 0 and fp == 0 and fn == 0:
                result[c] = None
                continue
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            result[c] = 2 * precision * recall / (precision + recall + eps)
        return result

    def coverage_report(self) -> dict:
        """{class_idx: (gt_images, pred_images, total_images)} -- the
        diagnostic that actually catches class collapse: a class present in
        many ground-truth images but predicted in zero of them is a total
        failure a bare F1 number can hide."""
        return {c: (self.gt_images[c], self.pred_images[c], self.n_images)
                for c in range(self.num_classes)}


def build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    if args.data_dir:
        full = SpaceNet8Dataset(args.data_dir, image_size=args.image_size)
        n_val = max(1, int(0.1 * len(full)))
        n_train = len(full) - n_val
        train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val])
    else:
        train_ds = SyntheticFloodDataset(length=args.synthetic_train_size, image_size=args.image_size, base_seed=0)
        val_ds = SyntheticFloodDataset(length=args.synthetic_val_size, image_size=args.image_size, base_seed=100_000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser(description="Train Dual-Axis GeoFormer")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Path to a preprocessed SpaceNet-8 directory. Omit to run the "
                         "built-in synthetic pipeline-validation dataset instead.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tversky-alpha", type=float, default=0.3)
    p.add_argument("--tversky-beta", type=float, default=0.7)
    p.add_argument("--synthetic-train-size", type=int, default=64)
    p.add_argument("--synthetic-val-size", type=int, default=16)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--log-csv", type=str, default="training_log.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="geoformer", choices=["geoformer", "baseline"],
                    help="'geoformer' = Dual-Axis GeoFormer (this thesis's proposed model). "
                         "'baseline' = SN8Baseline, a from-scratch U-Net/ResNet-34 reproduction "
                         "with no bi-temporal fusion (see baseline.py) -- the comparison point "
                         "for docs/MANUAL.md's Table 2.")
    p.add_argument("--no-grid-attention", action="store_true",
                    help="Ablation (geoformer only): disable grid (global) attention, "
                         "keeping only block (local) attention. Table 2's 'GeoFormer "
                         "- grid attention' row.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (CPU -- keep --image-size / --epochs modest)"))

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)}  Val samples: {len(val_loader.dataset)}"
          + ("" if args.data_dir else "  [SYNTHETIC -- pipeline validation, not real SpaceNet-8 data]"))

    if args.model == "baseline":
        model = SN8Baseline(num_classes=NUM_CLASSES).to(device)
    else:
        model = DualAxisGeoFormer(
            GeoFormerConfig(num_classes=NUM_CLASSES, use_grid_attention=not args.no_grid_attention)
        ).to(device)
        if args.no_grid_attention:
            print("Ablation: grid attention DISABLED (block attention only)")
    print(f"Model: {args.model}  parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta, num_classes=NUM_CLASSES)

    start_epoch = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model_type = ckpt.get("model_type", "geoformer")  # older checkpoints predate this field
        if ckpt_model_type != args.model:
            raise SystemExit(
                f"--resume checkpoint was trained with --model {ckpt_model_type}, "
                f"but --model {args.model} was requested. Match them, or drop --resume "
                f"to train {args.model} from scratch."
            )
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # BUG THIS FIXES: optimizer.load_state_dict restores the PREVIOUS run's
        # param_group lr too -- which, after a full cosine anneal, is ~0. Left
        # alone, that silently overrides whatever --lr was just requested, so a
        # "fine-tune at a new --lr" resume trains at an effectively frozen
        # learning rate (only BatchNorm running stats still move, from forward
        # passes in .train() mode, which looks like faint progress in the loss
        # curve but isn't real learning). Force the requested lr back in.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, lr reset to {args.lr:.2e}")

    # T_max is the REMAINING epoch count, not args.epochs -- a fresh
    # CosineAnnealingLR built with T_max=args.epochs on a resume would anneal
    # as if starting from epoch 0, reaching 0 far later than --epochs actually
    # ends (or, combined with the lr restored above, from the wrong lr entirely).
    remaining_epochs = max(1, args.epochs - start_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    # BUG THIS FIXES: if --resume's checkpoint epoch is already >= --epochs,
    # `range(start_epoch, args.epochs)` below is empty, log_rows stays empty,
    # and the CSV writer crashes at the end with a bare `IndexError: list
    # index out of range` -- true, but useless for explaining what actually
    # went wrong. Fail loudly and specifically, here, instead.
    if start_epoch >= args.epochs:
        raise SystemExit(
            f"Nothing to do: --resume checkpoint is already at epoch {start_epoch}, "
            f"which is >= --epochs {args.epochs}. Pass a larger --epochs to train "
            f"further (e.g. --epochs {start_epoch + 10})."
        )

    # On --resume, keep the existing log's history instead of starting the
    # CSV over from this run's first epoch -- otherwise the same --log-csv
    # path across a resume silently loses everything before start_epoch the
    # moment this run writes its first row.
    log_rows = []
    if args.resume and Path(args.log_csv).exists():
        with open(args.log_csv, newline="") as f:
            log_rows = list(csv.DictReader(f))
        print(f"Continuing existing log {args.log_csv} ({len(log_rows)} prior rows)")

    class_names = ["background", "building", "road", "flooded"]

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for pre, post, mask in train_loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(pre, post)
            loss = loss_fn(out["logits"], mask)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        acc = ConfusionAccumulator(NUM_CLASSES)
        with torch.no_grad():
            for pre, post, mask in val_loader:
                pre, post, mask = pre.to(device), post.to(device), mask.to(device)
                out = model(pre, post)
                val_loss_sum += loss_fn(out["logits"], mask).item()
                n_val_batches += 1
                acc.update(out["logits"], mask)
        val_loss = val_loss_sum / max(1, n_val_batches)
        f1_final = acc.f1()
        coverage = acc.coverage_report()
        dt = time.time() - t0

        def _fmt(c):
            f1v = f1_final[c]
            if f1v is None:
                return f"{class_names[c]}=n/a(never seen)"
            gt_n, pred_n, total_n = coverage[c]
            flag = "" if (gt_n == 0 or pred_n > 0) else "!COLLAPSED"
            return f"{class_names[c]}={f1v:.3f}{flag}"

        f1_str = "  ".join(_fmt(c) for c in range(NUM_CLASSES))
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_F1[{f1_str}]  lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")

        log_rows.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            **{f"val_f1_{class_names[c]}": (f1_final[c] if f1_final[c] is not None else float("nan"))
               for c in range(NUM_CLASSES)},
            **{f"val_coverage_{class_names[c]}_pred_images": coverage[c][1] for c in range(NUM_CLASSES)},
            "lr": scheduler.get_last_lr()[0], "seconds": round(dt, 2),
        })
        # BUG THIS FIXES: the CSV used to be written once, after the whole
        # epoch loop finished -- so a run killed mid-training (this
        # environment's OOM kills, repeatedly, this session) lost the entire
        # log, even though the model checkpoint and dataset index.json both
        # survive via their own per-step writes. Write it every epoch instead,
        # same incremental-save philosophy applied consistently.
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

        ckpt_payload = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_type": args.model,
            # stored as a plain dict, not the dataclass instance -- torch>=2.6
            # defaults torch.load(weights_only=True), which refuses to
            # unpickle arbitrary classes (including our own GeoFormerConfig).
            # SN8Baseline has no config dataclass -- None for that model type.
            "config_dict": dataclasses.asdict(model.cfg) if args.model == "geoformer" else None,
            "val_loss": val_loss, "best_val_loss": min(best_val_loss, val_loss),
            # BUG THIS FIXES: nothing previously recorded whether a checkpoint
            # was trained on real or synthetic data, or which real dataset --
            # a script showing a checkpoint's results had no reliable way to
            # say what it was actually trained on except a caller's assumption
            # (real_image_demo.py's caption used to hardcode "synthetic data
            # only" regardless of what checkpoint was actually passed in).
            "data_source": args.data_dir if args.data_dir else "synthetic",
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  -> new best (val_loss={val_loss:.4f}), saved {ckpt_dir / 'best.pt'}")

    print(f"\n{args.log_csv} is up to date (written every epoch). Best val_loss: {best_val_loss:.4f}. "
          f"Checkpoints in {ckpt_dir}/ (best.pt, last.pt).")


if __name__ == "__main__":
    main()
