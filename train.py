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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SyntheticFloodDataset, SpaceNet8Dataset, NUM_CLASSES
from losses import TverskyLoss, AsymmetricUnifiedFocalLoss, RegionMutualInformationLoss, TopKLoss
from model import DualAxisGeoFormer, GeoFormerConfig
from baseline import SN8Baseline
from db.db_logger import DBLogger


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


def checkpoint_score(metric: str, val_loss: float, f1_final: dict) -> float:
    """Lower is always better, whichever metric is chosen, so a single
    min()-based comparison drives checkpoint selection regardless.

    WHY THIS EXISTS (docs/MANUAL.md S12.12): val_loss-based "best"
    selection was found, concretely, to prefer a COLLAPSED checkpoint
    over a genuinely useful one. Tversky loss is dominated by whichever
    classes have the most pixels (background, then road); a model that
    stops predicting building/flooded at all can have a LOWER val_loss
    than one that predicts them imperfectly but for real -- confirmed on
    this exact project's own v5 run: epoch 2 (building F1 0.49, flooded
    F1 0.14, both with real per-image coverage) had val_loss 0.686;
    epoch 5 (building/flooded both exactly 0.000, totally collapsed) had
    val_loss 0.370 -- LOWER, so epoch 5 became "best.pt" and epoch 2's
    weights were never saved as best and are now gone. 'mean_f1' and
    'min_f1' score checkpoints by what this project actually cares about
    (balanced multi-class detection, not just loss magnitude) instead.
    """
    if metric == "val_loss":
        return val_loss
    f1_values = [v for v in f1_final.values() if v is not None]
    if not f1_values:
        return float("inf")  # no class has appeared in gt or pred yet -- nothing to score
    if metric == "mean_f1":
        return -(sum(f1_values) / len(f1_values))
    if metric == "min_f1":
        return -min(f1_values)
    raise ValueError(f"Unknown --checkpoint-metric '{metric}'")


def ema_init(model) -> dict:
    """A fresh EMA shadow state, a plain CPU-agnostic clone of the model's
    current weights -- called once at the start of training (or on resume,
    from the just-loaded raw weights, if the checkpoint predates EMA)."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def ema_update(ema_state: dict, model, momentum: float) -> None:
    """In-place: ema = ema * (1 - momentum) + raw_weights * momentum.
    Integer buffers (e.g. BatchNorm's num_batches_tracked) are copied
    directly rather than blended -- an averaged step COUNT is meaningless,
    unlike an averaged floating-point weight or running statistic."""
    raw_state = model.state_dict()
    with torch.no_grad():
        for k, ema_v in ema_state.items():
            raw_v = raw_state[k]
            if torch.is_floating_point(ema_v):
                ema_v.mul_(1 - momentum).add_(raw_v, alpha=momentum)
            else:
                ema_v.copy_(raw_v)


def reinit_flood_head(model, optimizer) -> None:
    """Replace model.flood_head's weights with a fresh nn.Conv2d init
    (same as a brand-new model's flood_head would get) and drop Adam's
    per-parameter moment estimates for those weights, leaving every
    other module (trunk/structure_head/backbone) exactly as loaded from
    the checkpoint.

    WHY THIS EXISTS (docs/MANUAL.md S12.24-S12.25): v11 showed that
    reweighting the flood loss more aggressively does NOT recover a
    flood head that has already saturated into always-predicting
    background -- a saturated logit has near-zero local gradient
    regardless of loss weight, so a stronger loss just multiplies a
    near-zero gradient by a bigger number and still gets a near-zero
    gradient. The fix has to give the head a fresh, non-saturated
    starting point instead of asking loss reweighting to argue it out
    of a state it can no longer see a gradient out of. Clearing the
    optimizer state matters too: leaving Adam's old exp_avg/exp_avg_sq
    in place would have its first several updates on the fresh weights
    still shaped by the collapsed run's stale gradient statistics.
    """
    fresh_flood_head = nn.Conv2d(model.flood_head.in_channels, model.flood_head.out_channels,
                                  model.flood_head.kernel_size)
    model.flood_head.load_state_dict(fresh_flood_head.state_dict())
    for p in model.flood_head.parameters():
        optimizer.state.pop(p, None)


def flood_head_patience_step(flood_f1_history, best_so_far, epochs_since_improved, patience,
                              smooth_window):
    """Pure decision function for the --flood-head-patience mechanism: given the
    val_f1_flooded history so far (this epoch's value already appended) and the
    tracker state, return (new_best_so_far, new_epochs_since_improved, should_freeze).

    Compares a SMOOTHED value (mean of the last `smooth_window` epochs) against the
    smoothed best-so-far, not the raw single-epoch value.

    WHY THIS EXISTS (docs/MANUAL.md S12.45): the original implementation compared
    raw per-epoch val_f1_flooded directly. On v14, flooded pixels are <1% of the
    data with only ~20-28 of 87 val tiles containing any -- single-epoch val_f1_flooded
    is noisy enough that 4 epochs of ordinary sampling variance (not genuine collapse)
    tripped patience=4 right after a real peak (epoch 51, F1=0.5463), permanently
    freezing flood_head for the remaining ~100 epochs of a 150-epoch run. Smoothing
    both sides of the comparison the same way absorbs that noise; smooth_window=1
    recovers the exact old raw-value behavior.
    """
    window = flood_f1_history[-smooth_window:]
    smoothed = sum(window) / len(window)
    if smoothed > best_so_far:
        return smoothed, 0, False
    epochs_since_improved += 1
    return best_so_far, epochs_since_improved, epochs_since_improved >= patience


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
        self.update_pred(logits.argmax(dim=1), target)

    def update_pred(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Same accumulation as update(), for callers that already have a
        final (B, H, W) class-index prediction rather than logits -- e.g.
        postprocess.py's suppress_isolated_flood_predictions, whose output
        isn't logits at all (docs/RESEARCH_NOTES.md item 7)."""
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
        # BUG THIS FIXES: random_split's split depends on len(full) and index
        # order, both of which change every time real_sn8_dataset_full grows
        # (this exact project's dataset went 202 -> 352 -> 801 tiles across
        # this session's runs). A hash-based split keeps each tile on the
        # same side of train/val forever -- see SpaceNet8Dataset.split's
        # docstring for the full story of what this actually broke.
        train_idx, val_idx = full.split(val_fraction=0.1)
        # A second dataset instance (same index.json, cheap to parse twice)
        # rather than a flag flipped on the shared one -- val must keep
        # seeing each tile in its one real orientation every epoch, or
        # held-out numbers stop being comparable epoch to epoch. Sharing
        # one instance would mean either both splits augment or neither
        # does; this keeps them independently controlled.
        train_source = SpaceNet8Dataset(args.data_dir, image_size=args.image_size, augment=args.augment,
                                         copy_paste_prob=args.copy_paste_prob) \
            if (args.augment or args.copy_paste_prob > 0) else full
        train_ds = torch.utils.data.Subset(train_source, train_idx)
        val_ds = torch.utils.data.Subset(full, val_idx)

        if args.oversample_rare_classes:
            weights = full.class_presence_weights(train_idx)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(train_idx), replacement=True
            )
            n_building = sum(1 for w in weights if w > 1.0)
            print(f"Oversampling rare classes: {n_building}/{len(train_idx)} train tiles "
                  f"contain building/flooded pixels and are drawn more often per epoch")
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    else:
        train_ds = SyntheticFloodDataset(length=args.synthetic_train_size, image_size=args.image_size, base_seed=0)
        val_ds = SyntheticFloodDataset(length=args.synthetic_val_size, image_size=args.image_size, base_seed=100_000)
        # Synthetic samples are generated in-process (make_synthetic_sample) --
        # cheap enough that worker-process overhead isn't worth it here even
        # if --num-workers is passed; real data is where it matters.
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers if args.data_dir else 0,
                             persistent_workers=(args.num_workers > 0) if args.data_dir else False)
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
    p.add_argument("--class-weights", type=str, default=None,
                    help="Comma-separated per-class weight for TverskyLoss's class average, in "
                         "background,building,road,flooded order (e.g. '1,3,1,5'). Default: uniform "
                         "(the original behavior). Complements --oversample-rare-classes: sampling "
                         "controls how OFTEN a rare-class tile is seen, this controls how much the "
                         "LOSS cares about that class once it is -- see docs/MANUAL.md S12.7-S12.8 "
                         "for why oversampling alone didn't prevent the building/flooded collapse.")
    p.add_argument("--focal-gamma", type=float, default=1.0,
                    help="Focal Tversky Loss exponent (Abraham & Khan 2018): raises each class's "
                         "(1 - Tversky index) to the power 1/gamma, concentrating gradient on pixels "
                         "the model still gets wrong within a class -- complements --class-weights "
                         "(which reweights classes against each other) rather than replacing it. "
                         "1.0 (default) is the identity power, exact original behavior. Typical "
                         "useful range 1.0-3.0. See docs/MANUAL.md S12.14-S12.15.")
    p.add_argument("--synthetic-train-size", type=int, default=64)
    p.add_argument("--synthetic-val-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                    help="Real data only (--data-dir): DataLoader worker processes for image "
                         "decode/resize, overlapping the NEXT batch's loading with the CURRENT "
                         "batch's forward/backward instead of doing both serially. Pure wall-clock "
                         "speedup, no effect on what gets trained -- try min(4, os.cpu_count()-1) "
                         "or so; ignored for the synthetic dataset (cheap enough in-process).")
    p.add_argument("--checkpoint-metric", type=str, default="val_loss",
                    choices=["val_loss", "mean_f1", "min_f1"],
                    help="What 'best.pt' tracks. 'val_loss' (default, exact prior behavior) can "
                         "prefer a collapsed checkpoint over a genuinely useful one -- see "
                         "docs/MANUAL.md S12.12. 'mean_f1' maximizes the average per-class F1 "
                         "across classes seen so far; 'min_f1' maximizes the WORST class's F1 "
                         "(the strictest anti-collapse choice -- a checkpoint can't be 'best' "
                         "while any seen class is at 0).")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=10,
                    help="Save a non-overwritten epoch_N.pt snapshot every N epochs, independent "
                         "of last.pt/best.pt -- a safety net so one bad epoch (e.g. a regression "
                         "right after --resume) can't erase the only recoverable checkpoint.")
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
    p.add_argument("--oversample-rare-classes", action="store_true",
                    help="Real data only (--data-dir): use a WeightedRandomSampler over the "
                         "TRAIN split so tiles containing building/flooded pixels are drawn "
                         "more often per epoch than their raw prevalence. Targets the "
                         "building/flooded total-collapse finding in docs/MANUAL.md S12.3-S13 "
                         "-- val stays unweighted so evaluation numbers stay honest.")
    p.add_argument("--augment", action="store_true",
                    help="Real data only (--data-dir): random flips + 90-degree rotations on the "
                         "TRAIN split (pre/post/mask transformed identically, val untouched). "
                         "Satellite imagery has no canonical orientation -- this project's real-data "
                         "training had no geometric augmentation at all before this flag existed. "
                         "See docs/MANUAL.md S12.16.")
    p.add_argument("--copy-paste-prob", type=float, default=0.0,
                    help="Real data only (--data-dir), TRAIN split only: probability per tile of "
                         "pasting a randomly-chosen OTHER tile's entire flooded-pixel footprint "
                         "(pre, post, AND mask together, at the same pixel coordinates) onto this "
                         "tile before training on it. 0.0 (default) disables it -- exact prior "
                         "behavior. Motivated by docs/MANUAL.md S12.33: four different model/loss-"
                         "level fixes for flooded's collapse all converged on nearly the same "
                         "training point, evidence for a data-exposure limit rather than a "
                         "model-side one -- this manufactures more flooded-pixel exposure per "
                         "epoch from the SAME 801 tiles already in hand (Ghiasi et al. 2021, "
                         "'Simple Copy-Paste is a Strong Data Augmentation Method'), testing that "
                         "hypothesis without waiting on new data collection.")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="geoformer + --pretrained-backbone only: freeze the pretrained backbone's "
                         "weights for this many epochs before unfreezing. Standard transfer-learning "
                         "practice -- less backward-pass compute and activation memory while frozen "
                         "(the backbone runs under torch.no_grad()), and protects the pretrained "
                         "ImageNet features from early, noisy gradients from an untrained decoder/head. "
                         "0 (default) never freezes -- exact prior behavior.")
    p.add_argument("--pretrained-backbone", type=str, default=None,
                    help="geoformer only: an ImageNet-pretrained timm model name (e.g. "
                         "'efficientnet_b0') supporting features_only=True at strides "
                         "4/8/16/32, used as the encoder's feature source instead of the "
                         "from-scratch stem -- Phase 2 of the thesis, see docs/MANUAL.md "
                         "S12.10-S12.11. Requires `pip install timm` and a first-run internet "
                         "download of the pretrained weights.")
    p.add_argument("--separate-flood-head", action="store_true",
                    help="geoformer only: decouple 'flooded' from the joint 4-way softmax into "
                         "its own binary head with its own gradient pathway, separate from the "
                         "3-way background/building/road structure head. Motivated by forensic "
                         "debugging of this project's own building/flooded collapse -- see "
                         "docs/MANUAL.md S12.14 item 1, S12.17-S12.18. Off by default: exact "
                         "prior single-head behavior, existing checkpoints unaffected.")
    p.add_argument("--flood-class-weight", type=float, default=None,
                    help="--separate-flood-head only: the flood loss's own [not-flooded, flooded] "
                         "class-weight ratio is [1.0, this]. Defaults to class_weights[3] if "
                         "--class-weights is set, else 1.0 (uniform) -- same as before this flag "
                         "existed. v10 (docs/MANUAL.md S12.23) showed the separate head alone "
                         "still lets flooded collapse in isolation (its own severe class "
                         "imbalance, not cross-class competition) once inherited from "
                         "--class-weights's structure-loss-oriented value (4.0) -- this flag lets "
                         "the flood loss use a much higher, independently-tuned weight instead.")
    p.add_argument("--flood-tversky-beta", type=float, default=None,
                    help="--separate-flood-head only: the flood loss's own false-negative weight "
                         "(beta). Defaults to --tversky-beta (shared with the structure loss) if "
                         "unset -- same as before this flag existed. A higher beta specifically "
                         "for flood pushes harder against missing flooded pixels without changing "
                         "how the structure loss weights building/road/background.")
    p.add_argument("--flood-bce-weight", type=float, default=None,
                    help="--separate-flood-head only: adds `weight * "
                         "binary_cross_entropy_with_logits(flood_logit, flood_target)` to the "
                         "flood loss, alongside the existing Tversky term. None (default) "
                         "disables it -- exact prior behavior. Motivated by real evidence "
                         "(docs/MANUAL.md S12.30): S12.24 diagnosed the flood head's collapse "
                         "as the logit saturating into a region where Tversky's gradient (a "
                         "global TP/FP/FN ratio) goes near-zero once predicted positives vanish "
                         "-- a known degenerate property of pure Dice/Tversky losses on an "
                         "emptying mask. BCE is computed per-pixel independently and never goes "
                         "flat at zero predicted positives, so it keeps injecting real gradient "
                         "in exactly the regime where Tversky alone currently stalls. This is "
                         "also literally what the SpaceNet-8 competition's own 5th-place "
                         "solution used for its flood head (`1*dice + 1*bce`) -- a weight of "
                         "1.0 matches their convention.")
    p.add_argument("--flood-loss-fn", choices=["tversky", "unified_focal"], default="tversky",
                    help="--separate-flood-head only: which loss computes the flood head's "
                         "primary term. 'tversky' (default) is exact prior behavior -- "
                         "TverskyLoss with --flood-tversky-beta. 'unified_focal' swaps in "
                         "AsymmetricUnifiedFocalLoss (Yeung et al. 2022, docs/RESEARCH_NOTES.md "
                         "item 4) instead -- a compound loss purpose-built for severe class "
                         "imbalance that treats the foreground/minority class and the background "
                         "class asymmetrically throughout, rather than applying the same focal "
                         "down-weighting to both the way --focal-gamma does.")
    p.add_argument("--flood-rmi-weight", type=float, default=None,
                    help="--separate-flood-head only: adds `weight * RegionMutualInformationLoss` "
                         "to the flood loss, alongside the primary --flood-loss-fn term. None "
                         "(default) disables it -- exact prior behavior. This is the actual flood "
                         "loss the SpaceNet-8 challenge's 1st-place team used (docs/RESEARCH_NOTES.md "
                         "item 3) -- unlike every other loss here, which scores each pixel "
                         "independently, RMI scores local NEIGHBORHOODS, rewarding a prediction "
                         "that gets the flood boundary's SHAPE right, not just its pixel count.")
    p.add_argument("--flood-topk-weight", type=float, default=None,
                    help="--separate-flood-head only: adds `weight * TopKLoss` (BCE computed only "
                         "on the hardest --flood-topk-fraction of pixels) to the flood loss. None "
                         "(default) disables it -- exact prior behavior. A different lever from "
                         "every other flood loss term: those change how much each pixel's error "
                         "counts, this changes which pixels get to contribute a gradient at all "
                         "(docs/RESEARCH_NOTES.md item 6).")
    p.add_argument("--flood-topk-fraction", type=float, default=0.15,
                    help="--flood-topk-weight only: fraction of pixels (by hardest per-pixel BCE) "
                         "that contribute to the TopK term. Default 0.15, unused unless "
                         "--flood-topk-weight is set.")
    p.add_argument("--ema-momentum", type=float, default=None,
                    help="Exponential moving average of model weights: after every optimizer "
                         "step, ema = ema * (1 - momentum) + raw_weights * momentum. The EMA "
                         "weights (not the raw ones) are what validation/checkpoint-selection "
                         "actually evaluates, and what get saved as the 'production' weights "
                         "(checkpoint_utils.load_checkpoint_model prefers ema_state when "
                         "present). None (default) disables EMA entirely -- exact prior "
                         "behavior. Motivated by real, external evidence (docs/MANUAL.md "
                         "S12.29): the SpaceNet-8 competition's own 5th-place solution "
                         "(github.com/motokimura/spacenet8_solution_5th-place) reports the "
                         "exact same flood-detection instability this project found (S12.17- "
                         "S12.28) -- 'the validation metric varied significantly from epoch to "
                         "epoch' -- and used EMA (momentum 2e-3) specifically to mitigate it. "
                         "Typical values are small, e.g. 0.002-0.01 -- a small momentum means "
                         "the EMA changes slowly, averaging out epoch-to-epoch noise/oscillation "
                         "rather than tracking every collapse-and-recover swing.")
    p.add_argument("--reinit-flood-head", action="store_true",
                    help="--resume + --separate-flood-head only: after loading the checkpoint, "
                         "reinitialize ONLY model.flood_head's own weights (fresh nn.Conv2d init) "
                         "instead of continuing from the checkpoint's flood_head weights, keeping "
                         "the trunk/structure_head/backbone's already-learned features intact. "
                         "Motivated by v11 (docs/MANUAL.md S12.24): reweighting the flood loss "
                         "alone did not recover a flood head that had already saturated into "
                         "always-predicting-background -- a saturated logit has near-zero local "
                         "gradient regardless of loss weight, so the fix has to give the head a "
                         "fresh, non-saturated starting point, not just a stronger pull on the "
                         "same starting point. No effect without --resume (a fresh model's "
                         "flood_head is already randomly initialized).")
    p.add_argument("--flood-head-patience", type=int, default=None,
                    help="--separate-flood-head only: once val_f1_flooded hasn't beaten its own "
                         "best-so-far for this many consecutive epochs, freeze flood_head's "
                         "parameters (requires_grad=False) for the REST of training -- it stops "
                         "learning at (approximately) its own peak instead of continuing to train "
                         "into the collapse S12.17-S12.33 characterized, while structure_head/"
                         "backbone/split_trunk keep training normally. None (default) disables "
                         "this -- exact prior behavior. IMPORTANT CAVEAT, stated honestly rather "
                         "than oversold: flood_head reads from split_trunk, which is SHARED with "
                         "structure_head and keeps training after the freeze (driven by the "
                         "structure loss) -- freezing flood_head does not fully insulate it from "
                         "the trunk's continuing evolution, so this is a real experiment, not a "
                         "guaranteed fix. min_f1 checkpoint selection already protects the best "
                         "checkpoint from being lost either way; this additionally tries to keep "
                         "TRAINING itself from moving away from a good flood state while still "
                         "letting building/road/background keep improving.")
    p.add_argument("--flood-head-patience-smooth-window", type=int, default=3,
                    help="--flood-head-patience only: instead of comparing each epoch's raw "
                         "val_f1_flooded against the best-so-far, compare the mean of the last N "
                         "epochs (N=this value) -- both the running best and the current value "
                         "are smoothed the same way. Bug found in practice (v14, docs/MANUAL.md "
                         "S12.45): with the raw per-epoch value and the default patience of 4, "
                         "flooded F1 peaked at epoch 51 (0.5463) then hit 4 epochs of ordinary "
                         "sampling noise (52-55, still in the 0.52-0.56 band it had been in all "
                         "along) and froze flood_head for the remaining ~100 epochs of a 150-epoch "
                         "run -- flooded pixels are under 1%% of the data and only ~20-28 of 87 val "
                         "tiles contain any, so single-epoch val_f1_flooded is noisy enough that "
                         "patience=4 on the raw value triggers on normal variance, not genuine "
                         "collapse. Set to 1 to recover the old raw-value behavior exactly.")
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
            GeoFormerConfig(num_classes=NUM_CLASSES, use_grid_attention=not args.no_grid_attention,
                             pretrained_backbone=args.pretrained_backbone,
                             separate_flood_head=args.separate_flood_head)
        ).to(device)
        if args.no_grid_attention:
            print("Ablation: grid attention DISABLED (block attention only)")
        if args.pretrained_backbone:
            print(f"Encoder: ImageNet-pretrained '{args.pretrained_backbone}' backbone "
                  f"(Phase 2) feeding the existing MaxViTBlock attention stages")
        if args.separate_flood_head:
            print("Separate flood head: 'flooded' decoupled from the joint softmax -- "
                  "own gradient pathway, own loss term (docs/MANUAL.md S12.17-S12.18)")
    print(f"Model: {args.model}  parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = None
    if args.class_weights:
        class_weights = [float(w) for w in args.class_weights.split(",")]
        if len(class_weights) != NUM_CLASSES:
            raise SystemExit(f"--class-weights needs {NUM_CLASSES} comma-separated values "
                              f"(background,building,road,flooded), got {len(class_weights)}: {args.class_weights}")
        print(f"Loss class weights: background={class_weights[0]} building={class_weights[1]} "
              f"road={class_weights[2]} flooded={class_weights[3]}")
    if args.focal_gamma != 1.0:
        print(f"Focal Tversky gamma={args.focal_gamma} (concentrates loss on still-hard pixels within each class)")

    flood_loss_fn = None
    flood_rmi_fn = None
    flood_topk_fn = None
    if args.separate_flood_head:
        structure_weights = class_weights[:3] if class_weights else None
        flood_class_weight = args.flood_class_weight
        if flood_class_weight is None:
            flood_class_weight = class_weights[3] if class_weights else 1.0
        flood_beta = args.flood_tversky_beta if args.flood_tversky_beta is not None else args.tversky_beta
        bce_note = f" + {args.flood_bce_weight}*BCE" if args.flood_bce_weight is not None else ""
        rmi_note = f" + {args.flood_rmi_weight}*RMI" if args.flood_rmi_weight is not None else ""
        topk_note = f" + {args.flood_topk_weight}*TopK({args.flood_topk_fraction})" if args.flood_topk_weight is not None else ""
        print(f"Flood loss [{args.flood_loss_fn}]: class_weight=[1.0, {flood_class_weight}] "
              f"beta={flood_beta}{bce_note}{rmi_note}{topk_note} "
              f"(independent of the structure loss's alpha={args.tversky_alpha}/beta={args.tversky_beta})")
        loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta, num_classes=3,
                               class_weights=structure_weights, focal_gamma=args.focal_gamma)
        if args.flood_loss_fn == "unified_focal":
            flood_loss_fn = AsymmetricUnifiedFocalLoss(delta=flood_beta)
        else:
            flood_loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=flood_beta, num_classes=2,
                                         class_weights=[1.0, flood_class_weight], focal_gamma=args.focal_gamma)
        if args.flood_rmi_weight is not None:
            flood_rmi_fn = RegionMutualInformationLoss(num_classes=2)
        if args.flood_topk_weight is not None:
            flood_topk_fn = TopKLoss(top_k_fraction=args.flood_topk_fraction)
    else:
        loss_fn = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta, num_classes=NUM_CLASSES,
                               class_weights=class_weights, focal_gamma=args.focal_gamma)

    def compute_loss(out: dict, mask: torch.Tensor) -> torch.Tensor:
        """Structure loss (background/building/road, computed only on
        non-flooded pixels) + an independent flood loss (binary,
        computed on every pixel) when the model has a separate flood
        head; the original single 4-class loss otherwise. Isolated here
        so train/val both use exactly the same combination logic."""
        if flood_loss_fn is None:
            return loss_fn(out["logits"], mask)
        valid = mask != 3  # exclude flooded pixels: their true structure class was already lost at rasterization
        structure_target = mask.clamp(max=2)  # placeholder value at excluded pixels; valid_mask zeroes their contribution
        structure_loss = loss_fn(out["structure_logits"], structure_target, valid_mask=valid)
        flood_target = (mask == 3).long()
        flood_logits_2ch = torch.cat([-out["flood_logit"], out["flood_logit"]], dim=1)
        flood_loss = flood_loss_fn(flood_logits_2ch, flood_target)
        if args.flood_bce_weight is not None:
            # out["flood_logit"] is (B, 1, H, W); flood_target is (B, H, W) of 0/1.
            # BCE stays computed per-pixel even once Tversky's gradient goes
            # near-zero on an emptying predicted mask (docs/MANUAL.md S12.30)
            # -- see --flood-bce-weight's own help text for the full rationale.
            bce = F.binary_cross_entropy_with_logits(out["flood_logit"], flood_target.unsqueeze(1).float())
            flood_loss = flood_loss + args.flood_bce_weight * bce
        if flood_rmi_fn is not None:
            flood_loss = flood_loss + args.flood_rmi_weight * flood_rmi_fn(flood_logits_2ch, flood_target)
        if flood_topk_fn is not None:
            flood_loss = flood_loss + args.flood_topk_weight * flood_topk_fn(out["flood_logit"], flood_target)
        return structure_loss + flood_loss

    start_epoch = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

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
        if args.reinit_flood_head:
            if not args.separate_flood_head or not hasattr(model, "flood_head"):
                raise SystemExit("--reinit-flood-head needs --separate-flood-head "
                                  "(the checkpoint's model has no flood_head to reinitialize).")
            reinit_flood_head(model, optimizer)
            print("Reinitialized flood_head's weights (fresh init) and cleared its optimizer "
                  "state -- trunk/structure_head/backbone kept their checkpoint weights.")
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
        # BUG THIS FIXES: best_val_loss used to be carried forward from the
        # checkpoint unconditionally. This project's real checkpoint chain
        # was, at some point, resumed from a SYNTHETIC-data checkpoint
        # (val_loss ~0.006 -- an easier task on a completely different loss
        # scale) and that best_val_loss kept propagating forward through
        # every subsequent real-data --resume's ckpt_payload forever, because
        # real val_loss (~0.4) can never beat it. The practical effect: on
        # this exact project, checkpoints/best.pt silently stopped updating
        # after the very first real-data run and stayed frozen at a stale
        # epoch-36 synthetic checkpoint for the rest of the project's
        # history -- discovered only when an oversampling run regressed
        # (val_loss 0.42 -> 0.45) and best.pt turned out to hold no
        # real-data safety net at all. Only trust a resumed best_val_loss
        # when it came from a checkpoint trained on the SAME data_source;
        # otherwise this is a regime change and "best" starts over.
        ckpt_data_source = ckpt.get("data_source")
        current_data_source = args.data_dir if args.data_dir else "synthetic"
        # Same regime-change logic as data_source above, extended to the
        # checkpoint metric: a best_score computed as -mean_f1 is not
        # comparable to one computed as val_loss, so only inherit it when
        # BOTH data_source and checkpoint_metric match this run's.
        ckpt_metric = ckpt.get("checkpoint_metric", "val_loss")  # older checkpoints predate this field
        if ckpt_data_source == current_data_source and ckpt_metric == args.checkpoint_metric:
            best_score = ckpt.get("best_score", best_score)
        else:
            print(f"Resumed checkpoint's (data_source='{ckpt_data_source}', metric='{ckpt_metric}') differs "
                  f"from this run's (data_source='{current_data_source}', metric='{args.checkpoint_metric}') "
                  f"-- treating this as a new regime, best_score restarts at inf instead of inheriting a "
                  f"value that isn't comparable.")
        print(f"Resumed from {args.resume} at epoch {start_epoch}, lr reset to {args.lr:.2e}")

    ema_state = None
    if args.ema_momentum is not None:
        if args.resume and ckpt.get("ema_state") is not None:
            ema_state = {k: v.clone() for k, v in ckpt["ema_state"].items()}
            print(f"Resumed EMA state from {args.resume} (ema_momentum={args.ema_momentum})")
        else:
            ema_state = ema_init(model)
            reason = "checkpoint predates EMA" if args.resume else "fresh model"
            print(f"EMA enabled (momentum={args.ema_momentum}), initialized fresh from {reason}")

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

    db_logger = DBLogger(
        run_name=Path(args.log_csv).name,
        model_type=args.model,
        data_source=args.data_dir if args.data_dir else "synthetic",
        config_dict=dataclasses.asdict(model.cfg) if args.model == "geoformer" else None,
        # ckpt["run_name"] is the --resume'd checkpoint's OWN recorded
        # run_name (see ckpt_payload below) -- None on a from-scratch run,
        # or if resuming from a checkpoint that predates this field.
        parent_run_name=ckpt.get("run_name") if args.resume else None,
    )

    best_flood_f1_so_far = -1.0
    epochs_since_flood_f1_improved = 0
    flood_head_frozen = False
    flood_f1_history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        # model.train() above puts every submodule (including the backbone)
        # back into train() mode -- re-apply the freeze state (which also
        # forces the backbone specifically back to eval()) every epoch,
        # not just once, or an epoch boundary would silently undo it.
        if args.model == "geoformer" and args.pretrained_backbone:
            should_freeze = epoch < args.freeze_backbone_epochs
            if epoch == start_epoch or should_freeze != (epoch - 1 < args.freeze_backbone_epochs):
                print(f"Backbone {'frozen' if should_freeze else 'unfrozen'} (epoch {epoch + 1})")
            model.encoder.set_backbone_frozen(should_freeze)
        train_loss_sum, n_batches = 0.0, 0
        for pre, post, mask in train_loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(pre, post)
            loss = compute_loss(out, mask)
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                ema_update(ema_state, model, args.ema_momentum)
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = train_loss_sum / max(1, n_batches)

        # Validate against the EMA weights, not the raw ones being trained --
        # the whole point (docs/MANUAL.md S12.29) is a smoothed, less noisy
        # signal for both the printed/logged metrics and checkpoint
        # selection. Swap the EMA weights in for eval, then restore the raw
        # ones so the NEXT epoch's training resumes from where the optimizer
        # actually left off, not from the smoothed snapshot.
        raw_state_for_restore = None
        if ema_state is not None:
            raw_state_for_restore = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        acc = ConfusionAccumulator(NUM_CLASSES)
        with torch.no_grad():
            for pre, post, mask in val_loader:
                pre, post, mask = pre.to(device), post.to(device), mask.to(device)
                out = model(pre, post)
                val_loss_sum += compute_loss(out, mask).item()
                n_val_batches += 1
                acc.update(out["logits"], mask)
        val_loss = val_loss_sum / max(1, n_val_batches)
        f1_final = acc.f1()
        coverage = acc.coverage_report()
        dt = time.time() - t0

        if args.flood_head_patience is not None and not flood_head_frozen:
            flood_f1_this_epoch = f1_final[3] if f1_final[3] is not None else -1.0
            flood_f1_history.append(flood_f1_this_epoch)
            best_flood_f1_so_far, epochs_since_flood_f1_improved, should_freeze = flood_head_patience_step(
                flood_f1_history, best_flood_f1_so_far, epochs_since_flood_f1_improved,
                args.flood_head_patience, args.flood_head_patience_smooth_window)
            if should_freeze:
                for p_ in model.flood_head.parameters():
                    p_.requires_grad_(False)
                flood_head_frozen = True
                print(f"  -> flood_head FROZEN (epoch {epoch + 1}): smoothed flooded F1 "
                      f"(mean of last {args.flood_head_patience_smooth_window} epochs) hasn't "
                      f"beaten {best_flood_f1_so_far:.4f} for {args.flood_head_patience} epochs "
                      f"(docs/MANUAL.md S12.34, S12.45)")

        if raw_state_for_restore is not None:
            model.load_state_dict(raw_state_for_restore)

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
        db_logger.log_epoch(log_rows[-1])

        ckpt_payload = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_type": args.model,
            # stored as a plain dict, not the dataclass instance -- torch>=2.6
            # defaults torch.load(weights_only=True), which refuses to
            # unpickle arbitrary classes (including our own GeoFormerConfig).
            # SN8Baseline has no config dataclass -- None for that model type.
            "config_dict": dataclasses.asdict(model.cfg) if args.model == "geoformer" else None,
            "val_loss": val_loss,
            "checkpoint_metric": args.checkpoint_metric,
            "best_score": min(best_score, checkpoint_score(args.checkpoint_metric, val_loss, f1_final)),
            # BUG THIS FIXES: nothing previously recorded whether a checkpoint
            # was trained on real or synthetic data, or which real dataset --
            # a script showing a checkpoint's results had no reliable way to
            # say what it was actually trained on except a caller's assumption
            # (real_image_demo.py's caption used to hardcode "synthetic data
            # only" regardless of what checkpoint was actually passed in).
            "data_source": args.data_dir if args.data_dir else "synthetic",
            "ema_state": ema_state, "ema_momentum": args.ema_momentum,
            # This run's own log-csv name -- lets a FUTURE --resume from
            # this checkpoint record accurate lineage (parent_run_id) in
            # Postgres, so the dashboard can show a resumed run's full
            # epoch history back through its ancestors, not just its own.
            "run_name": Path(args.log_csv).name,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        score = checkpoint_score(args.checkpoint_metric, val_loss, f1_final)
        if score < best_score:
            best_score = score
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  -> new best ({args.checkpoint_metric}, score={score:.4f}), saved {ckpt_dir / 'best.pt'}")
        # BUG THIS FIXES: last.pt is overwritten every single epoch and
        # best.pt only updates when val_loss improves -- so a run that goes
        # through one genuinely bad epoch (a bad --resume lr, an aggressive
        # sampler change, anything that spikes val_loss) leaves NO way back
        # to the good epoch right before it: last.pt is already overwritten,
        # and best.pt won't hold it either unless that specific epoch
        # happened to be the single best of the whole run. This is exactly
        # what destroyed this project's own epoch-104 checkpoint (see
        # docs/MANUAL.md S12.6) -- an --oversample-rare-classes resume
        # regressed val_loss on its very first epoch and there was no
        # milestone snapshot to fall back to. A non-overwritten snapshot
        # every --checkpoint-every epochs costs disk space, not correctness.
        if (epoch + 1) % args.checkpoint_every == 0:
            milestone_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
            torch.save(ckpt_payload, milestone_path)
            print(f"  -> milestone snapshot saved {milestone_path}")

    print(f"\n{args.log_csv} is up to date (written every epoch). Best {args.checkpoint_metric} "
          f"score: {best_score:.4f}. Checkpoints in {ckpt_dir}/ (best.pt, last.pt).")


if __name__ == "__main__":
    main()
