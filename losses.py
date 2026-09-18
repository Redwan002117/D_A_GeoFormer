"""Tversky loss (thesis Phase 3) -- weights false negatives above false
positives, since a missed flood is a worse operational error than a false
alarm on this dataset's class balance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, num_classes: int = 4,
                 class_weights: list[float] | None = None, focal_gamma: float = 1.0):
        """
        alpha weights false positives, beta weights false negatives.
        beta > alpha (default 0.7 / 0.3) means the loss is penalized more
        for MISSING a flooded pixel than for over-predicting one -- see
        thesis Phase 3.

        BUG THIS FIXES: `1.0 - tversky.mean()` averages the per-class
        Tversky index with EQUAL weight across all num_classes -- giving
        `flooded` (well under 0.1% of pixels dataset-wide) the same 25%
        share of the loss as `background` (~85%+ of pixels, and already
        near-perfect almost immediately). Standard practice for severe
        class imbalance is to combine Tversky/Dice with an explicit
        per-class weight, not a uniform mean (see e.g. "Unified Focal
        loss: Generalising Dice and cross entropy-based losses to handle
        class imbalanced ... segmentation", Yeung et al. 2022, and the
        original Focal Tversky Loss paper, Abraham & Khan 2018). Found
        while diagnosing this project's own real-data building/flooded
        collapse (docs/MANUAL.md S12.3-S12.7): tile-level oversampling
        alone (train.py --oversample-rare-classes) delayed but did not
        prevent the same collapse, which pointed at the LOSS's per-class
        weighting, not just how often a tile is sampled, as the other
        half of the fix. `class_weights` defaults to None (uniform mean,
        the original behavior) so existing callers/tests are unaffected.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        # Focal Tversky Loss (Abraham & Khan 2018): raising (1 - TI) to the
        # power 1/gamma additionally down-weights pixels the model already
        # gets right within a class, concentrating gradient on the hard
        # ones -- complementary to class_weights above (which reweights
        # CLASSES against each other), not a replacement for it. gamma=1.0
        # (default) is the identity power -- exact original behavior, so
        # existing callers/tests are unaffected unless they opt in.
        # gamma > 1 sharpens the focus on hard pixels; typical range 1-3
        # per the original paper. Untried before this project's own
        # building/flooded collapse (docs/MANUAL.md S12.13-S12.14) --
        # class_weights alone reweights which class the loss prioritizes,
        # this additionally reweights which PIXELS within that class it
        # prioritizes, a genuinely different lever.
        if focal_gamma <= 0:
            raise ValueError(f"focal_gamma must be > 0, got {focal_gamma}")
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(f"class_weights has {len(class_weights)} entries, expected {num_classes}")
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B, C, H, W); target: (B, H, W) long class indices
        # valid_mask: optional (B, H, W) bool -- pixels where False are
        # excluded from every tp/fp/fn sum entirely, as if they weren't
        # part of the image at all. Added for the separate-flood-head
        # architecture (docs/MANUAL.md S12.17-S12.18): the structure head
        # (background/building/road) has no ground truth for what a
        # FLOODED pixel's underlying structure was -- the original
        # rasterization already overwrote that information with class 3,
        # and recovering it needs the raw GeoJSON re-processed (see
        # docs/EXTERNAL_DATA_PLAN.md), out of scope here. Excluding those
        # pixels from the structure loss (rather than guessing a fallback
        # class for them) is the honest choice -- a guessed label would be
        # actively wrong training signal, worse than no signal.
        # None (default) means every pixel counts, unchanged prior
        # behavior, verified by test.
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if valid_mask is not None:
            m = valid_mask.unsqueeze(1).float()  # (B, 1, H, W), broadcasts over the class dim
            probs = probs * m
            target_onehot = target_onehot * m

        dims = (0, 2, 3)
        tp = (probs * target_onehot).sum(dim=dims)
        fp = (probs * (1 - target_onehot)).sum(dim=dims)
        fn = ((1 - probs) * target_onehot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        per_class_loss = (1.0 - tversky) ** (1.0 / self.focal_gamma)
        if self.class_weights is not None:
            weights = self.class_weights.to(per_class_loss.device)
            return (per_class_loss * weights).sum() / weights.sum()
        return per_class_loss.mean()


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Asymmetric Unified Focal loss (Yeung et al. 2022,
    https://arxiv.org/abs/2102.04525), binary-only (background vs.
    foreground). Ported faithfully from the paper's own reference PyTorch
    port (https://github.com/oikosohn/compound-loss-pytorch, itself ported
    from the official TF implementation https://github.com/mlyg/unified-
    focal-loss) -- the formula below is not re-derived from the paper text,
    it's checked line-for-line against that reference.

    Combines an ASYMMETRIC focal Tversky term (only the foreground class
    gets the extra focal down-weighting of already-easy pixels; background
    does not) with an ASYMMETRIC focal cross-entropy term (only background
    gets focal down-weighting; foreground CE is left at full strength), via
    `weight * focal_tversky + (1 - weight) * focal_ce`. This is a genuinely
    different lever from this project's existing `TverskyLoss(focal_gamma=
    ...)`: that applies the SAME focal down-weighting to every class
    uniformly, whereas the whole point of the asymmetric variant (per the
    paper) is treating the minority/foreground class differently from the
    majority/background class -- directly aimed at this project's actual
    situation (flooded pixels under 1% of the dataset, docs/MANUAL.md
    S12.45), not a generic imbalance trick.

    Deviation from the reference, documented rather than silent: the
    reference's `(1 - dice_class) ** -gamma` term for the foreground class
    can blow up toward infinity as dice_class approaches 1 (a near-perfect
    batch), since `epsilon^-gamma` is large for small epsilon and
    `gamma > 0`. The reference's own epsilon guards y_pred, not this
    specific term. Clamped with the same epsilon here so a lucky batch
    can't produce inf/nan and crash a long CPU training run.
    """
    def __init__(self, delta: float = 0.7, gamma: float = 0.5, weight: float = 0.5, epsilon: float = 1e-7):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B, 2, H, W); target: (B, H, W) long, 0=background, 1=foreground
        if logits.size(1) != 2:
            raise ValueError(f"AsymmetricUnifiedFocalLoss is binary-only, got {logits.size(1)} channels")
        probs = F.softmax(logits, dim=1).clamp(self.epsilon, 1.0 - self.epsilon)
        target_onehot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        if valid_mask is not None:
            m = valid_mask.unsqueeze(1).float()
            probs = probs * m
            target_onehot = target_onehot * m

        dims = (2, 3)
        tp = (target_onehot * probs).sum(dim=dims)
        fn = (target_onehot * (1 - probs)).sum(dim=dims)
        fp = ((1 - target_onehot) * probs).sum(dim=dims)
        tversky_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        back_dice = 1 - tversky_class[:, 0]
        fore_base = torch.clamp(1 - tversky_class[:, 1], min=self.epsilon)  # see class docstring
        fore_dice = fore_base * torch.pow(fore_base, -self.gamma)
        asym_ftl = torch.stack([back_dice, fore_dice], dim=-1).mean()

        cross_entropy = -target_onehot * torch.log(probs)
        back_ce = (1 - self.delta) * torch.pow(1 - probs[:, 0], self.gamma) * cross_entropy[:, 0]
        fore_ce = self.delta * cross_entropy[:, 1]
        asym_fl = (back_ce + fore_ce).mean()

        return self.weight * asym_ftl + (1 - self.weight) * asym_fl


def _local_neighbor_stack(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Turn each pixel into a vector of its radius*radius local neighborhood
    values, so region statistics (mean/covariance) can be computed over
    NEIGHBORHOODS rather than single pixels. x: (B, C, H, W) -> returns
    (B, C, radius*radius, H-radius+1, W-radius+1) (edges drop where a full
    neighborhood doesn't fit, same as valid-mode convolution)."""
    b, c, h, w = x.shape
    nh, nw = h - radius + 1, w - radius + 1
    shifted = [x[:, :, dy:dy + nh, dx:dx + nw] for dy in range(radius) for dx in range(radius)]
    return torch.stack(shifted, dim=2)


class RegionMutualInformationLoss(nn.Module):
    """Region Mutual Information loss (Zhao et al., NeurIPS 2019,
    https://arxiv.org/abs/1910.12037) -- the flood-detection loss actually
    used by the SpaceNet-8 challenge's 1st-place team (KARI-AI, whose code
    lives at github.com/SpaceNetChallenge/SpaceNet8/01-ohhan777, RMI
    hyperparameters left at the class's own defaults per that team's
    `flags.txt`, i.e. these ARE the winning values, not guesses).

    What makes this different from every other loss in this file: Tversky,
    BCE, Focal, and Unified Focal all score each PIXEL independently --
    two predictions with the identical count of right/wrong pixels score
    identically even if one is a clean, contiguous flood boundary and the
    other is salt-and-pepper noise scattered across the tile. RMI instead
    represents each pixel by the small neighborhood of pixels AROUND it
    (a `radius x radius` patch, flattened to a vector) and maximizes the
    mutual information between the predicted neighborhood-vectors and the
    ground-truth neighborhood-vectors -- so it directly rewards getting
    local SHAPE right, not just pixel count.

    This is an ADDITIVE term, not a complete drop-in loss like the
    reference's own `RMILoss` (which bundles its own internal BCE mixing).
    This project already composes BCE/Tversky/class-weight terms in
    train.py's `compute_loss` (see --flood-bce-weight); duplicating that
    blending inside this class too would just be two places doing the same
    job. Use this alongside the existing flood loss via a weight, the same
    pattern as --flood-bce-weight already uses.

    Deviation from the reference implementation, documented rather than
    silent: the original hardcodes `.type(torch.cuda.DoubleTensor)`, which
    is a hard crash on this project's CPU-only environment (docs/MANUAL.md
    S13's CPU bottleneck). Replaced with device-agnostic `.double()` --
    same numerical effect (double precision, needed because the covariance
    matrix inversion is numerically sensitive at float32), works on
    whatever device the tensors are already on. Also dropped the
    reference's max-pool/nearest-interpolation pooling variants (rmi_pool_
    way 0 and 2) and its 21-class Cityscapes-oriented generality -- this
    project only ever needs the avg-pool variant (rmi_pool_way=1, which is
    what the winning run actually used) on a binary flood channel, so
    carrying the unused branches forward would just be dead code pretending
    to be a feature.
    """
    def __init__(self, num_classes: int = 2, radius: int = 3, pool_size: int = 4, epsilon: float = 5e-4):
        super().__init__()
        self.num_classes = num_classes
        self.radius = radius
        self.pool_size = pool_size
        self.neighborhood_dim = radius * radius
        self.epsilon = epsilon  # regularizes the covariance matrix so it's always invertible

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B, num_classes, H, W); target: (B, H, W) long class indices
        probs = torch.sigmoid(logits).clamp(1e-6, 1.0)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        if valid_mask is not None:
            m = valid_mask.unsqueeze(1).float()
            probs = probs * m
            target_onehot = target_onehot * m

        p = self.pool_size
        if p > 1:
            probs = F.avg_pool2d(probs, kernel_size=p, stride=p, padding=p // 2)
            target_onehot = F.avg_pool2d(target_onehot, kernel_size=p, stride=p, padding=p // 2)
        if min(probs.shape[-2:]) < self.radius:
            # too small a tile (after pooling) to form even one full
            # neighborhood -- contribute nothing rather than crash on an
            # empty/negative-size tensor below. Real tiles never hit this;
            # it's a guard for tests and unusually small crops.
            return probs.sum() * 0.0

        b, c = target_onehot.shape[:2]
        labels_local = _local_neighbor_stack(target_onehot, self.radius).reshape(b, c, self.neighborhood_dim, -1).double()
        probs_local = _local_neighbor_stack(probs, self.radius).reshape(b, c, self.neighborhood_dim, -1).double()

        labels_local = labels_local - labels_local.mean(dim=3, keepdim=True)
        probs_local = probs_local - probs_local.mean(dim=3, keepdim=True)
        label_cov = labels_local @ labels_local.transpose(2, 3)
        cross_cov = labels_local @ probs_local.transpose(2, 3)
        probs_cov = probs_local @ probs_local.transpose(2, 3)

        eye = torch.eye(self.neighborhood_dim, dtype=torch.float64, device=logits.device)
        probs_cov_inv = torch.inverse(probs_cov + eye * self.epsilon)
        # conditional covariance of the label neighborhood given the
        # predicted neighborhood -- the smaller this is, the more the
        # prediction's local neighborhood explains the label's, which is
        # exactly what "high mutual information" between them means.
        conditional_cov = label_cov - cross_cov @ probs_cov_inv @ cross_cov.transpose(2, 3)
        conditional_cov = conditional_cov + eye * self.epsilon

        # log-det via Cholesky (log det(A) = 2 * sum(log(diag(chol(A))))) --
        # numerically stable and, unlike torch.logdet, never returns -inf
        # from a merely ill-conditioned (not actually singular) matrix.
        chol = torch.linalg.cholesky(conditional_cov)
        log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8).sum(dim=-1)

        rmi_per_class = (0.5 * log_det / self.neighborhood_dim).mean(dim=0)  # mean over the batch
        return rmi_per_class.sum().float()  # sum over classes, back to float32 for the training loop


class TopKLoss(nn.Module):
    """Binary cross-entropy computed only on the hardest `top_k_fraction`
    of pixels in the batch (ranked by per-pixel BCE), the rest excluded
    entirely -- not just down-weighted to near-zero. A genuinely different
    lever from every loss above: those change how much each pixel's error
    COUNTS; this changes which pixels get to contribute a gradient AT ALL,
    concentrating training capacity on genuinely hard cases once the model
    has already solved the easy majority (the vast, easy "definitely not
    flooded" background that dominates this dataset). Standard hard-example
    mining (see e.g. Shrivastava et al. 2016, OHEM), applied here to the
    binary flood channel specifically."""
    def __init__(self, top_k_fraction: float = 0.15):
        super().__init__()
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")
        self.top_k_fraction = top_k_fraction

    def forward(self, flood_logit: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # flood_logit: (B, 1, H, W) or (B, H, W) raw logit; target: (B, H, W) 0/1
        if flood_logit.dim() == 4:
            flood_logit = flood_logit.squeeze(1)
        per_pixel = F.binary_cross_entropy_with_logits(flood_logit, target.float(), reduction="none")
        if valid_mask is not None:
            per_pixel = per_pixel[valid_mask]
        else:
            per_pixel = per_pixel.reshape(-1)
        k = max(1, int(per_pixel.numel() * self.top_k_fraction))
        hardest, _ = per_pixel.reshape(-1).topk(min(k, per_pixel.numel()))
        return hardest.mean()


if __name__ == "__main__":
    loss_fn = TverskyLoss()
    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    print("Tversky loss on random tensors:", loss_fn(logits, target).item())
