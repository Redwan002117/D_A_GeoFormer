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

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); target: (B, H, W) long class indices
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

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


if __name__ == "__main__":
    loss_fn = TverskyLoss()
    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    print("Tversky loss on random tensors:", loss_fn(logits, target).item())
