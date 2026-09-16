"""Tversky loss (thesis Phase 3) -- weights false negatives above false
positives, since a missed flood is a worse operational error than a false
alarm on this dataset's class balance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, num_classes: int = 4):
        """
        alpha weights false positives, beta weights false negatives.
        beta > alpha (default 0.7 / 0.3) means the loss is penalized more
        for MISSING a flooded pixel than for over-predicting one -- see
        thesis Phase 3.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); target: (B, H, W) long class indices
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * target_onehot).sum(dim=dims)
        fp = (probs * (1 - target_onehot)).sum(dim=dims)
        fn = ((1 - probs) * target_onehot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


if __name__ == "__main__":
    loss_fn = TverskyLoss()
    logits = torch.randn(2, 4, 32, 32)
    target = torch.randint(0, 4, (2, 32, 32))
    print("Tversky loss on random tensors:", loss_fn(logits, target).item())
