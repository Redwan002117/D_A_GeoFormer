"""Loss-function tests -- bounds, perfect/worst-case predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from losses import TverskyLoss


def test_loss_is_near_zero_for_a_perfect_prediction():
    target = torch.randint(0, 4, (1, 8, 8))
    # Build logits that argmax to exactly `target` with high confidence.
    logits = torch.full((1, 4, 8, 8), -10.0)
    for c in range(4):
        mask = target == c
        logits[:, c][mask.unsqueeze(1).expand(-1, 1, -1, -1)[:, 0]] = 10.0
    loss_fn = TverskyLoss()
    loss = loss_fn(logits, target)
    assert loss.item() < 0.05, f"Expected near-zero loss for a perfect prediction, got {loss.item()}"


def test_loss_is_bounded_between_zero_and_one():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    loss_fn = TverskyLoss()
    loss = loss_fn(logits, target)
    assert 0.0 <= loss.item() <= 1.0


def test_beta_greater_than_alpha_penalizes_false_negatives_more():
    """Tversky's whole point (thesis Phase 3): missing a flooded pixel
    should cost more than a false alarm. Isolate beta's effect by holding
    alpha FIXED and only raising beta -- an earlier version of this test
    varied both at once and accidentally compared the wrong thing (a drop
    in alpha lowering an unrelated class's loss enough to mask beta's own,
    correctly-directed effect). Holding alpha constant removes that confound."""
    target = torch.ones(1, 4, 4, dtype=torch.long)  # every pixel is class 1

    # Predicts class 0 everywhere -- every class-1 pixel is a false negative,
    # and there are zero false positives for class 1 anywhere.
    fn_logits = torch.zeros(1, 4, 4, 4)
    fn_logits[:, 0] = 10.0

    loss_fn_beta_low = TverskyLoss(alpha=0.5, beta=0.3)
    loss_fn_beta_high = TverskyLoss(alpha=0.5, beta=0.9)

    beta_low = loss_fn_beta_low(fn_logits, target).item()
    beta_high = loss_fn_beta_high(fn_logits, target).item()
    assert beta_high > beta_low, (
        "With alpha held constant, raising beta (false-negative weight) should score "
        f"an all-false-negative prediction worse: beta=0.3 -> {beta_low}, beta=0.9 -> {beta_high}"
    )


if __name__ == "__main__":
    test_loss_is_near_zero_for_a_perfect_prediction()
    test_loss_is_bounded_between_zero_and_one()
    test_beta_greater_than_alpha_penalizes_false_negatives_more()
    print("All tests passed.")
