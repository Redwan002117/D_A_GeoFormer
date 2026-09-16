"""Tests for train.py's per_class_f1 -- includes a regression test for the
absent-class scoring bug (see train.py's per_class_f1 docstring)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from train import per_class_f1


def test_perfect_prediction_scores_f1_one_for_every_class():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 4, 2, 2), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 1, 0] = 10.0
    logits[0, 3, 1, 1] = 10.0
    f1s = per_class_f1(logits, target, num_classes=4)
    for c in range(4):
        assert f1s[c] > 0.999, f"class {c}: expected ~1.0, got {f1s[c]}"


def test_class_absent_from_both_pred_and_target_scores_perfect_not_zero():
    """Regression test for the real bug this fixes: a class with zero true
    positives, zero false positives, and zero false negatives (e.g.
    'flooded' evaluated on a batch of entirely-dry tiles, correctly
    predicted as dry) used to score 0.0 -- the eps-only formula divides
    0/(0+eps) unconditionally. It should score 1.0: nothing to find, and
    nothing wrongly found, is a correct answer, not a failure."""
    # target and prediction both use only classes 0 and 2 -- class 1 and
    # class 3 never appear in either.
    target = torch.tensor([[[0, 2], [2, 0]]])
    logits = torch.full((1, 4, 2, 2), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 2, 0, 1] = 10.0
    logits[0, 2, 1, 0] = 10.0
    logits[0, 0, 1, 1] = 10.0

    f1s = per_class_f1(logits, target, num_classes=4)
    assert f1s[1] == 1.0, f"absent class 1 should score 1.0 (trivially correct), got {f1s[1]}"
    assert f1s[3] == 1.0, f"absent class 3 should score 1.0 (trivially correct), got {f1s[3]}"
    # classes that ARE present and perfectly predicted should still score 1.0
    assert f1s[0] > 0.999
    assert f1s[2] > 0.999


def test_totally_wrong_prediction_scores_low_but_not_crash():
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    logits = torch.full((1, 4, 2, 2), -10.0)
    logits[0, 3] = 10.0  # predicts class 3 everywhere; target is all class 0
    f1s = per_class_f1(logits, target, num_classes=4)
    assert f1s[0] == 0.0  # class 0 had 4 false negatives, 0 true positives
    assert f1s[3] == 0.0  # class 3 had 4 false positives, 0 true positives
    assert f1s[1] == 1.0  # class 1 never appears anywhere -- trivially correct
    assert f1s[2] > 0.999


if __name__ == "__main__":
    test_perfect_prediction_scores_f1_one_for_every_class()
    test_class_absent_from_both_pred_and_target_scores_perfect_not_zero()
    test_totally_wrong_prediction_scores_low_but_not_crash()
    print("All tests passed.")
