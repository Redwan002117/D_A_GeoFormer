"""Tests for train.py's per_class_f1 -- includes a regression test for the
absent-class scoring bug (see train.py's per_class_f1 docstring)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from train import per_class_f1, ConfusionAccumulator


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


def test_confusion_accumulator_does_not_mask_total_class_failure():
    """Regression test for the real bug found by actually running this on
    real data: averaging per_class_f1's per-BATCH output across many
    batches let a class the model predicts NOWHERE still score a high F1,
    because most batches happened to have no ground truth for it either
    (scoring the per_class_f1 "trivially correct" 1.0 on those, which is
    right for THAT one image but wrong to average in with the images that
    actually fail). ConfusionAccumulator must not reproduce this: a class
    predicted in zero images, despite being present in most of them, has to
    score a low F1 from the global counts -- not something inflated by
    per-image absences elsewhere.
    """
    acc = ConfusionAccumulator(num_classes=2)
    # 8 of 10 "images" (batch entries) genuinely contain class 1; the model
    # never predicts class 1 anywhere, in any of them (a real collapse).
    for i in range(10):
        target = torch.ones(1, 4, 4, dtype=torch.long) if i < 8 else torch.zeros(1, 4, 4, dtype=torch.long)
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0] = 10.0  # always predicts class 0, never class 1
        acc.update(logits, target)

    f1 = acc.f1()
    coverage = acc.coverage_report()
    assert f1[1] is not None
    assert f1[1] < 0.1, f"a class predicted nowhere despite being present in 8/10 images should score near 0, got {f1[1]}"
    gt_n, pred_n, total_n = coverage[1]
    assert gt_n == 8
    assert pred_n == 0  # this is the number that actually flags the collapse


def test_confusion_accumulator_undefined_when_class_never_appears_at_all():
    """A class absent from every image's ground truth AND every prediction
    is genuinely undefined (not seen at all in this evaluation) -- reported
    as None, not silently scored 1.0 the way the old per-batch-averaging
    bug effectively did in aggregate."""
    acc = ConfusionAccumulator(num_classes=3)
    for _ in range(5):
        target = torch.zeros(1, 4, 4, dtype=torch.long)  # only ever class 0
        logits = torch.zeros(1, 3, 4, 4)
        logits[:, 0] = 10.0  # only ever predicts class 0
        acc.update(logits, target)

    f1 = acc.f1()
    assert f1[0] is not None and f1[0] > 0.99  # class 0: perfect, genuinely evaluated
    assert f1[1] is None  # class 1: never appeared anywhere -- undefined
    assert f1[2] is None  # class 2: same


if __name__ == "__main__":
    test_perfect_prediction_scores_f1_one_for_every_class()
    test_class_absent_from_both_pred_and_target_scores_perfect_not_zero()
    test_totally_wrong_prediction_scores_low_but_not_crash()
    test_confusion_accumulator_does_not_mask_total_class_failure()
    test_confusion_accumulator_undefined_when_class_never_appears_at_all()
    print("All tests passed.")
