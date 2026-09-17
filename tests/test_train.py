"""Tests for train.py's checkpoint-selection logic (checkpoint_score) --
the fix for a real bug where val_loss-based "best" selection preferred a
collapsed checkpoint over a genuinely useful one (docs/MANUAL.md S12.12)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from train import checkpoint_score


def test_val_loss_metric_matches_prior_behavior():
    """The default metric must be exactly val_loss, unchanged -- existing
    callers/docs that don't pass --checkpoint-metric see no behavior change."""
    assert checkpoint_score("val_loss", 0.42, {0: 0.9, 1: 0.0, 2: 0.3, 3: 0.0}) == 0.42
    assert checkpoint_score("val_loss", 0.05, {}) == 0.05


def test_mean_f1_prefers_broad_detection_over_lower_loss():
    """The exact real-world scenario this fix targets: a lower val_loss
    epoch that collapsed two classes should score WORSE than a higher
    val_loss epoch that detected all four -- reproducing this project's
    own v5 run (epoch 2 vs epoch 5, see docs/MANUAL.md S12.12)."""
    epoch2 = checkpoint_score("mean_f1", val_loss=0.686,
                               f1_final={0: 0.953, 1: 0.491, 2: 0.342, 3: 0.144})
    epoch5 = checkpoint_score("mean_f1", val_loss=0.370,
                               f1_final={0: 0.969, 1: 0.0, 2: 0.332, 3: 0.0})
    assert epoch2 < epoch5, (
        "epoch2 (broad detection, higher val_loss) must score BETTER (lower) than "
        f"epoch5 (collapsed, lower val_loss) under mean_f1: epoch2={epoch2}, epoch5={epoch5}"
    )


def test_min_f1_is_stricter_than_mean_f1_against_collapse():
    """min_f1 can't be fooled by three strong classes masking one collapsed
    one -- mean_f1 might still call this "good," min_f1 must not."""
    f1s = {0: 0.98, 1: 0.95, 2: 0.90, 3: 0.0}  # three great classes, one total collapse
    mean_score = checkpoint_score("mean_f1", val_loss=0.1, f1_final=f1s)
    min_score = checkpoint_score("min_f1", val_loss=0.1, f1_final=f1s)
    # min_f1's score is -0.0 = 0.0 (worst possible short of "no data"), mean_f1's
    # is well below zero (looks good) -- min_f1 must not reward this the way mean_f1 does
    assert min_score == 0.0
    assert mean_score < min_score


def test_none_entries_are_excluded_from_the_average():
    """A class that never appeared in ground truth OR prediction at all
    (F1 = None, per ConfusionAccumulator.f1()'s own contract) must not
    silently count as a zero and drag the average down."""
    f1_final = {0: 0.9, 1: None, 2: 0.5, 3: None}
    score = checkpoint_score("mean_f1", val_loss=0.2, f1_final=f1_final)
    assert score == pytest.approx(-0.7)  # -(0.9 + 0.5) / 2, not / 4


def test_no_classes_seen_yet_scores_as_worst_possible():
    """Before any class has appeared at all (e.g. the very first batches
    of a synthetic smoke test), there's nothing to score -- must not
    crash, and must never look like a "good" checkpoint."""
    assert checkpoint_score("mean_f1", val_loss=0.5, f1_final={0: None, 1: None}) == float("inf")
    assert checkpoint_score("min_f1", val_loss=0.5, f1_final={0: None, 1: None}) == float("inf")


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="checkpoint-metric"):
        checkpoint_score("not_a_real_metric", 0.5, {0: 0.9})


if __name__ == "__main__":
    test_val_loss_metric_matches_prior_behavior()
    test_mean_f1_prefers_broad_detection_over_lower_loss()
    test_min_f1_is_stricter_than_mean_f1_against_collapse()
    test_none_entries_are_excluded_from_the_average()
    test_no_classes_seen_yet_scores_as_worst_possible()
    print("All tests passed.")
