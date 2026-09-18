"""Tests for train.py's checkpoint-selection logic (checkpoint_score) --
the fix for a real bug where val_loss-based "best" selection preferred a
collapsed checkpoint over a genuinely useful one (docs/MANUAL.md S12.12)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from train import checkpoint_score, reinit_flood_head, ema_init, ema_update, flood_head_patience_step
from model import DualAxisGeoFormer, GeoFormerConfig


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


def _tiny_separate_head_config() -> GeoFormerConfig:
    return GeoFormerConfig(
        stem_channels=8, stage_dims=(8, 16), stage_windows=(4, 2), stage_grids=(4, 2),
        num_heads=2, num_classes=4, separate_flood_head=True,
    )


def test_reinit_flood_head_changes_only_the_flood_head():
    """The exact real bug this exists to fix (docs/MANUAL.md S12.24-S12.25):
    a --resume'd flood head that has already saturated needs a fresh
    starting point, not just a stronger loss pulling on the same
    saturated weights. Verifies reinit_flood_head changes flood_head's
    weights while leaving every other module (here, split_trunk) exactly
    as it was -- reinitializing the wrong thing, or too much, would
    silently discard checkpoint progress this feature is meant to keep."""
    model = DualAxisGeoFormer(_tiny_separate_head_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Simulate a "trained" checkpoint: one real optimizer step so Adam has
    # per-parameter state to verify gets cleared for flood_head specifically.
    pre = torch.randn(1, 3, 32, 32)
    post = torch.randn(1, 3, 32, 32)
    out = model(pre, post)
    (out["structure_logits"].sum() + out["flood_logit"].sum()).backward()
    optimizer.step()

    flood_head_params = list(model.flood_head.parameters())
    assert all(p in optimizer.state for p in flood_head_params), \
        "test setup: Adam should have state for flood_head after a real step"

    trunk_weight_before = model.split_trunk[0].weight.clone()
    flood_weight_before = model.flood_head.weight.clone()

    reinit_flood_head(model, optimizer)

    assert torch.equal(model.split_trunk[0].weight, trunk_weight_before), \
        "reinit_flood_head must not touch split_trunk's weights"
    assert not torch.equal(model.flood_head.weight, flood_weight_before), \
        "reinit_flood_head must actually change flood_head's weights"
    assert not any(p in optimizer.state for p in flood_head_params), \
        "reinit_flood_head must clear flood_head's stale Adam moment estimates"


def test_ema_update_moves_toward_raw_weights_by_momentum():
    """The exact formula this exists to implement (docs/MANUAL.md S12.29,
    matching the SpaceNet-8 5th-place solution's own convention): ema =
    ema * (1 - momentum) + raw * momentum. Verified numerically, not just
    that it changes -- a wrong blend direction or factor would still
    "change the weights" but score wrong on this test."""
    model = DualAxisGeoFormer(_tiny_separate_head_config())
    ema_state = ema_init(model)
    # Perturb the raw model so ema_state (a separate clone) and the model's
    # live weights genuinely differ -- otherwise this test can't tell a
    # correct blend from a no-op.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema_before = ema_state["flood_head.weight"].clone()
    raw_now = model.state_dict()["flood_head.weight"].clone()
    momentum = 0.1
    ema_update(ema_state, model, momentum)
    expected = ema_before * (1 - momentum) + raw_now * momentum
    assert torch.allclose(ema_state["flood_head.weight"], expected, atol=1e-6)


def test_ema_init_is_an_independent_clone_not_a_reference():
    """A shared reference would make ema_state silently track the live
    model instead of lagging behind it -- the entire point of EMA."""
    model = DualAxisGeoFormer(_tiny_separate_head_config())
    ema_state = ema_init(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    assert not torch.allclose(ema_state["flood_head.weight"], model.state_dict()["flood_head.weight"])


def test_ema_update_copies_integer_buffers_instead_of_blending():
    """num_batches_tracked (a BatchNorm buffer, integer dtype) must track
    the raw model exactly, not get averaged -- an averaged step COUNT
    (e.g. 3.7) is meaningless, unlike an averaged float weight."""
    model = DualAxisGeoFormer(_tiny_separate_head_config())
    ema_state = ema_init(model)
    tracked_keys = [k for k, v in model.state_dict().items() if not torch.is_floating_point(v)]
    assert tracked_keys, "test setup: this model should have at least one non-float buffer (e.g. BatchNorm's num_batches_tracked)"
    with torch.no_grad():
        model.state_dict()[tracked_keys[0]].add_(7)
    ema_update(ema_state, model, momentum=0.01)
    assert torch.equal(ema_state[tracked_keys[0]], model.state_dict()[tracked_keys[0]])


def test_flood_head_patience_step_raw_value_regression_case():
    """This is the exact v14 bug (docs/MANUAL.md S12.45), reproduced with the
    REAL val_f1_flooded values actually logged in
    training_log_geoformer_801_v14.csv for epochs 48-55: with smooth_window=1
    (the old raw-value behavior), the peak at epoch 51 (0.54628) is followed
    by 4 epochs of ordinary noise in the same band it had already been living
    in -- and patience=4 freezes flood_head at epoch 55, exactly as the real
    run did."""
    history = []
    best, since, freeze = -1.0, 0, False
    # epochs 48-55, val_f1_flooded, verbatim from training_log_geoformer_801_v14.csv
    values = [0.53613, 0.53757, 0.54149, 0.54628, 0.53520, 0.54622, 0.53071, 0.51700]
    for v in values:
        history.append(v)
        best, since, freeze = flood_head_patience_step(history, best, since, patience=4, smooth_window=1)
    assert freeze, "test setup: this is the real sequence that froze flood_head in v14 at epoch 55"


def test_flood_head_patience_step_smoothing_delays_the_same_noise():
    """The fix, checked against the same real data: with smooth_window=3 and
    the same patience=4, the real epoch 48-55 sequence must NOT have frozen
    yet by epoch 55 -- smoothing absorbs enough single-epoch noise to buy
    real additional training time, even though (see the next test) it can't
    promise to prevent freezing forever on this noisy a metric."""
    history = []
    best, since, freeze = -1.0, 0, False
    values = [0.53613, 0.53757, 0.54149, 0.54628, 0.53520, 0.54622, 0.53071, 0.51700]
    for v in values:
        history.append(v)
        best, since, freeze = flood_head_patience_step(history, best, since, patience=4, smooth_window=3)
    assert not freeze, "smoothing should have delayed the freeze past epoch 55, unlike the raw value"


def test_flood_head_patience_step_even_smoothing_cannot_promise_forever():
    """Honest limitation, not swept under the rug: flooded pixels are <1% of
    this dataset (only ~20-28 of 87 val tiles have any), so even a generous
    smooth_window=3 + patience=8 eventually reads a long enough quiet stretch
    as a plateau -- real epochs 48-61 do this at epoch 61, two epochs before
    the run's actual best smoothed value shows up at epoch 74 (0.5512, from
    val_f1_flooded values 0.54822/0.56076/0.54472 around it). This is why the
    v15 launch recommendation is to leave --flood-head-patience unset rather
    than just raise the number -- see docs/MANUAL.md S12.45."""
    history = []
    best, since, freeze = -1.0, 0, False
    # epochs 48-61, val_f1_flooded, verbatim from training_log_geoformer_801_v14.csv
    values = [0.53613, 0.53757, 0.54149, 0.54628, 0.53520, 0.54622, 0.53071, 0.51700,
              0.53804, 0.55150, 0.53710, 0.53276, 0.54587, 0.54357]
    for v in values:
        history.append(v)
        best, since, freeze = flood_head_patience_step(history, best, since, patience=8, smooth_window=3)
    assert freeze, "even smoothing+patience=8 catches this real noisy stretch -- tuning the number isn't a full fix"


def test_flood_head_patience_step_still_freezes_on_genuine_sustained_decline():
    """Smoothing must not defeat the mechanism's actual purpose -- a real,
    sustained decline (not noise) should still trigger the freeze."""
    history = []
    best, since, freeze = -1.0, 0, False
    values = [0.55, 0.54, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]  # genuine collapse
    for v in values:
        history.append(v)
        best, since, freeze = flood_head_patience_step(history, best, since, patience=4, smooth_window=3)
    assert freeze, "a genuine sustained decline must still freeze the head"


def test_flood_head_patience_step_improvement_resets_the_counter():
    """A new best (even a smoothed one) must reset epochs_since_improved to
    0 -- otherwise a head that keeps setting new records could still freeze
    if patience epochs happen to follow immediately after."""
    history = [0.3, 0.3, 0.3]
    best, since = 0.3, 3
    best2, since2, freeze2 = flood_head_patience_step(history + [0.9], best, since, patience=4, smooth_window=1)
    assert since2 == 0, "a new best-so-far must reset the non-improvement counter"
    assert not freeze2


if __name__ == "__main__":
    test_val_loss_metric_matches_prior_behavior()
    test_mean_f1_prefers_broad_detection_over_lower_loss()
    test_min_f1_is_stricter_than_mean_f1_against_collapse()
    test_none_entries_are_excluded_from_the_average()
    test_no_classes_seen_yet_scores_as_worst_possible()
    test_reinit_flood_head_changes_only_the_flood_head()
    test_ema_update_moves_toward_raw_weights_by_momentum()
    test_ema_init_is_an_independent_clone_not_a_reference()
    test_ema_update_copies_integer_buffers_instead_of_blending()
    test_flood_head_patience_step_raw_value_regression_case()
    test_flood_head_patience_step_smoothing_delays_the_same_noise()
    test_flood_head_patience_step_even_smoothing_cannot_promise_forever()
    test_flood_head_patience_step_still_freezes_on_genuine_sustained_decline()
    test_flood_head_patience_step_improvement_resets_the_counter()
    print("All tests passed.")
