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


def test_class_weights_none_matches_uniform_mean():
    """Default behavior (class_weights=None) must be unchanged -- a uniform
    mean across classes, same as before this feature existed."""
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    unweighted = TverskyLoss()(logits, target).item()
    explicit_uniform = TverskyLoss(class_weights=[1.0, 1.0, 1.0, 1.0])(logits, target).item()
    assert abs(unweighted - explicit_uniform) < 1e-6


def test_class_weights_upweight_a_poorly_predicted_rare_class():
    """A model that gets class 3 (flooded) entirely wrong should be
    penalized more when class 3's weight is raised, all else equal --
    the whole point of weighting it against background's dominant share."""
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 0, 0] = 3  # exactly one flooded pixel, everything else background

    # Model confidently predicts background everywhere -- misses the one
    # flooded pixel entirely (a total false negative on class 3).
    logits = torch.zeros(1, 4, 8, 8)
    logits[:, 0] = 10.0

    loss_uniform = TverskyLoss(class_weights=[1.0, 1.0, 1.0, 1.0])(logits, target).item()
    loss_flooded_upweighted = TverskyLoss(class_weights=[1.0, 1.0, 1.0, 10.0])(logits, target).item()
    assert loss_flooded_upweighted > loss_uniform, (
        "Upweighting the rare, missed class should raise the loss relative to a "
        f"uniform mean: uniform={loss_uniform}, upweighted={loss_flooded_upweighted}"
    )


def test_class_weights_wrong_length_raises():
    import pytest
    with pytest.raises(ValueError, match="class_weights"):
        TverskyLoss(class_weights=[1.0, 2.0])  # only 2 entries for num_classes=4


def test_focal_gamma_one_matches_original_behavior():
    """gamma=1.0 (default) must be the exact identity power -- byte-for-byte
    the pre-focal loss, so existing callers/tests are unaffected."""
    torch.manual_seed(2)
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    plain = TverskyLoss()(logits, target).item()
    explicit_gamma_one = TverskyLoss(focal_gamma=1.0)(logits, target).item()
    assert abs(plain - explicit_gamma_one) < 1e-6


def test_focal_gamma_raises_loss_for_a_partially_wrong_prediction():
    """The whole point of the focal term: gamma > 1 should score a
    partially-wrong prediction (moderate per-class Tversky index, neither
    perfect nor total failure) worse than gamma=1 does, since
    (1-TI)^(1/gamma) > (1-TI) when 0 < (1-TI) < 1 and gamma > 1."""
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :4, :] = 1  # half building, half background

    # Confident but imperfect: gets most of the building region right,
    # a quarter of it wrong -- a genuine partial-credit case, not 0 or 1.
    logits = torch.zeros(1, 4, 8, 8)
    logits[:, 0] = 5.0
    logits[0, 1, :3, :] = 8.0  # override: 3/4 rows of building correctly predicted

    plain = TverskyLoss(focal_gamma=1.0)(logits, target).item()
    focal = TverskyLoss(focal_gamma=2.0)(logits, target).item()
    assert focal > plain, (
        f"Focal (gamma=2.0) should score a partially-wrong prediction worse than "
        f"plain Tversky (gamma=1.0): plain={plain}, focal={focal}"
    )


def test_focal_gamma_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="focal_gamma"):
        TverskyLoss(focal_gamma=0.0)
    with pytest.raises(ValueError, match="focal_gamma"):
        TverskyLoss(focal_gamma=-1.0)


def test_valid_mask_none_matches_prior_behavior():
    """The default (valid_mask=None) must be unchanged -- every pixel
    counts, exact original behavior."""
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    plain = TverskyLoss()(logits, target).item()
    explicit_all_valid = TverskyLoss()(logits, target, valid_mask=torch.ones(2, 8, 8, dtype=torch.bool)).item()
    assert abs(plain - explicit_all_valid) < 1e-6


def test_valid_mask_excludes_pixels_entirely():
    """A pixel masked out must have NO effect on the loss, not even a
    diluted one -- verified by making an invalid pixel maximally wrong
    (would spike the loss if counted) and confirming the loss matches a
    version of the same tensors with that pixel simply removed."""
    logits = torch.zeros(1, 4, 1, 3)
    logits[:, 0] = 10.0  # confidently predicts class 0 everywhere
    target = torch.tensor([[[0, 0, 3]]])  # last pixel is class 3 -- a total miss if counted

    mask_last_pixel_out = torch.tensor([[[True, True, False]]])
    loss_with_bad_pixel_excluded = TverskyLoss()(logits, target, valid_mask=mask_last_pixel_out).item()

    # Same loss computed on just the two valid pixels, mask omitted entirely --
    # must match, proving the masked pixel contributed nothing, not a
    # smaller-but-nonzero amount.
    loss_pixel_physically_removed = TverskyLoss()(logits[:, :, :, :2], target[:, :, :2]).item()
    assert abs(loss_with_bad_pixel_excluded - loss_pixel_physically_removed) < 1e-5


def test_tversky_gradient_vanishes_on_saturated_prediction_but_bce_does_not():
    """The exact real mechanism motivating --flood-bce-weight (docs/MANUAL.md
    S12.30): S12.24 diagnosed the flood head's collapse as the logit
    saturating into a confidently-negative ("never flooded") region where
    Tversky's gradient goes near-zero even though real flooded pixels are
    present in the target -- Tversky is a GLOBAL tp/fp/fn ratio, and once
    predicted positives are ~0, that ratio's gradient w.r.t. the logits
    is tiny. BCE is computed per-pixel independently and has no such
    collapse. Verified numerically, not just asserted: on the same
    deeply-saturated logits and the same target (some real positives
    present), Tversky's gradient norm is near-zero while BCE's is not --
    this is the actual justification for adding BCE as a second term."""
    torch.manual_seed(0)
    # A flood_logit that has already saturated: confidently negative
    # ("not flooded") everywhere, even at the 5 pixels the target says
    # ARE flooded -- exactly the collapsed state S12.24 found in practice.
    saturated_logit = torch.full((1, 1, 8, 8), -12.0, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, :2, :3] = 1  # 6 real flooded pixels the saturated model is missing entirely

    two_channel = torch.cat([-saturated_logit, saturated_logit], dim=1)
    tversky_loss = TverskyLoss(num_classes=2)(two_channel, target)
    tversky_loss.backward()
    tversky_grad_norm = saturated_logit.grad.norm().item()

    saturated_logit2 = saturated_logit.detach().clone().requires_grad_(True)
    import torch.nn.functional as F
    bce_loss = F.binary_cross_entropy_with_logits(saturated_logit2, target.unsqueeze(1).float())
    bce_loss.backward()
    bce_grad_norm = saturated_logit2.grad.norm().item()

    assert tversky_grad_norm < 1e-3, (
        f"Tversky's gradient should be ~vanished on a saturated prediction (this is the "
        f"failure mode S12.24 diagnosed), got norm={tversky_grad_norm}"
    )
    assert bce_grad_norm > 1e-2, (
        f"BCE must keep producing real gradient in exactly the regime Tversky stalls -- "
        f"the whole point of adding it -- got norm={bce_grad_norm}"
    )
    assert bce_grad_norm > 100 * tversky_grad_norm


if __name__ == "__main__":
    test_loss_is_near_zero_for_a_perfect_prediction()
    test_loss_is_bounded_between_zero_and_one()
    test_beta_greater_than_alpha_penalizes_false_negatives_more()
    test_class_weights_none_matches_uniform_mean()
    test_class_weights_upweight_a_poorly_predicted_rare_class()
    test_focal_gamma_one_matches_original_behavior()
    test_focal_gamma_raises_loss_for_a_partially_wrong_prediction()
    test_focal_gamma_must_be_positive()
    print("All tests passed.")
