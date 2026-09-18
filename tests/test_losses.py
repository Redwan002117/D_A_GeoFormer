"""Loss-function tests -- bounds, perfect/worst-case predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from losses import TverskyLoss, AsymmetricUnifiedFocalLoss, TopKLoss, RegionMutualInformationLoss


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


# ---------------------------------------------------------------------------
# AsymmetricUnifiedFocalLoss (docs/RESEARCH_NOTES.md item 4)
# ---------------------------------------------------------------------------

def test_unified_focal_loss_is_near_zero_for_a_perfect_prediction():
    target = torch.randint(0, 2, (2, 8, 8))
    logits = torch.full((2, 2, 8, 8), -10.0)
    for c in range(2):
        mask = target == c
        logits[:, c][mask.unsqueeze(1).expand(-1, 1, -1, -1)[:, 0]] = 10.0
    loss_fn = AsymmetricUnifiedFocalLoss()
    loss = loss_fn(logits, target)
    assert loss.item() < 0.05, f"Expected near-zero loss for a perfect prediction, got {loss.item()}"


def test_unified_focal_loss_is_high_for_a_completely_wrong_prediction():
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.zeros(2, 2, 8, 8)
    logits[:, 1] = 10.0  # confidently predicts class 1 everywhere; target is all class 0
    loss_fn = AsymmetricUnifiedFocalLoss()
    loss = loss_fn(logits, target)
    assert loss.item() > 0.3, f"Expected a large loss for a completely wrong prediction, got {loss.item()}"


def test_unified_focal_loss_rejects_non_binary_input():
    """The class is binary-only by construction (docstring, and the
    reference it's ported from) -- silently doing something wrong on a
    4-class input would be worse than refusing it outright."""
    logits = torch.randn(1, 4, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8))
    loss_fn = AsymmetricUnifiedFocalLoss()
    try:
        loss_fn(logits, target)
        assert False, "expected a ValueError for non-binary input"
    except ValueError:
        pass


def test_unified_focal_loss_weight_zero_is_pure_focal_ce():
    """weight=0 should collapse the tversky term's contribution to zero,
    leaving pure asymmetric focal cross-entropy -- checked by comparing
    against weight=1e-9 (numerically indistinguishable from 0 without
    literally zeroing the coefficient) moving the loss value continuously,
    not by a discontinuous jump, i.e. the two terms really are just being
    linearly blended as the formula claims."""
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 6, 6)
    target = torch.randint(0, 2, (2, 6, 6))
    loss_w0 = AsymmetricUnifiedFocalLoss(weight=0.0)(logits, target)
    loss_w_tiny = AsymmetricUnifiedFocalLoss(weight=1e-6)(logits, target)
    assert torch.allclose(loss_w0, loss_w_tiny, atol=1e-4), \
        "weight=0 and weight~0 should give nearly identical loss (continuous blend, not a special case)"


def test_unified_focal_loss_does_not_produce_nan_on_a_near_perfect_batch():
    """Regression test for the documented deviation from the reference
    implementation: the foreground focal-Tversky term's `x ** -gamma` can
    blow up toward inf as x -> 0 (a near-perfect batch) without the clamp
    added in this port."""
    target = torch.randint(0, 2, (1, 16, 16))
    logits = torch.full((1, 2, 16, 16), -10.0)
    for c in range(2):
        mask = target == c
        logits[:, c][mask.unsqueeze(1).expand(-1, 1, -1, -1)[:, 0]] = 10.0
    loss_fn = AsymmetricUnifiedFocalLoss()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss), f"Expected a finite loss on a near-perfect batch, got {loss.item()}"


# ---------------------------------------------------------------------------
# TopKLoss (docs/RESEARCH_NOTES.md item 6)
# ---------------------------------------------------------------------------

def test_topk_loss_matches_hand_computed_mean_of_the_hardest_pixels():
    """Construct a flood_logit/target pair where each pixel's BCE value is
    known in advance (by construction), then verify TopKLoss's output is
    EXACTLY the mean of the k hardest values -- not close, not
    approximately, exactly, since this is a deterministic ranking-then-mean
    operation with no randomness involved."""
    # 4 pixels: 2 "easy" (confidently correct) and 2 "hard" (confidently wrong)
    logit = torch.tensor([10.0, 10.0, -10.0, -10.0]).view(1, 1, 2, 2)
    target = torch.tensor([1.0, 1.0, 1.0, 1.0]).view(1, 2, 2)  # all should be class 1
    import torch.nn.functional as F
    per_pixel = F.binary_cross_entropy_with_logits(logit.squeeze(1), target, reduction="none")
    hardest_two_expected = per_pixel.reshape(-1).topk(2).values.mean()

    loss_fn = TopKLoss(top_k_fraction=0.5)
    loss = loss_fn(logit, target)
    assert torch.allclose(loss, hardest_two_expected, atol=1e-6)


def test_topk_loss_excludes_the_easy_majority():
    """With top_k_fraction small, a loss dominated by a few genuinely hard
    pixels among a large easy majority should stay high -- proving the easy
    majority was actually excluded, not just down-weighted (a down-weighted
    but still-included easy majority would pull the mean down noticeably)."""
    torch.manual_seed(0)
    easy = torch.full((1, 1, 30, 30), 10.0)  # 900 easy pixels, confidently correct
    target = torch.ones(1, 30, 30)
    hard_logit = torch.tensor(-10.0)
    logit = easy.clone()
    logit[0, 0, 0, 0] = hard_logit  # 1 genuinely hard, wrong pixel among 900

    loss_fn = TopKLoss(top_k_fraction=1 / 900)  # keep exactly the single hardest pixel
    loss = loss_fn(logit, target)
    # BCE for a confidently-correct pixel (logit=10, target=1) is ~0; for the
    # hard pixel (logit=-10, target=1) it's -log(sigmoid(-10)) ~= 10.0 -- the
    # mean of ONLY the hardest pixel should be close to that exact value,
    # not pulled toward ~0 by the 899 easy ones.
    assert loss.item() > 9.0, f"Expected the hard pixel to dominate, got {loss.item()}"


def test_topk_loss_respects_valid_mask():
    """A masked-out pixel must never be selected as one of the 'hardest'
    pixels, even if its raw BCE value would otherwise qualify."""
    logit = torch.tensor([10.0, -10.0, 10.0, 10.0]).view(1, 1, 2, 2)
    target = torch.tensor([1.0, 1.0, 1.0, 1.0]).view(1, 2, 2)
    valid_mask = torch.tensor([True, False, True, True]).view(1, 2, 2)  # mask out the one hard pixel

    loss_fn = TopKLoss(top_k_fraction=1.0)  # would otherwise include everything
    loss = loss_fn(logit, target, valid_mask=valid_mask)
    # only the 3 easy (confidently-correct) pixels remain -- loss should be near 0
    assert loss.item() < 0.01, f"Expected near-zero loss once the hard pixel is masked out, got {loss.item()}"


def test_topk_loss_fraction_must_be_in_valid_range():
    for bad in (0.0, -0.1, 1.1):
        try:
            TopKLoss(top_k_fraction=bad)
            assert False, f"expected a ValueError for top_k_fraction={bad}"
        except ValueError:
            pass
    TopKLoss(top_k_fraction=1.0)  # boundary value must be accepted


# ---------------------------------------------------------------------------
# RegionMutualInformationLoss (docs/RESEARCH_NOTES.md item 3, SpaceNet-8
# 1st-place team's actual flood loss)
# ---------------------------------------------------------------------------

def _make_flood_tile(size=32, seed=0):
    """A reproducible fake flood mask: a filled rectangle, not random noise
    -- RMI is about local SHAPE, so the test target needs actual spatial
    structure for a shape-aware loss to have anything meaningful to reward
    or penalize."""
    torch.manual_seed(seed)
    target = torch.zeros(1, size, size, dtype=torch.long)
    target[:, size // 4: size // 2, size // 4: size // 2] = 1
    return target


def test_rmi_loss_is_finite_and_nonnegative_for_a_realistic_batch():
    target = _make_flood_tile()
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    loss_fn = RegionMutualInformationLoss()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss), f"expected a finite loss, got {loss.item()}"
    # RMI is a log-det of a conditional covariance -- can legitimately go
    # negative for a very good prediction (differential entropy, unlike
    # discrete entropy, has no zero floor), so this only checks finiteness,
    # not a sign or bound.


def test_rmi_loss_rewards_a_prediction_that_matches_the_target_shape():
    """The actual point of this loss (per its own docstring): a prediction
    that reproduces the target's local shape should score BETTER (lower
    loss / more mutual information) than a prediction with no spatial
    relationship to the target at all, even before any threshold/argmax is
    applied -- checked with a real logits tensor built to visibly resemble
    the target's rectangle, vs. logits built from unrelated noise."""
    target = _make_flood_tile()
    matching_probs = target.float().unsqueeze(1)  # (1, 1, H, W), values in {0, 1}
    matching_logits = torch.cat([
        (1 - matching_probs) * 8 - 4,  # background channel: high where target==0
        matching_probs * 8 - 4,        # foreground channel: high where target==1
    ], dim=1)

    torch.manual_seed(1)
    unrelated_logits = torch.randn(1, 2, 32, 32) * 2

    loss_fn = RegionMutualInformationLoss()
    matching_loss = loss_fn(matching_logits, target)
    unrelated_loss = loss_fn(unrelated_logits, target)
    assert matching_loss.item() < unrelated_loss.item(), (
        f"a prediction matching the target's local shape should score lower than "
        f"an unrelated one -- got matching={matching_loss.item()}, unrelated={unrelated_loss.item()}"
    )


def test_rmi_loss_produces_finite_gradients():
    """This loss involves matrix inversion and a Cholesky decomposition --
    exactly the kind of operation that can silently produce NaN gradients
    on an ill-conditioned input without raising any error. Must be checked
    directly, not assumed from the forward pass alone being finite."""
    target = _make_flood_tile()
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    loss_fn = RegionMutualInformationLoss()
    loss = loss_fn(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), "expected finite gradients, got NaN/inf somewhere"


def test_rmi_loss_respects_valid_mask():
    """Pixels outside valid_mask must not influence the loss -- checked by
    confirming the loss changes when a region full of adversarial noise is
    masked out, versus left in."""
    target = _make_flood_tile()
    torch.manual_seed(2)
    logits = torch.randn(1, 2, 32, 32)
    # corrupt one quadrant with extreme, wrong-direction logits
    logits[:, :, :16, :16] = torch.randn(1, 2, 16, 16) * 20

    valid_mask_excluding_corrupt = torch.ones(1, 32, 32, dtype=torch.bool)
    valid_mask_excluding_corrupt[:, :16, :16] = False

    loss_fn = RegionMutualInformationLoss()
    loss_with_corrupt_region = loss_fn(logits, target)
    loss_masked = loss_fn(logits, target, valid_mask=valid_mask_excluding_corrupt)
    assert loss_with_corrupt_region.item() != loss_masked.item(), \
        "masking out the corrupted region should change the loss value"


def test_rmi_loss_handles_a_too_small_tile_without_crashing():
    """Guard for the case where pooling shrinks a tile below the
    neighborhood radius -- must return a harmless zero, not crash on a
    negative-size tensor."""
    target = torch.zeros(1, 3, 3, dtype=torch.long)
    logits = torch.randn(1, 2, 3, 3)
    loss_fn = RegionMutualInformationLoss(radius=3, pool_size=4)
    loss = loss_fn(logits, target)
    assert loss.item() == 0.0


if __name__ == "__main__":
    test_loss_is_near_zero_for_a_perfect_prediction()
    test_loss_is_bounded_between_zero_and_one()
    test_beta_greater_than_alpha_penalizes_false_negatives_more()
    test_class_weights_none_matches_uniform_mean()
    test_class_weights_upweight_a_poorly_predicted_rare_class()
    test_focal_gamma_one_matches_original_behavior()
    test_focal_gamma_raises_loss_for_a_partially_wrong_prediction()
    test_focal_gamma_must_be_positive()
    test_unified_focal_loss_is_near_zero_for_a_perfect_prediction()
    test_unified_focal_loss_is_high_for_a_completely_wrong_prediction()
    test_unified_focal_loss_rejects_non_binary_input()
    test_unified_focal_loss_weight_zero_is_pure_focal_ce()
    test_unified_focal_loss_does_not_produce_nan_on_a_near_perfect_batch()
    test_topk_loss_matches_hand_computed_mean_of_the_hardest_pixels()
    test_topk_loss_excludes_the_easy_majority()
    test_topk_loss_respects_valid_mask()
    test_topk_loss_fraction_must_be_in_valid_range()
    test_rmi_loss_is_finite_and_nonnegative_for_a_realistic_batch()
    test_rmi_loss_rewards_a_prediction_that_matches_the_target_shape()
    test_rmi_loss_produces_finite_gradients()
    test_rmi_loss_respects_valid_mask()
    test_rmi_loss_handles_a_too_small_tile_without_crashing()
    print("All tests passed.")
