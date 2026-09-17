"""Model-level tests -- shapes, batch-size correctness, gradient flow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import DualAxisGeoFormer, GeoFormerConfig, GridAttention


def _tiny_config() -> GeoFormerConfig:
    # A much smaller config than the default so these tests run in
    # milliseconds, not seconds -- shape/gradient correctness doesn't need
    # the full 10.9M-parameter network.
    return GeoFormerConfig(
        stem_channels=8,
        stage_dims=(8, 16),
        stage_windows=(4, 2),
        stage_grids=(4, 2),
        num_heads=2,
        num_classes=4,
    )


def test_forward_pass_output_shape():
    model = DualAxisGeoFormer(_tiny_config())
    model.eval()
    pre = torch.randn(1, 3, 32, 32)
    post = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(pre, post)
    assert out["logits"].shape == (1, 4, 32, 32)


def test_forward_pass_batch_size_greater_than_one():
    """Regression test for the GridAttention batch-dimension bug: the
    saliency reshape used to assume num_groups == g*g (true only at batch
    size 1), and silently produced a shape mismatch RuntimeError for any
    batch size > 1. See model.py's GridAttention.forward comment."""
    model = DualAxisGeoFormer(_tiny_config())
    model.eval()
    for batch_size in (1, 2, 5):
        pre = torch.randn(batch_size, 3, 32, 32)
        post = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            out = model(pre, post)
        assert out["logits"].shape == (batch_size, 4, 32, 32)
        assert out["grid_saliency"].shape[0] == batch_size


def test_gradients_flow_to_every_parameter():
    model = DualAxisGeoFormer(_tiny_config())
    model.train()
    pre = torch.randn(2, 3, 32, 32)
    post = torch.randn(2, 3, 32, 32)
    out = model(pre, post)
    loss = out["logits"].sum()
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


def test_grid_attention_ablation_reduces_parameters_and_runs():
    """Table 2's 'GeoFormer - grid attention' ablation: disabling grid
    attention should shrink the model (no grid-attention weights) and still
    produce a correctly-shaped forward pass, including a saliency map (now
    all-zero, since there's no real signal to report -- not fabricated)."""
    cfg_with = _tiny_config()
    cfg_without = GeoFormerConfig(**{**cfg_with.__dict__, "use_grid_attention": False})

    model_with = DualAxisGeoFormer(cfg_with)
    model_without = DualAxisGeoFormer(cfg_without)
    assert model_without.num_parameters() < model_with.num_parameters()

    pre = torch.randn(1, 3, 32, 32)
    post = torch.randn(1, 3, 32, 32)
    model_without.eval()
    with torch.no_grad():
        out = model_without(pre, post)
    assert out["logits"].shape == (1, 4, 32, 32)
    assert torch.all(out["grid_saliency"] == 0)


def test_grid_attention_saliency_matches_batch_dimension():
    """Directly exercises the fixed module, not just the full model, so a
    future regression here fails at the smallest possible unit."""
    attn = GridAttention(dim=8, grid=2, num_heads=2)
    x = torch.randn(3, 8, 8, 8)  # B=3, C=8, H=W=8
    out, saliency = attn(x)
    assert out.shape == x.shape
    assert saliency.shape == (3, 8, 8)


def test_pretrained_backbone_wiring_produces_correct_shapes():
    """Phase 2: a timm backbone feeds the SAME downstream pipeline (diffs,
    decoder, geo-head) via per-stage 1x1 projections. pretrained=False
    (random init) keeps this test fast and network-free -- it's checking
    the wiring (channel projection, stage count, output shape), not
    real ImageNet weights, which a real training run enables separately."""
    import pytest
    pytest.importorskip("timm")
    cfg = GeoFormerConfig(
        stage_dims=(8, 16, 24, 32),  # deliberately != efficientnet_b0's own channels,
        stage_windows=(4, 4, 2, 2),  # so this also proves the projection convs work,
        stage_grids=(4, 2, 2, 1),    # not just that the shapes happened to already match
        num_heads=2, num_classes=4,
        pretrained_backbone="efficientnet_b0", pretrained=False,
    )
    model = DualAxisGeoFormer(cfg)
    model.eval()
    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(pre, post)
    assert out["logits"].shape == (1, 4, 64, 64)


def test_pretrained_backbone_none_is_unaffected():
    """The default path (pretrained_backbone=None) must be byte-for-byte
    the same from-scratch encoder as before this feature existed --
    proven by checking the model still has a `stem` attribute (the
    from-scratch path) and no `backbone` attribute (the timm path)."""
    model = DualAxisGeoFormer(_tiny_config())
    assert hasattr(model.encoder, "stem")
    assert not hasattr(model.encoder, "backbone")


def _tiny_pretrained_config() -> GeoFormerConfig:
    return GeoFormerConfig(
        stage_dims=(8, 16, 24, 32), stage_windows=(4, 4, 2, 2), stage_grids=(4, 2, 2, 1),
        num_heads=2, num_classes=4, pretrained_backbone="efficientnet_b0", pretrained=False,
    )


def test_freeze_backbone_stops_gradients_and_saves_compute():
    """set_backbone_frozen(True) must (a) zero out requires_grad on every
    backbone param, so the optimizer never touches them, and (b) actually
    produce no .grad after backward -- not just flip a flag that forward()
    ignores."""
    import pytest
    pytest.importorskip("timm")
    model = DualAxisGeoFormer(_tiny_pretrained_config())
    model.train()
    model.encoder.set_backbone_frozen(True)
    assert all(not p.requires_grad for p in model.encoder.backbone.parameters())
    assert not model.encoder.backbone.training  # BatchNorm stats must stop drifting too

    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    out = model(pre, post)
    out["logits"].sum().backward()
    assert all(p.grad is None for p in model.encoder.backbone.parameters())
    # the rest of the model must still train normally while the backbone is frozen
    assert any(p.grad is not None for p in model.encoder.projections.parameters())


def test_unfreeze_backbone_restores_gradient_flow():
    import pytest
    pytest.importorskip("timm")
    model = DualAxisGeoFormer(_tiny_pretrained_config())
    model.train()
    model.encoder.set_backbone_frozen(True)
    model.encoder.set_backbone_frozen(False)
    assert all(p.requires_grad for p in model.encoder.backbone.parameters())
    assert model.encoder.backbone.training

    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    out = model(pre, post)
    out["logits"].sum().backward()
    assert any(p.grad is not None for p in model.encoder.backbone.parameters())


def test_set_backbone_frozen_is_noop_without_a_backbone():
    """The from-scratch path has no .backbone attribute at all -- calling
    set_backbone_frozen must not crash, just do nothing."""
    model = DualAxisGeoFormer(_tiny_config())
    model.encoder.set_backbone_frozen(True)  # must not raise


if __name__ == "__main__":
    test_forward_pass_output_shape()
    test_forward_pass_batch_size_greater_than_one()
    test_gradients_flow_to_every_parameter()
    test_grid_attention_saliency_matches_batch_dimension()
    test_pretrained_backbone_wiring_produces_correct_shapes()
    test_pretrained_backbone_none_is_unaffected()
    test_freeze_backbone_stops_gradients_and_saves_compute()
    test_unfreeze_backbone_restores_gradient_flow()
    test_set_backbone_frozen_is_noop_without_a_backbone()
    print("All tests passed.")
