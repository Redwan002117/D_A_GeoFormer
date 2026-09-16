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


def test_grid_attention_saliency_matches_batch_dimension():
    """Directly exercises the fixed module, not just the full model, so a
    future regression here fails at the smallest possible unit."""
    attn = GridAttention(dim=8, grid=2, num_heads=2)
    x = torch.randn(3, 8, 8, 8)  # B=3, C=8, H=W=8
    out, saliency = attn(x)
    assert out.shape == x.shape
    assert saliency.shape == (3, 8, 8)


if __name__ == "__main__":
    test_forward_pass_output_shape()
    test_forward_pass_batch_size_greater_than_one()
    test_gradients_flow_to_every_parameter()
    test_grid_attention_saliency_matches_batch_dimension()
    print("All tests passed.")
