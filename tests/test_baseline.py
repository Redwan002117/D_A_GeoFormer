"""Tests for baseline.py's SN8Baseline (the U-Net/ResNet-34 comparison point)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from baseline import SN8Baseline


def test_forward_pass_output_shape():
    model = SN8Baseline(num_classes=4)
    model.eval()
    pre = torch.randn(1, 3, 64, 64)
    post = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(pre, post)
    assert out["logits"].shape == (1, 4, 64, 64)


def test_batch_size_greater_than_one():
    model = SN8Baseline(num_classes=4)
    model.eval()
    pre = torch.randn(3, 3, 64, 64)
    post = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = model(pre, post)
    assert out["logits"].shape == (3, 4, 64, 64)


def test_gradients_flow_to_every_parameter():
    model = SN8Baseline(num_classes=4)
    model.train()
    pre = torch.randn(2, 3, 64, 64)
    post = torch.randn(2, 3, 64, 64)
    out = model(pre, post)
    out["logits"].sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"No gradient reached: {missing}"


if __name__ == "__main__":
    test_forward_pass_output_shape()
    test_batch_size_greater_than_one()
    test_gradients_flow_to_every_parameter()
    print("All tests passed.")
