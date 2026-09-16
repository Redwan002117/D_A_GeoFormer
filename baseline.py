"""
SN-8 baseline reproduction -- U-Net with a ResNet-34-style encoder,
for the comparison in the thesis proposal's Table 2 / docs/MANUAL.md.

Scope honesty: this is NOT a byte-for-byte copy of SpaceNet-8's own
baseline implementation (its source isn't vendored here) -- it is a
from-scratch ResNet-34 encoder (standard BasicBlock stack, no torchvision
dependency) + U-Net decoder, matching the baseline's DESCRIBED
architecture from docs/MANUAL.md's literature review: a U-Net/ResNet-34
encoder with NO bi-temporal fusion mechanism. Pre- and post-event images
are simply channel-stacked (6 input channels) rather than processed
through a Siamese encoder + difference module -- exactly the limitation
Section 2.1 of the literature review identifies in the real baseline.

This exists to make the "Compute cost" / "Feature coverage" comparison
tables in the report and dashboard artifacts real numbers instead of
citations to someone else's paper -- run it through the same train.py-style
loop and compare its own loss/F1 curve against Dual-Axis GeoFormer's,
on the same data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (used by ResNet-18/34)."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock(out_ch, out_ch, 1))
    return nn.Sequential(*layers)


class ResNet34Encoder(nn.Module):
    """The [3, 4, 6, 3] BasicBlock stack that makes a ResNet "34" -- the
    same block counts as torchvision.models.resnet34, reimplemented here so
    this repo doesn't need torchvision as a dependency just for one
    comparison baseline."""

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64, 64, 3, stride=1)
        self.layer2 = _make_layer(64, 128, 4, stride=2)
        self.layer3 = _make_layer(128, 256, 6, stride=2)
        self.layer4 = _make_layer(256, 512, 3, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]  # shallow -> deep, for U-Net skip connections


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SN8Baseline(nn.Module):
    """U-Net / ResNet-34, channel-stacked pre+post input, no bi-temporal
    fusion, no attention -- the comparison point Table 1/2 cite."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=6)
        dims = [64, 128, 256, 512]
        self.up3 = _UpBlock(dims[3], dims[2], dims[2])
        self.up2 = _UpBlock(dims[2], dims[1], dims[1])
        self.up1 = _UpBlock(dims[1], dims[0], dims[0])
        self.final_up = nn.ConvTranspose2d(dims[0], dims[0] // 2, 2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(dims[0] // 2, dims[0] // 2, 3, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0] // 2, num_classes, 1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        x = torch.cat([pre, post], dim=1)  # naive 6-channel stack, no diff module
        f1, f2, f3, f4 = self.encoder(x)
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.final_up(x)
        x = F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(x)
        return {"logits": logits}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = SN8Baseline()
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    out = model(pre, post)
    print("logits:", tuple(out["logits"].shape))
    print(f"parameters: {model.num_parameters():,}")
