"""
Shared checkpoint-loading logic for evaluate.py, demo.py, and serve.py.

BUG THIS FIXES: each of those three scripts used to duplicate its own
`GeoFormerConfig(**ckpt["config_dict"]) if "config_dict" in ckpt else
GeoFormerConfig()` line, unconditionally building a DualAxisGeoFormer.
Two real problems followed from that once baseline.py's SN8Baseline was
added:

1. A baseline checkpoint stores `"config_dict": None` (SN8Baseline has no
   config dataclass) -- not absent, `None`. `"config_dict" in ckpt` is True
   for a `None` value too, so `GeoFormerConfig(**None)` was reached and
   raised `TypeError: argument of type 'NoneType' is not iterable`.
2. Even with that fixed, all three scripts would still unconditionally
   construct a DualAxisGeoFormer regardless of what the checkpoint actually
   is, and `load_state_dict` would fail on a baseline checkpoint with a
   confusing "Missing key(s)"/"Unexpected key(s)" wall of text instead of a
   clear "this is a baseline checkpoint" message.

Centralizing the logic here means the fix (and any future one) is made
once, not three times with three chances to drift out of sync.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model import DualAxisGeoFormer, GeoFormerConfig
from baseline import SN8Baseline


def load_checkpoint_model(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Loads a checkpoint saved by train.py and returns (model, checkpoint_dict).

    The returned model is already in eval() mode with weights loaded. The
    raw checkpoint dict is returned too, for callers that want
    epoch/val_loss/model_type for logging.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "geoformer")  # older checkpoints predate this field

    if model_type == "baseline":
        model = SN8Baseline(num_classes=4)
    elif model_type == "geoformer":
        config_dict = ckpt.get("config_dict")
        cfg = GeoFormerConfig(**config_dict) if config_dict else GeoFormerConfig()
        model = DualAxisGeoFormer(cfg)
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in checkpoint {checkpoint_path}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
