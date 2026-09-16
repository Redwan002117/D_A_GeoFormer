"""Minimal tests -- run with: python -m pytest tests/  (or just `python tests/test_dataset.py`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dataset import SpaceNet8Dataset, SyntheticFloodDataset


def test_synthetic_dataset_shapes():
    ds = SyntheticFloodDataset(length=3, image_size=32)
    assert len(ds) == 3
    pre, post, mask = ds[0]
    assert pre.shape == (3, 32, 32)
    assert post.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.min() >= 0 and mask.max() <= 3


def test_synthetic_dataset_reproducible_per_index():
    ds = SyntheticFloodDataset(length=3, image_size=32, base_seed=0)
    pre_a, _, _ = ds[1]
    pre_b, _, _ = ds[1]
    assert (pre_a == pre_b).all()


def test_dataset_spacenet8_missing_dir_fails_loudly(tmp_path):
    """A directory not in the expected preprocessed layout should raise a
    clear, actionable error -- not a silent empty dataset or a cryptic
    KeyError deep inside __getitem__."""
    with pytest.raises(FileNotFoundError, match="index.json"):
        SpaceNet8Dataset(str(tmp_path))


if __name__ == "__main__":
    test_synthetic_dataset_shapes()
    test_synthetic_dataset_reproducible_per_index()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            SpaceNet8Dataset(d)
            raise SystemExit("FAIL: expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "index.json" in str(e)
    print("All tests passed.")
