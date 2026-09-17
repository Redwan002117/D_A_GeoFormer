"""Minimal tests -- run with: python -m pytest tests/  (or just `python tests/test_dataset.py`)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dataset import SpaceNet8Dataset, SyntheticFloodDataset


def _make_fake_index(tmp_path, tile_ids):
    """A SpaceNet8Dataset with just enough on disk for .split() to work --
    split() only reads index.json's tile_id field, never opens an image."""
    (tmp_path / "index.json").write_text(json.dumps([
        {"tile_id": tid, "pre": f"pre/{tid}.png", "post": f"post/{tid}.png", "mask": f"mask/{tid}.png"}
        for tid in tile_ids
    ]))
    return SpaceNet8Dataset(str(tmp_path))


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


def test_split_covers_every_tile_exactly_once(tmp_path):
    tile_ids = [f"tile_{i}" for i in range(200)]
    ds = _make_fake_index(tmp_path, tile_ids)
    train_idx, val_idx = ds.split(val_fraction=0.1)
    assert set(train_idx) | set(val_idx) == set(range(200))
    assert set(train_idx) & set(val_idx) == set()
    assert 10 <= len(val_idx) <= 30  # roughly 10%, hashing won't be exact


def test_split_is_reproducible_across_calls(tmp_path):
    tile_ids = [f"tile_{i}" for i in range(100)]
    ds = _make_fake_index(tmp_path, tile_ids)
    train_idx_a, val_idx_a = ds.split()
    train_idx_b, val_idx_b = ds.split()
    assert train_idx_a == train_idx_b
    assert val_idx_a == val_idx_b


def test_split_assignment_is_stable_when_the_dataset_grows(tmp_path):
    """Regression test for the real bug this fixes: torch.utils.data.random_split
    reshuffles the whole train/val partition whenever the dataset's length
    changes -- exactly what happened repeatedly in this project as
    real_sn8_dataset_full grew (202 -> 352 -> 801 tiles) across sessions. A
    tile_id-hash-based split must keep every tile on the same side
    regardless of how many other tiles exist alongside it."""
    original_ids = [f"tile_{i}" for i in range(50)]
    ds_small = _make_fake_index(tmp_path, original_ids)
    _, val_idx_small = ds_small.split()
    val_ids_small = {original_ids[i] for i in val_idx_small}

    grown_ids = original_ids + [f"tile_{i}" for i in range(50, 300)]  # same 50 first, plus 250 more
    ds_grown = _make_fake_index(tmp_path, grown_ids)
    _, val_idx_grown = ds_grown.split()
    val_ids_grown_from_original = {grown_ids[i] for i in val_idx_grown if grown_ids[i] in original_ids}

    assert val_ids_small == val_ids_grown_from_original, (
        "a tile's train/val assignment must not change just because the dataset grew"
    )


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
