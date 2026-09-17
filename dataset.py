"""
Datasets for Dual-Axis GeoFormer.

Two datasets live here:

- `SyntheticFloodDataset` -- procedurally generates randomized pre/post tile
  pairs with ground-truth masks, entirely on the fly, no files needed. This
  is what `train.py` uses by default. Its job is to prove the TRAINING
  PIPELINE is correct (loss goes down, checkpoints save, metrics compute) --
  it is NOT a substitute for SpaceNet-8 and a model trained on it should
  never be described as a flood detector.

- `SpaceNet8Dataset` -- loads a real, preprocessed SpaceNet-8 directory (see
  docs/MANUAL.md "Preparing real SpaceNet-8 data" for the exact layout and
  the preprocessing step that gets you there from the official download).
  Not runnable without that data; it is exercised by
  `test_dataset_spacenet8_missing_dir` in `tests/` only to confirm it fails
  loudly and helpfully rather than silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 4  # background, building, road, flooded (building or road)


def _base_tile(size: int, rng: np.random.Generator) -> np.ndarray:
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return img


def make_synthetic_sample(size: int, rng: np.random.Generator):
    """Returns (pre_img, post_img, mask) as float32 HWC in [0,1] / int64 HW."""
    pre = _base_tile(size, rng)
    post = _base_tile(size, rng)
    mask = np.zeros((size, size), dtype=np.int64)

    # a curved road, position/curvature randomized per sample
    ys = np.arange(size)
    center0 = rng.uniform(size * 0.35, size * 0.65)
    amp = rng.uniform(0, size * 0.06)
    freq = rng.uniform(20, 60)
    road_center = center0 + amp * np.sin(ys / freq)
    road_width = rng.integers(4, 8)
    road_mask = np.abs(np.arange(size)[None, :] - road_center[:, None]) < road_width
    for img in (pre, post):
        img[road_mask] = [0.25, 0.25, 0.28]
    mask[road_mask] = 2  # road

    # buildings: a random number of small rectangles
    n_buildings = rng.integers(3, 8)
    building_mask = np.zeros((size, size), dtype=bool)
    for _ in range(n_buildings):
        bw, bh = rng.integers(size // 12, size // 7, size=2)
        x0 = rng.integers(0, max(1, size - bw))
        y0 = rng.integers(0, max(1, size - bh))
        box = np.zeros((size, size), dtype=bool)
        box[y0:y0 + bh, x0:x0 + bw] = True
        box &= ~road_mask
        building_mask |= box
        color = [0.75, 0.55, 0.35]
        pre[box] = color
        post[box] = color
    mask[building_mask & (mask == 0)] = 1  # building (roads keep priority)

    # flood: present in ~70% of samples, post-event only
    flooded_pixels = np.zeros((size, size), dtype=bool)
    if rng.uniform() < 0.7:
        cy = rng.uniform(size * 0.25, size * 0.75)
        cx = rng.uniform(size * 0.25, size * 0.75)
        ry = rng.uniform(size * 0.08, size * 0.2)
        rx = rng.uniform(size * 0.1, size * 0.25)
        yy, xx = np.mgrid[0:size, 0:size]
        flood_region = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy) ** 2) / (ry ** 2) <= 1
        post[flood_region] = [0.15, 0.35, 0.55]
        flooded_pixels = flood_region & (road_mask | building_mask)

    mask[flooded_pixels] = 3  # flooded (either class)

    return np.clip(pre, 0, 1), np.clip(post, 0, 1), mask


class SyntheticFloodDataset(Dataset):
    """A fixed-length, deterministic-per-index synthetic dataset (seeded by
    index, so `len()` is well-defined and repeated epochs see varied but
    reproducible samples across runs with the same base seed)."""

    def __init__(self, length: int = 200, image_size: int = 128, base_seed: int = 0):
        self.length = length
        self.image_size = image_size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        pre, post, mask = make_synthetic_sample(self.image_size, rng)
        pre_t = torch.from_numpy(pre).permute(2, 0, 1).float()
        post_t = torch.from_numpy(post).permute(2, 0, 1).float()
        pre_t = (pre_t - 0.5) / 0.5
        post_t = (post_t - 0.5) / 0.5
        mask_t = torch.from_numpy(mask).long()
        return pre_t, post_t, mask_t


class SpaceNet8Dataset(Dataset):
    """
    Real SpaceNet-8 loader. Expects a directory prepared as:

        data_dir/
          index.json                 # list of {"pre","post","mask"} relative paths
          pre/AOI_x_tile_y.tif       # pre-event image
          post/AOI_x_tile_y.tif      # post-event image
          mask/AOI_x_tile_y.png      # single-channel class-index mask
                                      #   0=background 1=building 2=road 3=flooded

    SpaceNet-8's raw release is GeoTIFF + per-class shapefiles/GeoJSON, not
    this layout -- see docs/MANUAL.md "Preparing real SpaceNet-8 data" for
    the conversion step (rasterizing labels to a class-index PNG per tile)
    before pointing `--data-dir` at a directory built this way.
    """

    def __init__(self, data_dir: str, image_size: int = 256):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        index_path = self.data_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index.json found under {self.data_dir}. This directory is not in "
                "the preprocessed SpaceNet-8 layout this loader expects -- see "
                "docs/MANUAL.md 'Preparing real SpaceNet-8 data'."
            )
        with open(index_path) as f:
            self.entries = json.load(f)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        pre = Image.open(self.data_dir / entry["pre"]).convert("RGB").resize(
            (self.image_size, self.image_size))
        post = Image.open(self.data_dir / entry["post"]).convert("RGB").resize(
            (self.image_size, self.image_size))
        mask = Image.open(self.data_dir / entry["mask"]).resize(
            (self.image_size, self.image_size), Image.NEAREST)

        pre_t = torch.from_numpy(np.array(pre)).permute(2, 0, 1).float() / 255.0
        post_t = torch.from_numpy(np.array(post)).permute(2, 0, 1).float() / 255.0
        pre_t = (pre_t - 0.5) / 0.5
        post_t = (post_t - 0.5) / 0.5
        mask_t = torch.from_numpy(np.array(mask)).long()
        return pre_t, post_t, mask_t

    def split(self, val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
        """Returns (train_indices, val_indices), assigning each tile to a
        side by hashing its own tile_id -- NOT by torch.utils.data.random_split
        on the dataset's current length.

        BUG THIS FIXES: random_split's split depends on len(dataset) and the
        dataset's current index order. This dataset grows across separate
        prepare_real_data.py runs (this exact project's real_sn8_dataset_full
        went 202 -> 352 -> 801 tiles over several sessions) and --resume
        training runs reload it fresh each time -- so the previous approach
        silently drew a DIFFERENT random train/val split every time the
        dataset's size changed, even at a fixed torch.manual_seed. A tile
        that was held out for validation in one run could end up in the next
        run's training set with no warning, and "held-out validation" claims
        made across resumed runs on a growing dataset were not actually
        comparing against a stable held-out set. Hashing each tile's own
        `tile_id` assigns it to the same side forever, regardless of how many
        other tiles exist or what order the index lists them in.
        """
        val_indices, train_indices = [], []
        for i, entry in enumerate(self.entries):
            tile_id = entry.get("tile_id", entry["pre"])  # tile_id if present, else its own path
            h = int(hashlib.md5(tile_id.encode()).hexdigest(), 16)
            if (h % 10_000) < int(val_fraction * 10_000):
                val_indices.append(i)
            else:
                train_indices.append(i)
        return train_indices, val_indices

    def class_presence_weights(self, indices: list[int],
                                boost_building: float = 4.0, boost_flooded: float = 6.0) -> list[float]:
        """Per-index sampling weight for a WeightedRandomSampler: tiles that
        contain building and/or flooded pixels get boosted, so those tiles
        appear more often per epoch than their raw prevalence in `indices`.

        WHY: this project's own real-data runs show `building`/`flooded`
        F1 stuck at exactly 0.000 with zero predicted-in images across every
        checkpoint tried (see docs/MANUAL.md S12.3-S12.5) while `road`
        (present in nearly every tile) shows real, climbing F1. Tversky's
        alpha/beta alone reweight the LOSS per-pixel; they do nothing about
        how often a tile containing those classes is even SEEN during an
        epoch, and `flooded` in particular covers well under a tenth of a
        percent of pixels dataset-wide. This is docs/MANUAL.md S13's
        "weighted/oversampled DataLoader" recommendation, made concrete.

        `indices` should be a TRAIN split's indices only (e.g. from
        `.split()[0]`) -- val must keep sampling its true, unweighted
        real-world distribution so evaluation numbers stay honest.
        """
        weights = []
        for i in indices:
            counts = self.entries[i].get("class_pixel_counts", {})
            has_building = counts.get("1", 0) > 0
            has_flooded = counts.get("3", 0) > 0
            w = 1.0
            if has_building:
                w *= boost_building
            if has_flooded:
                w *= boost_flooded
            weights.append(w)
        return weights


if __name__ == "__main__":
    ds = SyntheticFloodDataset(length=4, image_size=64)
    pre, post, mask = ds[0]
    print("pre:", tuple(pre.shape), "post:", tuple(post.shape), "mask:", tuple(mask.shape))
    print("mask class distribution:", {int(c): int((mask == c).sum()) for c in mask.unique()})
