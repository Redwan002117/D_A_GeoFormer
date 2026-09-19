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
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 4  # background, building, road, flooded (building or road)


def _copy_paste_flood(pre: Image.Image, post: Image.Image, mask: Image.Image,
                       donor_pre: Image.Image, donor_post: Image.Image, donor_mask: Image.Image):
    """Pastes the donor tile's ENTIRE flooded-pixel footprint onto the
    target tile at the same pixel coordinates, in pre, post, AND mask
    together, so the three stay consistent (a pasted "flooded" mask
    pixel must actually show flooded-looking pixels in both images, not
    just a relabeled mask -- otherwise this would just be a more
    elaborate way to feed the model a wrong label).

    WHY (docs/MANUAL.md S12.34): S12.17-S12.33 characterized flooded's
    collapse as resistant to four different architecture/loss-level
    fixes, each converging on nearly the same training point --
    consistent with a training-DATA limit (not enough real flooded-
    pixel exposure per epoch), not a model-side one. Copy-paste
    (Ghiasi et al. 2021, "Simple Copy-Paste is a Strong Data
    Augmentation Method") tests that specific hypothesis using ONLY
    the 801 tiles already in hand -- no new data collection -- by
    directly manufacturing more flooded-pixel exposure per epoch than
    the real geography alone provides. If this measurably delays or
    prevents the collapse, that's evidence FOR the "not enough
    exposure" reading; if it doesn't, that points more toward "not
    enough distinct real examples" specifically (which copy-paste,
    reusing the same 198 flooded tiles' own content, can't manufacture).
    """
    donor_mask_arr = np.array(donor_mask)
    flood_pixels = donor_mask_arr == 3
    if not flood_pixels.any():
        return pre, post, mask  # donor had no flooded pixels after all -- no-op, not an error
    paste_mask_img = Image.fromarray((flood_pixels * 255).astype(np.uint8))
    pre = Image.composite(donor_pre, pre, paste_mask_img)
    post = Image.composite(donor_post, post, paste_mask_img)
    mask_arr = np.array(mask).copy()
    mask_arr[flood_pixels] = 3
    mask = Image.fromarray(mask_arr)
    return pre, post, mask


def parse_tile_grid_id(tile_id: str) -> tuple[str, int, int] | None:
    """A SpaceNet-8 tile_id encodes its real position in the source AOI's
    tiling grid: `<aoi>_<row>_<col>` (Germany, e.g. "0_41_58") or
    `<AOI_Name>__<aoi>_<row>_<col>` (Louisiana-East, e.g.
    "Louisiana-East_Training_Public__2_16_49"). Returns (aoi_key, row, col)
    so genuinely adjacent tiles can be found for mosaicing -- adjacency
    here means real, physical adjacency on the ground, not just "two
    tiles that happen to sit next to each other in index.json". Returns
    None for a tile_id that doesn't match this pattern (a handful don't;
    those simply aren't eligible as a mosaic anchor)."""
    parts = tile_id.rsplit("_", 2)
    if len(parts) != 3:
        return None
    aoi_key, row_str, col_str = parts
    try:
        return aoi_key, int(row_str), int(col_str)
    except ValueError:
        return None


def find_mosaic_groups(entries: list[dict]) -> dict[int, tuple[int, int, int]]:
    """For every tile that has real, physically-adjacent right/down/
    diagonal neighbors in the same AOI, map its index -> (right_idx,
    down_idx, diag_idx). Only about ~40% of tiles have a complete 2x2
    neighborhood (edge tiles and gaps in the downloaded AOI coverage don't)
    -- those simply aren't eligible as a mosaic anchor, which is fine,
    this only needs SOME eligible anchors, not all of them.

    WHY THIS EXISTS (docs/RESEARCH_NOTES.md item 2): the SpaceNet-8
    5th-place solution's single most-cited fix for flood-class data
    scarcity was joining 4 REAL adjacent tiles into one composite training
    sample -- every resulting pixel is still real imagery, just more
    flood-dense per sample than any single 256x256 crop happens to be.
    Different from --copy-paste-prob (which pastes an unrelated donor
    tile's flood footprint onto a possibly-unrelated background): mosaic
    uses tiles that are actually next to each other on the ground.
    """
    by_coord: dict[tuple[str, int, int], int] = {}
    for i, entry in enumerate(entries):
        parsed = parse_tile_grid_id(entry.get("tile_id", ""))
        if parsed is not None:
            by_coord[parsed] = i

    groups: dict[int, tuple[int, int, int]] = {}
    for (aoi_key, row, col), i in by_coord.items():
        right = by_coord.get((aoi_key, row, col + 1))
        down = by_coord.get((aoi_key, row + 1, col))
        diag = by_coord.get((aoi_key, row + 1, col + 1))
        if right is not None and down is not None and diag is not None:
            groups[i] = (right, down, diag)
    return groups


def _mosaic_tiles(quad_images: list[tuple[Image.Image, Image.Image, Image.Image]],
                   image_size: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Compose 4 real, adjacent tiles (top-left, top-right, bottom-left,
    bottom-right order) into one 2x2 grid, then resize back down to
    image_size -- the resize is what actually increases flood-pixel
    DENSITY per training sample: 4 tiles' worth of real flood-labeled
    pixels get compressed into the same crop size the model always sees,
    rather than diluted across 4 separate low-flood-density samples.
    Mask resize uses NEAREST (discrete class indices; any other
    interpolation would invent invalid in-between class values)."""
    half = image_size  # each source tile is already resized to image_size before mosaicing
    canvas_pre = Image.new("RGB", (half * 2, half * 2))
    canvas_post = Image.new("RGB", (half * 2, half * 2))
    canvas_mask = Image.new("L", (half * 2, half * 2))
    for (pre, post, mask), (x, y) in zip(quad_images, [(0, 0), (half, 0), (0, half), (half, half)]):
        canvas_pre.paste(pre, (x, y))
        canvas_post.paste(post, (x, y))
        canvas_mask.paste(mask, (x, y))
    pre = canvas_pre.resize((image_size, image_size))
    post = canvas_post.resize((image_size, image_size))
    mask = canvas_mask.resize((image_size, image_size), Image.NEAREST)
    return pre, post, mask


def _augment_tile(pre: Image.Image, post: Image.Image, mask: Image.Image):
    """Random dihedral-group (D4) transform applied IDENTICALLY to pre,
    post, and mask, so the three stay spatially aligned.

    WHY: satellite/aerial imagery has no canonical "up" -- north isn't
    meaningfully different from any other direction for what this model
    needs to learn (a building looks like a building rotated 90 degrees;
    a flooded road is still a flooded road mirrored). This project's
    real-data training had NO geometric augmentation at all before this --
    every epoch saw each of the 801 tiles in exactly one fixed orientation,
    which is real, unused headroom for a model that's shown to overfit or
    collapse on rare classes rather than generalize (see docs/MANUAL.md
    S12.13-S12.15) -- one physical tile can now teach the model up to 8
    orientation-equivalent views of the same real content instead of 1.

    `mask` uses NEAREST resampling explicitly (not the PIL default for
    every mode) -- a mask's pixel values are discrete class indices, and
    any interpolation between them would fabricate a class index that was
    never in the original label.
    """
    if random.random() < 0.5:
        pre = pre.transpose(Image.FLIP_LEFT_RIGHT)
        post = post.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        pre = pre.transpose(Image.FLIP_TOP_BOTTOM)
        post = post.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    k = random.choice([0, 1, 2, 3])
    if k:
        angle = k * 90
        pre = pre.rotate(angle)
        post = post.rotate(angle)
        mask = mask.rotate(angle, resample=Image.NEAREST)
    return pre, post, mask


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

    def __init__(self, data_dir: str, image_size: int = 256, augment: bool = False,
                 copy_paste_prob: float = 0.0, mosaic_prob: float = 0.0):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        # Default False -- exact prior behavior for existing callers/tests.
        # Only meant to be True for a TRAIN split; a validation set must
        # keep seeing tiles in their one real orientation so held-out
        # numbers stay comparable epoch to epoch (see train.py, which
        # builds a second, augment=False instance for val instead of
        # reusing one instance for both splits).
        self.augment = augment
        # Default 0.0 -- exact prior behavior. Only meant for the TRAIN
        # split, same reasoning as `augment` above -- val must stay
        # untouched so held-out numbers mean what they say.
        self.copy_paste_prob = copy_paste_prob
        # Default 0.0 -- exact prior behavior. Only meant for the TRAIN
        # split, same reasoning as copy_paste_prob above (docs/
        # RESEARCH_NOTES.md item 2 -- see find_mosaic_groups()'s docstring
        # for what this actually does and why).
        self.mosaic_prob = mosaic_prob
        index_path = self.data_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index.json found under {self.data_dir}. This directory is not in "
                "the preprocessed SpaceNet-8 layout this loader expects -- see "
                "docs/MANUAL.md 'Preparing real SpaceNet-8 data'."
            )
        with open(index_path) as f:
            self.entries = json.load(f)
        # Precomputed once, only when actually needed -- reading every
        # mask's array at init is real I/O this loader otherwise never
        # does eagerly, so it's skipped entirely when copy-paste is off
        # (the default, and every non-train-split instance).
        self._flooded_donor_indices: list[int] | None = None
        if self.copy_paste_prob > 0:
            self._flooded_donor_indices = [
                i for i, entry in enumerate(self.entries)
                if (np.array(Image.open(self.data_dir / entry["mask"])) == 3).any()
            ]
            if not self._flooded_donor_indices:
                raise ValueError(
                    "copy_paste_prob > 0 but no tile in this dataset has any flooded "
                    "pixels to donate -- copy-paste has nothing real to paste from."
                )
        self._mosaic_groups: dict[int, tuple[int, int, int]] | None = None
        if self.mosaic_prob > 0:
            self._mosaic_groups = find_mosaic_groups(self.entries)
            if not self._mosaic_groups:
                raise ValueError(
                    "mosaic_prob > 0 but no tile in this dataset has a full set of "
                    "physically-adjacent neighbors -- mosaicing has nothing real to compose."
                )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_tile(self, entry: dict) -> tuple[Image.Image, Image.Image, Image.Image]:
        pre = Image.open(self.data_dir / entry["pre"]).convert("RGB").resize(
            (self.image_size, self.image_size))
        post = Image.open(self.data_dir / entry["post"]).convert("RGB").resize(
            (self.image_size, self.image_size))
        mask = Image.open(self.data_dir / entry["mask"]).resize(
            (self.image_size, self.image_size), Image.NEAREST)
        return pre, post, mask

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        pre, post, mask = self._load_tile(entry)

        # Mutually exclusive with copy-paste below -- one flood-density
        # augmentation per sample, not both stacked, to keep what's
        # actually driving a training-signal change unambiguous.
        did_mosaic = False
        if (self._mosaic_groups is not None and idx in self._mosaic_groups
                and random.random() < self.mosaic_prob):
            right_idx, down_idx, diag_idx = self._mosaic_groups[idx]
            quad = [(pre, post, mask),
                    self._load_tile(self.entries[right_idx]),
                    self._load_tile(self.entries[down_idx]),
                    self._load_tile(self.entries[diag_idx])]
            pre, post, mask = _mosaic_tiles(quad, self.image_size)
            did_mosaic = True

        if not did_mosaic and self._flooded_donor_indices and random.random() < self.copy_paste_prob:
            donor_idx = random.choice(self._flooded_donor_indices)
            donor_entry = self.entries[donor_idx]
            donor_pre, donor_post, donor_mask = self._load_tile(donor_entry)
            pre, post, mask = _copy_paste_flood(pre, post, mask, donor_pre, donor_post, donor_mask)

        if self.augment:
            pre, post, mask = _augment_tile(pre, post, mask)

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
                                boost_building: float = 2.0, boost_flooded: float = 3.0) -> list[float]:
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

        DEFAULTS REDUCED from an earlier 4.0/6.0 (up to 24x combined) after
        a real regression: resuming a 104-epoch real-data checkpoint with
        those boosts and batch_size=1 collapsed `road` F1 from 0.237 to
        0.000 within a single epoch and it stayed at 0.000 the epoch after,
        with val_loss going UP, not down (see docs/MANUAL.md S12.6). The
        combined 24x weight on tiles with both classes, at batch size 1,
        meant the model saw an extreme, narrow slice of the data
        distribution repeatedly -- plausibly enough to catastrophically
        overwrite what it had already learned about `road`. 2.0/3.0 (6x
        combined) is a real but much gentler nudge; this needs its own
        honest before/after comparison, not just a lower number assumed
        safe.
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
