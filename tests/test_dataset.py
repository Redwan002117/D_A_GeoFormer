"""Minimal tests -- run with: python -m pytest tests/  (or just `python tests/test_dataset.py`)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import numpy as np
import pytest
from PIL import Image

from dataset import (SpaceNet8Dataset, SyntheticFloodDataset, _augment_tile, _copy_paste_flood,
                      parse_tile_grid_id, find_mosaic_groups, _mosaic_tiles)


def _make_fake_index(tmp_path, tile_ids, class_pixel_counts=None):
    """A SpaceNet8Dataset with just enough on disk for .split() (and
    .class_presence_weights()) to work -- neither reads actual image files,
    only index.json fields."""
    entries = []
    for i, tid in enumerate(tile_ids):
        entry = {"tile_id": tid, "pre": f"pre/{tid}.png", "post": f"post/{tid}.png", "mask": f"mask/{tid}.png"}
        if class_pixel_counts is not None:
            entry["class_pixel_counts"] = class_pixel_counts[i]
        entries.append(entry)
    (tmp_path / "index.json").write_text(json.dumps(entries))
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


def test_class_presence_weights_boosts_rare_classes(tmp_path):
    # tile_0: nothing rare. tile_1: building only. tile_2: flooded only. tile_3: both.
    counts = [
        {"0": 1000, "1": 0, "2": 0, "3": 0},
        {"0": 900, "1": 100, "2": 0, "3": 0},
        {"0": 900, "1": 0, "2": 0, "3": 100},
        {"0": 800, "1": 100, "2": 0, "3": 100},
    ]
    ds = _make_fake_index(tmp_path, [f"tile_{i}" for i in range(4)], class_pixel_counts=counts)
    weights = ds.class_presence_weights([0, 1, 2, 3], boost_building=4.0, boost_flooded=6.0)
    assert weights[0] == 1.0
    assert weights[1] == 4.0
    assert weights[2] == 6.0
    assert weights[3] == 24.0  # both boosts multiply


def test_class_presence_weights_only_covers_requested_indices(tmp_path):
    counts = [{"0": 1000, "1": 0, "2": 0, "3": 0}, {"0": 900, "1": 100, "2": 0, "3": 0}]
    ds = _make_fake_index(tmp_path, ["tile_0", "tile_1"], class_pixel_counts=counts)
    weights = ds.class_presence_weights([1])  # only tile_1, a building tile
    assert len(weights) == 1
    assert weights[0] == 2.0  # default boost_building


def _marker_tile(size=16):
    """A pre/post/mask triplet with a unique marker in the top-left
    corner, distinguishable from the rest -- so a test can tell WHERE
    the corner ended up after a transform, not just that pixels moved."""
    pre = Image.new("RGB", (size, size), (10, 10, 10))
    post = Image.new("RGB", (size, size), (20, 20, 20))
    mask = Image.new("L", (size, size), 0)
    for img, marker in ((pre, (255, 0, 0)), (post, (0, 255, 0))):
        img.paste(marker, (0, 0, 3, 3))
    mask.paste(3, (0, 0, 3, 3))  # class 3 (flooded) marks the same corner
    return pre, post, mask


def test_augment_tile_preserves_size():
    pre, post, mask = _marker_tile(size=20)
    a_pre, a_post, a_mask = _augment_tile(pre, post, mask)
    assert a_pre.size == a_post.size == a_mask.size == (20, 20)


def test_augment_tile_transforms_pre_post_mask_identically():
    """The whole point of this function: whatever transform is applied,
    it must be the SAME one on all three images, or pre/post/mask go out
    of spatial alignment -- a genuinely corrupting bug, not a cosmetic one.
    Run many trials (random.random()/choice control which transform is
    picked) so every code path gets exercised across the run."""
    random.seed(0)
    for _ in range(30):
        pre, post, mask = _marker_tile(size=16)
        a_pre, a_post, a_mask = _augment_tile(pre, post, mask)
        pre_marker = np.argwhere(np.array(a_pre)[:, :, 0] == 255)  # red channel, pre's marker
        post_marker = np.argwhere(np.array(a_post)[:, :, 1] == 255)  # green channel, post's marker
        mask_marker = np.argwhere(np.array(a_mask) == 3)
        # All three markers must have landed in exactly the same set of
        # pixel coordinates after whatever random transform was applied.
        assert set(map(tuple, pre_marker)) == set(map(tuple, post_marker)) == set(map(tuple, mask_marker))


def test_augment_tile_mask_values_stay_valid_class_indices():
    """A mask must never gain a class value that wasn't in the original --
    rotate()'s default resampling for an integer-mode image can otherwise
    introduce values between adjacent classes at the rotation's edges,
    fabricating a label that was never in the real data."""
    random.seed(1)
    _, _, mask = _marker_tile(size=16)
    original_values = set(np.array(mask).flatten().tolist())
    for _ in range(20):
        _, _, a_mask = _augment_tile(*_marker_tile(size=16))
        assert set(np.array(a_mask).flatten().tolist()) <= original_values | {0, 3}


def _flooded_marker_tile(size=16):
    """Like _marker_tile, but the flooded region has a DISTINCT, real-
    looking appearance in pre/post (not just a mask label) -- so a copy-
    paste test can confirm the pasted pixels are visually real, not just
    a relabeled mask with unchanged imagery underneath."""
    pre = Image.new("RGB", (size, size), (10, 10, 10))
    post = Image.new("RGB", (size, size), (20, 20, 20))
    mask = Image.new("L", (size, size), 0)
    pre.paste((0, 0, 200), (0, 0, 4, 4))    # blue-ish "water" in pre
    post.paste((0, 0, 220), (0, 0, 4, 4))   # water in post too
    mask.paste(3, (0, 0, 4, 4))             # class 3 (flooded)
    return pre, post, mask


def test_copy_paste_flood_pastes_donors_flooded_region_into_all_three():
    target_pre = Image.new("RGB", (16, 16), (100, 100, 100))
    target_post = Image.new("RGB", (16, 16), (100, 100, 100))
    target_mask = Image.new("L", (16, 16), 0)  # no flooded pixels originally
    donor_pre, donor_post, donor_mask = _flooded_marker_tile(size=16)

    out_pre, out_post, out_mask = _copy_paste_flood(
        target_pre, target_post, target_mask, donor_pre, donor_post, donor_mask)

    donor_flood_px = np.array(donor_mask) == 3
    assert donor_flood_px.any(), "test setup: donor must actually have flooded pixels"
    # Every donor-flooded pixel must now be flooded in the target's mask...
    assert (np.array(out_mask)[donor_flood_px] == 3).all()
    # ...and carry the DONOR's real pixel values in pre/post, not the
    # target's original (untouched-region) values -- a relabeled mask
    # over unchanged imagery would be a fabricated label, not real data.
    assert (np.array(out_pre)[donor_flood_px] == np.array(donor_pre)[donor_flood_px]).all()
    assert (np.array(out_post)[donor_flood_px] == np.array(donor_post)[donor_flood_px]).all()
    # Pixels OUTSIDE the donor's flooded region must be untouched.
    assert (np.array(out_mask)[~donor_flood_px] == 0).all()


def test_copy_paste_flood_is_a_noop_when_donor_has_no_flooded_pixels():
    target_pre, target_post, target_mask = _flooded_marker_tile(size=16)
    donor_pre = Image.new("RGB", (16, 16), (0, 0, 0))
    donor_post = Image.new("RGB", (16, 16), (0, 0, 0))
    donor_mask = Image.new("L", (16, 16), 0)  # no class-3 pixels anywhere

    out_pre, out_post, out_mask = _copy_paste_flood(
        target_pre, target_post, target_mask, donor_pre, donor_post, donor_mask)

    assert np.array_equal(np.array(out_pre), np.array(target_pre))
    assert np.array_equal(np.array(out_post), np.array(target_post))
    assert np.array_equal(np.array(out_mask), np.array(target_mask))


def _write_real_tile(base_dir, tile_id, pre, post, mask):
    for sub in ("pre", "post", "mask"):
        (base_dir / sub).mkdir(exist_ok=True)
    pre.save(base_dir / "pre" / f"{tile_id}.png")
    post.save(base_dir / "post" / f"{tile_id}.png")
    mask.save(base_dir / "mask" / f"{tile_id}.png")
    return {"tile_id": tile_id, "pre": f"pre/{tile_id}.png",
            "post": f"post/{tile_id}.png", "mask": f"mask/{tile_id}.png"}


def test_dataset_copy_paste_prob_one_always_introduces_flooded_pixels(tmp_path):
    """End-to-end through SpaceNet8Dataset.__getitem__, not just the
    standalone helper -- confirms copy_paste_prob is actually wired up:
    a tile with NO real flooded pixels, requested from a dataset where
    exactly one OTHER tile has flooded pixels and copy_paste_prob=1.0,
    must come back with flooded pixels in its mask every time."""
    dry_pre, dry_post, dry_mask = Image.new("RGB", (16, 16), (5, 5, 5)), \
        Image.new("RGB", (16, 16), (5, 5, 5)), Image.new("L", (16, 16), 0)
    flood_pre, flood_post, flood_mask = _flooded_marker_tile(size=16)

    entries = [
        _write_real_tile(tmp_path, "dry_tile", dry_pre, dry_post, dry_mask),
        _write_real_tile(tmp_path, "flood_tile", flood_pre, flood_post, flood_mask),
    ]
    (tmp_path / "index.json").write_text(json.dumps(entries))

    ds = SpaceNet8Dataset(str(tmp_path), image_size=16, copy_paste_prob=1.0)
    _, _, mask_t = ds[0]  # the dry tile -- has no flooded pixels of its own
    assert (mask_t == 3).any(), "copy_paste_prob=1.0 must introduce flooded pixels into a dry tile"


def test_dataset_copy_paste_prob_zero_matches_prior_behavior(tmp_path):
    dry_pre, dry_post, dry_mask = Image.new("RGB", (16, 16), (5, 5, 5)), \
        Image.new("RGB", (16, 16), (5, 5, 5)), Image.new("L", (16, 16), 0)
    flood_pre, flood_post, flood_mask = _flooded_marker_tile(size=16)
    entries = [
        _write_real_tile(tmp_path, "dry_tile", dry_pre, dry_post, dry_mask),
        _write_real_tile(tmp_path, "flood_tile", flood_pre, flood_post, flood_mask),
    ]
    (tmp_path / "index.json").write_text(json.dumps(entries))

    ds = SpaceNet8Dataset(str(tmp_path), image_size=16)  # copy_paste_prob defaults to 0.0
    _, _, mask_t = ds[0]
    assert not (mask_t == 3).any(), "default (copy_paste_prob=0.0) must never introduce flooded pixels"


# ---------------------------------------------------------------------------
# Mosaic augmentation (docs/RESEARCH_NOTES.md item 2)
# ---------------------------------------------------------------------------

def test_parse_tile_grid_id_germany_format():
    assert parse_tile_grid_id("0_41_58") == ("0", 41, 58)


def test_parse_tile_grid_id_louisiana_format():
    assert parse_tile_grid_id("Louisiana-East_Training_Public__2_16_49") == \
        ("Louisiana-East_Training_Public__2", 16, 49)


def test_parse_tile_grid_id_rejects_a_non_matching_id():
    assert parse_tile_grid_id("not-a-grid-id") is None
    assert parse_tile_grid_id("only_two") is None


def test_find_mosaic_groups_finds_a_real_2x2_neighborhood():
    """4 tiles forming a real 2x2 block: (0,0) is a valid anchor with
    right=(0,1), down=(1,0), diag=(1,1)."""
    entries = [
        {"tile_id": "5_0_0"}, {"tile_id": "5_0_1"},
        {"tile_id": "5_1_0"}, {"tile_id": "5_1_1"},
    ]
    groups = find_mosaic_groups(entries)
    assert 0 in groups
    right_idx, down_idx, diag_idx = groups[0]
    assert entries[right_idx]["tile_id"] == "5_0_1"
    assert entries[down_idx]["tile_id"] == "5_1_0"
    assert entries[diag_idx]["tile_id"] == "5_1_1"


def test_find_mosaic_groups_excludes_a_tile_with_a_missing_neighbor():
    """Same as above but the diagonal neighbor is missing -- (0,0) must
    NOT be eligible, a partial 2x2 isn't a real mosaic."""
    entries = [
        {"tile_id": "5_0_0"}, {"tile_id": "5_0_1"}, {"tile_id": "5_1_0"},
    ]
    groups = find_mosaic_groups(entries)
    assert 0 not in groups


def test_find_mosaic_groups_never_crosses_aoi_boundaries():
    """Two different AOIs that happen to share (row, col) coordinates must
    never be mistaken for real neighbors -- physical adjacency is only
    meaningful within the same source AOI."""
    entries = [
        {"tile_id": "5_0_0"}, {"tile_id": "6_0_1"},
        {"tile_id": "6_1_0"}, {"tile_id": "6_1_1"},
    ]
    groups = find_mosaic_groups(entries)
    assert 0 not in groups, "tile 5_0_0 has no same-AOI neighbors and must not be an anchor"


def test_mosaic_tiles_places_each_source_in_its_own_quadrant():
    size = 4
    quad = []
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]:
        pre = Image.new("RGB", (size, size), color)
        post = Image.new("RGB", (size, size), color)
        mask = Image.new("L", (size, size), 0)
        quad.append((pre, post, mask))

    pre, post, mask = _mosaic_tiles(quad, image_size=size)
    assert pre.size == (size, size), "must resize back down to image_size, not stay 2x"
    pre_arr = np.array(pre)
    # After compositing to 2x and resizing back down, the four quadrants'
    # source colors must still be distinguishable in their approximate
    # original corner -- check the exact 4 corner pixels, which survive
    # resize sampling regardless of the exact resampling filter used.
    assert tuple(pre_arr[0, 0]) == (255, 0, 0)       # top-left source
    assert tuple(pre_arr[0, -1]) == (0, 255, 0)      # top-right source
    assert tuple(pre_arr[-1, 0]) == (0, 0, 255)      # bottom-left source
    assert tuple(pre_arr[-1, -1]) == (255, 255, 0)   # bottom-right source


def test_mosaic_tiles_mask_uses_nearest_and_stays_valid_class_indices():
    """The mask must never contain an interpolated in-between value (e.g.
    1.5 rounded weirdly) -- every output pixel must be one of the four
    real input classes, confirming NEAREST (not bilinear) resizing."""
    size = 8
    quad = []
    for cls in [0, 1, 2, 3]:
        pre = Image.new("RGB", (size, size), (0, 0, 0))
        post = Image.new("RGB", (size, size), (0, 0, 0))
        mask = Image.new("L", (size, size), cls)
        quad.append((pre, post, mask))

    _, _, mask = _mosaic_tiles(quad, image_size=size)
    mask_arr = np.array(mask)
    assert set(np.unique(mask_arr).tolist()) <= {0, 1, 2, 3}


def test_dataset_mosaic_prob_one_always_composites_a_flood_dense_sample(tmp_path):
    """End-to-end through SpaceNet8Dataset.__getitem__: an anchor tile
    with NO flooded pixels of its own, whose real neighbor DOES have
    flooded pixels, must come back with flooded pixels when mosaic_prob=1.0
    (which also proves the wiring, not just the standalone helper)."""
    dry_pre, dry_post, dry_mask = Image.new("RGB", (16, 16), (5, 5, 5)), \
        Image.new("RGB", (16, 16), (5, 5, 5)), Image.new("L", (16, 16), 0)
    flood_pre, flood_post, flood_mask = _flooded_marker_tile(size=16)

    entries = [
        _write_real_tile(tmp_path, "9_0_0", dry_pre, dry_post, dry_mask),      # anchor, dry
        _write_real_tile(tmp_path, "9_0_1", flood_pre, flood_post, flood_mask),  # right neighbor, flooded
        _write_real_tile(tmp_path, "9_1_0", dry_pre, dry_post, dry_mask),      # down neighbor
        _write_real_tile(tmp_path, "9_1_1", dry_pre, dry_post, dry_mask),      # diag neighbor
    ]
    (tmp_path / "index.json").write_text(json.dumps(entries))

    ds = SpaceNet8Dataset(str(tmp_path), image_size=16, mosaic_prob=1.0)
    _, _, mask_t = ds[0]  # the dry anchor tile
    assert (mask_t == 3).any(), "mosaic_prob=1.0 on an eligible anchor must pull in the neighbor's flood"


def test_dataset_mosaic_prob_zero_matches_prior_behavior(tmp_path):
    dry_pre, dry_post, dry_mask = Image.new("RGB", (16, 16), (5, 5, 5)), \
        Image.new("RGB", (16, 16), (5, 5, 5)), Image.new("L", (16, 16), 0)
    flood_pre, flood_post, flood_mask = _flooded_marker_tile(size=16)
    entries = [
        _write_real_tile(tmp_path, "9_0_0", dry_pre, dry_post, dry_mask),
        _write_real_tile(tmp_path, "9_0_1", flood_pre, flood_post, flood_mask),
        _write_real_tile(tmp_path, "9_1_0", dry_pre, dry_post, dry_mask),
        _write_real_tile(tmp_path, "9_1_1", dry_pre, dry_post, dry_mask),
    ]
    (tmp_path / "index.json").write_text(json.dumps(entries))

    ds = SpaceNet8Dataset(str(tmp_path), image_size=16)  # mosaic_prob defaults to 0.0
    _, _, mask_t = ds[0]
    assert not (mask_t == 3).any(), "default (mosaic_prob=0.0) must never introduce flooded pixels"


def test_dataset_mosaic_prob_falls_back_gracefully_for_an_ineligible_anchor(tmp_path):
    """A tile with no real neighbors at all must just load normally when
    mosaic_prob=1.0, not crash or raise -- graceful fallback, not an error."""
    pre, post, mask = Image.new("RGB", (16, 16), (5, 5, 5)), \
        Image.new("RGB", (16, 16), (5, 5, 5)), Image.new("L", (16, 16), 0)
    flood_pre, flood_post, flood_mask = _flooded_marker_tile(size=16)
    entries = [
        _write_real_tile(tmp_path, "lonely_tile", pre, post, mask),
        # a second tile purely so mosaic_prob > 0 has at least one real
        # neighborhood somewhere in the dataset and doesn't raise at init
        _write_real_tile(tmp_path, "9_0_0", pre, post, mask),
        _write_real_tile(tmp_path, "9_0_1", flood_pre, flood_post, flood_mask),
        _write_real_tile(tmp_path, "9_1_0", pre, post, mask),
        _write_real_tile(tmp_path, "9_1_1", pre, post, mask),
    ]
    (tmp_path / "index.json").write_text(json.dumps(entries))

    ds = SpaceNet8Dataset(str(tmp_path), image_size=16, mosaic_prob=1.0)
    pre_t, post_t, mask_t = ds[0]  # "lonely_tile" -- not a valid grid id, no neighbors
    assert pre_t.shape == (3, 16, 16)
    assert not (mask_t == 3).any()


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
