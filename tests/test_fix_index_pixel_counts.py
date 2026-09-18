"""Tests for fix_index_pixel_counts.py -- the S12.48 data-migration fix for
index.json's stale, pre-resize class_pixel_counts (docs/MANUAL.md S12.48)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fix_index_pixel_counts import real_class_pixel_counts, correct_entries


def test_real_class_pixel_counts_matches_a_hand_built_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 2
    mask[2, 2] = 3
    # 13 background, 1 building, 1 road, 1 flooded
    counts = real_class_pixel_counts(mask)
    assert counts == {"0": 13, "1": 1, "2": 1, "3": 1}


def test_real_class_pixel_counts_omits_absent_classes():
    """A class with zero pixels must not appear as a "0" count -- absence,
    not a zero entry, matching np.unique's own behavior and what
    dataset.py's class_sampling_weights() checks with .get(key, 0)."""
    mask = np.zeros((4, 4), dtype=np.uint8)  # all background, nothing else
    counts = real_class_pixel_counts(mask)
    assert counts == {"0": 16}
    assert "3" not in counts


def test_correct_entries_fixes_the_exact_s12_48_regression_case():
    """This is the real bug, reproduced: an entry's class_pixel_counts
    claims flood pixels (from stale pre-resize metadata), but the actual
    mask that would be loaded for training has none -- the entry must be
    corrected and counted as a flood-presence fix."""
    entries = [{
        "tile_id": "0_24_68",
        "mask": "mask/0_24_68.png",
        "class_pixel_counts": {"0": 1676423, "1": 538, "2": 13038, "3": 1},  # stale, pre-resize
    }]
    real_mask = np.zeros((256, 256), dtype=np.uint8)
    real_mask[10:15, 10:15] = 2  # some road, no flood at all

    fixed, n_changed, n_flood_fixed = correct_entries(entries, lambda e: real_mask)

    assert n_changed == 1
    assert n_flood_fixed == 1, "flood presence flipped from True to False -- must be counted"
    assert "3" not in fixed[0]["class_pixel_counts"]
    assert fixed[0]["class_pixel_counts"] == {"0": 65511, "2": 25}


def test_correct_entries_does_not_flag_flood_presence_when_it_was_already_correct():
    entries = [{
        "tile_id": "genuinely_flooded",
        "mask": "mask/x.png",
        "class_pixel_counts": {"0": 60000, "3": 5536},  # already wrong scale, but flood was and stays present
    }]
    real_mask = np.zeros((256, 256), dtype=np.uint8)
    real_mask[0:20, 0:20] = 3  # 400 real flood pixels -- still present, just a different count

    fixed, n_changed, n_flood_fixed = correct_entries(entries, lambda e: real_mask)

    assert n_changed == 1, "the counts themselves changed (60000/5536 -> the real values)"
    assert n_flood_fixed == 0, "presence didn't flip -- flood was True before and after"
    assert fixed[0]["class_pixel_counts"]["3"] == 400


def test_correct_entries_leaves_an_already_correct_entry_unchanged():
    real_mask = np.zeros((4, 4), dtype=np.uint8)
    real_mask[0, 0] = 1
    entries = [{"tile_id": "t", "mask": "m.png", "class_pixel_counts": {"0": 15, "1": 1}}]

    fixed, n_changed, n_flood_fixed = correct_entries(entries, lambda e: real_mask)

    assert n_changed == 0
    assert n_flood_fixed == 0


if __name__ == "__main__":
    test_real_class_pixel_counts_matches_a_hand_built_mask()
    test_real_class_pixel_counts_omits_absent_classes()
    test_correct_entries_fixes_the_exact_s12_48_regression_case()
    test_correct_entries_does_not_flag_flood_presence_when_it_was_already_correct()
    test_correct_entries_leaves_an_already_correct_entry_unchanged()
    print("All tests passed.")
