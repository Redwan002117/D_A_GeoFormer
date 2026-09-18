"""Tests for postprocess.py's bridge_road_gaps (Phase 4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from postprocess import bridge_road_gaps, suppress_isolated_flood_predictions


def test_bridges_a_gap_when_attention_is_high():
    mask = np.zeros((64, 64), dtype=bool)
    mask[30, 5:29] = True
    mask[30, 35:59] = True
    saliency = np.full((64, 64), 0.9)

    bridged, bridges = bridge_road_gaps(mask, saliency, max_gap_px=15, saliency_threshold=0.5)
    assert len(bridges) == 1
    assert bridged[30, 30:35].any(), "the gap should now have some connecting pixels"


def test_does_not_bridge_when_attention_is_low():
    mask = np.zeros((64, 64), dtype=bool)
    mask[30, 5:29] = True
    mask[30, 35:59] = True
    saliency = np.zeros((64, 64))  # the model has no reason to think these connect

    bridged, bridges = bridge_road_gaps(mask, saliency, max_gap_px=15, saliency_threshold=0.5)
    assert len(bridges) == 0


def test_does_not_bridge_gaps_beyond_max_distance():
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 5:20] = True
    mask[50, 80:95] = True  # a 60px gap
    saliency = np.full((100, 100), 0.9)

    bridged, bridges = bridge_road_gaps(mask, saliency, max_gap_px=10, saliency_threshold=0.5)
    assert len(bridges) == 0, "a gap far beyond max_gap_px should not be bridged"


def test_handles_a_mask_with_no_road_pixels():
    """Should not crash on an all-background mask (a real, likely scenario
    -- a dry tile with no road predicted at all)."""
    mask = np.zeros((32, 32), dtype=bool)
    saliency = np.full((32, 32), 0.5)
    bridged, bridges = bridge_road_gaps(mask, saliency)
    assert bridged.sum() == 0
    assert bridges == []


# ---------------------------------------------------------------------------
# suppress_isolated_flood_predictions (docs/RESEARCH_NOTES.md item 7,
# KARI-AI's actual 1st-place false-positive-suppression heuristic)
# ---------------------------------------------------------------------------

def _fake_output(structure_argmax: torch.Tensor, flood_mask: torch.Tensor) -> dict:
    """Build a minimal fake model output dict: confident logits that argmax
    to exactly `structure_argmax` for structure, and a flood_logit whose
    sign matches `flood_mask` exactly."""
    b, h, w = structure_argmax.shape
    structure_logits = torch.full((b, 3, h, w), -10.0)
    for c in range(3):
        structure_logits[:, c][structure_argmax == c] = 10.0
    flood_logit = torch.where(flood_mask.bool(), torch.tensor(10.0), torch.tensor(-10.0)).unsqueeze(1)
    return {"structure_logits": structure_logits, "flood_logit": flood_logit}


def test_suppresses_a_small_isolated_flood_blob():
    structure = torch.zeros(1, 32, 32, dtype=torch.long)  # background everywhere
    flood = torch.zeros(1, 32, 32, dtype=torch.bool)
    flood[0, 10, 10] = True  # a single isolated flood pixel -- 1/1024 of the tile
    out = _fake_output(structure, flood)

    result = suppress_isolated_flood_predictions(out, min_component_fraction=0.003)  # min_pixels = int(0.003*1024) = 3, a 1px blob is clearly below it
    assert result[0, 10, 10].item() == 0, "a single-pixel isolated flood blob should be suppressed to background"


def test_keeps_a_large_flood_region():
    structure = torch.zeros(1, 32, 32, dtype=torch.long)
    flood = torch.zeros(1, 32, 32, dtype=torch.bool)
    flood[0, 5:20, 5:20] = True  # a real 15x15=225px flood region, ~22% of the tile
    out = _fake_output(structure, flood)

    result = suppress_isolated_flood_predictions(out, min_component_fraction=0.003)
    assert (result[0, 5:20, 5:20] == 3).all(), "a large, genuine flood region must not be suppressed"


def test_suppressed_pixels_fall_back_to_the_structure_prediction():
    """A suppressed flood pixel should become whatever the structure head
    predicted underneath it -- building, road, or background -- not
    unconditionally background."""
    structure = torch.zeros(1, 32, 32, dtype=torch.long)
    structure[0, 10, 10] = 1  # the structure head thinks this pixel is a building
    flood = torch.zeros(1, 32, 32, dtype=torch.bool)
    flood[0, 10, 10] = True  # but flood fired on just this one isolated pixel
    out = _fake_output(structure, flood)

    result = suppress_isolated_flood_predictions(out, min_component_fraction=0.003)
    assert result[0, 10, 10].item() == 1, "should fall back to 'building', not blank to background"


def test_handles_a_mask_with_no_flood_predicted():
    """Must not crash when a tile has zero predicted flood pixels at all
    (the common case for a dry tile)."""
    structure = torch.zeros(1, 16, 16, dtype=torch.long)
    flood = torch.zeros(1, 16, 16, dtype=torch.bool)
    out = _fake_output(structure, flood)
    result = suppress_isolated_flood_predictions(out)
    assert (result == 0).all()


def test_handles_a_batch_of_more_than_one_tile():
    structure = torch.zeros(2, 32, 32, dtype=torch.long)
    flood = torch.zeros(2, 32, 32, dtype=torch.bool)
    flood[0, 10, 10] = True          # tile 0: isolated, should be suppressed
    flood[1, 5:20, 5:20] = True      # tile 1: real region, should survive
    out = _fake_output(structure, flood)

    result = suppress_isolated_flood_predictions(out, min_component_fraction=0.003)
    assert result[0, 10, 10].item() == 0
    assert (result[1, 5:20, 5:20] == 3).all()


if __name__ == "__main__":
    test_bridges_a_gap_when_attention_is_high()
    test_does_not_bridge_when_attention_is_low()
    test_does_not_bridge_gaps_beyond_max_distance()
    test_handles_a_mask_with_no_road_pixels()
    test_suppresses_a_small_isolated_flood_blob()
    test_keeps_a_large_flood_region()
    test_suppressed_pixels_fall_back_to_the_structure_prediction()
    test_handles_a_mask_with_no_flood_predicted()
    test_handles_a_batch_of_more_than_one_tile()
    print("All tests passed.")
