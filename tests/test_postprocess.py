"""Tests for postprocess.py's bridge_road_gaps (Phase 4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from postprocess import bridge_road_gaps


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


if __name__ == "__main__":
    test_bridges_a_gap_when_attention_is_high()
    test_does_not_bridge_when_attention_is_low()
    test_does_not_bridge_gaps_beyond_max_distance()
    test_handles_a_mask_with_no_road_pixels()
    print("All tests passed.")
