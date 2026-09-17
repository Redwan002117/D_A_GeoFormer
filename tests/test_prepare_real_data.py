"""Tests for prepare_real_data.py's mask rasterization."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from prepare_real_data import rasterize_mask


def _identity_pixel(lon, lat):
    # A trivial "geo transform" that just treats (lon, lat) as (col, row)
    # directly -- lets tests write pixel coordinates straight into the
    # fake GeoJSON below without a real GeoTIFF's tiepoint/scale tags.
    return lon, lat


def test_building_polygon_fills_interior():
    geojson = {"features": [{
        "properties": {"building": "yes", "highway": None, "flooded": None},
        "geometry": {"type": "Polygon", "coordinates": [
            [[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]],
        ]},
    }]}
    path = Path("_test_building.geojson")
    path.write_text(json.dumps(geojson))
    try:
        mask = rasterize_mask(path, _identity_pixel, width=64, height=64)
    finally:
        path.unlink()
    assert mask[25, 25] == 1  # well inside the square -- building
    assert mask[5, 5] == 0    # well outside -- background


def test_building_polygon_with_hole_leaves_hole_as_background():
    """Regression test for the real bug this fixes: a Polygon's rings after
    the first are interior holes (GeoJSON spec), not more solid area.
    Verified against the real dataset: 3 of 17,474 real polygons checked
    across real_sn8_dataset_full actually have a second ring."""
    geojson = {"features": [{
        "properties": {"building": "yes", "highway": None, "flooded": None},
        "geometry": {"type": "Polygon", "coordinates": [
            [[10, 10], [50, 10], [50, 50], [10, 50], [10, 10]],   # exterior
            [[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]],   # hole (courtyard)
        ]},
    }]}
    path = Path("_test_building_hole.geojson")
    path.write_text(json.dumps(geojson))
    try:
        mask = rasterize_mask(path, _identity_pixel, width=64, height=64)
    finally:
        path.unlink()
    assert mask[15, 15] == 1  # inside exterior, outside hole -- building
    assert mask[25, 25] == 0, "the hole must be cut out, not filled solid"


def test_flooded_building_marks_flooded_class():
    geojson = {"features": [{
        "properties": {"building": "yes", "highway": None, "flooded": "yes"},
        "geometry": {"type": "Polygon", "coordinates": [
            [[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]],
        ]},
    }]}
    path = Path("_test_flooded.geojson")
    path.write_text(json.dumps(geojson))
    try:
        mask = rasterize_mask(path, _identity_pixel, width=64, height=64)
    finally:
        path.unlink()
    assert mask[25, 25] == 3  # flooded overrides building


if __name__ == "__main__":
    test_building_polygon_fills_interior()
    test_building_polygon_with_hole_leaves_hole_as_background()
    test_flooded_building_marks_flooded_class()
    print("All tests passed.")
