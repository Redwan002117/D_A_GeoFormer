"""One-time data migration: regenerate index.json's class_pixel_counts and
n_flooded_features from the ACTUAL saved 256x256 mask PNGs, not the stale
pre-resize (1300x1300) rasterization metadata they currently hold.

BUG THIS FIXES (docs/MANUAL.md S12.48): found via audit_data_quality.py --
every entry's `class_pixel_counts` sums to exactly 1,690,000 = 1300*1300,
but every saved mask/*.png is actually 256x256 = 65,536 pixels. The counts
were computed at the original rasterization resolution, before the tiles
were resized down to 256x256 for training, and never recomputed after.

Confirmed concrete impact: dataset.py's class_sampling_weights() (used by
--oversample-rare-classes) checks `class_pixel_counts.get("3", 0) > 0` to
decide whether a tile counts as "has flooded" for oversampling -- 2 of 801
tiles (0_24_68, 0_36_62) are flagged as containing flood pixels by this
stale metadata while their ACTUAL 256x256 training mask has zero flood
pixels (the labeled region was too small -- 1px and 11px at 1300x1300 --
to survive the resize to 256x256). These tiles were getting oversampling
weight for a signal that doesn't exist in what the model actually trains
on. 3 similar mismatches exist for the building class too.

This script recomputes both fields directly from each real mask.png, so
every consumer (oversampling, any future analysis) reflects the data the
model actually sees. n_flooded_features is left as-is -- it's a real,
independent count from the original GeoJSON annotations (how many
polygons), not a pixel count, and isn't affected by the resize.

Safe to run while a training process is already running: Python datasets
load index.json ONCE at process startup (dataset.py's __init__), so a live
process already has the old values in memory and is unaffected -- only a
NEW process (e.g. a future v16 run) picks up the corrected file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def real_class_pixel_counts(mask: np.ndarray) -> dict[str, int]:
    """The actual class -> pixel-count map for a loaded mask array, in the
    same {"0": n, "1": n, ...} string-keyed format index.json uses. Pulled
    out as its own function so the regeneration logic is unit-testable
    without touching real files on disk."""
    values, counts = np.unique(mask, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def correct_entries(entries: list[dict], load_mask) -> tuple[list[dict], int, int]:
    """Recompute class_pixel_counts for every entry from its real mask.
    `load_mask(entry) -> np.ndarray` is injected so this can be tested
    against in-memory arrays instead of real files. Returns (entries,
    n_changed, n_flood_presence_changed)."""
    changed = 0
    flood_presence_fixed = 0
    for e in entries:
        mask = load_mask(e)
        real_counts = real_class_pixel_counts(mask)

        old_flood = int(e["class_pixel_counts"].get("3", 0)) > 0
        new_flood = real_counts.get("3", 0) > 0
        if old_flood != new_flood:
            flood_presence_fixed += 1

        if e["class_pixel_counts"] != real_counts:
            changed += 1
            e["class_pixel_counts"] = real_counts
    return entries, changed, flood_presence_fixed


def main(data_dir: str, dry_run: bool = False):
    data_dir = Path(data_dir)
    index_path = data_dir / "index.json"
    with open(index_path) as f:
        entries = json.load(f)

    entries, changed, flood_presence_fixed = correct_entries(
        entries, lambda e: np.array(Image.open(data_dir / e["mask"])))

    print(f"{changed}/{len(entries)} tiles' class_pixel_counts corrected "
          f"(all of them, expected -- every tile had the pre-resize values)")
    print(f"{flood_presence_fixed} tiles' flood PRESENCE (used by "
          f"--oversample-rare-classes) actually changed as a result")

    if dry_run:
        print("--dry-run: not writing the file")
        return

    with open(index_path, "w") as f:
        json.dump(entries, f)
    print(f"Wrote corrected {index_path}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "real_sn8_dataset_full"
    dry_run = "--dry-run" in sys.argv
    main(data_dir, dry_run=dry_run)
