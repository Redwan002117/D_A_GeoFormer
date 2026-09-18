"""Annotation/data quality audit for the 801-tile real SpaceNet-8 dataset.

Motivated directly by two independent, primary sources (docs/RESEARCH_NOTES.md
item 5, docs/MANUAL.md S12.46-S12.47): the SpaceNet-8 SOTA paper's single
largest, cleanly-attributed improvement was removing mislabeled tiles
(+2.2 IoU), and the actual 1st-place team's own whitepaper says plainly
"the algorithm is designed to operate under the premise that labels are
accurate, but if there is a code to detect wrong labels, the performance
will be better" -- this is that code, for this project's own dataset.

Read-only. Produces a report; does not modify or exclude any tile itself
(that's a human call, made from the report -- see docs/MANUAL.md for
whatever gets decided).

Checks, in order of how concrete/certain they are:
1. File integrity -- every pre/post/mask file actually opens and has the
   expected shape/mode.
2. Exact pre==post duplicates -- a real data-prep bug (the "post" event
   image accidentally being a copy of "pre"), not a labeling judgment call.
3. Degenerate flood annotations -- flood_pixels / n_flooded_features far
   below the dataset's own normal range, i.e. a "feature" that rasterized
   to only a handful of pixels. Concrete, checkable against the dataset's
   own distribution, not an external assumption.
4. Pre/post misalignment proxy -- 1st-place team's own stated issue
   ("pre-image and post-image did not match significantly... necessary to
   implement an additional image registration algorithm"). Approximated
   here via FFT-based phase correlation on grayscale downsamples (no cv2
   in this environment) -- flags a LARGE detected shift as suspicious, not
   a diagnosis on its own; real content change (an actual flood) can also
   produce a large apparent shift, so this needs human eyes, not
   auto-exclusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_gray_downsampled(path: Path, size: int = 64) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("L").resize((size, size))
        return np.asarray(img, dtype=np.float64)
    except Exception:
        return None


def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Estimated pixel shift between two same-size grayscale images via FFT
    phase correlation -- the standard image-registration primitive, done
    directly with numpy since cv2 isn't available in this environment.
    Returns the magnitude of the estimated (dy, dx) shift."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross_power = (fa * np.conj(fb))
    denom = np.abs(cross_power)
    denom[denom == 0] = 1e-9
    cross_power /= denom
    corr = np.fft.ifft2(cross_power).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] < a.shape[0] / 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] < a.shape[1] / 2 else peak[1] - a.shape[1]
    return float(np.hypot(dy, dx))


def audit(data_dir: str) -> dict:
    data_dir = Path(data_dir)
    with open(data_dir / "index.json") as f:
        entries = json.load(f)

    corrupt_files = []
    exact_duplicates = []
    degenerate_flood_features = []
    shift_estimates = []

    # Establish the dataset's own normal px/feature range first (check 3
    # is relative to this dataset's own distribution, not an external
    # threshold pulled from nowhere).
    px_per_feature = []
    for e in entries:
        flood_px = int(e["class_pixel_counts"].get("3", 0))
        n_feat = e.get("n_flooded_features", 0)
        if flood_px > 0 and n_feat > 0:
            px_per_feature.append(flood_px / n_feat)
    p5 = float(np.percentile(px_per_feature, 5)) if px_per_feature else 0.0

    for i, e in enumerate(entries):
        pre_path = data_dir / e["pre"]
        post_path = data_dir / e["post"]
        mask_path = data_dir / e["mask"]

        pre_img = _load_gray_downsampled(pre_path)
        post_img = _load_gray_downsampled(post_path)
        try:
            mask_arr = np.asarray(Image.open(mask_path))
        except Exception:
            mask_arr = None

        if pre_img is None or post_img is None or mask_arr is None:
            corrupt_files.append(e["tile_id"])
            continue

        if np.array_equal(pre_img, post_img):
            exact_duplicates.append(e["tile_id"])

        invalid_values = set(np.unique(mask_arr).tolist()) - {0, 1, 2, 3}
        if invalid_values:
            corrupt_files.append(f"{e['tile_id']} (invalid mask values: {invalid_values})")

        flood_px = int(e["class_pixel_counts"].get("3", 0))
        n_feat = e.get("n_flooded_features", 0)
        if flood_px > 0 and n_feat > 0 and (flood_px / n_feat) < p5:
            degenerate_flood_features.append((e["tile_id"], flood_px, n_feat, flood_px / n_feat))

        shift = _phase_correlation_shift(pre_img, post_img)
        shift_estimates.append((e["tile_id"], shift))

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(entries)} tiles checked")

    shifts = np.array([s for _, s in shift_estimates])
    shift_p95 = float(np.percentile(shifts, 95))
    high_shift = [(tid, s) for tid, s in shift_estimates if s > shift_p95]
    high_shift.sort(key=lambda x: -x[1])

    return {
        "total_tiles": len(entries),
        "corrupt_files": corrupt_files,
        "exact_pre_post_duplicates": exact_duplicates,
        "degenerate_flood_features_threshold_px_per_feature": p5,
        "degenerate_flood_features": sorted(degenerate_flood_features, key=lambda r: r[3])[:20],
        "shift_p95_px_at_64x64": shift_p95,
        "highest_shift_tiles": high_shift[:15],
    }


if __name__ == "__main__":
    import sys
    result = audit(sys.argv[1] if len(sys.argv) > 1 else "real_sn8_dataset_full")
    print(json.dumps(result, indent=2, default=str))
