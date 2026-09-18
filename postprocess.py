"""
Phase 4 -- infrastructure-aware post-processing.

Skeletonizes the predicted road mask, finds skeleton endpoints that sit close
together but are not connected (a "gap"), and reconnects a gap only when the
model's own grid-attention saliency map (Section 3, Phase 1's grid attention
"receiving" score) is high along the straight-line path between them -- i.e.
only when the model already thinks the two points are related. This replaces
the fixed-radius dilation used by prior SN-8 solutions (see thesis Section
2.2 / Table 1) with a learned, image-specific signal.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize
from scipy import ndimage


def _skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """A skeleton pixel with exactly one skeleton neighbour is an endpoint."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints = np.argwhere(skeleton & (neighbor_count == 1))
    return [tuple(p) for p in endpoints]


def _sample_line(p0: tuple[int, int], p1: tuple[int, int], n: int = 20) -> np.ndarray:
    ys = np.linspace(p0[0], p1[0], n)
    xs = np.linspace(p0[1], p1[1], n)
    return np.stack([ys, xs], axis=1)


def bridge_road_gaps(
    road_mask: np.ndarray,
    grid_saliency: np.ndarray,
    max_gap_px: int = 25,
    saliency_threshold: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """
    road_mask: (H, W) bool, the model's predicted road pixels (any flood state).
    grid_saliency: (H, W) float in roughly [0, 1] after normalization -- the
      encoder's own grid-attention "received attention" map, upsampled to
      the mask's resolution.

    Returns (bridged_mask, bridges) where `bridges` records every gap that
    was closed, for presentation/inspection.
    """
    H, W = road_mask.shape
    skeleton = skeletonize(road_mask)
    endpoints = _skeleton_endpoints(skeleton)

    sal = grid_saliency.copy()
    sal_range = sal.max() - sal.min()
    if sal_range > 1e-8:
        sal = (sal - sal.min()) / sal_range

    bridged = skeleton.copy()
    bridges = []
    used = set()

    for i, p0 in enumerate(endpoints):
        if i in used:
            continue
        best_j, best_dist, best_score = None, None, None
        for j, p1 in enumerate(endpoints):
            if j == i or j in used:
                continue
            dist = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
            if dist == 0 or dist > max_gap_px:
                continue
            line = _sample_line(p0, p1)
            ys = np.clip(line[:, 0].round().astype(int), 0, H - 1)
            xs = np.clip(line[:, 1].round().astype(int), 0, W - 1)
            score = float(sal[ys, xs].mean())
            if score < saliency_threshold:
                continue
            if best_score is None or score > best_score:
                best_j, best_dist, best_score = j, dist, score

        if best_j is not None:
            p1 = endpoints[best_j]
            line = _sample_line(p0, p1, n=max(2, int(best_dist)))
            ys = np.clip(line[:, 0].round().astype(int), 0, H - 1)
            xs = np.clip(line[:, 1].round().astype(int), 0, W - 1)
            bridged[ys, xs] = True
            used.add(i)
            used.add(best_j)
            bridges.append({
                "from": p0, "to": p1,
                "gap_px": round(best_dist, 1),
                "attention_score": round(best_score, 3),
            })

    return bridged, bridges


def suppress_isolated_flood_predictions(out: dict, min_component_fraction: float = 0.0005) -> torch.Tensor:
    """Return a (B, H, W) predicted class mask (0=background, 1=building,
    2=road, 3=flooded) from a separate-flood-head model's output, with
    small isolated blobs of predicted flood reassigned to whatever
    background/building/road the structure head would have predicted on
    its own -- not blanked to background outright, since a flooded
    building that gets its flood flag suppressed is still, most likely, a
    building.

    Motivated by the SpaceNet-8 challenge's actual 1st-place team (KARI-AI):
    their own whitepaper (github.com/SpaceNetChallenge/SpaceNet8/
    01-ohhan777/Whitepaper_KARI-AI.docx, section 3) states plainly that
    "false positives for flood detection significantly impact the score,"
    and describes their fix: "when flooded buildings and roads occur at a
    low rate in the image, they were considered false detection, and all
    of them were treated as non-flooded." Adapted here, not copied: their
    model scores flooded-building and flooded-road as two separate binary
    channels, so they check each independently; this project's model has
    one combined flood channel, so there's one spatial check -- is a given
    blob of predicted flood pixels large enough on its own to plausibly be
    a real flood region, or small enough to plausibly be prediction noise.

    This changes nothing about the model -- it's a pure post-processing
    step on an already-trained model's predictions (docs/RESEARCH_NOTES.md
    item 7), separate from and stackable with the training-time changes in
    docs/MANUAL.md S12.46.

    Args:
        out: model output dict with "structure_logits" (B, 3, H, W) and
            "flood_logit" (B, 1, H, W) -- the same separate-flood-head
            output train.py's compute_loss and model.py's combined "logits"
            already consume (docs/MANUAL.md S12.44).
        min_component_fraction: a connected blob of predicted-flooded
            pixels smaller than this fraction of the tile's total pixel
            count gets suppressed. 0.0005 (0.05% of a tile -- ~13 pixels on
            a 256x256 tile) is a starting point, not a value taken from the
            source team (their exact threshold isn't stated in the
            whitepaper) -- cheap to sweep against a fixed checkpoint's
            validation F1 rather than guess.
    """
    structure_pred = out["structure_logits"].argmax(dim=1)  # (B, H, W), 0/1/2
    # matches the S12.44 combined-argmax decision boundary exactly: sigmoid(flood_logit) > 0.5 <=> flood_logit > 0
    flood_pred = (out["flood_logit"].squeeze(1) > 0).long()  # (B, H, W), 0/1

    combined = torch.where(flood_pred.bool(), torch.full_like(structure_pred, 3), structure_pred)

    b, h, w = combined.shape
    min_pixels = max(1, int(min_component_fraction * h * w))
    result = combined.clone()
    flood_np = flood_pred.cpu().numpy()
    for i in range(b):
        labeled, n_components = ndimage.label(flood_np[i])
        if n_components == 0:
            continue
        sizes = ndimage.sum(flood_np[i], labeled, index=range(1, n_components + 1))
        for comp_id, size in enumerate(sizes, start=1):
            if size < min_pixels:
                suppress = torch.from_numpy(labeled == comp_id)
                result[i][suppress] = structure_pred[i][suppress]
    return result


if __name__ == "__main__":
    # Tiny synthetic sanity check: a road with a 6px gap, plus a uniform
    # saliency field so the bridge should always fire.
    mask = np.zeros((64, 64), dtype=bool)
    mask[30, 5:29] = True
    mask[30, 35:59] = True
    saliency = np.full((64, 64), 0.9)

    bridged, bridges = bridge_road_gaps(mask, saliency, max_gap_px=15, saliency_threshold=0.5)
    print(f"Endpoints found, gaps bridged: {len(bridges)}")
    for b in bridges:
        print(" ", b)
