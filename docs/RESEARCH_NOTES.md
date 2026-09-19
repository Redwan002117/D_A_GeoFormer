# Research Notes: Candidate Next Steps for Flood Detection

**Update (docs/MANUAL.md S12.46): items 3, 4, and 6 are now implemented,
tested code** (`losses.py`'s `RegionMutualInformationLoss`,
`AsymmetricUnifiedFocalLoss`, `TopKLoss`, wired into `train.py` as opt-in
flags), not just research anymore -- what's still missing is a real
training run using them. Their sections below are left as originally
written (the reasoning is still the reasoning); S12.46 has the
implementation details.

**Also found since this was first written**: `github.com/
SpaceNetChallenge/SpaceNet8` (the official challenge repo) contains the
actual submitted code for all five top-placing teams, not just the
5th-place writeup already cited below -- including 1st place (KARI-AI /
`01-ohhan777`), which the original web search for this document could not
locate. That's where item 3's RMI loss defaults were confirmed as the
actual winning hyperparameters, not estimates.

**Status of the remaining items (1, 2, 5, 7, 8): still research only,
nothing implemented.** This started as a literature/competition-solution
review to find evidence-based next steps for the flood-detection plateau
(docs/MANUAL.md S12.44-S12.45), not a to-do list committed to all at once.
Every item is cross-checked against what this project has ALREADY tried
(per MANUAL.md's history) so nothing here duplicates existing work. Sources
are linked; nothing is taken on faith from a single blog post.

**The actual bottleneck, restated plainly**: flooded pixels are under 1% of
the 801-tile real dataset, with only ~20-28 of 87 validation tiles
containing any flood coverage at all (S12.45). Every idea below is judged
first by whether it attacks THAT constraint directly (more effective signal
per labeled flood pixel, or literally more flood-relevant training signal)
versus generically tuning an already-reasonable loss/architecture.

---

## Tier 1 — directly attacks the data-scarcity bottleneck, not yet tried at all

### 1. Semi-supervised self-training on the unused Louisiana-West imagery
**This project already has the exact infrastructure this needs, unused for
this purpose.** `docs/MANUAL.md` §12.2 confirms a third AOI, Louisiana-West,
exists as **imagery with no public labels** (SN-8's own blind test set) and
is currently not downloaded or used at all. Meanwhile `train.py` already
implements an EMA teacher (`ema_init`/`ema_update`, S12.29) — but purely to
smooth the *validation metric*, not to generate pseudo-labels.

The Mean Teacher framework (Tarvainen & Valpola, NeurIPS 2017; still the
basis of current semi-supervised segmentation work, e.g. "Edge Guided
Dynamic Mean Teacher for Semi-Supervised Remote Sensing Image Segmentation",
2025) is exactly this: an EMA "teacher" model generates pseudo-labels on
*unlabeled* imagery, and a consistency loss trains the "student" to agree
with the teacher under perturbation. This turns Louisiana-West from
completely unused pixels into real (if noisy) training signal — the single
most direct attack on "not enough labeled flood examples" available,
because it doesn't need new labels at all.

**Concrete next step, if this gets picked up**: download Louisiana-West
imagery (`prepare_real_data.py` already supports `--aoi all`), and extend
the existing `ema_update` machinery so the EMA weights periodically
generate pseudo-masks on Louisiana-West tiles, consistency-training the
student against them (with a confidence threshold so low-confidence
flood-class pseudo-labels don't get trusted blindly — this is the part
every source above stresses as the actual hard part of doing this well).
**This is also the highest-effort item on this list** — it's a real new
training loop, not a flag change.

Sources: [Edge Guided Dynamic Mean Teacher (2025)](https://link.springer.com/chapter/10.1007/978-981-95-5702-8_31),
[Correlation-based switching mean teacher (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004904),
[Flood-MATE: Mean Teacher + ensemble for imbalanced urban flood segmentation (2025)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70023) —
this last one is the closest analog to this exact problem (imbalanced flood
segmentation via Mean Teacher) but its full text is paywalled; only the
existence and framing were confirmed, not its specific numbers.

### 2. Mosaicing augmentation (distinct from the copy-paste already in use)
The SpaceNet-8 5th-place solution's single most-cited fix for flood-class
scarcity was **not** a loss change — it was generating new training tiles by
joining 4 adjacent real tiles into one, which mechanically increases flood
pixel density per training sample without inventing any fake pixels (every
pixel is still real, just recomposited). This is different from this
project's existing `--copy-paste-prob 0.3` (which pastes a flood *region*
onto a possibly-unrelated background — a real technique, but a different
one). Mosaicing is comparatively cheap to implement (`dataset.py`-level
change, no training loop change) and directly increases how much flood
signal the model sees per epoch.

Source: [motokimura/spacenet8_solution_5th-place](https://github.com/motokimura/spacenet8_solution_5th-place)
(already cited in this project's own docs for its loss-weighting and EMA
choices — this specific technique from the same source was not adopted).

---

## Tier 2 — proven in the exact SpaceNet-8 domain, moderate effort

### 3. RMI (Region Mutual Information) loss instead of/alongside Tversky+BCE
**[IMPLEMENTED, S12.46]** `losses.py`'s `RegionMutualInformationLoss`, opt-in
via `--flood-rmi-weight`. Confirmed directly in the 1st-place team's own
code (`github.com/SpaceNetChallenge/SpaceNet8/01-ohhan777`, found after
this note was first written) that RMI was their actual flood loss, not
just referenced by a secondary paper -- see below.

The SOTA SpaceNet-8 result found in this search (arXiv 2404.18235) used a
**Siamese HRNet+OCR model with RMI loss**, not a Dice/Tversky/BCE
combination. RMI's actual mechanism (NeurIPS 2019) is meaningfully
different from every loss this project has tried: instead of treating each
pixel as an independent sample (which is what Dice, Tversky, and BCE all
still do, just with different weighting), RMI maximizes mutual information
between a pixel *and its neighbors* in prediction vs. ground truth — i.e.
it directly rewards getting the local *shape* of a flood region right, not
just the per-pixel count. The paper reports "no overhead during testing"
and modest extra compute during training. This is a genuinely different
lever from anything in the flood-loss history (S12.17-S12.44 all vary
weightings of pixel-independent losses).

Sources: [RMI paper (NeurIPS 2019)](https://arxiv.org/abs/1910.12037),
[official code](https://github.com/ZJULearning/RMI),
[SpaceNet-8 SOTA reference using it](https://arxiv.org/html/2404.18235v1)
(this same paper also reports **+2.6 F1 / +4.5 IoU purely from removing
mislabeled tiles** — see item 5 below, same source).

### 4. Unified Focal Loss
**[IMPLEMENTED, S12.46]** `losses.py`'s `AsymmetricUnifiedFocalLoss`,
opt-in via `--flood-loss-fn unified_focal`.

A 2021 loss (Yeung et al., cited widely, existing PyTorch port available)
that generalises Focal + Focal Tversky into one framework specifically
built for severe class imbalance, explicitly reducing the number of
hyperparameters to search versus hand-combining Focal and Tversky
separately (which is what `--flood-class-weight`, `--flood-tversky-beta`,
`--flood-bce-weight` currently do, as 3 separate hand-tuned knobs). A
recent survey ("Loss Functions in the Era of Semantic Segmentation", 2023)
lists it alongside Focal Tversky as a top candidate for exactly this
imbalance regime.

Sources: [Unified Focal loss paper](https://arxiv.org/abs/2102.04525),
[PyTorch port](https://github.com/oikosohn/compound-loss-pytorch),
[Loss function survey](https://arxiv.org/html/2312.05391)

### 5. Systematic annotation error removal
**[DONE — first pass, S12.48.]** `audit_data_quality.py` (file integrity,
duplicate detection, degenerate-flood-feature check, misalignment proxy)
run against all 801 real tiles, with every flagged tile actually visually
inspected before acting on it, not just trusted as a heuristic. Result: no
corrupted files or exact duplicates; found and visually confirmed at least
2 genuinely cloud-covered `post` images (not yet acted on further); found
and visually confirmed a clean false positive (the misalignment proxy
flagging a tile that turned out to contain a real flood event, not
misalignment); and — bigger than the tile-level annotation check itself —
found that `index.json`'s `class_pixel_counts` were stale, computed at the
1300×1300 pre-resize rasterization resolution rather than the actual
256×256 training masks, a systematic (not random) mismatch across all 801
tiles. This had a real, measured effect on `--oversample-rare-classes`:
2 tiles were getting flood-oversampling weight for a signal that doesn't
exist in their actual training mask. Fixed by `fix_index_pixel_counts.py`
(5 tests), applied to the real dataset. Full writeup: `docs/MANUAL.md`
S12.48.

**[CLOSED OUT, S12.50.]** All 15 misalignment-flagged tiles individually
checked (15/15): 10 genuine cloud cover, 1 no-data/swath-edge gap, 1 false
positive (a real flood event, not misalignment), 1 real flood partially
under cloud, 2 benign. Then quantified across the **whole** 801-tile
dataset using the now-validated brightness/contrast proxy: **33 tiles
(4.1%) are cloud-affected, and zero of them carry a flood label** — the
data-quality issues found are real but don't directly corrupt
flood-detection training signal. Full detail in `docs/MANUAL.md` S12.48-
S12.50. The SOTA paper's own broader "remove mislabeled tiles" sweep
(they don't publish which tiles or how many, so it's not directly
replicable) is still not attempted here — this audit found and fixed one
concrete, real bug (the pixel-count resolution mismatch) and fully
characterized what the misalignment proxy catches, which is a complete,
bounded piece of work, not a partial one.

The same SpaceNet-8 SOTA paper (arXiv 2404.18235) reports its single
largest, cleanly-attributed improvement came from **removing mislabeled
tiles**, not from any architecture or loss change: IoU 0.727 → 0.749 (+2.2
points), with precision +5% and F1 +2.6% specifically credited to data
cleaning. The 5th-place solution independently did the same thing
(blacklisting specific tiles with known annotation errors) and also
discarded misaligned/cloud-covered pre/post image pairs using an MSE
alignment check.

Source: [arXiv 2404.18235](https://arxiv.org/html/2404.18235v1)

---

## Tier 3 — smaller, proven, low-effort

### 6. TopK loss (hard-pixel mining)
**[IMPLEMENTED, S12.46]** `losses.py`'s `TopKLoss`, opt-in via
`--flood-topk-weight` / `--flood-topk-fraction`.

Selects only the hardest-to-classify pixels for backpropagation each step.
Flagged by the same 2023 loss survey as a distinct imbalance strategy from
Focal/Tversky-family losses (it changes *which pixels contribute gradient*,
not how much each pixel's loss is weighted). Cheap to try as an ablation
alongside the existing flood loss, not a replacement architecture.

### 7. Post-processing: conservative flood threshold + isolated-detection
filtering
**[IMPLEMENTED AND TESTED AGAINST REAL DATA, S12.47 — RESULT: DOES NOT
HELP HERE.]** `postprocess.py`'s `suppress_isolated_flood_predictions()`,
opt-in via `evaluate.py --suppress-isolated-flood`. Confirmed directly in
the winning team's own whitepaper (`Whitepaper_KARI-AI.docx`, found after
this note was first written — no longer just a secondary summary): "false
positives for flood detection significantly impact the score... when
flooded buildings and roads occur at a low rate in the image, they were
considered false detection." Tested against `checkpoints_v14/best.pt` on
the real 87-tile held-out split, sweeping the size threshold: flooded F1
went from 0.5438 (baseline) to as low as 0.4676 at the largest threshold
tried, monotonically worse as the threshold grows, with only noise-level
(+0.001-ish) movement at the smallest. See S12.47 for the full sweep table
and reasoning on why a heuristic that won for KARI-AI doesn't transfer to
this dataset's flood-extent characteristics. Kept in the codebase
(correct, tested, and cheap to re-check against a future checkpoint) but
not enabled anywhere by default.

The SpaceNet-8 overall winner (KARI-AI) also mentioned a conservative
flood-probability threshold as a separate lever from the isolated-blob
filtering above — not implemented/tested here yet, a distinct experiment
from the connected-component approach S12.47 tested.

Sources: `github.com/SpaceNetChallenge/SpaceNet8/01-ohhan777/
Whitepaper_KARI-AI.docx` (primary source, found after this note was first
written), [SpaceNet 8 winners announcement](https://medium.com/@SpaceNet_Project/the-spacenet-8-flood-detection-challenge-announcing-the-winners-c74d4619195b)
(secondary summary, corroborated by the primary source above)

### 8. Targeted transfer learning from a related task (not just ImageNet)
5th-place solution reports concrete, measured leaderboard gains from
fine-tuning from **SpaceNet-5 (road extraction, +0.89 points)** and
**xView2 (building damage detection, +0.32 to +0.56 points)** specifically,
on top of ImageNet pretraining. This project's `--pretrained-backbone
efficientnet_b0` is ImageNet-only (§14, corrected in S12.45). A
domain-adjacent pretraining step (if a suitable checkpoint or dataset is
findable) is a different lever from swapping to a bigger ImageNet backbone
like MaxViT-Base (which is still the other open item in §14).

Source: [motokimura/spacenet8_solution_5th-place](https://github.com/motokimura/spacenet8_solution_5th-place)

---

## What this search did NOT find (originally)

- ~~No first-place (KARI-AI) team code repository~~ **Found since**:
  `github.com/SpaceNetChallenge/SpaceNet8/01-ohhan777` is their actual
  submitted code — HRNet-W48+OCR, `RMILoss(num_classes=5)` for flood
  specifically (a 5-class target combining building/road/flood into one
  space, not a separate binary flood head like this project's
  architecture), deep supervision with `[0.4, 1.0]` auxiliary/main output
  weights, and task-level loss weighting `0.6*building + 0.4*road +
  0.4*flood`. The official repo (`github.com/SpaceNetChallenge/SpaceNet8`)
  turns out to contain all five top teams' code, not just the baseline —
  worth checking `02_number13`, `03-sianalytics`, `04-ZABURO` too if this
  gets revisited.
- The Flood-MATE paper (item 1's closest analog) is paywalled; its
  reported numbers were not verified, only its existence and general
  approach.
- The general loss-function survey's full PDF didn't extract cleanly on
  first attempt; findings above came from the HTML rendering, cross-checked
  against the abstract page.

## Suggested priority if/when this gets acted on

Cheapest-to-verify-value-first, not necessarily implementation order:
**(6) TopK loss ablation** and **(4) Unified Focal Loss** are both drop-in
loss swaps testable in a single run each, low risk. **(5) annotation audit**
is investigation, not code, and might explain some of the plateau on its
own before touching anything else. **(1) semi-supervised Louisiana-West**
and **(3) RMI loss** are the two with the strongest theoretical case for
being genuinely different levers (not just re-tuned versions of what's
already tried), but also the two most expensive to build and validate
correctly.
