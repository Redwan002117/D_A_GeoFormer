# Dual-Axis GeoFormer — Usage Manual

Version: prototype / proposal stage &middot; 2026-09-17

This manual covers installation, running the pipeline end to end, every
CLI flag, the data format (synthetic and real), the checkpoint format, and
troubleshooting. If you only read one section, read **"What 'trained'
means here"** — it's the difference between an honest and a misleading
presentation of this work.

---

## 1. What this project is

A real, runnable implementation of the architecture proposed in the thesis
*"Dual-Axis GeoFormer: Exploiting Global-Local Dependencies for Multiclass
Flood Segmentation of Narrow Infrastructure via MaxViT."* It includes:

- The model (`model.py`) — a Siamese encoder built from simplified
  MaxViT-style blocks (MBConv + windowed Block Attention + strided Grid
  Attention), a bi-temporal Difference Module, a U-shaped decoder, and a
  4-channel Geo-Head.
- A loss function (`losses.py`) — Tversky loss, weighted against missed
  floods.
- Post-processing (`postprocess.py`) — skeletonization + attention-guided
  road-gap bridging.
- A full training loop (`train.py`), evaluation script (`evaluate.py`), and
  end-to-end demo (`demo.py`).
- Two datasets (`dataset.py`) — a procedural synthetic generator (works out
  of the box, no download) and a loader for real, preprocessed SpaceNet-8
  data (needs data you provide — see §6).

## 2. What "trained" means here — read this before presenting

**The model in `checkpoints/best.pt` is trained on the synthetic dataset,
not on SpaceNet-8.** It was trained specifically to validate that the
pipeline is correct end to end: that gradients flow, the loss decreases,
checkpointing/resuming works, and per-class F1 computes correctly. It
succeeds completely at that — loss drops from 0.57 to ~0.01 over 30 epochs
and validation F1 reaches ~0.98–1.00 on every class (see
`training_curve.png`).

That number is **not a SpaceNet-8 accuracy figure**, and saying so out loud
is what keeps this honest: the synthetic generator draws clean geometric
shapes (a smooth curved road, rectangular buildings, an elliptical flood
region) that are far easier to segment than real satellite imagery, which
has texture, shadows, occlusion, seasonal variation, and genuine ambiguity.
A model reaching 0.98 F1 on this synthetic task tells you the
**architecture and training loop are implemented correctly** — it tells you
nothing about flood-detection accuracy in the real world.

| Claim | Is it true of this prototype? |
|---|---|
| "The architecture builds and runs a real forward + backward pass" | **Yes** |
| "The training loop, checkpointing, and metrics are implemented correctly" | **Yes** — demonstrated by a real, reproducible 30-epoch run |
| "Phase 4's road-bridging uses the model's own learned attention, not a fixed heuristic" | **Yes** — `postprocess.py` reads `grid_saliency` from a real forward pass |
| "This model detects floods in real satellite imagery" | **No** — never trained on SpaceNet-8 or any real imagery |
| "This is the MaxViT-Base backbone from the thesis's Phase 2" | **No** — this is an ~11M-parameter, randomly-initialized-then-synthetically-trained stand-in; MaxViT-Base (ImageNet-21k, ~120M params, via `timm`) is a Phase-2 swap, not yet done |

## 3. Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Tested against PyTorch 2.12 (CPU build) and Python
3.14. A CUDA-capable GPU is **not required** — every script auto-detects and
uses one if present (`torch.cuda.is_available()`), and falls back to CPU
otherwise. All figures and checkpoints in this repository were produced on
CPU.

## 4. Quickstart

```bash
# 1. Sanity-check each piece in isolation
python model.py         # builds the network, prints output shape + param count
python losses.py         # Tversky loss on random tensors
python postprocess.py    # gap-bridging on a synthetic mask

# 2. Train (synthetic data, ~7 minutes on CPU for the defaults below)
python train.py --epochs 30 --batch-size 4 --image-size 128 \
                 --synthetic-train-size 120 --synthetic-val-size 24 --lr 1e-3

# 3. Evaluate the best checkpoint
python evaluate.py --checkpoint checkpoints/best.pt

# 4. Run the full pipeline demo (figure + terminal walkthrough)
python demo.py

# 5. Plot the training curve
python plot_training_curve.py
```

`train.py`'s own defaults are deliberately smaller (8 epochs, 64 synthetic
samples) so a bare `python train.py` finishes in well under a minute as a
quick check; the command above is the configuration that actually produced
`checkpoints/best.pt` and `training_curve.png` in this repository.

## 5. CLI reference

### `train.py`

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `None` | Path to a preprocessed SpaceNet-8 directory (§6). Omit to use the synthetic dataset. |
| `--image-size` | `128` | Tile side length in pixels. Larger = slower on CPU. |
| `--epochs` | `8` | Training epochs. |
| `--batch-size` | `4` | Samples per gradient step. |
| `--lr` | `1e-3` | Peak learning rate (AdamW, cosine-annealed to 0 over `--epochs`). |
| `--tversky-alpha` / `--tversky-beta` | `0.3` / `0.7` | False-positive / false-negative weights. `beta > alpha` penalizes missed floods more. |
| `--synthetic-train-size` / `--synthetic-val-size` | `64` / `16` | Number of synthetic samples per epoch (ignored if `--data-dir` is set). |
| `--checkpoint-dir` | `checkpoints` | Where `best.pt` / `last.pt` are written. |
| `--resume` | `None` | Path to a checkpoint to resume optimizer + model state from. |
| `--log-csv` | `training_log.csv` | Per-epoch CSV log (loss, per-class F1, lr, wall time). |
| `--seed` | `0` | Torch random seed. |

### `evaluate.py`

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | *(required)* | Checkpoint to load. |
| `--data-dir` | `None` | Real SpaceNet-8 directory; omit to regenerate the synthetic validation split. |
| `--image-size` | `128` | Must match how the checkpoint was trained. |
| `--batch-size` | `4` | |
| `--synthetic-val-size` | `24` | |

### `demo.py`

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Checkpoint to load. Pass `--checkpoint ""` to force random-init (untrained) weights. |

### `plot_training_curve.py`

| Flag | Default | Meaning |
|---|---|---|
| `--log-csv` | `training_log.csv` | Input CSV from `train.py`. |
| `--out` | `training_curve.png` | Output figure path. |

## 6. Preparing real SpaceNet-8 data

The official SpaceNet-8 release ships as GeoTIFF imagery plus per-class
labels (buildings/roads as GeoJSON, flood attributes as tabular data), via
AWS S3 — see the [SN8 challenge page](https://spacenet.ai/sn8-challenge/)
and the [official baseline repo](https://github.com/SpaceNetChallenge/SpaceNet8)
for the download and its own preprocessing scripts. `SpaceNet8Dataset` in
`dataset.py` expects a **converted** layout:

```
data_dir/
  index.json              # [{"pre": "pre/x.tif", "post": "post/x.tif", "mask": "mask/x.png"}, ...]
  pre/AOI_x_tile_y.tif     # pre-event image
  post/AOI_x_tile_y.tif    # post-event image
  mask/AOI_x_tile_y.png    # single-channel class-index PNG:
                           #   0 = background, 1 = building, 2 = road, 3 = flooded
```

Getting from the raw release to this layout means rasterizing each tile's
building/road/flood labels into one class-index PNG per tile (rasterizing
GeoJSON polygons at the image's resolution and transform) and writing
`index.json` to list every tile triple. This is real, non-trivial data
engineering — it is the next concrete task in the thesis timeline (§4 /
"Weeks 1–2: Data pipeline + baseline reproduction" in the proposal), not
something this manual can shortcut.

Once `data_dir` is in that layout:

```bash
python train.py --data-dir path/to/spacenet8 --image-size 256 --epochs 60
python evaluate.py --checkpoint checkpoints/best.pt --data-dir path/to/spacenet8
```

No other code changes are needed — `SpaceNet8Dataset` returns the same
`(pre_tensor, post_tensor, mask_tensor)` triple as `SyntheticFloodDataset`,
so the model, loss, training loop, and metrics are all already compatible.

## 7. Checkpoint format

`train.py` saves `checkpoints/last.pt` every epoch and `checkpoints/best.pt`
whenever validation loss improves. Each checkpoint is a dict:

```python
{
    "epoch": int,
    "model_state": ...,        # model.state_dict()
    "optimizer_state": ...,    # optimizer.state_dict()
    "config_dict": {...},      # dataclasses.asdict(GeoFormerConfig) -- NOT the
                                # dataclass instance itself, so torch>=2.6's
                                # default weights_only=True load still works
    "val_loss": float,
    "best_val_loss": float,
}
```

Load one manually:

```python
import torch
from model import DualAxisGeoFormer, GeoFormerConfig

ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
model = DualAxisGeoFormer(GeoFormerConfig(**ckpt["config_dict"]))
model.load_state_dict(ckpt["model_state"])
model.eval()
```

## 8. Architecture reference

| Module | File | Role |
|---|---|---|
| `BlockAttention` | `model.py` | Local windowed self-attention — fine building edges. |
| `GridAttention` | `model.py` | Strided global self-attention — full-image reach in one layer; also returns a per-pixel "received attention" saliency map, consumed by Phase 4. |
| `MBConv` | `model.py` | Inverted-residual conv block, the "MB" half of each MaxViT-style stage. |
| `MaxViTBlock` | `model.py` | One `MBConv → BlockAttention → GridAttention` stage unit. |
| `SiameseMaxViTEncoder` | `model.py` | Runs the same weights over the pre- and post-event tile, once each. |
| `DiffModule` | `model.py` | `D = |F_pre − F_post|`, fused with `F_post` via a 1×1 conv, per stage. |
| `UpBlock` / decoder | `model.py` | U-shaped decoder with skip connections back to each encoder stage. |
| `DualAxisGeoFormer` | `model.py` | The full assembled model; `forward()` returns `{"logits", "grid_saliency"}`. |
| `TverskyLoss` | `losses.py` | `alpha` weights false positives, `beta` weights false negatives. |
| `bridge_road_gaps` | `postprocess.py` | Skeletonizes a road mask, closes gaps whose straight-line path scores high on `grid_saliency`. |

Default `GeoFormerConfig` (see `model.py`): 4 stages, channel dims
`(64, 128, 256, 512)`, ~10.9M parameters total.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `_pickle.UnpicklingError: Weights only load failed` on an *older* checkpoint | Checkpoint was saved before the `config_dict` fix (an earlier version of this repo pickled the `GeoFormerConfig` dataclass instance directly, which `torch>=2.6`'s default `weights_only=True` load refuses) | Re-train, or load with `torch.load(path, weights_only=False)` **only if you trust the checkpoint's source**. |
| `RuntimeError: shape '[...]' is invalid for input of size ...` from `GridAttention` | Running a batch size > 1 against a version of `model.py` predating the batch-dimension fix in `GridAttention.forward` | Update to the current `model.py` — the saliency reshape now carries `B` through explicitly instead of assuming batch size 1. |
| `demo.py`'s panels look mostly empty / the road is missing | `IMG_SIZE` was changed without checking `make_synthetic_pair()`'s hand-placed coordinates | They're expressed relative to a 256px canvas and rescaled by `s = IMG_SIZE / 256` — if you edit `IMG_SIZE`, confirm the scene still composes sensibly by looking at `demo_output.png`, don't assume it does. |
| `FileNotFoundError` from `SpaceNet8Dataset` | `--data-dir` doesn't contain `index.json` in the expected layout | See §6. |
| Training loss is `nan` | Learning rate too high for a given `--image-size`/`--batch-size` combination | Lower `--lr` (try `3e-4`) or reduce `--batch-size`. |
| CPU training feels slow | Default `--image-size 128` / small batches should be CPU-tractable (~15s/epoch for 120 synthetic samples on the reference machine) | Reduce `--image-size`, `--synthetic-train-size`, or run on a CUDA GPU if available (auto-detected, no flag needed). |
| `--resume` with a new `--lr` trains at an effectively frozen learning rate (loss barely moves, or moves only from BatchNorm stat drift) | `optimizer.load_state_dict()` restores the *previous* run's param-group `lr` too — after a full cosine anneal that's ~0, silently overriding whatever `--lr` was just requested | Fixed in the current `train.py`: after loading the optimizer state, every param group's `lr` is explicitly reset to `args.lr`, and the scheduler is built with `T_max` = the *remaining* epoch count, not the full `--epochs`. If you're on an older copy of this file, apply the same fix before trusting any `--resume` run's numbers. |

## 10. Serving &amp; deployment

`serve.py` wraps the model in a FastAPI inference API (`/health`,
`/predict`), and `Dockerfile` packages it to run on any container platform
— local machine, any cloud VM, or a PaaS (Render, Railway, Fly.io, Cloud
Run, App Runner, Azure Container Apps). Full instructions, environment
variables, and per-platform notes are in **`docs/DEPLOYMENT.md`**.

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000     # run locally
docker build -t geoformer-proto . && docker run -p 8000:8000 geoformer-proto  # run anywhere
```

## 11. Real-photograph robustness check

`real_image_demo.py` runs the trained pipeline on a real, public-domain
aerial photograph (`real_flood_sample.jpg`, NIST image #15001 — a genuine
Hurricane Katrina flood photo, a U.S. government work) instead of synthetic
data. There is no real pre-event photo of that exact site available here,
so it's paired with a generated dry "stand-in" pre-image — this makes it a
**robustness check** ("does the real architecture run on messy, real
photographic texture without crashing"), **not** a validated bi-temporal
flood prediction. Run it and read `real_image_demo_output.png`'s own
caption before showing it to anyone.

```bash
python real_image_demo.py
```

## 12. Real-data experiment — what actually happened

`prepare_real_data.py` downloads real SpaceNet-8 imagery + labels directly
from the public, **unsigned-access** `spacenet-dataset` S3 bucket (the
official host — no AWS account or credentials needed) and rasterizes real
OpenStreetMap-derived building/road GeoJSON labels (with a real
`flooded: "yes"` flag SN-8's annotators set) into the mask format
`SpaceNet8Dataset` expects, using a pure-Python lon/lat→pixel transform read
straight from the GeoTIFFs' own tags (no GDAL/rasterio dependency). This is
genuinely real SpaceNet-8 data from the 2021 Germany flood AOI — verified by
eye (`real_mask_sanity_check.png` overlays the rasterized mask on the real
photo; building footprints line up correctly).

**The fine-tune result on this data was a real, honest negative result, not
a success — reporting it accurately matters more than reporting a good
number.** Fine-tuning the synthetic-converged checkpoint on a 20-tile sample
(18 train / 2 val, deliberately biased toward flood-heavy tiles to guarantee
some `flooded`-class signal) for 20 epochs:

- Training loss genuinely decreased (0.76 → 0.56) — this time with a
  correctly-applied nonzero learning rate (see §9's `--resume` bug entry).
- Validation loss on the 2-tile split plateaued around 0.70 and building/road
  F1 collapsed toward 0 — the model overfit to background/flooded and
  stopped discriminating buildings from roads.
- Evaluated on the full 20-tile set (train tiles included, so this is an
  optimistic number, not a held-out generalization test):
  `background 0.94 · building 0.26 · road 0.14 · flooded 0.23` F1.

**Why, honestly:** 20 tiles is nowhere near enough data for a 4-class
segmentation task, made worse by deliberately oversampling flood-heavy tiles
for the demo (real class balance would be far more background/building/road
and far less flooded than this sample). SN-8's own Germany AOI alone has 202
tiles; the full multi-AOI benchmark has far more. This result is exactly the
kind of finding that supports the thesis's own "Weeks 1–2: data pipeline"
timeline item being real, scoped work, not a formality — **the checkpointing
logic worked correctly and protected against this**: `checkpoints/best.pt`
was never overwritten by the degraded real-data run (its val_loss never beat
the synthetic-converged checkpoint's 0.0064), so it still holds the better
synthetic-trained weights. `checkpoints/last.pt` holds the real-data-tuned
(but degraded) weights, kept for inspection, not as the recommended
checkpoint to demo from.

**What this experiment proves**: the entire real-data path — S3 download, no
credentials needed, pure-Python georeferencing, correct label rasterization,
training on real tensors, checkpoint-selection safety — works end to end.
**What it doesn't prove**: that the model is accurate on real imagery. That
needs the full dataset (hundreds–thousands of tiles across all AOIs, not 20
biased ones) and, ideally, a GPU — see `notebooks/train_on_colab.ipynb` for
a ready-to-run path to get both.

```bash
python prepare_real_data.py --n-tiles 24 --out-dir real_sn8_dataset
python train.py --data-dir real_sn8_dataset --resume checkpoints/last.pt \
                 --epochs <start_epoch + N> --image-size 256 --lr 2e-4
```

### 12.1 The follow-up: two AOIs, higher beta, and a real improvement

After the finding above, two changes were made and re-tested, not just
proposed: (1) `prepare_real_data.py` pulled a second, genuinely different
AOI (Louisiana-East, real 2021 hurricane flooding), bringing the real
dataset to 352 tiles across two geographies instead of 202 from one; (2)
`--tversky-alpha 0.15 --tversky-beta 0.85` pushed harder against missed
floods, on top of the earlier default 0.3/0.7.

The 90-epoch Germany-only run (§12's original result, extended) converged
cleanly — val_loss plateaus ~0.52–0.53 for its last 40 epochs, not
undertrained — but **`flooded` F1 stayed at exactly 0.000 for all 90
epochs.** The model never learned the one class that is the actual point of
this thesis, on that AOI's real class balance (`flooded` is ~0.7% of
pixels; see `sample_outputs/training_curve_real_full_90ep.png`).

Resuming onto the combined 352-tile dataset with the higher beta changed
that immediately: `flooded` F1 jumped to **0.333 on the held-out validation
split from the very first epoch**, and stayed there
(`sample_outputs/training_curve_combined.png`). A full-dataset evaluation
(optimistic — includes tiles the model trained on, not a clean held-out
number) put it at **0.58**. Building and road F1 also both improved (to
~0.21 and ~0.19 respectively) over the same run.

**One thing flagged, not glossed over**: the held-out `flooded` F1 line is
not just stable, it is bit-identical (exactly 0.333...) across all 19
epochs, while training loss keeps moving and building/road F1 visibly
fluctuate epoch to epoch. The most likely explanation is mundane, not a
bug: F1 is computed on **discrete argmax predictions**, and with a small
validation split (35 tiles, 18 batches), the same pixels can keep landing
on the same side of the argmax decision boundary — giving an identical
true/false-positive/negative count — even while the underlying logits
continue shifting from further training. This was **not verified against
actual per-epoch prediction masks** (older epochs' checkpoints aren't kept,
only `last.pt`/`best.pt`, so the exact epoch-by-epoch masks can't be
replayed after the fact) — flagged here as the honest state of
investigation, not presented as a confirmed root cause.

### 12.2 The full public dataset, and a baseline comparison run

All 801 real, publicly-labeled SpaceNet-8 tiles (Germany's 202 + all 599 of
Louisiana-East) are downloaded in `real_sn8_dataset_full/` — this is the
complete public-labeled dataset, not a sample of it. Pulling the last ~180
tiles surfaced a real environment finding worth recording: two background
jobs (this download plus a concurrent baseline training run) both got
killed by the OS for memory pressure, twice, on a shared machine with only
~2.5-4GB free at the time — and a single job alone was *also* killed once.
No lingering process was left behind by any of these kills, and no data was
lost either time: `prepare_real_data.py`'s per-tile `index.json` write and
`train.py`'s per-epoch checkpoint both did exactly what they were built to
do, and each resumed cleanly from where it left off. **Practical lesson**:
on a memory-constrained, shared machine, don't assume two background jobs
that each look individually lightweight are safe to run concurrently —
run one at a time, and lean on the incremental-save design to make
interruption cheap rather than trying to avoid it.

`baseline.py`'s SN8Baseline (U-Net/ResNet-34, no bi-temporal fusion — the
literature review's comparison point) trained for 30 epochs on the full
801-tile set, surviving the same OOM-kill/resume cycle via `--resume`
(§12.2's own lesson, applied). Results (held-out validation split):

| Class | Baseline F1 (801 tiles) | GeoFormer F1 (352 tiles, §12.1) |
|---|---|---|
| background | 0.978 | 0.98 |
| building | 0.300 | 0.21 |
| road | 0.348 | 0.19 |
| flooded | 0.662 | 0.333 |

**Read this table carefully, not as "the baseline wins"**: GeoFormer's
number here is from the smaller 352-tile run — the baseline has more than
double the training data. GeoFormer is now training on the same full
801-tile set (`training_log_geoformer_801.csv`) specifically so this
comparison can be made on identical data; until that finishes, this table
is not yet the real Table 2 comparison, only the baseline's own real,
verified result.

The same suspicious pattern noted in §12.1 recurs here, and is now
stronger: looking at the actual plotted curve
(`sample_outputs/training_curve_baseline_801.png`), **both** `flooded`
(0.662) **and** `building` (0.300) are exactly flat for the entire
10-epoch window shown, while `road` visibly fluctuates and train_loss
keeps decreasing the whole time. Two different classes landing on
bit-identical F1 simultaneously, while a third genuinely varies, is harder
to wave off as one coincidental boundary-stability case — this needs
actual investigation (e.g. logging per-epoch prediction masks, not just
the aggregate F1 number) before the "argmax boundary" explanation should
be trusted. Recorded here as an open question, explicitly not resolved.

Full-dataset (optimistic, train-included) evaluation of the baseline:
`background 0.981, building 0.050, road 0.405, flooded 0.741` — note
building F1 is *lower* here than on the held-out split alone (0.05 vs.
0.30), an inversion worth noting rather than explaining away; it wasn't
investigated further given time spent already on this comparison.

### 12.3 CORRECTION — §12.1 and §12.2's flooded/building numbers were a metric artifact

**The "flat line" flagged as unexplained in both §12.1 and §12.2 above
turned out to have a real cause, and it changes the honest conclusion.**
Investigated by comparing raw predictions between two nearby checkpoints
(`checkpoints_baseline/best.pt` epoch 27 vs. `last.pt` epoch 30) on the
same real validation tiles, then confirmed on 40 random tiles across the
whole dataset:

**The baseline predicts zero `building` pixels and zero `flooded` pixels
on every single one of 40 random real tiles checked — including the 27
tiles that genuinely contain buildings and the 15 that genuinely contain
flooding.** The reported F1 numbers (0.300 / 0.662) are almost exactly
what pure class-prevalence arithmetic predicts if the model predicts
nothing at all: 13/40 tiles lack building in ground truth → 0.325 expected
vs. 0.300 observed; 25/40 lack flooding → 0.625 expected vs. 0.662
observed. This is not a coincidence.

**Root cause, now understood**: §9's earlier per-class-F1 fix ("a class
absent from both prediction and target should score 1.0, not 0.0") is
correct for a *single* image, but `train.py`/`evaluate.py` were averaging
that per-image judgment across many images as if it measured overall class
performance. A model that never predicts a rare class still scores 1.0 on
every image that happens to lack that class in ground truth too — diluting
total failure on the images that DO have it into a misleadingly high
aggregate. Fixed properly in `train.py`'s new `ConfusionAccumulator`:
raw TP/FP/FN are now summed across the ENTIRE validation pass first, and
F1 is computed once from those global totals — plus a coverage report
(`gt_images` vs. `pred_images` per class) that makes total collapse
visible instead of hidden. `per_class_f1` itself is unchanged and still
correct for what it actually is: a single-call primitive, now documented
as unsafe to average across batches. See `tests/test_metrics.py`'s two new
`ConfusionAccumulator` tests for the regression coverage.

**The corrected, honest picture** (`python evaluate.py`, full 801-tile
dataset, both models):

| Class | Baseline F1 | GT images | Predicted-in images | GeoFormer F1 (epoch 56, still training) |
|---|---|---|---|---|
| background | 0.981 | 801/801 | 801/801 | 0.978 |
| building | **0.000 — collapsed** | 507/801 | **0/801** | **0.000 — collapsed** |
| road | 0.431 | 704/801 | 751/801 | 0.072 (early; training continues) |
| flooded | **0.000 — collapsed** | 198/801 | **0/801** | **0.000 — collapsed** |

**Neither model has learned to detect buildings or flooding on real
SpaceNet-8 imagery at all.** `road` is the only class either model shows
genuine, partial success on. The earlier "flooded F1 0.000 → 0.333 → 0.58,
a real improvement" narrative in §12.1/§12.2 was built on the flawed
metric and should be read as superseded by this section, not as still
true alongside it — it is kept above, uncorrected in place, as an honest
record of the investigation rather than quietly edited away.

**UPDATE, superseded by §12.11 -- this specific "0/801, never" claim no
longer holds.** It was true of the from-scratch, randomly-initialized
encoder used everywhere above. Once a pretrained backbone was added
(§12.11, `--pretrained-backbone efficientnet_b0`), the very first
training epoch produced real, non-zero predictions for BOTH classes with
genuine per-image coverage: `building` F1 0.454 (80/87 held-out images),
`flooded` F1 0.117 (85/87 images) -- both improved further by epoch 2
(`building` 0.491, `flooded` 0.144). So "detects buildings or flooding at
all" is no longer accurate as an absolute; what's still true, and now the
more precise honest claim, is that this project has not yet produced a
checkpoint where that detection is STABLE across many epochs -- both
classes narrowed back toward collapse by epoch 3-6 of that same run (see
§12.11's full table). Read as: the pretrained backbone proved the
representations needed for real detection exist and are reachable, not
that the stability problem is solved.

**What this actually means for next steps**: more real data (§12.2's 801
tiles vs. §12's 202) did NOT fix the rare-class collapse — the metric bug
just made it look like it did. The real open problems are the model
capacity/training-recipe ones §13 was already gesturing at (class
imbalance, no pretrained backbone, limited epochs on real photographic
complexity vs. clean synthetic shapes) — not primarily a data-volume
problem, which is a materially different, harder diagnosis than this
document previously gave.

### 12.4 GeoFormer retrained on the full 801-tile set — road genuinely improves

With the metric fixed, GeoFormer was retrained from its 352-tile checkpoint
onto the full 801-tile combined dataset. This run hit the OOM-kill pattern
described in §12.2 **five times in a row** (a session record) — each kill
recovered from via `--resume`, no progress lost, three real evaluation
checkpoints captured along the way:

| Epoch | road F1 (corrected metric) | building / flooded |
|---|---|---|
| 56 | 0.072 | 0.000 / 0.000 (unchanged throughout) |
| 64 | 0.159 | 0.000 / 0.000 |
| 74 | 0.203 | 0.000 / 0.000 |
| 93 | 0.275* | 0.000 / 0.000 |

`road` is genuinely, steadily improving with more training (0.072 → 0.159 →
0.203 → 0.275) — real learning, not a metric artifact (it has real,
non-zero prediction coverage throughout, unlike building/flooded, which
stay at exactly 0 predicted-in images every single checkpoint checked).
Approaching the baseline's 0.431; whether it would eventually close or
exceed that gap with more epochs than this environment can run in one
sitting is genuinely unknown, not implied either way.

*Epoch 93's number isn't perfectly apples-to-apples with the three before
it: this checkpoint was the first evaluated after §12.5's train/val split
fix (`SpaceNet8Dataset.split()`, a stable hash-based split replacing
`random_split`), so it's measured against a genuinely different -- if now
correctly stable going forward -- 87-tile held-out set than epochs 56-74
were. The direction (continued real improvement) is consistent either way;
the exact number isn't a clean continuation of the same held-out set.

**A second real bug found and fixed in the course of this**: the training
log CSV was only written once, at the very end of the full epoch loop —
so every one of these five kills silently lost the entire per-epoch log,
even though the checkpoint (weights) and `real_sn8_dataset_full/index.json`
both survived via their own incremental writes. `train.py` now writes the
log after every epoch, the same philosophy applied consistently, and
`--resume` now loads and continues an existing log instead of restarting
it — verified with a real train-then-resume test showing one continuous 4-row
log across two separate process invocations. This is why the table above
is three hand-captured `evaluate.py` snapshots rather than a training
curve plot: the curve for epochs 56-74 was already lost to the kills that
predated this fix.

**On the repeated kills themselves**: five consecutive OOM kills on jobs
that were already reduced to batch size 1 is a strong, consistent signal
that this specific machine does not have enough free memory available for
this workload *right now*, for reasons outside this process's own control
(free memory was observed fluctuating between ~2.4GB and ~5.3GB across the
session with no lingering processes of this pipeline's own found after any
kill). The honest response to that is stopping the retry loop and reporting
the real, current state — which is what happened — rather than continuing
to retry the same thing repeatedly on the chance it works. `notebooks/
train_on_colab.ipynb` sidesteps this entirely for anyone who wants to
continue this specific run further.

### 12.5 Train/val split instability — found during a general bug sweep, not this experiment

Separately from the OOM/metric issues above, a systematic bug sweep (asked
for directly: "make it more perfect and error free") found that
`build_dataloaders`' `torch.utils.data.random_split` draws a split that
depends on the dataset's current length and index order — both of which
changed every time `real_sn8_dataset_full` grew across this project's
sessions (202 → 352 → 801 tiles). A tile held out for validation in one
run could silently end up in the next `--resume`'d run's training set,
with no warning printed either way. **Every "held-out validation" F1
reported anywhere above this section, across every resumed run on a
growing dataset, was not actually measured against one stable held-out
set** — the split moved under it each time the dataset grew.

Fixed with `SpaceNet8Dataset.split()`: each tile's own `tile_id` is hashed
to assign it to train or val, independent of dataset size or list order,
so a tile stays on the same side forever once assigned. Verified with a
regression test that a tile's assignment is provably unchanged when 250
more tiles are added around it. `evaluate.py` gained a `--held-out-only`
flag to evaluate specifically against this same stable split.

This does not retroactively invalidate the qualitative findings above (the
building/flooded collapse was confirmed by prediction-coverage checks
across the *entire* dataset, train and val combined, which the split
choice doesn't affect) — but any specific held-out F1 number reported
*before* this fix landed should be read as measured against whatever split
happened to exist at that moment, not a consistent one across runs. Every
held-out number reported from §12.4's epoch 93 checkpoint onward uses the
corrected, stable split.

### 12.6 Continued training to epoch 104, then a class-weighted oversampling run

Continuing from §12.5's epoch 93 checkpoint (stable split, `road` F1
0.275) to epoch 104, `road` F1 stopped climbing and started oscillating:

| epoch | val_loss | road F1 | building/flooded F1 |
|---|---|---|---|
| 95 | 0.4037 | 0.269 | 0.000 / 0.000 |
| 97 | 0.4139 | 0.256 | 0.000 / 0.000 |
| 100 | 0.4299 | 0.196 | 0.000 / 0.000 |
| 101 | 0.4324 | 0.167 | 0.000 / 0.000 |
| 104 | 0.4157 | 0.237 | 0.000 / 0.000 |

`building` and `flooded` stayed at exactly 0.000 with zero predicted-in
images across every one of these epochs -- unchanged from §12.3-§12.5.
`road`'s oscillation (0.28 → 0.17 → 0.24, not a clean plateau) most likely
reflects `lr` restarting near its scheduled floor as each `--resume`
segment's cosine schedule re-anneals over a short remaining-epoch window,
not a genuine ceiling on what the architecture can learn -- but distinguishing
those two explanations needs a longer uninterrupted run, not another
short resume.

**Concrete next step taken, not just recommended**: §13's own
"weighted/oversampled DataLoader" suggestion is now real code, not just a
sentence. `SpaceNet8Dataset.class_presence_weights()` reads the
`class_pixel_counts` field `prepare_real_data.py` already writes into
`index.json` (no new download needed) and returns a per-tile sampling
weight -- boosted 4x for tiles containing any building pixels, 6x for
flooded, multiplicatively for tiles with both. On the real 714-tile train
split, this boosts 528 of 714 tiles (74%). `train.py --oversample-rare-classes`
wires this into a `torch.utils.data.WeightedRandomSampler` over the TRAIN
split only -- validation keeps sampling the true, unweighted real
distribution, so held-out F1 stays comparable to every number above.

The Colab notebook's two main training cells were also fixed (a genuine,
separate bug: their multi-line `!python` commands had a corrupted
line-continuation -- literal `\n` text instead of an actual line break,
left over from an earlier notebook edit -- that would have broken both
cells the moment anyone ran them) and updated to use the same flag, so a
GPU run gets the same boost at real scale.

### 12.7 The oversampling resume regressed, and it destroyed the epoch-104 checkpoint -- a real incident, not a projection

The resumed run mentioned above (epoch 104 + `--oversample-rare-classes
--epochs 200`, boost 4x/6x as first written) did not go well:

| epoch | val_loss | road F1 | building/flooded F1 |
|---|---|---|---|
| 104 (resume point) | 0.4157 | 0.237 | 0.000 / 0.000 |
| 105 | 0.4496 | **0.000** | 0.000 / 0.000 |
| 106 | 0.4495 | **0.000** | 0.000 / 0.000 |

`road` F1 -- the one class this project had real, if modest, learning
on -- collapsed to exactly 0.000 on the very first oversampled epoch and
stayed there, with `val_loss` going up and then flatlining almost
exactly (0.44955 -> 0.44955), not down. This is a real regression, not
noise: two consecutive epochs pointing the same direction with a frozen
loss is the signature of the model settling into a degenerate
background-only local optimum, not a one-epoch blip. The run was stopped
after epoch 106 rather than left to keep degrading.

**The bigger problem it exposed**: recovering from this should have meant
"go back to the epoch-104 checkpoint and try a gentler configuration." It
couldn't -- `checkpoints/last.pt` is overwritten every single epoch
unconditionally, so it already held the collapsed epoch-106 weights, and
`checkpoints/best.pt` turned out to still be an epoch-36 **synthetic**-only
checkpoint from early in this project (`val_loss` 0.0064, a completely
different loss regime, `data_source: None` -- it predates several fields
this project added since). It had never been touched by any real-data
run because `best_val_loss` was carried forward, unconditionally, from
every `--resume` -- including, at some point far earlier, from that
synthetic checkpoint -- and real-data `val_loss` (~0.40-0.45) can never
beat a synthetic-task `val_loss` of 0.0064. **Epoch 104's actual trained
weights -- this project's best real-data GeoFormer checkpoint -- are
unrecoverable.** They were overwritten with no snapshot anywhere holding
them.

Two real fixes landed in `train.py` because of this, not just one lowered
number:

1. **`best_val_loss` no longer carries forward across a data_source
   change on `--resume`.** If the resumed checkpoint's `data_source`
   doesn't match the current run's, `best_val_loss` restarts at infinity
   instead of inheriting an incomparable value -- this is what let
   `best.pt` silently stop updating for this project's entire real-data
   history.
2. **`--checkpoint-every N` (default 10)** now saves a non-overwritten
   `epoch_N.pt` snapshot independent of `last.pt`/`best.pt`, so one bad
   epoch can never again be the difference between "recoverable" and
   "gone."

The oversampling boost defaults were also lowered (`class_presence_weights`:
4.0/6.0 -> 2.0/3.0, so the combined multiplier for a tile with both rare
classes drops from 24x to 6x) on the reasoning that the combined 24x
weight at `--batch-size 1` was plausibly too sharp a distribution shift
for the optimizer to survive without catastrophic forgetting -- stated as
reasoning, not yet as a proven fix, since it hasn't been tested against a
real collapse yet either.

**Training was restarted from epoch 0** (`training_log_geoformer_801_v2.csv`,
the old log preserved as `training_log_geoformer_801_epoch1-106_lost.csv`
rather than deleted) with the gentler oversampling boost active from the
very first epoch and milestone snapshots on. Starting fresh from epoch 0
with oversampling already active is a deliberately different situation
from a resume: there's no already-converged optimum to catastrophically
forget out of.

That run (v2) reproduced the same failure, just delayed: `building` and
`flooded` coverage decayed monotonically from strong (81/87, 82/87 images
at epoch 1) to exactly zero by epoch 4, and stayed at exactly zero through
epoch 8 -- five straight epochs. Oversampling alone (fixing how OFTEN a
rare-class tile is seen) delayed the collapse by a few epochs but did not
prevent it, which pointed at the other half of the problem: how much the
LOSS itself cares about that class once a tile is seen.

### 12.8 Researching how the actual SpaceNet-8 winners handled this, and a real loss bug it surfaced

Rather than keep guessing at oversampling multipliers, this stopped to
research how SpaceNet-8's actual competitors handled the exact same
"flooded" class-imbalance problem (flooded pixels are well under 1% of
this dataset). Findings, sourced:

- The **5th-place solution** (motokimura) does not use tile-level
  oversampling at all. It handles the imbalance with (a) **mosaicing**
  adjacent tiles together to synthesize more flood-containing training
  crops, (b) **pretrained backbones** -- fine-tuning the winning
  SpaceNet-5 road model (SE-ResNeXt-50) and an xView2 building model
  (DenseNet-161) at a low learning rate (1e-5), stated to have
  "significantly improved the score," and (c) treating flood detection as
  a **separate model** (a Siamese U-Net) from building/road segmentation,
  not one joint multiclass head.
  [github.com/motokimura/spacenet8_solution_5th-place](https://github.com/motokimura/spacenet8_solution_5th-place)
- Other top solutions used **swin-transformer backbones pretrained on
  ImageNet-22K**, with UPerNet/Segformer decoders per task.
  [SpaceNet 8: A Closer Look at the Winning Approaches](https://medium.com/@SpaceNet_Project/spacenet-8-a-closer-look-at-the-winning-approaches-75ff4033bf53)
- Separately, the class-imbalanced-segmentation literature (e.g. "Unified
  Focal loss: Generalising Dice and cross entropy-based losses to handle
  class imbalanced ... segmentation," Yeung et al. 2022; the original
  Focal Tversky Loss paper, Abraham & Khan 2018) is consistent that
  Tversky/Dice losses need an **explicit per-class weight** for severe
  imbalance, not just alpha/beta's precision/recall tradeoff.

**This surfaced a real, previously-unnoticed bug in `losses.py`**:
`TverskyLoss.forward` computed `1.0 - tversky.mean()` -- averaging the
per-class Tversky index across all 4 classes with **equal weight**.
`flooded` (well under 0.1% of pixels dataset-wide) was getting exactly
the same 25% share of the loss as `background` (~85%+ of pixels, and
already near-perfect almost immediately). This had been sitting in the
loss the entire project, silently undermining every real-data run,
independent of the oversampling experiments above.

Two architectural findings from this research are real but out of scope
to act on immediately: a pretrained backbone (this project's MaxViT
encoder is randomly initialized, not ImageNet-pretrained like the winners'
encoders) and a separate flood-classification head/model (rather than one
joint 4-way per-pixel softmax forcing `flooded` to compete directly
against `background`/`building`/`road` in a single decision) are both
bigger changes than a loss-function fix -- named here honestly as the
next real lever if the fix below still isn't enough, not silently
deferred.

**Fix applied**: `TverskyLoss` now takes an optional `class_weights`
list -- a weighted average instead of a uniform mean (`class_weights=None`
preserves the exact original behavior, verified by test). `train.py`
gained `--class-weights` (comma-separated, `background,building,road,flooded`
order). Training was restarted a third time
(`training_log_geoformer_801_v3.csv`, v1/v2 logs preserved, not deleted)
combining both fixes: `--oversample-rare-classes --class-weights "1,3,1,8"`
-- building weighted 3x, flooded 8x relative to background/road's
baseline 1x, chosen to reflect flooded being the rarer and harder of the
two collapsed classes.

### 12.9 The v3 weights (1,3,1,8) traded one collapse for another -- `road` this time

| epoch | val_loss | building F1 (coverage) | road F1 (coverage) | flooded F1 (coverage) |
|---|---|---|---|---|
| 1 | 0.8706 | 0.123 (84/87) | **0.000 (0/87)** | 0.071 (84/87) |
| 2 | 0.8563 | 0.256 (77/87) | **0.000 (0/87)** | 0.049 (86/87) |

Genuinely different from v2's failure: `building` and `flooded` both had
real, non-zero coverage from epoch 1 onward this time. But `road` --
this project's one class with real prior success -- collapsed to exactly
0.000 in both epochs measured. Weighting `background`/`road` both at 1x
against `building` at 3x and `flooded` at 8x left `road` undefended once
the other two classes started pulling harder on the same per-pixel
softmax; two-for-two epochs pointing the same direction was treated as
strong enough evidence to intervene rather than wait for more.

Rebalanced to `1,2,2,4` (`road` raised from 1x to 2x, `flooded` lowered
from 8x to 4x, `building` from 3x to 2x) -- softening the gap between all
three foreground classes relative to background, rather than sacrificing
`road` to chase `flooded`. Training restarted a fourth time
(`training_log_geoformer_801_v4.csv`, v1/v2/v3 logs all preserved).
Results appended here as real epochs land.

**A separate bug found and fixed while reviewing this session's own newer
code** (asked directly to "fix all bugs and errors," so the dashboard/db
code that hadn't had a dedicated review pass yet got one): `dashboard_server.py`'s
sample-upload endpoint built the saved file path directly from the
client-supplied multipart filename with no sanitization -- a crafted
filename like `../../../../evil.txt` resolved outside the intended upload
directory entirely (confirmed directly against the running server, not
just reasoned about) -- a real path-traversal arbitrary-file-write.
Fixed by reducing the filename to a sanitized basename before it ever
touches a path. Separately, `db/db_logger.py`'s `log_epoch` never rolled
back after a failed query, which would have left the DB connection stuck
in Postgres's "aborted transaction" state for the rest of a run after any
single transient failure -- reproduced directly against the live database
and fixed with an explicit `rollback()`.

### 12.10 v4 (weights 1,2,2,4) through its first milestone checkpoint -- a clearer, still honest, still not-solved picture

| epoch | val_loss | building F1 (cov) | road F1 (cov) | flooded F1 (cov) |
|---|---|---|---|---|
| 1 | 0.830 | 0.197 (83/87) | 0.100 (81/87) | 0.068 (83/87) |
| 2 | 0.831 | 0.140 (84/87) | 0.095 (85/87) | 0.040 (81/87) |
| 3 | 0.795 | 0.180 (82/87) | 0.142 (85/87) | 0.064 (85/87) |
| 4 | 0.734 | **0.000 (0/87)** | 0.202 (85/87) | 0.087 (86/87) |
| 5 | 0.727 | 0.000 (0/87) | 0.212 (85/87) | 0.089 (86/87) |
| 6 | 0.735 | 0.000 (0/87) | 0.199 (85/87) | 0.071 (85/87) |
| 7 | 0.733 | 0.000 (0/87) | 0.184 (85/87) | 0.054 (86/87) |
| 8 | 0.740 | 0.000 (0/87) | 0.189 (84/87) | 0.055 (84/87) |
| 9 | 0.724 | 0.000 (0/87) | 0.217 (86/87) | **0.098 (83/87)** (best flooded yet) |
| 10 | 0.427 | 0.000 (0/87) | 0.211 (83/87) | **0.000 (0/87)** |

The clearest signal in this project's real-data history: `road` -- 1x
weight in v3, 2x weight here, present in nearly every tile regardless of
oversampling -- is the ONE class that stayed non-zero across every single
one of these 10 epochs, every configuration tried so far. `building`
collapsed at epoch 4 and never recovered through epoch 10. `flooded` held
on through epoch 9 (its best epoch of the whole project, F1 0.098) and
then also collapsed exactly at epoch 10, the same epoch `val_loss` dropped
sharply (0.724 -> 0.427) -- consistent with the model finding a lower-loss
solution that fits background+road more confidently and abandons the
harder, rarer classes rather than one that actually improved on all four.

**Honest synthesis across v2/v3/v4** (three different sampling/weighting
configurations, all summarized here rather than left scattered): every
configuration tried produces SOME partial collapse, just a different one
each time -- v2 (oversampling only) collapsed both building and flooded;
v3 (aggressive 1,3,1,8 weights) collapsed road instead; v4 (moderate
1,2,2,4 weights) holds building/flooded longer but building still
collapses by epoch 4 and flooded eventually follows. This is consistent
with, not contradicting, the S12.8 research finding: a single joint 4-way
per-pixel softmax forces every foreground class to compete directly for
the same probability mass, and sampling/loss-weighting tricks can shift
WHICH class loses that competition without changing that a competition
exists at all. The real fix the actual SpaceNet-8 winners used --
decoupling flood detection into its own model/head instead of one joint
softmax, plus a pretrained backbone -- remains the next real lever, named
honestly rather than implied to be solved by hyperparameter tuning alone.

The epoch-10 checkpoint is safely preserved as `checkpoints/epoch_10.pt`
(the `--checkpoint-every` fix from S12.7), independent of whatever
`last.pt` looks like by the time anyone reads this.

v4 was stopped at epoch 12 (`road` F1 0.211 -> 0.229 -> 0.249, still
climbing; `building`/`flooded` still exactly 0.000) once the decision was
made to move to the pretrained-backbone fix below rather than keep tuning
this run's hyperparameters further -- not because it broke.

### 12.11 Phase 2, for real: an ImageNet-pretrained backbone

Given a choice between another round of hyperparameter tuning on the
from-scratch encoder and the architectural fix the actual SpaceNet-8
winners used, the pretrained-backbone half of that fix was built --
self-contained, no dataset reprocessing needed (unlike a separate flood
head, which needs the original per-tile GeoJSON re-rasterized to keep
building/road identity alongside a flooded attribute -- out of scope for
this pass, named honestly, not silently dropped).

`GeoFormerConfig.pretrained_backbone` (default `None`, exact prior
behavior) takes any `timm` model name supporting `features_only=True` at
strides 4/8/16/32 -- e.g. `"efficientnet_b0"`, confirmed to work in this
CPU environment (not `timm/MaxViT-Base` itself: a ~120M-parameter
network this environment can't run at real scale, per S1's own scope
note; EfficientNet-B0 is the CPU-tractable pretrained option available
here, at 5.3M of its own parameters). `SiameseMaxViTEncoder` runs the
backbone once per timestamp, projects each of its 4 stages onto
`cfg.stage_dims` via a 1x1 conv, and feeds the SAME `MaxViTBlock`
attention stages (block + grid) used by the from-scratch path -- the
thesis's actual proposed attention mechanism and Phase 4's
grid-saliency-based bridging are unchanged either way. `train.py
--pretrained-backbone efficientnet_b0` wires it in.

Verified, not just wired: a full forward pass at production resolution
(256x256, default `stage_dims`) with REAL downloaded ImageNet-1k weights
produces correctly-shaped logits (13.16M total parameters, vs 10.9M for
the from-scratch default), and a checkpoint saved with
`pretrained_backbone` set round-trips correctly through
`checkpoint_utils.load_checkpoint_model` (config reconstruction, state
dict load, a second forward pass on the reloaded model). Two new
regression tests use `pretrained=False` (random init, no network call)
to exercise the wiring itself -- channel projection, stage count, output
shape -- fast and deterministically, separate from the one-time manual
verification that real pretrained weights actually download and load.

One real, disclosed tradeoff: reloading ANY checkpoint saved with
`pretrained_backbone` set re-runs `timm.create_model(..., pretrained=True)`
first (downloading/loading real ImageNet weights, cached after the first
time) even though `load_checkpoint_model`'s subsequent `load_state_dict`
immediately overwrites them with the checkpoint's own trained weights --
so `evaluate.py`/`demo.py`/`serve.py`/`dashboard/process_samples.py` all
now need network access (or a warm `timm` cache) and `timm` installed the
first time they load such a checkpoint. Not fixed here -- the wasted
download is a one-time, cached cost, not a correctness problem -- but
named so it isn't a surprise.

**Training restarted a fifth time** with this backbone
(`training_log_geoformer_801_v5.csv`, v1-v4 logs all preserved), combined
with `--oversample-rare-classes --class-weights "1,2,2,4"`:

| epoch | val_loss | building F1 (cov) | road F1 (cov) | flooded F1 (cov) |
|---|---|---|---|---|
| 1 | 0.734 | **0.454 (80/87)** | 0.269 (84/87) | **0.117 (85/87)** |
| 2 | 0.686 | **0.491 (78/87)** (best ever, either class) | 0.342 (83/87) | **0.144 (84/87)** (best ever) |
| 3 | 0.381 | 0.018 (25/87) | 0.337 (83/87) | 0.000 (0/87) |
| 4 | 0.375 | 0.000 (0/87) | 0.361 (83/87) | 0.000 (0/87) |
| 5 | 0.375 | 0.000 (0/87) | 0.332 (84/87) | 0.000 (0/87) |
| 6 | 0.370 | 0.000 (0/87) | 0.362 (81/87) | 0.000 (0/87) |
| 7 | 0.378 | 0.000 (0/87) | 0.325 (80/87) | 0.000 (0/87) |

The single most important result of this entire project's real-data
history is right there in epochs 1-2: `building` F1 0.491 and `flooded`
F1 0.144, BOTH with real, high per-image coverage, simultaneously with
`road` also working. No prior configuration (from-scratch encoder, any
oversampling/weighting combination) ever produced that. It did not hold
-- by epoch 4 both collapsed back to exactly 0.000, the same shape seen
before, just delayed further and from a much higher peak. `road` alone
has stayed stable across all 7 epochs (0.27-0.36 throughout), the same
pattern as every prior run.

Read plainly: the pretrained backbone gives the network the CAPACITY to
represent building/flooded well (proven, not theoretical -- epoch 2 is
real evidence). What it hasn't fixed is why that capability doesn't
survive continued training -- still consistent with the S12.10 synthesis
that a single joint 4-way softmax makes every foreground class compete
for the same probability mass, and now with additional evidence that
better features raise the PEAK before the collapse rather than preventing
it. Training continues, and building/flooded stayed at exactly 0.000
through epoch 10 (`road` also degraded there, F1 0.171, coverage 49/87 --
possibly the start of a further slide, not yet confirmed as one).

### 12.12 `best.pt` was tracking the wrong thing this whole time -- a real, costly bug

Checking `checkpoints/best.pt` directly (asked to "fine tune the model
for best output results") found it holds **epoch 5** -- a fully
COLLAPSED checkpoint (`building` F1 0.000, `flooded` F1 0.000). Epoch 2
-- `building` F1 0.491, `flooded` F1 0.144, both with genuine per-image
coverage, this project's actual best real-data result -- was never saved
as "best" and its weights are now gone (overwritten by `last.pt`'s
continued progress, with no milestone at epoch 2 since
`--checkpoint-every 10` only lands on multiples of 10).

**Root cause**: `best.pt` is selected by lowest `val_loss`, and Tversky
loss is dominated by whichever classes have the most pixels
(`background`, then `road`). A checkpoint that stops predicting
`building`/`flooded` at all can have a LOWER val_loss than one that
predicts them imperfectly but for real:

| epoch | val_loss | building F1 | flooded F1 | selected as "best" by val_loss? |
|---|---|---|---|---|
| 2 | 0.686 | 0.491 | 0.144 | no |
| 5 | 0.370 | 0.000 | 0.000 | **yes** |

So the checkpoint-selection logic was actively working against the
project's actual goal the entire time -- not just on this run.
`checkpoints_baseline/`'s "best" and every prior GeoFormer run's "best"
inherit the same risk; they simply weren't checked this closely before.

**Fixed**: `train.py --checkpoint-metric {val_loss, mean_f1, min_f1}`
(default `val_loss`, exact prior behavior -- existing docs/callers see no
change unless they opt in). `mean_f1` maximizes the average F1 across
classes seen so far; `min_f1` maximizes the WORST class's F1 -- the
strictest choice, since a checkpoint can't be "best" while any seen class
is still at exactly 0. `checkpoint_score()` is a pure function (lower
always better, regardless of metric) with 6 new regression tests,
including one that reproduces this exact epoch-2-vs-epoch-5 scenario
numerically and asserts `mean_f1` would have chosen epoch 2.

Training restarted a sixth time (`training_log_geoformer_801_v6.csv`)
with the same config as v5 (pretrained backbone, oversampling, class
weights) plus `--checkpoint-metric min_f1` -- from epoch 0, not resumed,
since epoch 10's state (still collapsed, `road` also degrading) isn't a
useful base and the actually-useful epoch-2 state is unrecoverable. This
was a cheap restart to make, not a wasted one: the good result appeared
in the first 2 epochs last time, not after 100+.

### 12.13 v7 (backbone freezing + min_f1): the fix works, the collapse still doesn't

v6 was stopped after 1 epoch once `--checkpoint-metric min_f1` was
confirmed protecting the good result correctly; v7 adds
`--freeze-backbone-epochs 3` (S12.11.1, verified ~1.6x faster per step
while frozen) on top of the same config.

| epoch | val_loss | building F1 (cov) | road F1 (cov) | flooded F1 (cov) |
|---|---|---|---|---|
| 1 | 0.730 | 0.371 (82/87) | 0.305 (86/87) | 0.161 (82/87) |
| 2 | 0.401 | 0.355 (35/87) | 0.297 (87/87) | 0.000 (0/87) |
| 3 | 0.375 | 0.000 (0/87) | 0.350 (82/87) | 0.000 (0/87) |
| 4 | 0.399 | 0.000 (0/87) | 0.189 (87/87) | 0.000 (0/87) |
| 5 | 0.378 | 0.000 (0/87) | 0.336 (86/87) | 0.000 (0/87) |
| 6 | 0.385 | 0.000 (0/87) | 0.236 (87/87) | 0.000 (0/87) |
| 7 | 0.407 | 0.000 (0/87) | 0.169 (87/87) | 0.000 (0/87) |
| 8 | 0.387 | 0.000 (0/87) | 0.236 (86/87) | 0.000 (0/87) |

**The `min_f1` checkpoint fix works exactly as designed, verified in a
real extended run, not just a unit test**: `checkpoints/best.pt` is
still epoch 1 (`best_score` = -0.161, `flooded`'s F1 at that epoch, the
worst class present) after 8 further epochs of collapse -- confirmed by
loading the checkpoint file directly and reading its own recorded
metadata. The exact loss the S12.12 bug caused did not repeat.

**The backbone-freeze/unfreeze boundary (epoch 3->4) produced no visible
change in the collapse pattern** -- `building`/`flooded` were already
collapsed by epoch 3, still frozen, and stayed collapsed straight through
unfreezing at epoch 4 with no recovery. This is real, useful negative
evidence: it argues against "the pretrained backbone's own features are
being disrupted by early gradients" as the mechanism, and for the S12.10
synthesis instead -- a single joint 4-way softmax head forcing every
foreground class to compete for the same probability mass, independent
of what's feeding it. `road` itself never fully collapsed across all 8
epochs (oscillating 0.17-0.36, no clear trend either direction).

### 12.14 What would actually need to change next

Named plainly, in rough order of how directly each one addresses the
mechanism above rather than working around it:

1. **Decouple flood detection from the joint softmax** (the change the
   competition research in S12.8 points at most directly). Concretely:
   a second, binary output head (flooded / not-flooded) trained with its
   own loss, added to the existing building/road/background 3-way head's
   logits rather than competing inside one 4-way softmax with them. This
   also matches SpaceNet-8's own real label semantics better -- flooding
   is an attribute of a building or road, not a mutually exclusive
   category -- but needs the original per-tile GeoJSON re-rasterized to
   recover that attribute information, since this project's current mask
   format already collapsed it to a single class index before this
   architecture question came up.
2. **A bigger pretrained backbone.** `efficientnet_b0` is the CPU-tractable
   choice available in this environment; a `timm` MaxViT or ConvNeXt
   variant closer to the thesis's own Phase 2 description needs a GPU
   (Colab) to be practical, and hasn't been tried.
3. **A focal-Tversky variant of the loss** (`(1-TI)^(1/gamma)` per class
   before weighting), on top of the class-weighted mean already built --
   literature-supported (Abraham & Khan 2018) as complementary to, not a
   replacement for, class weighting: it additionally down-weights
   already-easy pixels within a class rather than only reweighting
   classes against each other.
4. **A held-out early-stopping criterion tied to `min_f1` specifically**,
   not just checkpoint *selection* -- e.g. stop increasing an unfrozen
   backbone's learning rate, or reduce it, the moment `min_f1` regresses
   for N consecutive epochs, instead of letting a full cosine schedule
   run to completion regardless of what the rare classes are doing.
5. **More epochs on the SAME config**, least likely to help on current
   evidence (v7's collapse pattern looks settled by epoch 3, not still
   converging) but cheapest to just try given how fast a good result
   appeared last time (2 epochs, not 100+).

### 12.15 Acting on item 3: Focal Tversky loss

v7 was stopped after 10 epochs -- ten straight of `building`/`flooded`
collapse, and `road` itself starting to degrade (F1 0.394 -> 0.182,
coverage dropping to 41/87 at epoch 10), a genuinely worse trend than
letting it continue was likely to reverse on its own.

Implemented item 3 from S12.14: `TverskyLoss` gained an optional
`focal_gamma` parameter (`(1 - TI) ** (1/gamma)` per class, applied
before the existing `class_weights` averaging -- Abraham & Khan 2018).
Where `class_weights` reweights which CLASS the loss prioritizes, this
additionally reweights which PIXELS within that class it prioritizes,
concentrating gradient on ones the model still gets wrong rather than
ones it's already confident about -- a genuinely different lever than
anything tried so far, not a rename of an existing one. `focal_gamma=1.0`
(default) is the identity power, exact prior behavior, verified by test.
`train.py --focal-gamma` wires it through. 3 new regression tests,
including one that isolates the focal term's effect on a genuine
partial-credit prediction (neither perfect nor total failure), where the
difference actually shows up.

Training restarted an eighth time (`training_log_geoformer_801_v8.csv`)
combining every fix so far plus `--focal-gamma 2.0` (the original paper's
typical range is 1-3; 2.0 is the middle of it, not yet tuned against this
specific problem).

**Result, real and negative**: `focal_gamma=2.0` collapsed FASTER than
any prior configuration -- `building` and `flooded` both fully gone
(0.000, zero coverage) by epoch 2-3, compared to epoch 3-4 in every
config without it.

| epoch | building F1 (cov) | road F1 (cov) | flooded F1 (cov) |
|---|---|---|---|
| 1 | 0.420 (78/87) | 0.295 (85/87) | 0.137 (85/87) |
| 2 | 0.002 (6/87) | 0.305 (80/87) | 0.000 (0/87) |
| 3 | 0.000 (0/87) | 0.366 (80/87) | 0.000 (0/87) |

A plausible mechanism, stated as reasoning rather than proven: the focal
term's whole design amplifies gradient on pixels a model is already
struggling with -- exactly the collapsed classes' own pixels, once they
start slipping. Instead of pulling the model back toward predicting
them, that extra gradient magnitude may have pushed it faster toward the
same degenerate background+road-only solution the un-focused loss
reaches more slowly. Stopped after 3 epochs rather than let a
demonstrably-worse trend continue -- this is useful negative evidence
for the next attempt (a lower gamma, or focal weighting applied only to
the ALREADY-boosted classes rather than uniformly across all four), not
a dead end to just retry unchanged.

### 12.16 Geometric augmentation (D4): a genuinely untried lever, not a variant of one already tried

Every configuration attempted so far (v1-v8) shared one thing in common:
**zero data augmentation**. Every real tile was shown to the model in
exactly one fixed orientation, every epoch, for the project's entire
history. Satellite/aerial imagery has no canonical "up" -- a building
rotated 90 degrees is still a building, a flooded road mirrored is still
a flooded road -- so this was real, unused headroom, and a genuinely
different kind of lever than anything tried in S12.5-S12.15 (all of
which changed how the loss or sampler weighted existing tiles, never
what the tiles themselves looked like).

`dataset.py`'s `SpaceNet8Dataset` gained `augment: bool = False` (default
preserves exact prior behavior). When enabled, `_augment_tile()` applies
a random dihedral-group (D4) transform -- horizontal flip, vertical
flip, and a 0/90/180/270-degree rotation, each independently chosen --
identically to the pre-event image, post-event image, AND mask, so the
three stay spatially aligned. The mask specifically uses NEAREST
resampling on rotation (not left to PIL's default for its image mode),
since any interpolation between adjacent class indices would fabricate
a label value that was never in the real GeoJSON data. `train.py
--augment` wires it through -- a second `SpaceNet8Dataset` instance
backs the TRAIN split only (val keeps seeing each tile in its one real
orientation, so held-out numbers stay comparable epoch to epoch;
sharing one instance between both splits was rejected specifically to
avoid coupling that choice). 3 new regression tests verify the
transform preserves image size, applies identically across all three
images (checked via a corner marker present in all three, asserted to
land in the same output coordinates regardless of which random
transform got picked, run across 30 trials to exercise every code
path), and never introduces a mask value that wasn't in the original.

Training restarted a ninth time (`training_log_geoformer_801_v9.csv`)
dropping the focal term back to its default (`focal_gamma=1.0`, given
S12.15's negative result) and adding `--augment` to the rest of the
stack (pretrained backbone, oversampling, class-weighted loss, backbone
freezing, `min_f1` checkpoint selection).

**First 3 epochs, real and mixed**:

| epoch | building F1 (cov) | road F1 (cov) | flooded F1 (cov) |
|---|---|---|---|
| 1 | 0.365 (81/87) | 0.209 (86/87) | 0.069 (86/87) |
| 2 | 0.432 (52/87) | 0.268 (86/87) | **0.187 (80/87)** — best flooded coverage of any run so far |
| 3 | 0.000 (0/87) | 0.295 (82/87) | 0.000 (0/87) |

Augmentation improved epoch-2 quality specifically (`flooded`'s best F1
AND best coverage of the entire project, beating v5's epoch-2 0.144 and
v7's epoch-1 0.137) but did **not** delay or prevent the epoch-3 collapse
itself — same timing as every from-scratch-augmentation run before it.
Read plainly: augmentation appears to raise the ceiling of what the
model reaches before collapsing, without addressing why it collapses at
all. Consistent with, not contradicting, the S12.10/S12.13 synthesis
that the joint 4-way softmax's class competition is the actual
mechanism -- better features (pretrained backbone) and better-conditioned
training (augmentation) both raise the peak; neither has yet changed
whether the peak holds.

`docs/EXTERNAL_DATA_PLAN.md` names the next data-side lever (external
building-footprint data) along with explicit trigger conditions for when
to actually build it -- not now, since S12.14's architecture-level fix
(a separate flood head) hasn't been tried yet and would need to be ruled
out first before concluding this is a data-volume problem rather than an
architecture one.

### 12.17 Debugging the collapse mechanism directly, not just its symptoms

Every section above establishes *that* building/flooded collapse happens
and *when*. This section is the first to actually look *inside* a
collapsed checkpoint to find out *how*, with three short, targeted
tests against v9's own live checkpoint (epoch 3, the collapsed one) --
not guessed, measured.

**Test 1 -- is it gradient instability?** Ran 60 real training steps
(same sampler, same loss config, backbone frozen to match v9's own
epoch 1-3 state) logging the per-step gradient norm before any optimizer
update. Result: stable throughout, 0.04-0.29, no spikes, no correlation
with which classes were present in that step's tile. **Ruled out**:
gradient clipping would have nothing to clip here.

**Test 2 -- is the model narrowly losing the argmax, or has it truly
abandoned the class?** Loaded the actual collapsed checkpoint
(`checkpoints/last.pt`, epoch 3) and evaluated its raw softmax
probability at every real building-labeled pixel of a genuine
building-containing held-out tile (2,476 real building pixels). Result:
mean predicted `building` probability = **0.0000** (max across all
2,476 pixels: 0.0001); mean `background` probability at those same
pixels = 0.9699. This is not a close competition the model is
narrowly losing -- it has confidently, near-totally eliminated
`building` as a possibility everywhere, including on pixels it was
literally shown are buildings.

**Test 3 -- is this a dead output channel (shallow, cheaply fixable) or
a deeper representational collapse?** Inspected the Geo-Head's final
1x1 conv layer's actual weights per class, on the same checkpoint:

| class | bias | weight norm |
|---|---|---|
| background | +0.611 | 2.569 |
| building | -0.337 | 1.596 |
| road | -0.202 | 1.326 |
| flooded | -0.104 | 1.725 |

`building`'s output weights are NOT degenerate -- a norm of 1.6 is
substantial, not a dead/zeroed channel, and the bias gap to background
(~0.95) is far too small on its own to explain a probability of
0.0000 vs 0.97. The near-total collapse must therefore come from
**upstream**: the shared 32-channel decoder features feeding into this
layer no longer contain a discriminative signal for `building` that this
weight vector can act on, at least not at the pixels that matter.
Background's own output weight vector has the LARGEST norm of all four
classes AND the most favorable bias -- consistent with the shared
representation itself having been shaped disproportionately around
recognizing background (present in effectively every pixel of every
tile) at the expense of the rarer classes' own discriminative features.

**What this rules out and what it strengthens, concretely**: not
gradient instability (Test 1), not a shallow/cheaply-reinitializable
output-layer problem (Test 3) -- both would have been quick wins if
true, and neither is. What remains consistent with all three tests is
the S12.10/S12.13 synthesis: a single joint softmax, with one dominant
background-like class and several rare ones sharing the SAME upstream
representation, lets the dominant class's training signal reshape that
shared representation around itself. A separate, decoupled output head
for the rare classes (S12.14 item 1) would give them their own gradient
pathway into a representation that doesn't have to also serve
background's overwhelming pixel-count advantage -- this debugging pass
is real evidence FOR that fix, not just a restatement of the plan.

### 12.18 Implementing the separate flood head (S12.14 item 1)

Acted directly on S12.17's evidence. `GeoFormerConfig` gained a
`separate_flood_head: bool = False` field. When set, `DualAxisGeoFormer`
replaces the single 4-class `geo_head` with a shared `split_trunk`
feeding two independent final layers: `structure_head` (3-way,
background/building/road) and `flood_head` (1-channel binary). Each
gets its own weights, so `flooded`'s gradient no longer has to share a
representation -- or an output layer -- with the classes S12.17 showed
are crowding it out. `forward()` still returns a synthesized 4-channel
`out["logits"]` (`cat([structure_logits, flood_logit])`) purely for
backward compatibility with `evaluate.py`/`demo.py`/`serve.py`/
`ConfusionAccumulator`/the dashboard, none of which needed to change.

**Backward compatibility, deliberately protected**: the default
(`separate_flood_head=False`) path's `geo_head` module is untouched,
byte-for-byte the same `nn.Sequential` as before this feature existed --
verified by a test that checks `hasattr(model, "geo_head")` is true and
`hasattr(model, "structure_head"/"flood_head"/"split_trunk")` is false
on the default path. This was caught as a bug in my own first draft:
an earlier version renamed the shared trunk to `geo_trunk` on BOTH
paths, which would have silently broken `load_state_dict` for every
checkpoint saved before this change (key mismatch, e.g.
`geo_head.0.weight` vs `geo_trunk.0.weight`). Fixed before any test ran.

**Loss side**: `TverskyLoss.forward()` gained an optional `valid_mask`
parameter -- pixels where it's `False` are excluded from every
tp/fp/fn sum entirely, not diluted. This matters specifically for the
structure loss: a pixel labeled `flooded` in the ground truth has no
recoverable label for what its underlying structure (building/road/
background) actually was -- the original rasterization already
overwrote that information (see docs/EXTERNAL_DATA_PLAN.md for what
recovering it would take, out of scope here). Guessing a fallback
class for those pixels would be actively wrong training signal;
excluding them via `valid_mask=(mask != 3)` is the honest choice.
Verified by a test that a masked pixel, even made maximally wrong,
produces byte-identical loss to the same tensor with that pixel
physically removed -- not just "small effect", genuinely zero.

`train.py` wiring: a new `--separate-flood-head` flag constructs two
`TverskyLoss` instances (structure: `num_classes=3`; flood:
`num_classes=2`, treated as background-vs-flooded binary via
`cat([-flood_logit, flood_logit])`), reusing the run's existing
`--tversky-alpha/beta`/`--focal-gamma` and splitting `--class-weights`
across both (`[:3]` for structure, `[1.0, w_flooded]` for flood). A
`compute_loss(out, mask)` helper isolates the branch so train and val
loops share identical combination logic -- `structure_loss(structure_
target=mask.clamp(max=2), valid_mask=mask!=3) + flood_loss(target=
mask==3)`.

**Testing**: 6 new tests (4 in `tests/test_model.py`, 2 in
`tests/test_losses.py`) -- shape correctness, the backward-compat
`hasattr` check, gradient flow to both new heads independently, the
4-class-only guard rejecting `separate_flood_head=True` with
`num_classes != 4`, `valid_mask=None` matching prior behavior exactly,
and the masked-pixel-contributes-zero proof. Full suite: 57 passed
(`python -m pytest tests/ -q`), including a real end-to-end smoke test
of `train.py --separate-flood-head` on synthetic data (ran, logged a
checkpoint, no crash; that smoketest run was deleted from the
Postgres `training_runs`/`epoch_logs` tables afterward so it doesn't
pollute the dashboard's real experiment history).

**Status**: implemented and tested, not yet validated on real data.
v9 (augmentation, no separate head) was stopped at epoch 4 -- its log
confirms the collapse pattern is exactly as settled as S12.16 already
documented (`val_f1_building=0.0`, `val_f1_flooded=0.0` at both epoch
3 and 4) -- freeing the GPU for v10, which combines this fix with the
rest of the proven stack (pretrained backbone, oversampling,
class-weighted loss, D4 augmentation, `min_f1` checkpoint selection,
backbone freezing). v10's first several epochs are the real test of
whether S12.17's hypothesis holds: if `flooded`/`building` F1 stays
nonzero past epoch 3 with a separate head, the collapse was indeed a
shared-representation problem, not something the loss/backbone/
augmentation levers alone could fix.

### 12.19 v10 result: the separate flood head breaks the collapse (provisional, epochs 1-3)

v10 (separate flood head + pretrained backbone + oversampling +
class-weighted loss + D4 augmentation + backbone freezing + `min_f1`
checkpoint selection, `training_log_geoformer_801_v10.csv`) is the direct
test of S12.17's hypothesis. Real result, epochs 1-3:

| epoch | building F1 | road F1 | flooded F1 | val_loss |
|---|---|---|---|---|
| 1 | 0.272 | 0.324 | 0.189 | 1.329 |
| 2 | 0.472 | 0.358 | 0.168 | 1.234 |
| 3 | **0.529** | 0.385 | **0.383** | 1.225 |

**Epoch 3 is exactly where every prior run (v1-v9) collapsed to
`building=0.000, flooded=0.000`.** v10 does not collapse there: building
F1 rises monotonically across all three epochs (0.272 -> 0.472 -> 0.529)
and flooded F1 reaches 0.383 at epoch 3 -- the best flooded F1 recorded
at ANY epoch of this entire project (previous best was v9 epoch 2's
0.187, S12.16). val_loss is also falling smoothly, not spiking.

This is consistent with, not just hoped-for by, the S12.17 diagnosis:
once `flooded` has its own output head and its own loss term instead of
sharing softmax competition with `background`'s overwhelming pixel-count
advantage, the mechanism that caused every previous collapse has no
direct pathway to act on it.

**Marked provisional deliberately**: three epochs is not proof the
collapse is gone for good -- S12.13's backbone-freezing run also looked
stable before further training; unfreezing at epoch 4 or continued
training could still reveal a later collapse or a different failure
mode. Monitoring continues through at least epoch 6-10 before this
result is called anything stronger than "the fix is working so far."

### 12.20 Epochs 4-5: F1 collapse still hasn't happened, but a slower, different warning sign has appeared

| epoch | building F1 | road F1 | flooded F1 | flooded coverage (of 87 val images) |
|---|---|---|---|---|
| 3 | 0.529 | 0.385 | 0.383 | 61/87 |
| 4 | 0.523 | 0.351 | **0.396** (new project-best) | 36/87 |
| 5 | 0.561 | 0.392 | 0.250 | 33/87 |

**The good news, unchanged**: building F1 keeps climbing (0.529 ->
0.523 -> 0.561, essentially flat-to-rising, not the sharp drop to 0.000
every prior run showed here) and flooded F1 stayed well above zero
through epoch 5 -- the collapse mechanism from S12.17 genuinely has not
recurred in its original form.

**A real, separate concern, reported honestly rather than glossed
over**: `flooded` *coverage* -- the number of validation images where
the model predicts flooded pixels at all -- has fallen every single
epoch: 73 -> 70 -> 61 -> 36 -> 33 out of 87. F1 is computed only over
images where the class is predicted or present, so a high F1 on a
shrinking set of images is compatible with the model quietly narrowing
*where* it's willing to predict flooded, even while staying accurate
when it does. This is a genuinely different mechanism from S12.17's
finding (a total probability collapse across every pixel) -- here the
per-pixel signal within a prediction is fine, but the model is
predicting flooded in fewer images epoch over epoch. Worth naming
plainly: this could be (a) benign -- the model correctly learning that
fewer validation tiles actually contain real flooding as it stops
over-predicting early on, which oversampling and class weighting can
induce transiently, or (b) an early, slower version of the same
representational crowding-out S12.17 diagnosed, just acting on
*which images* trigger any flooded prediction instead of collapsing
the probability everywhere at once. Not distinguishable from 5 epochs
alone -- continuing to monitor whether coverage keeps falling toward 0
(pointing to (b)) or stabilizes (pointing to (a)).

### 12.21 Epochs 6-7: the coverage question resolves toward the benign explanation

| epoch | building F1 | road F1 | flooded F1 | flooded coverage (of 87) |
|---|---|---|---|---|
| 5 | 0.561 | 0.392 | 0.250 | 33/87 |
| 6 | 0.574 | 0.415 | 0.396 | 28/87 |
| 7 | 0.562 | 0.372 | **0.482** (new project best) | 41/87 |

S12.20 asked whether falling `flooded` coverage would keep declining
toward 0 (a slower collapse) or stabilize (benign). It bottomed at 28
(epoch 6) and **recovered** to 41 (epoch 7) -- not a monotonic decline
toward zero. Combined with flooded F1 reaching a new project-best of
0.482 at epoch 7 (previous best 0.396, epoch 4), this leans toward the
benign reading: the model's flooded predictions are fluctuating
epoch-to-epoch the way a genuinely learning classifier's do, not
narrowing toward silence. Building F1 also remains stable in the
0.52-0.57 range across epochs 3-7, no sign of the joint-softmax
collapse recurring through 7 real epochs -- more than double the
longest any prior run survived without it (every one of v1-v9 collapsed
by epoch 3-4). Still calling this provisional, not concluded --
continuing to epoch 10+.

**Correction, epoch 8-9 (see S12.22 below): this read was premature.**
flooded DID collapse, just later and differently than before. Left here
unedited, not quietly fixed, because the reasoning at the time was
honest given the data available then -- S12.22 is the correction.

### 12.22 Epochs 8-9: flooded collapses anyway, but NOT together with building/road this time

| epoch | building F1 | road F1 | flooded F1 | flooded coverage | val_loss |
|---|---|---|---|---|---|
| 7 | 0.562 | 0.372 | 0.482 | 41/87 | 1.159 |
| 8 | 0.530 | 0.314 | **0.000** | **0/87** | 0.727 |
| 9 | 0.540 | 0.410 | **0.000** | **0/87** | 0.680 |

**The honest headline: flooded collapsed, at epoch 8.** S12.20-S12.21's
optimistic read of the coverage trend was wrong -- it wasn't
stabilizing, epoch 7 was a local peak before the same kind of
total-collapse S12.17 found (0.000 F1, 0 coverage, not a gradual
decline).

**What's genuinely different from every prior run (v1-v9), and matters
for what this proves**: `building` and `road` did NOT collapse with it.
building F1 stayed at 0.530/0.540 (barely moved from epoch 7's 0.562)
and road at 0.314/0.410 (within its normal epoch-to-epoch range) --
completely unlike v1-v9, where building and flooded always collapsed
together in the same epoch because they shared one softmax. The
separate-head architecture change DID achieve its narrow goal: it
decoupled flooded's fate from the other classes'. What it did NOT do is
prevent flooded's OWN collapse within its own binary head.

**Revised understanding of the mechanism**: S12.17 diagnosed a shared-
representation problem where the dominant class (background) reshapes
upstream features at the expense of rare ones sharing its softmax.
Giving flooded its own head removed that specific cross-class
competition -- and building/road's stability here confirms that removal
worked. But flooded is *still* a severe class-imbalance problem even in
isolation (well under 0.1% of pixels project-wide, S12.3), and a binary
head with a Tversky loss can still find "predict not-flooded everywhere"
as a loss-reducing local minimum on its own -- val_loss dropping sharply
(1.159 -> 0.727 -> 0.680) exactly when flooded's F1 hit zero is
consistent with this: the flood_loss term shrinks a lot when flooded
predictions vanish, because so few pixels are actually flooded, so total
loss drops even as the class is abandoned. This means S12.14's item 1
(separate head) was a real, partial fix -- it solved the cross-class
crowding-out -- but not a complete fix for flooded specifically, which
still needs its own severe-imbalance handling (stronger class weighting
on the flood loss specifically, a higher beta there, or genuinely more
flooded-labeled data per docs/EXTERNAL_DATA_PLAN.md) independent of the
architecture question.

Training continues past epoch 9 to see whether flooded's collapse here
is as permanent as v1-v9's was, or whether -- now that it's isolated
from building/road -- it's able to recover on its own without dragging
the other two classes down with it, which the architecture change would
still have earned credit for even if flooded itself needs a second,
separate intervention.

### 12.23 Epochs 10-12: flooded's collapse looks permanent; building/road stay healthy and improving

| epoch | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| 8 | 0.530 | 0.314 | 0.000 | 0/87 |
| 9 | 0.540 | 0.410 | 0.000 | 0/87 |
| 10 | 0.551 | 0.394 | 0.000 | 0/87 |
| 11 | 0.593 | 0.415 | 0.000 | 0/87 |
| 12 | **0.599** | 0.402 | 0.000 | 0/87 |

**flooded**: 5 consecutive epochs at exactly 0.000 F1 / 0 coverage
(epochs 8-12). This is no longer read as a transient dip -- it matches
the shape of every prior permanent collapse in this project (once F1
hits exactly 0.000 and coverage hits exactly 0, no run has ever
recovered from that state on its own). Calling this the likely
conclusion for flooded in v10's current configuration, barring a
change of intervention.

**building/road**: the real, durable positive result. Across 5 more
epochs since flooded's collapse, building kept RISING (0.530 -> 0.599,
new project best, still trending up) and road stayed in its normal
0.31-0.42 band -- neither shows any sign of being dragged down by
flooded's collapse. This is the clearest evidence yet that the
separate-head architecture change achieved its actual goal: it broke
the cross-class coupling that made v1-v9's collapses total (all three
classes together). What remains is a second, different problem --
flooded's own severe class imbalance is apparently still enough to
collapse its own isolated binary head, independent of any competition
with other classes.

**Implication for what to try next** (not yet implemented, S12.14's
still-open items): since flooded's own head can collapse in complete
isolation, the fix has to act on flooded's OWN loss/sampling, not on
inter-class competition (already solved). Concretely, in roughly
likely-effort order: (1) a flood-specific beta higher than the shared
`--tversky-beta` (this run still uses 0.7 default, shared with
structure classes -- the flood loss branch could use its own, more
aggressive alpha/beta independent of the structure loss's), (2) a much
larger flood-class weight specifically in the binary flood loss (this
run's `--class-weights "1,2,2,4"` only weights the STRUCTURE loss;
`compute_loss()` in train.py currently hard-codes the flood loss's own
weights as `[1.0, class_weights[3]]` = `[1.0, 4.0]` -- a much higher
ratio, e.g. 1:20, is untried), (3) genuinely more flooded-labeled data
(docs/EXTERNAL_DATA_PLAN.md's trigger conditions are now closer to met,
since the architecture-level fix has been tried and flooded's collapse
persists in a form not explained by cross-class competition).

### 12.24 v11: flood-specific loss reweighting does NOT recover an already-collapsed head

Implemented S12.23 items 1-2 together: `--flood-class-weight` and
`--flood-tversky-beta` let the flood loss use its own independently-tuned
class weight/beta instead of inheriting the structure loss's values.
v11 resumed from v10's epoch-12 checkpoint (building F1=0.599, road
F1=0.402, flooded collapsed to 0.000 for the prior 5 epochs) with
`--flood-class-weight 20 --flood-tversky-beta 0.9` -- 5x the inherited
flood weight (was 4.0) and a higher false-negative penalty (was 0.7).

| epoch | building F1 | road F1 | flooded F1 |
|---|---|---|---|
| 12 (v10, before the new weighting) | 0.599 | 0.402 | 0.000 |
| 13 | 0.578 | 0.425 | 0.000 |
| 14 | 0.571 | 0.411 | 0.000 |
| 15 | 0.570 | 0.419 | 0.000 |
| 16 | 0.591 | **0.443** | **0.000** |

**Real, honest negative result**: 4 full epochs with a 5x stronger,
independently-tuned flood loss -- flooded stayed at exactly 0.000
throughout. Loss reweighting alone does not recover a flood head that
has already fully collapsed. Building/road stayed healthy and
undisrupted by the resume (confirming the resume mechanics themselves
weren't the problem) and road even reached a new project-best (0.443).

**Working explanation, not yet independently verified**: once a binary
classification head's logit has converged to a confidently negative
value (always predicting "not flooded"), the local gradient of a
softmax/sigmoid-based loss near that saturation point is close to zero
-- multiplying a near-zero gradient by a larger loss weight still
produces a near-zero gradient. A stronger loss can't out-argue an
already-saturated activation; it would need to have been applied
BEFORE the head saturated, not after (v10's own first several epochs,
before the epoch-8 collapse, showed flooded actively learning under
the weaker default weighting -- the head was never stuck in a poor
region while gradients were still flowing).

**Implication for the next real experiment**: rather than resuming
training with a stronger loss, the more promising next test is
reinitializing ONLY the flood head's own weights (fresh random init,
keeping the trunk/structure head/backbone's already-learned features
intact) before resuming with the stronger flood-specific weighting from
S12.24 -- giving the flood head a non-saturated starting point instead
of asking loss reweighting to reverse a state it can no longer see a
gradient out of. Not yet implemented or tested; this is a documented,
reasoned next step, not a claimed result.

### 12.25 Implementing flood-head reinitialization

`train.py` gained a standalone `reinit_flood_head(model, optimizer)`
function and a `--reinit-flood-head` CLI flag (`--resume` +
`--separate-flood-head` only). It replaces `model.flood_head`'s weights
with a fresh `nn.Conv2d` init -- the same init a brand-new model's
flood_head would get -- while leaving `split_trunk`/`structure_head`/the
backbone exactly as loaded from the checkpoint. It also clears Adam's
per-parameter moment estimates (`exp_avg`/`exp_avg_sq`) for just
`flood_head`'s parameters, so its first several post-reinit updates
aren't still shaped by the collapsed run's stale gradient statistics.

**Verified, not just written**: a real before/after test -- trained a
tiny model for 1 real step (so Adam has genuine per-parameter state),
called `reinit_flood_head`, and confirmed (a) `flood_head.weight`
actually changed, (b) `split_trunk[0].weight` did NOT change at all
(byte-identical), (c) `flood_head`'s Adam state was cleared. Also
smoke-tested at the CLI level (`--resume ... --reinit-flood-head`) on a
real checkpoint: the reinit message printed, and directly diffing the
before/after checkpoint files confirmed `flood_head.weight` changed
while `structure_head.weight` and `split_trunk.0.weight` only moved by
the small amount one real optimizer step would produce, not a reinit's
worth of change. 1 new test, 58 passing overall
(`python -m pytest tests/ -q`).

v12 launches next: resume from v10's checkpoint (same one v11 started
from -- building F1 0.599, road F1 0.402, flooded collapsed) with
`--reinit-flood-head` PLUS v11's flood-specific weighting
(`--flood-class-weight 20 --flood-tversky-beta 0.9`), so the fresh flood
head starts learning under the stronger loss from a non-saturated
point, instead of v11's mistake of applying the stronger loss to
weights that had already saturated.

### 12.26 v12 epoch 13: the fresh flood head recovers immediately

v12 resumed from the exact same v10 checkpoint v11 used (building
F1=0.599, road F1=0.402, flooded collapsed to 0.000), with the same
stronger flood-specific loss v11 used (`--flood-class-weight 20
--flood-tversky-beta 0.9`) -- the ONLY difference from v11 is
`--reinit-flood-head`, giving flood_head fresh weights instead of
resuming its already-saturated ones.

| | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| v11 epoch 13 (saturated head, same stronger loss) | 0.578 | 0.425 | 0.000 | 0/87 |
| **v12 epoch 13 (fresh head, same stronger loss)** | **0.603** (new project best) | 0.405 | **0.443** | **51/87** |

**Real, immediate recovery.** One epoch after reinitialization, flooded
F1 is 0.443 with real coverage across 51 of 87 validation images --
compare v11's IDENTICAL loss configuration producing exactly 0.000
across 4 full epochs on the same (but saturated) head. This is the
strongest direct confirmation yet of the diagnostic chain built across
S12.17 (the collapse is a representational problem, not a dead output
layer) through S12.24 (a saturated logit has near-zero local gradient
regardless of loss weight) -- the reinit's whole premise (give the head
a fresh, non-saturated starting point) produced exactly the outcome
that premise predicts.

Building also reached a new project-best F1 (0.603), confirming the
reinit didn't disturb the parts of the checkpoint it wasn't supposed to
touch (`split_trunk`/`structure_head`/backbone).

**Appropriate caution, not overclaiming from one epoch**: v10 itself
looked this promising for 7 straight epochs (flooded F1 up to 0.482)
before collapsing at epoch 8 (S12.22-S12.23). A single strong epoch is
real, positive evidence, not proof the fresh head won't eventually
saturate the same way under continued training. Monitoring continues
past the epoch-8-equivalent point (v12's epoch ~20, i.e. 8 epochs after
the reinit) before this is called a durable fix rather than a promising
early sign.

### 12.27 Epochs 14-16: flooded F1 holds and improves, but coverage is declining again

| epoch | building F1 | road F1 | flooded F1 | flooded coverage (of 87) |
|---|---|---|---|---|
| 13 | 0.603 | 0.405 | 0.443 | 51/87 |
| 14 | 0.603 | 0.416 | 0.396 | 46/87 |
| 15 | 0.566 | 0.407 | 0.411 | 26/87 |
| 16 | 0.561 | 0.427 | **0.535** (new project best) | 25/87 |

**The good news**: flooded F1 has stayed well above zero for 4
consecutive epochs since the reinit and just reached a new project
best (0.535) at epoch 16 -- no collapse, unlike v10 (collapsed epoch 8)
or v11 (never recovered). This is the longest a reinitialized/recovered
flood head has held real signal in this project.

**The honest caveat, reported plainly rather than glossed over**:
coverage is declining across the same 4 epochs -- 51 -> 46 -> 26 -> 25
of 87 validation images. This is the exact same shape S12.20 flagged
for v10 (73->70->61->36->33) before v10 went on to fully collapse at
epoch 8. S12.21 read a similar recovery-then-dip pattern in v10 as
benign, and that read turned out to be wrong (S12.22). Naming that
directly this time instead of repeating the same optimistic read: F1
staying high on a SHRINKING set of images the model is willing to
predict flooded in is compatible with a slower version of the same
narrowing-then-collapsing mechanism, not necessarily a sign the fix
has fully worked. Not calling this resolved either way yet -- continuing
to monitor specifically for whether coverage keeps shrinking toward 0
(pointing to a delayed collapse, same failure mode, just pushed back
several epochs by the reinit) or stabilizes/recovers (pointing to a
genuinely different, healthier training trajectory this time).

### 12.28 Epochs 19-20: it collapsed again -- reinitialization delays the collapse, doesn't fix it

| epoch | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| 18 | 0.543 | 0.449 | 0.329 | 18/87 |
| **19** | 0.552 | 0.403 | **0.000** | **0/87** |
| **20** | 0.566 | 0.405 | **0.000** | **0/87** |

**The honest, full result of the reinitialization experiment (S12.25-S12.28),
stated plainly**: the fresh flood head produced a real, substantial
recovery -- 6 epochs (13-18) with genuine flooded detection, a new
project-best F1 of 0.535 at epoch 16 -- but then collapsed again at
epoch 19, exactly the shape S12.20 and S12.27 both flagged as a warning
sign and both times correctly. Reinitializing the head delayed the
collapse (7 epochs post-reinit vs v10's 8 epochs from a cold start) and
raised the peak (0.535 vs v10's 0.482), but did not produce a durable
fix. Building/road stayed healthy and decoupled throughout, including
through the recollapse (0.552-0.566 / 0.403-0.449 at epochs 19-20) --
that part of S12.17-S12.25's diagnosis remains solid and repeatedly
confirmed across v10, v11, and v12.

**What this changes about the overall picture**: the separate-head
architecture (S12.14 item 1, S12.17-S12.18) successfully and repeatedly
solves the CROSS-class problem -- building/road no longer collapse
together with flooded, confirmed across three independent runs now.
But flooded's OWN collapse is not an architecture problem or a loss-
weighting problem in the way tested here -- three different
interventions (separate head alone/v10, stronger flood-specific loss on
a saturated head/v11, stronger flood-specific loss on a FRESH
head/v12) all eventually produced the same outcome: flooded's
prediction narrows and then collapses, on a timescale of roughly
7-8 epochs regardless of starting point. The consistent recurrence
across genuinely different interventions is itself informative: it
points toward the amount and diversity of real flooded-pixel training
data being the actual bottleneck, not the model architecture or the
loss function's weighting -- both of which have now been tried and
both improved matters without fixing the underlying problem.

**Recommendation**: `docs/EXTERNAL_DATA_PLAN.md`'s trigger condition
("the architecture-level fix has been tried and ruled out first") is
now genuinely met -- not just the architecture fix, but a loss-weighting
fix and a head-reinitialization fix on top of it, all three tried, all
three real, none durable. The next real lever left un-tried is more/
different flooded-labeled training data, per that plan's Phase 1
(Microsoft Global ML Building Footprints) recommendation. The best
checkpoint from this whole v10-v12 sequence -- v12's epoch 16
(building F1 0.561, road F1 0.427, flooded F1 0.535, all three
detecting) -- is preserved at `checkpoints_v12/best.pt` (min_f1
selection correctly protected it from being overwritten by the later
collapse) and is this project's best real result to date, honestly
reported with its own limitation: it is a peak the training run passed
through, not a state it converged to and stayed at.

### 12.29 External validation, and a genuinely new lever: EMA of model weights

Before assuming S12.28's instability was specific to this project's own
implementation, checked the actual SpaceNet-8 competition's public
winner writeups (spacenet.ai/sn8-challenge, and
github.com/motokimura/spacenet8_solution_5th-place, a real 5th-place
competition solution with its methodology published). Real, directly
relevant finding: a top-5 team on the SAME dataset, with a genuinely
different architecture (Siamese U-Net, not GeoFormer's MaxViT/grid-
attention design), reports the exact same symptom this project spent
S12.17-S12.28 diagnosing -- their own words: **"the validation metric
varied significantly from epoch to epoch"** for flood detection,
despite their own mitigations. This is real, external confirmation
that flood-detection instability on this exact dataset is a known,
genuinely hard problem, not a bug or a modeling mistake unique to this
project's implementation -- valuable both as a sanity check and as a
citable point for the thesis's honesty about what's actually solved
vs. what's still an open, real limitation of the underlying task.

Their specific mitigation, not yet tried here: **EMA (exponential
moving average) of model weights**, momentum 2e-3, applied every
epoch. Implemented as `--ema-momentum` in `train.py`: after every
optimizer step, `ema = ema * (1 - momentum) + raw_weights * momentum`
(standalone `ema_init`/`ema_update` functions, unit-tested for the
exact blend formula, for cloning-not-referencing the raw weights, and
for copying rather than blending integer buffers like BatchNorm's
`num_batches_tracked`). Validation and checkpoint SELECTION run against
the EMA weights, not the raw training ones -- the raw weights keep
training normally each epoch (EMA is swapped in only for the
validation pass, then swapped back out), and `checkpoint_utils.
load_checkpoint_model` prefers `ema_state` over `model_state` by
default so every inference consumer (`demo.py`, `evaluate.py`,
`serve.py`, `dashboard/inference.py`) benefits automatically. `None`
(default) disables EMA entirely -- exact prior behavior, unaffected.

**Why this is a plausible fix for THIS project's specific failure
mode, not just a generic trick to try**: S12.17-S12.28 characterized
the collapse as the flood head's raw weights swinging into a
saturated, always-predicts-background state after several epochs of
real (if noisy) learning. EMA doesn't prevent the raw weights from
still swinging -- but the EMA snapshot, updated slowly (small
momentum), lags behind and averages across exactly that kind of swing,
so even if the raw weights collapse, the EMA weights (which is what
actually gets evaluated, checkpointed, and used for inference) may
not, as long as the collapse doesn't persist long enough to drag the
slow-moving average down with it. Untested claim, not yet a result --
v13 (next) is the real test.

Verified: 3 new unit tests (blend formula correctness, independent
clone not a reference, integer-buffer handling) plus an end-to-end
smoke test -- trained 2 synthetic epochs with `--ema-momentum 0.1`,
resumed for a 3rd epoch and confirmed EMA state round-trips through
`--resume` correctly, and confirmed `load_checkpoint_model` loads the
EMA weights by default (`prefer_ema=False` recovers the raw ones).
61 tests passing overall.

**Also fixed while implementing this**: the local dashboard's live
detection (added this session) defaulted to `checkpoints/best.pt`,
which is NOT this project's actual best real result -- that's
`checkpoints_v12/best.pt` (S12.28). The dashboard was silently serving
predictions from a weaker, older checkpoint than the one this project
actually has. Fixed by pointing the default at the real best checkpoint.

### 12.30 A research pass before the next intervention, and a second new lever: compound BCE+Tversky loss

Before choosing what to try after EMA, dispatched a research pass
(citing real sources: the SpaceNet-8 5th-place solution's own README,
Isensee et al. 2021's nnU-Net paper, Ghiasi et al. 2021's Simple
Copy-Paste augmentation paper, Tian et al.'s Recall Loss paper, and
Zhong et al. 2023's "Understanding Imbalanced Semantic Segmentation
Through Neural Collapse") specifically asking what's left to try for
THIS project's exact failure mode (a binary head that trains well for
several epochs, then saturates into always-predicting-background),
given everything already ruled out in S12.14-S12.29.

**Top recommendation, implemented**: a compound loss -- Tversky (already
used) PLUS binary cross-entropy, matching the SpaceNet-8 5th-place
solution's own `1*dice + 1*bce` for its flood head. The mechanism is
specific, not generic: S12.24 diagnosed the collapse as the flood
logit saturating into a region where Tversky's gradient (a GLOBAL
tp/fp/fn ratio) goes near-zero once predicted positives are ~0, even
though real flooded pixels are present in the target. BCE is computed
per-pixel independently and has no such collapse. **Verified
numerically, not just argued**: a new test constructs a deliberately
saturated logit (confidently "not flooded" everywhere) with real
flooded pixels in the target, and confirms Tversky's gradient norm on
it is <1e-3 (effectively vanished) while BCE's gradient norm on the
IDENTICAL input is over 100x larger -- this is the literal, measured
justification for adding it, not an assumption.

`--flood-bce-weight` (default `None`, disabled, exact prior behavior)
adds `weight * binary_cross_entropy_with_logits(flood_logit,
flood_target)` to the flood loss. 1 new unit test (the gradient-
vanishing proof above) plus an end-to-end smoke test. 62 tests passing.

**Other research findings, ranked, not yet implemented** (candidates
for the run after next, in roughly this priority order):
1. Freezing `flood_head`'s parameters specifically once its own F1
   plateaus/declines (per-head early stopping) -- operationalizes
   S12.14 item 4 surgically instead of a project-wide LR change.
2. A wider (7x7, vs the current 1x1) kernel in `flood_head` -- the
   SAME SpaceNet-8 5th-place solution's other mitigation alongside EMA,
   for the same reported symptom. Mechanism: spatial context smooths
   per-pixel logit noise, since flood regions are large/contiguous, not
   point-like. Their own source is explicit this only "mitigate[d] the
   instability to SOME extent," not a full fix. Higher cost here than
   the other items: changes `flood_head`'s parameter shape, so it can't
   load directly from an existing checkpoint's weights (needs fresh
   init or a center-weight transplant).
3. Weight decay specifically on `flood_head`'s own optimizer param
   group, to bound logit magnitude and make the S12.24 saturation
   regime harder to reach in the first place.
4. Copy-paste augmentation (Ghiasi et al. 2021): paste real flooded
   regions from flooded tiles onto non-flooded tiles, to test whether
   S12.28's "more data" conclusion is really about volume or about
   per-epoch exposure, using only the existing 801 tiles, before
   committing to `docs/EXTERNAL_DATA_PLAN.md`'s larger effort.
5. Recall Loss (Tian et al.) as a self-adjusting replacement for the
   now hand-tuned `--flood-class-weight` -- lower priority, same family
   as weighting already tried.

### 12.31 v13 epochs 17-18: EMA looks genuinely stabilizing so far (early, provisional)

v13 (separate head + EMA momentum 0.002 + the S12.24-S12.25 flood-
specific weighting, resumed from v12's best checkpoint at epoch 16 --
so v13's epoch numbering continues at 17) gives the cleanest possible
comparison: v12 and v13 share the exact same starting weights, same
loss config, same data, differing ONLY in EMA. Direct comparison at
the same two epochs:

| epoch | v12 flooded F1 (no EMA) | v13 flooded F1 (EMA) | v12 coverage | v13 coverage |
|---|---|---|---|---|
| 17 | 0.430 | **0.518** | 25/87 | 23/87 |
| 18 | 0.329 (declining) | **0.519** (flat) | 18/87 | 22/87 |

v13 is both higher AND markedly more stable epoch-to-epoch (0.518 ->
0.519, essentially flat) than v12's declining trajectory (0.430 ->
0.329, the beginning of the slide that led to v12's epoch-19 collapse)
at the identical relative point. This is consistent with, and so far
supports, S12.29's hypothesis: the EMA-smoothed weights (which is what
gets validated and checkpointed) may not swing into the raw weights'
oscillation the same way.

**Explicitly not calling this resolved**: v12 itself didn't collapse
until epoch 19-20 -- two epochs is not enough to know whether EMA
prevents that collapse or merely delays/smooths its visible symptoms
by a similar margin to what reinitialization already achieved (S12.25-
S12.28's own lesson: early, multi-epoch stability has repeatedly looked
like a fix before, and wasn't, twice). Monitoring continues specifically
through and past v13's own epoch 19-20 before this gets called
anything stronger than "a promising, measurably different trajectory
so far."

### 12.32 v13 epoch 19: the drop has started -- and a real dashboard lineage feature/bug found while checking it

| epoch | flooded F1 | flooded coverage |
|---|---|---|
| 18 | 0.519 | 22/87 |
| **19** | **0.006** | **5/87** |

The drop v12 also showed right before its own epoch-19 collapse is
visible here too -- coverage fell from 22 to 5, F1 from 0.519 to
essentially zero. Not yet a full collapse (0.006 and 5/87 are not
exactly 0/0 the way v10's and v12's actual collapse epochs were), but
this is very likely the start of the same event, one epoch later than
v12's. Continuing to watch epoch 20+ to see whether it fully bottoms
out or, unlike every prior run, recovers.

**Separately, while checking this live on the dashboard**: the user
asked for the "Latest run" F1 trend chart to show full history --
correctly identified a real gap. Every `--resume` (v11, v12, v13...)
had been creating a NEW, disconnected `training_runs` row whose own
`epoch_logs` only start wherever it resumed from (v13's own rows: epoch
17 on), so the chart looked like the current run had no history before
that point, even though it's a real continuation of v10's training.

Fixed properly, not just papered over: `training_runs` gained
`parent_run_id`; `train.py` now records its own run_name in every
checkpoint (`ckpt_payload["run_name"]`), and `db_logger.py`'s
`DBLogger` accepts a `parent_run_name` (read back from the --resume'd
checkpoint's own recorded name) and resolves it to a Postgres id.
`/api/runs/{id}/epochs` walks the parent chain and merges every
ancestor's epochs into one continuous series -- v13's chart now
correctly shows v10 (epochs 1-12) -> v12 (13-16) -> v13 (17+) in one
view. Since a resume chain can BRANCH (v11 and v12 both resumed from
the same v10 checkpoint), this walks one run's own real ancestry, never
a sibling's -- v13's chart never shows v11's separate epochs.

**A real bug caught testing this live, not assumed correct**: v12 kept
running for a few epochs after v13 resumed from its epoch-16
checkpoint, so v12's OWN epoch_logs table has an epoch 20 -- a
DIFFERENT, diverged training trajectory than what v13 eventually reaches
at its own epoch 20. The first version of the merge overwrote by epoch
NUMBER alone, so once v12's epoch-20 row existed, it got displayed as
if it were v13's current state, even before v13's own training had
actually reached epoch 20 (confirmed live: the dashboard showed "epoch
20" with numbers that turned out to be v12's, while v13's own CSV log
was still only at epoch 19). Fixed: once a descendant run has ANY
epoch data, its epoch range fully supersedes its ancestor's from its
own first epoch onward, even for epoch numbers the descendant hasn't
logged YET -- a chart that stops at the run's real latest epoch is
correct; one that borrows a diverged sibling/ancestor's future epoch
is not. Existing v10-v13 rows backfilled with their real, known
lineage (v11.parent=v10, v12.parent=v10, v13.parent=v12) since they
predate this feature and can't record it retroactively on their own.

### 12.33 v13 epoch 20: collapsed -- and at nearly the SAME ABSOLUTE epoch as v12, despite EMA

| epoch | flooded F1 | flooded coverage |
|---|---|---|
| 19 | 0.006 | 5/87 |
| **20** | **0.000** | **0/87** |

Confirmed, not just the beginning of a drop: v13 has collapsed.

**The genuinely new, important observation is WHEN**: v13's checkpoint
lineage is v10 (epochs 1-12) -> v12 (13-16, reinit) -> v13 (17-20, EMA
added on top). v13's collapse landed at **absolute epoch 20** -- almost
exactly the same absolute epoch where v12 ALSO collapsed (epoch 19),
even though v13 had an entirely additional intervention (EMA) that v12
never had. If EMA (or reinit, or the stronger flood-specific loss) were
independently delaying the collapse by some fixed NUMBER OF EPOCHS from
whenever each was applied, v13 should have collapsed later than v12,
proportional to its own extra epochs of EMA-smoothed training. It
didn't -- it collapsed at essentially the same point in the OVERALL
training trajectory (total real epochs since v10's original start),
regardless of which combination of interventions was active for the
epochs leading up to it.

**Why this matters more than any single run's result**: three
substantively different interventions -- reinitialization alone (v12,
S12.25-S12.28), reinitialization plus a much stronger flood-specific
loss (v12 continued, same run), and all of that plus EMA weight
smoothing (v13, S12.29-S12.33) -- have now all failed at nearly the
same absolute epoch count on this same 801-tile dataset. That
consistency is itself the finding: it's much more consistent with a
fixed limit tied to this dataset/training-regime (how much real
flooded-pixel signal 801 tiles' worth of oversampled epochs can
actually sustain) than with any one of the three mechanisms being
individually fixable by a smarter loss, a smarter head, or a smarter
optimizer trick.

**Where this leaves the project, stated plainly**: the separate-head
architecture (S12.14 item 1) is a real, validated, positive
contribution -- building and road reliably stop collapsing together
with flooded, confirmed across v10/v11/v12/v13, four independent runs.
Flooded's own residual instability has now resisted four different,
individually well-motivated interventions (separate head, loss
reweighting, reinitialization, EMA), converging on nearly the same
failure point each time. `docs/EXTERNAL_DATA_PLAN.md`'s trigger
condition -- try the architecture-level and training-level fixes first,
then treat data volume as the remaining explanation -- is met with more
evidence behind it now than when S12.28 first said so. The honest
recommendation is to treat this as this project's real, documented
finding rather than keep cycling through further loss/architecture
variants: report it as a genuine, characterized limitation (with the
full diagnostic chain S12.17-S12.33 as evidence of how thoroughly it
was investigated), and treat more/different flooded-labeled training
data as the next real lever, not another same-day training-loop tweak.

### 12.34 A formal benchmark, and two more in-hand-data levers before concluding S12.33

S12.33 recommended external data as the next step. Before committing to
that larger effort, two more genuinely different, well-motivated
interventions were implemented and tested -- both still using only the
801 tiles already in hand, per `docs/EXTERNAL_DATA_PLAN.md`'s own
trigger condition (external data only once in-hand-data levers are
actually exhausted, not just architecture/loss ones).

**Formal benchmark established**: `evaluate.py --held-out-only` (a
flag that already existed but hadn't been run as a standalone,
independently-verified check) against `checkpoints/best.pt` (v12
epoch 16) on the SAME 87-tile held-out split train.py uses:

| class | F1 | GT imgs | Pred imgs |
|---|---|---|---|
| background | 0.9755 | 87 | 87 |
| building | 0.5610 | 57 | 67 |
| road | 0.4275 | 80 | 78 |
| flooded | 0.5354 | 20 | 25 |

This exactly matches v12 epoch 16's own training-time logged numbers
(cross-verified, not just assumed consistent) -- confirming the
checkpoint, the dashboard, and this independent evaluation script all
agree on the same real result. This is now the formal baseline any
future run is compared against.

**New lever 1 -- freeze `flood_head` at its own peak**:
`--flood-head-patience K` tracks `val_f1_flooded`'s own best-so-far;
once K consecutive epochs pass without beating it, `flood_head`'s
parameters are frozen (`requires_grad=False`) for the rest of training,
while `structure_head`/backbone/`split_trunk` keep training normally.
Directly targets the S12.17-S12.33 characterized failure (continued
training past the peak is what drags flooded down) rather than trying
to prevent the peak from being reached in the first place. Honest
caveat, stated in the flag's own help text: `flood_head` reads from
the SHARED `split_trunk`, which keeps evolving from the structure
loss's gradients after the freeze -- this does not fully insulate
flooded's read-out from the trunk's continuing drift, so it is a real
experiment, not a guaranteed fix.

**New lever 2 -- copy-paste augmentation**: `--copy-paste-prob P`
pastes a randomly-chosen donor tile's entire flooded-pixel footprint
(pre, post, AND mask together, at the same coordinates) onto the
current tile before training on it, manufacturing more flooded-pixel
exposure per epoch from the SAME 801 tiles (Ghiasi et al. 2021,
"Simple Copy-Paste is a Strong Data Augmentation Method") -- directly
tests whether S12.33's "data-exposure" reading is right (this should
help) or whether it's really "not enough DISTINCT real examples"
(reusing the same 198 flooded tiles' own content can't manufacture
that, so this wouldn't help much). 198 of 801 tiles have flooded
pixels to donate from, confirmed.

**Verified, not just written**: 4 new tests for copy-paste (the
donor's real pixel values land in pre/post, not just a relabeled mask
over unchanged imagery; a no-flood donor is a correct no-op; the
dataset-level wiring actually introduces flooded pixels at
`copy_paste_prob=1.0` and never does at the default 0.0) plus an
end-to-end real-data smoke test combining both new flags with the
full existing stack (augment, oversampling, class weights, separate
head, BCE) -- started clean, no crash, correct config printed. 66
tests passing overall (`python -m pytest tests/ -q`).

v14 (next) combines both new levers with the full proven stack, resumed
from `checkpoints/best.pt` (v12 epoch 16, this section's own formal
benchmark) -- reported once real epochs land.

### 12.35 v14 epochs 17-18: healthy start, too early to attribute anything yet

v14 resumed from `checkpoints/best.pt` (v12 epoch 16) with
`--flood-head-patience 4 --copy-paste-prob 0.3 --flood-bce-weight 1.0
--ema-momentum 0.002` on top of the full proven stack.

| epoch | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| 17 | 0.593 | 0.443 | 0.489 | 24/87 |
| 18 | **0.608** (new project best) | 0.442 | 0.448 | 26/87 |

Building reached a new project best (0.608). flooded F1 (0.489, 0.448)
is in the same general range as prior runs at this point in the
lineage, not clearly higher than v13's own epoch 17-18 (0.518, 0.519) --
too early and too close to call the new levers helping or not helping
yet. No `flood_head FROZEN` message has printed (expected: `--flood-
head-patience 4` needs 4 consecutive epochs without beating the
running-best flooded F1, and only 2 epochs have happened, with no
established decline yet). No crash; copy-paste augmentation ran
without issue across a full real epoch. Continuing to monitor through
the epoch ~20 window that has been decisive in every prior run in this
lineage (v10, v12, v13).

### 12.36 v14 epochs 19-21: flooded F1 stays nonzero through the critical window -- first time in this sequence

| epoch | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| 18 | 0.608 | 0.442 | 0.448 | 26/87 |
| 19 | 0.593 | 0.429 | 0.327 | 26/87 |
| 20 | 0.593 | 0.425 | 0.310 | 28/87 |
| 21 | 0.591 | 0.435 | **0.301** | **27/87** |

**Real, not-yet-seen-before result**: every prior run in this lineage
collapsed to exactly 0.000 flooded F1 / 0 coverage at this point --
v10 at its own epoch 8, v12 at epoch 19, v13 at epoch 20. v14 has not.
Flooded F1 is declining (0.448 -> 0.327 -> 0.310 -> 0.301) but stays
genuinely nonzero, and coverage is stable (26-28 of 87), not
collapsing toward zero the way it did in every earlier run at this
exact point. This is the first real durability signal across the
whole v10-v14 sequence.

**flood_head_patience's math, computed from the CSV** (best flooded F1
was 0.489 at epoch 17; epochs 18-21 are four consecutive epochs
without beating it): `--flood-head-patience 4` should trigger the
freeze right at epoch 21. The printed confirmation message can't be
directly verified yet -- this run was launched without unbuffered
output (missing `-u`), so stdout is sitting in a buffer not yet
flushed to disk; only the CSV (written directly, not through the
buffered stream) is confirmable right now. Epoch 22 onward is the
real test of what the freeze actually does: does flooded F1 stabilize
near its current ~0.30 (the freeze doing its job), or does it keep
declining anyway (confirming the flag's own documented caveat --
`flood_head` reads from the shared `split_trunk`, which keeps
evolving from the structure loss after the freeze, so freezing the
head's own weights may not be enough to insulate it)?

**Not yet attributing this to any one of the four combined levers**
(flood-head-patience, copy-paste, BCE loss, EMA) -- they were tested
together for practical reasons (CPU-only, ~600s/epoch), and this
result cannot yet say which one (or which combination) is responsible.
If the durability holds through several more epochs, an ablation
(testing each lever alone against this same starting checkpoint) would
be the honest next step to find out which one actually matters, rather
than assuming all four are necessary or crediting one without evidence.

### 12.37 v14 epochs 22-23: flooded F1 has genuinely flattened, not just delayed its decline

| epoch | building F1 | road F1 | flooded F1 | flooded coverage |
|---|---|---|---|---|
| 20 | 0.593 | 0.425 | 0.310 | 28/87 |
| 21 | 0.591 | 0.435 | 0.301 | 27/87 |
| 22 | 0.600 | 0.439 | 0.307 | 30/87 |
| 23 | **0.607** | 0.440 | 0.302 | 30/87 |

**A pattern not seen anywhere else in this project**: flooded F1
declined for 4 straight epochs (0.489 -> 0.448 -> 0.327 -> 0.310 ->
0.301) then FLATTENED -- 3 epochs now oscillating narrowly around
0.30-0.31 (0.301, 0.307, 0.302), with coverage actually recovering
slightly (27 -> 30 -> 30). This is a real plateau, not a slower version
of the same slide to exactly 0.000 every prior run in this lineage
(v10, v12, v13) showed. Building also keeps improving (0.607, near
its own project best) alongside the flattened flooded curve.

This shape -- decline, then hold -- is exactly what a working
`flood_head_patience` freeze would produce: training stops pulling
flood_head's own weights further down once patience is exhausted
(computed to trigger at epoch 21, S12.36), so the read-out itself
stops degrading even though `split_trunk` (shared, still training on
the structure loss) keeps slowly moving underneath it -- consistent
with a plateau rather than a full recovery. Still cannot directly
confirm the "flood_head FROZEN" print message landed (this run's
stdout remains unflushed on disk as of this check), but the CSV's own
shape is strong indirect evidence consistent with the freeze having
triggered and being effective.

**Appropriately cautious framing**: 3 flat epochs is meaningful, real
progress -- the first stabilization in this entire project -- but not
yet long enough to call this durably solved. Continuing to monitor for
whether the plateau holds for many more epochs (the real target,
since `--epochs 150` leaves a great deal of training still ahead) or
eventually resumes declining despite the freeze.

### 12.38 Correction, epochs 24-26: not a plateau -- a slower, continued decline

| epoch | flooded F1 | flooded coverage |
|---|---|---|
| 23 | 0.302 | 30/87 |
| 24 | 0.302 | 26/87 |
| 25 | 0.289 | 26/87 |
| 26 | 0.271 | 22/87 |

S12.37's "plateau" read was premature -- flooded F1 is still declining,
just much more slowly than before the freeze (roughly -0.01/epoch over
epochs 23-26, versus roughly -0.047/epoch during the epoch 17-21
slide, a real ~5x reduction in decline rate, not zero). Coverage is
drifting down too (30 -> 26 -> 26 -> 22). Correcting the record rather
than letting the more optimistic framing stand.

**What is still real and different from every prior run**: by epoch
26, flooded F1 is 0.271 -- v10/v12/v13 were all already at exactly
0.000 well before their own equivalent points. Whatever the freeze
(and/or the other three combined levers) is doing, it has clearly
slowed the decline dramatically even if it hasn't stopped it. Building
keeps improving throughout (0.610 at epoch 26, a new high) with no
sign of the frozen flood pathway hurting the rest of the model.

**Honest open question**: does this slow decline eventually reach
0 anyway (a delayed version of the same collapse, just gentler), or
does it continue decelerating toward a genuine floor above zero? Both
are plausible from six epochs of data. Continuing to monitor rather
than calling either outcome yet.

### 12.39 Epochs 27-28: deltas shrinking and turning noisy -- leaning toward a real floor, not yet conclusive

| epoch | flooded F1 | epoch-over-epoch delta | coverage |
|---|---|---|---|
| 25 | 0.289 | -0.013 | 26/87 |
| 26 | 0.271 | -0.017 | 22/87 |
| 27 | 0.273 | **+0.002** (upward) | 22/87 |
| 28 | 0.266 | -0.007 | 21/87 |

8 epochs past the peak (0.489, epoch 17), flooded F1 is 0.266 -- still
meaningfully nonzero. The epoch-over-epoch deltas have shrunk to
roughly +-0.01-0.02 and turned NOISY (epoch 27 actually rose slightly)
rather than continuing the earlier steady monotonic decline. This
pattern -- small, noisy fluctuation rather than a steady approach to
zero -- is more consistent with the decline decelerating toward a real
floor than with a delayed version of the same collapse. Not
conclusive from 8 epochs, and this project's own history (S12.21's
premature optimism, corrected in S12.22; S12.37's premature "plateau"
read, corrected in S12.38) is a direct reason not to over-claim this
yet either. Building reached another new high (0.612). Continuing to
monitor for a longer, more convincing stretch before calling this
durable.

## 13. Bottlenecks, honestly, and how to actually overcome each one

Four real bottlenecks were hit while building this, in this environment
(Windows, CPU-only, ~16GB RAM, shared with a browser and other apps). Each
one below is what was actually observed, not a generic list.

| Bottleneck | What was actually observed | How it's addressed here | What still needs a bigger machine |
|---|---|---|---|
| **Compute (no GPU)** | A single forward+backward pass at 256px/batch 8 took ~4.5s on CPU; a full epoch over ~180 real tiles took 90–100s | Batch size and image size are kept modest by default so training stays CPU-tractable at all | `notebooks/train_on_colab.ipynb` gets a free GPU and can run much larger batches/epochs in the same wall-clock time |
| **Memory** | Training at `--batch-size 8, --image-size 256` got the process **killed by the OS** for memory, with zero Python traceback — see `docs/INSTALL.md` §7 | `--batch-size 2` is now the real-data default; INSTALL.md documents the batch-size-vs-free-RAM tradeoff as measured, not guessed | A dedicated machine or Colab (with more headroom, or a GPU where activations live in VRAM, not system RAM) can safely use larger batches |
| **Real data volume** | One AOI (Germany, 202 tiles) is not enough — 20 biased tiles overfit badly (§12); the full 202-tile Germany AOI alone is still one geography, and flooded F1 stayed at 0.000 for 90 epochs on it | `prepare_real_data.py` pulls from multiple real AOIs (`--aoi Germany_Training_Public,Louisiana-East_Training_Public` or `--aoi all`), namespaced so tile ids never collide, appendable across runs (`--append`) so an interrupted pull doesn't lose progress. **All 801 real, publicly-labeled tiles across both AOIs are now downloaded** (Germany 202 + Louisiana-East 599, the entire labeled public SN-8 dataset — Louisiana-West is imagery-only, SN-8's own blind test set) — see §12.2 | This is the ceiling of what's publicly labeled; a genuinely large benchmark result beyond this needs either unpublished held-out labels or a different, larger dataset entirely |
| **S3 listing/scanning speed** | Scanning every AOI's annotations for flood content is 200–600 sequential `s3.get_object` calls, one tile at a time — the slow, silent part of `prepare_real_data.py` (no per-file progress printed during this phase, which looked like a hang the first time it happened) | Known and documented here, not hidden | A real fix (not yet done): parallelize the scan with a thread pool (the original draft repo's `download_data.py` already did this for its own download step) — worth doing before pulling `--aoi all` at real scale |

The honest summary: every bottleneck above has either already been worked
around in this repo, or has a named, concrete next step (Colab, more AOIs,
parallelizing the scan) rather than a vague "needs more resources."

## 14. Known, deliberate limitations of this prototype

- **No pretrained backbone.** `GeoFormerConfig`'s ~11M-parameter encoder is
  randomly initialized, not MaxViT-Base/ImageNet-21k. Swapping in a `timm`
  backbone is a `SiameseMaxViTEncoder` change, not a redesign — deliberately
  deferred past proposal stage.
- **No real SpaceNet-8 data yet.** `SpaceNet8Dataset` is implemented and
  covered by a "fails loudly, not silently" test, but has not been run
  against real data in this environment.
- **`bridge_road_gaps` is O(endpoints²) per image** — fine for the sparse
  endpoint counts a single tile produces, not written for batched/GPU
  execution. It runs on CPU, post-inference, per image.
