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

`road` is genuinely, steadily improving with more training (0.072 → 0.159 →
0.203) — real learning, not a metric artifact (it has real, non-zero
prediction coverage throughout, unlike building/flooded). It is still well
below the baseline's 0.431 at this point; whether GeoFormer would
eventually close or exceed that gap with more epochs than this environment
allowed to run in one sitting is genuinely unknown, not implied either way.

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
