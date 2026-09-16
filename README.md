# Dual-Axis GeoFormer — architecture prototype

Real, runnable PyTorch code for the architecture proposed in the thesis
*"Dual-Axis GeoFormer: Exploiting Global-Local Dependencies for Multiclass
Flood Segmentation of Narrow Infrastructure via MaxViT."* Includes a
completed training run, a real-SpaceNet-8-data experiment (downloaded
directly from the public S3 bucket — no credentials needed), a deployable
inference API, and a real-photograph robustness check.

**Full documentation lives in `docs/MANUAL.md`** (usage, CLI reference,
data format, real-data findings, troubleshooting) and `docs/DEPLOYMENT.md`
(running it anywhere/in the cloud). This file is a quick orientation only.
Sample outputs (figures, training curves, logs) referenced below live in
`sample_outputs/`; checkpoints and downloaded data are `.gitignore`d
(regenerate with the commands in Quickstart) since a trained checkpoint is
~126MB, over GitHub's plain-file limit.

## What's real vs. simplified — say this when you present

| | This prototype | Full thesis system |
|---|---|---|
| Block + grid attention mechanism | **Real**, implemented from the MaxViT paper's design | Same |
| Encoder size | ~11M params | MaxViT-Base, ImageNet-21k pretrained (~120M params, via `timm`) |
| Training (synthetic) | **Real** — 30-epoch run, loss 0.57→0.01, val F1 up to ~0.98–1.00 | — |
| Training (real SpaceNet-8) | **Real data, honest negative result** — see below | Full benchmark training |
| Diff module, U-decoder, Geo-Head | **Real**, runs end to end | Same design |
| Phase 4 (skeleton + attention-guided bridging) | **Real** — genuinely reads the model's own grid-attention map | Same |
| Deployment | **Real** — FastAPI + Docker, runs on any cloud | Same, pointed at a real-data checkpoint |
| Real-photo test | **Real photo run** (`real_image_demo.py`) — robustness check, not accuracy evidence | Trained + evaluated on real imagery |

## The real-data experiment, honestly

`prepare_real_data.py` pulls real, labeled SpaceNet-8 tiles (2021 Germany
flood AOI) straight from the public `spacenet-dataset` S3 bucket and
rasterizes the real GeoJSON building/road/`flooded` labels into training
masks — no GDAL/rasterio needed, just the GeoTIFF's own georeferencing tags.
Fine-tuning on a 20-tile sample of this real data produced a genuine
learning curve (loss 0.76→0.56) but **validation building/road F1 collapsed
— 20 tiles, biased toward flood-heavy ones, is not enough data.** This is
reported as a finding, not hidden: see `docs/MANUAL.md` §12 for the full
numbers and reasoning, and `sample_outputs/training_curve_real.png` /
`sample_outputs/real_mask_sanity_check.png` for the evidence. The
checkpoint-selection logic correctly protected against this — the shipped
"best" checkpoint is still the better synthetic-converged one.

The honest one-line summary for your defense: *"the architecture, the real
training loop, and the real-SpaceNet-8 data pipeline are all working
end-to-end — what's missing is enough real data and compute to actually
converge on it, which is exactly the thesis's own next milestone."*

## Files

- `model.py` — `DualAxisGeoFormer`: Siamese MaxViT-style encoder, `DiffModule`, U-decoder, `Geo-Head`.
- `losses.py` — `TverskyLoss`.
- `postprocess.py` — `bridge_road_gaps` (Phase 4).
- `dataset.py` — `SyntheticFloodDataset` (built-in) and `SpaceNet8Dataset` (real data).
- `prepare_real_data.py` — downloads + rasterizes a real SpaceNet-8 sample from the public S3 bucket.
- `train.py` / `evaluate.py` / `plot_training_curve.py` — training, evaluation, and reporting.
- `demo.py` — synthetic end-to-end demo, loads `checkpoints/best.pt` by default.
- `real_image_demo.py` — runs the pipeline on a real public-domain flood photo.
- `serve.py` / `Dockerfile` — deployable inference API.
- `tests/` — pytest suite.
- `sample_outputs/` — figures, training curves, and logs from the runs described above.

## Quickstart

```bash
pip install -r requirements.txt

# Synthetic pipeline-validation training (reproduces sample_outputs/training_curve.png)
python train.py --epochs 30 --batch-size 4 --image-size 128 \
                 --synthetic-train-size 120 --synthetic-val-size 24 --lr 1e-3

# Real SpaceNet-8 data (no AWS account needed — public bucket)
python prepare_real_data.py --n-tiles 24 --out-dir real_sn8_dataset
python train.py --data-dir real_sn8_dataset --resume checkpoints/last.pt \
                 --epochs 50 --image-size 256 --lr 2e-4

python demo.py                    # trained-model demo, synthetic tile
python real_image_demo.py         # trained model on a real photo
uvicorn serve:app --port 8000     # run the inference API locally
```

See `docs/MANUAL.md` for everything else, including how to scale
`prepare_real_data.py` past this repo's 20-tile proof-of-concept sample.
