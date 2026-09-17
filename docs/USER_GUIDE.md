# User Guide — Usage, Data, and FAQ

A practical, task-oriented guide to actually using this project: running
it, finding/reading data, training, evaluating, and the dashboard. For
setup instructions (installing dependencies, virtual environments), see
`docs/INSTALL.md`. For the detailed technical log of every experiment and
bug fix, see `docs/MANUAL.md`. For the readable summary of what's been
tried and why, see `docs/PROJECT_HISTORY.md`.

## 1. The five-minute orientation

This project predicts, from a pair of before/after satellite images, one
of four things per pixel: `background`, `building`, `road`, or `flooded`.
The main pieces:

| File/folder | What it's for |
|---|---|
| `model.py` | The neural network architecture itself |
| `train.py` | Trains the model |
| `evaluate.py` | Scores a trained checkpoint against held-out data |
| `demo.py` / `real_image_demo.py` | Run the model on one image pair and see the output |
| `prepare_real_data.py` | Downloads and prepares real training data |
| `dashboard/` | A web UI for training history and submitting new samples |
| `serve.py` | A deployable HTTP API for running predictions |
| `checkpoints/` | Where trained model weights are saved (not in git — too large) |

## 2. Where the data comes from, and how to get more

**Where it comes from**: the public SpaceNet-8 dataset, hosted on a
public Amazon S3 bucket (`spacenet-dataset`) that needs **no AWS account
or credentials** — it's openly readable. The dataset covers two
real-world flood events with published, ground-truth labels: a 2021
flood in Germany, and Hurricane Ida's effects in Louisiana.

**How to download it**:

```bash
# One region:
python prepare_real_data.py --aoi Germany_Training_Public --out-dir real_sn8_dataset

# Everything with public labels (recommended — 801 tiles total):
python prepare_real_data.py --aoi all --n-tiles 599 --out-dir real_sn8_dataset_full
```

This writes a folder containing `pre/`, `post/`, and `mask/`
subdirectories (the before-image, after-image, and ground-truth label for
each tile) plus an `index.json` that lists every tile and some metadata
about it (which classes it contains, how many pixels of each).

**Is there more data available anywhere else?** No additional SpaceNet-8
tiles exist beyond what's already downloaded here — the full public
portion (801 tiles) is already included. A third region (Louisiana-West)
exists in the same S3 bucket but was deliberately never given public
labels by SpaceNet — it's their own blind competition test set. Genuinely
different datasets exist (e.g. xView2/xBD for building damage, which is
what some competition winners fine-tuned from) but they use a different
label format and would need real integration work, not just a download.

**How to read the data yourself** (e.g. to sanity-check a tile):

```python
from dataset import SpaceNet8Dataset
ds = SpaceNet8Dataset("real_sn8_dataset_full", image_size=256)
pre, post, mask = ds[0]          # first tile: pre-image, post-image, label mask
print(pre.shape, post.shape, mask.shape)
print(mask.unique())             # which classes (0-3) are present in this tile
```

`mask` is a single-channel image where each pixel's value is a class
index: `0` = background, `1` = building, `2` = road, `3` = flooded
(either a flooded building or flooded road — the two aren't
distinguished in this format).

## 3. How to train

The simplest possible run (no download needed — uses synthetic,
procedurally-generated data to prove the pipeline itself works):

```bash
python train.py --epochs 10
```

A real run on real downloaded data, using everything this project has
learned so far about what actually helps:

```bash
python train.py \
  --data-dir real_sn8_dataset_full \
  --image-size 256 --batch-size 1 --epochs 150 \
  --lr 5e-4 --tversky-alpha 0.15 --tversky-beta 0.85 \
  --oversample-rare-classes --class-weights "1,2,2,4" \
  --pretrained-backbone efficientnet_b0 --freeze-backbone-epochs 3 \
  --num-workers 4 --checkpoint-metric min_f1 --checkpoint-every 10
```

**What each flag actually does**, in plain terms:

| Flag | What it does |
|---|---|
| `--data-dir` | Points at a real, downloaded dataset. Omit to use synthetic data instead. |
| `--batch-size` | How many image pairs to process at once. Keep this low (1-2) on a normal computer without a lot of free memory. |
| `--tversky-alpha` / `--tversky-beta` | Controls whether the training loss penalizes false alarms (`alpha`) or missed detections (`beta`) more. Higher `beta` = "don't miss anything," at the cost of more false positives. |
| `--oversample-rare-classes` | Shows the model tiles containing rare classes (building/flooded) more often per training epoch than their natural frequency. |
| `--class-weights` | Makes the training loss itself care more about specific classes — four comma-separated numbers for background,building,road,flooded. |
| `--pretrained-backbone` | Uses an ImageNet-pretrained feature extractor instead of starting from random weights. Almost always worth using if you have internet access on first run. |
| `--freeze-backbone-epochs` | Keeps the pretrained backbone's weights fixed for the first N epochs — faster, and gives the rest of the network time to adapt first. |
| `--num-workers` | How many background processes load/decode images in parallel. Set this close to (but below) your CPU's core count. |
| `--checkpoint-metric` | What "best" means when deciding which checkpoint to keep. `min_f1` is the safest choice — see the FAQ below. |
| `--resume` | Continues training from an existing checkpoint file instead of starting over. |

Training writes progress to a CSV log (every epoch, so nothing is lost if
training is interrupted) and saves model checkpoints to
`--checkpoint-dir` (default `checkpoints/`):

- `last.pt` — always the most recent epoch, overwritten every time.
- `best.pt` — the best epoch seen so far, by whatever `--checkpoint-metric` you chose.
- `epoch_N.pt` — a permanent snapshot every `--checkpoint-every` epochs, never overwritten.

## 4. How to check whether it's actually working

Don't just look at the F1 number. Run:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data-dir real_sn8_dataset_full --held-out-only
```

and read the **"Pred imgs"** column, not just the F1 column. If a class
shows `0` predicted-in images despite a large ground-truth count, that
class has completely collapsed — the model isn't detecting it anywhere,
no matter what its F1 score says. This is the single most important habit
for working with this project honestly.

## 5. Running the model on one image (demo)

```bash
python demo.py --checkpoint checkpoints/best.pt          # synthetic example
python real_image_demo.py --checkpoint checkpoints/best.pt  # a real photo
```

Both save an annotated figure showing the prediction. `demo.py --skip-bridging`
shows the raw road prediction without Phase 4's gap-bridging step, useful
for seeing what that step actually adds.

## 6. The dashboard

A local web app showing every training run's history and letting you
submit new image pairs for prediction.

```bash
cp .env.example .env   # fill in a Postgres connection string
python db/init_db.py   # create the database tables (once)
python db/migrate_csv_logs.py  # optional: import existing training logs
uvicorn dashboard.dashboard_server:app --port 8080
```

Then open `http://localhost:8080`. To actually run predictions on a
submitted sample: `python dashboard/process_samples.py`.

## 7. Serving predictions over HTTP

```bash
uvicorn serve:app --port 8000
# or: docker build -t geoformer . && docker run -p 8000:8000 geoformer
```

`POST /predict` with `pre` and `post` image files returns a prediction.
See `docs/DEPLOYMENT.md` for cloud deployment options.

## 8. FAQ

**Q: My training run's F1 numbers look great — is the model actually
working?**
A: Check the coverage columns (or `evaluate.py`'s "Pred imgs" column)
before believing it. A model that predicts nothing for a rare class can
still show a deceptively reasonable *aggregate* F1 if you're not
checking per-image coverage — this exact mistake happened earlier in
this project and is documented in detail in `docs/MANUAL.md`.

**Q: Why does `best.pt` sometimes seem to get *worse* over time?**
A: If you're using the default `--checkpoint-metric val_loss`, this can
genuinely happen — a checkpoint that has stopped detecting a rare class
entirely can have a *lower* loss than one detecting it imperfectly but
for real. Use `--checkpoint-metric min_f1` to avoid this.

**Q: Training seems to have stopped making progress / crashed. Did I
lose everything?**
A: No. The CSV log and `last.pt` are both written after every single
epoch, and a permanent snapshot is saved every `--checkpoint-every`
epochs. Resume with `--resume checkpoints/last.pt`.

**Q: I resumed training and the learning rate looks wrong / training
seems frozen.**
A: This was a real bug that's since been fixed — `--resume` now
explicitly resets the learning rate to whatever `--lr` you pass, rather
than silently inheriting the previous run's fully-annealed (near-zero)
rate.

**Q: How much data / how many epochs do I actually need?**
A: More than you might think, and it depends heavily on which classes
you care about. `road` (present in almost every tile) learns
meaningfully within the first 50-100 epochs on the full dataset.
`building`/`flooded` (much rarer) are considerably harder — see
`docs/PROJECT_HISTORY.md` for the full account of what has and hasn't
helped so far.

**Q: Can I train on my own images instead of SpaceNet-8?**
A: Yes, if you can produce the same directory layout `SpaceNet8Dataset`
expects (`pre/`, `post/`, `mask/` folders plus an `index.json` listing
each tile) with masks using the same 0-3 class-index scheme.

**Q: What's the difference between the "baseline" model and
"GeoFormer"?**
A: The baseline (`--model baseline`) is a standard U-Net/ResNet-34 with
no bi-temporal fusion — it's the comparison point the thesis measures
GeoFormer against, not the proposed architecture itself.

**Q: My computer doesn't have a GPU. Can I still train this?**
A: Yes — everything here runs on CPU by default, with settings (small
batch size, modest image size) chosen to make that practical. It will be
considerably slower than a GPU. `notebooks/train_on_colab.ipynb` sets up
the same training run on a free Google Colab GPU if you want more speed.

**Q: Where do I report a bug or check what's already been found?**
A: `docs/MANUAL.md` is the running, honest log of every bug found and
fixed in this project, in the order they were found — check there first.
