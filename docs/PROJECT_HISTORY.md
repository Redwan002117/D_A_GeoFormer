# Project History — What Was Built, and Every Approach Tried to Fine-Tune It

This is the narrative version of the project's development. For the full,
chronological, evidence-by-evidence technical log (every table, every
exact number, every regression test), see `docs/MANUAL.md` — this
document is the readable summary of *why* things changed, in the order
they changed, without the raw log format.

## 1. What this project is

Dual-Axis GeoFormer is a real, runnable implementation of a proposed
thesis architecture for multiclass flood segmentation: given a pair of
satellite images (before and after a flood event), predict, per pixel,
one of four classes — `background`, `building`, `road`, or `flooded`.

The architecture:

- **Siamese MaxViT-style encoder** — the same encoder (shared weights)
  processes the pre-event and post-event image separately. Each stage
  combines a convolutional block (MBConv), local windowed self-attention
  (Block Attention, for sharp edges), and strided global attention (Grid
  Attention, for full-image context in one layer).
- **Difference Module** — combines the pre/post feature maps
  (`|pre - post|`, fused with the post-event features) at every stage, so
  the network reasons about *change*, not just static content.
- **U-shaped decoder** — upsamples back to full resolution with skip
  connections from each encoder stage.
- **Geo-Head** — a final convolution producing 4-channel per-pixel class
  logits.
- **Phase 4 post-processing** — bridges small gaps in the predicted road
  network, using the model's own grid-attention map (not a generic
  image-processing heuristic) to decide which gaps are worth bridging.

A from-scratch **baseline** (`baseline.py`, a ResNet-34 + U-Net with no
bi-temporal fusion) exists purely as the comparison point the thesis's
own tables call for.

## 2. The real data pipeline

`prepare_real_data.py` downloads real, labeled SpaceNet-8 tiles directly
from the public `spacenet-dataset` S3 bucket (no AWS credentials needed —
unsigned access) and rasterizes the accompanying GeoJSON building/road/
flood labels into training masks, using the GeoTIFF's own georeferencing
tags (no GDAL/rasterio dependency). The complete publicly-labeled portion
of SpaceNet-8 — 801 tiles across two AOIs (Germany, 202 tiles; Louisiana-
East, 599 tiles) — is downloaded and used. A third AOI (Louisiana-West)
exists in the public bucket but ships imagery only, with no public
labels — it's SpaceNet-8's own blind competition test set.

## 3. Early bug-fixing passes

Before any real-data training happened, and repeatedly afterward as new
issues surfaced, this project went through several rounds of "find and
fix real bugs," verified by actually running the code and inspecting
real output — not by inspection alone. The headline fixes:

- **Per-batch F1 averaging bug**: an early metric implementation averaged
  F1 scores across validation batches, which let a class the model never
  predicted anywhere score deceptively well whenever it happened to be
  absent from a batch's ground truth too. Replaced with a proper
  corpus-level accumulator (`ConfusionAccumulator`) that sums raw
  true/false positive/negative counts across an entire validation pass
  before computing F1 once — plus a coverage report (how many images a
  class is ever predicted in at all), which is what actually catches
  total class collapse.
- **Train/val split instability**: the original split (`random_split`)
  silently drew a different random partition every time the dataset grew
  across sessions, so "held-out validation" wasn't comparing against a
  stable set. Fixed by hashing each tile's own ID to decide its side,
  independent of dataset size or order.
- **Checkpoint provenance**: checkpoints didn't record what they were
  actually trained on, so downstream scripts guessed (and were
  sometimes wrong). Every checkpoint now records its model type, config,
  and data source.
- **Incremental saves everywhere**: the training log CSV, the dataset
  index, and model checkpoints are all now written continuously (every
  epoch/tile/step) rather than only at the end — this environment's
  repeated out-of-memory kills would otherwise have silently destroyed
  hours of progress.

## 4. The honest, central finding: building/flooded detection collapses

Once real training was underway, the same result kept reappearing no
matter how far training ran: the model achieves genuine, useful accuracy
on `road`, but predicts **zero** `building` or `flooded` pixels anywhere
across the entire real dataset. This was caught by checking *prediction
coverage* (how many images a class is ever predicted in at all), not
just an aggregate F1 number — an aggregate number alone can look
deceptively reasonable even when a class has totally collapsed.

This became the project's central problem, and everything from here is
an attempt to understand and fix it.

## 5. Approaches tried, in order, with what each one actually did

### 5.1 Class-weighted oversampling

**Idea**: if `building`/`flooded`-containing tiles are rare, show them to
the model more often per epoch than their raw prevalence.

**What happened**: this alone delayed the collapse by a few epochs but
did not prevent it — the model still ended up ignoring the rare classes,
just slightly later.

### 5.2 Researching what the actual SpaceNet-8 competition winners did

Rather than keep guessing, this stopped to research how real competitors
handled the identical imbalance problem. Findings:

- Winning solutions used **ImageNet-pretrained backbones**, crediting
  this with "significantly" improving their score.
- Some winning solutions treated flood detection as a **separate model**
  from building/road segmentation, rather than one joint classification
  decision.
- The broader literature on class-imbalanced segmentation consistently
  recommends **explicit per-class loss weighting**, not just a general
  precision/recall tradeoff.

That last point surfaced a real bug: this project's loss function was
averaging its four per-class scores with **equal weight**, giving the
rare `flooded` class the same 25% share as the dominant `background`
class. Fixed by adding an explicit, configurable per-class weight.

### 5.3 Combining oversampling with class-weighted loss

Several rounds of tuning the specific weight values followed, each
producing a *different* partial collapse pattern (one run's `road`
collapsed instead; another's `building` collapsed while `flooded`
briefly recovered). The pattern across every attempt: something always
loses ground, consistent with a single joint 4-way classification head
forcing every foreground class to compete for the same probability mass.

### 5.4 An ImageNet-pretrained backbone

The most consequential single change: the network's own convolutional
feature extractor was replaced with a real ImageNet-pretrained backbone
(fed into the same attention stages, decoder, and output head — not a
redesign). The very first training epoch afterward produced, for the
first time in the project's history, genuine, simultaneous, non-zero
predictions for **all four classes**, including `building` and `flooded`
with real per-image coverage. This proved the representational capacity
was reachable — it did not, by itself, make that state stable across
many further epochs (the same competition/collapse pattern still
eventually reasserted itself).

### 5.5 A critical bug in checkpoint selection

Investigating why the good early result kept disappearing surfaced
another real bug: the logic that decides which checkpoint is "best" was
using the lowest validation loss — and a collapsed checkpoint (predicting
nothing for the rare classes) can have a *lower* loss than a genuinely
useful one, because the loss is dominated by whichever classes have the
most pixels. Concretely, this project's own best early result was
overwritten and lost because of exactly this. Fixed by adding an
alternative selection criterion based on the worst (or average) per-class
F1 score instead of raw loss — a checkpoint can no longer be called
"best" while a class it once detected has since gone to zero.

### 5.6 Backbone freezing (efficiency and stability)

Standard transfer-learning practice: the pretrained backbone's weights
are frozen (no gradient updates) for the first several epochs, giving
the rest of the network time to adapt to the new features before the
backbone itself starts changing. Verified to give a genuine ~1.6x
speedup per training step while frozen (fewer gradients to compute), and
may also reduce the risk of the pretrained features being disrupted by
early, noisy gradients from an untrained decoder.

### 5.7 Parallel data loading

A separate, unrelated efficiency fix: image loading was running strictly
serially in the main training process even though the machine has
multiple CPU cores available. Parallel data-loading workers now overlap
the next batch's image decoding with the current batch's computation.

## 6. Where things stand

Training continues, combining every fix above: an ImageNet-pretrained
backbone, oversampling, class-weighted loss, correct checkpoint
selection, and backbone freezing for early stability/speed. `docs/MANUAL.md`
is updated with each new result as it lands, including the honest
outcome whether or not it's positive.

Two larger architectural changes were identified through this process but
not yet built, named honestly as the likely next step if the current
combination isn't enough:

1. **A bigger pretrained backbone** (the current one is a CPU-tractable
   choice; a larger one needs a GPU).
2. **Decoupling flood detection into its own model or output head**,
   instead of one joint 4-way classification decision — this is closer
   to how the real SpaceNet-8 label data is structured (flooding is an
   attribute of a building or road, not a mutually exclusive category)
   and to what several winning competition solutions actually did.
