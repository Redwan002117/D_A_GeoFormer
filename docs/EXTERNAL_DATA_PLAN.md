# External Data Integration — Detailed Plan (Not Yet Implemented)

This is a plan, not a change log entry — nothing described here has been
built yet. It exists so the decision of *whether* and *when* to build it
is made deliberately, against real trigger conditions, not on a whim
mid-session. See the bottom section for exactly when to revisit this.

## 1. Why consider this at all

The project's central, unresolved problem (`docs/MANUAL.md` §12.3
onward) is that `building` and `flooded` detection collapses under
sustained training, while `road` doesn't. Every fix tried so far
(oversampling, class-weighted loss, a pretrained backbone, focal loss,
geometric augmentation) has operated on the **801 real tiles already in
hand** — reweighting or reshaping how the model sees the same fixed
pool of labeled examples. None of them add genuinely new *information*.
External data is the one lever in this project's whole history that
would.

## 2. Candidate sources, evaluated honestly

### 2a. Microsoft Global ML Building Footprints

- **What it is**: ~1.4 billion building footprint polygons, machine-generated
  from Bing Maps satellite imagery via a Microsoft-trained segmentation
  model, released as open data (ODbL license) on Azure Blob Storage,
  organized by country and quadkey tile.
- **Access**: No API key or registration needed. Files are GeoJSONL,
  downloadable directly via HTTPS from a public Azure container
  (`https://minedbuildings.z5.web.core.windows.net/...` — the exact
  index changes periodically; the current dataset index/links page is at
  `github.com/microsoft/GlobalMLBuildingFootprints`).
- **What it gives us**: A huge volume of building *footprints only* — no
  flood labels, no road labels, no pre/post-event pairing. Not usable
  for end-to-end training on this task directly.
- **How it could actually help**: as **pretraining data for the building
  class specifically**, before fine-tuning on the real 801-tile
  SpaceNet-8 set. The idea: pull building footprints for the same or
  similar geographies (Germany, Louisiana), rasterize them the same way
  `prepare_real_data.py` already rasterizes SpaceNet-8's own GeoJSON
  buildings, and run a building-only (or building+background) 2-class
  pretraining pass on that data before the real 4-class fine-tune. This
  targets the exact class (`building`) that's one of the two collapsing.
- **Caveat, stated plainly**: these footprints have no corresponding
  satellite *imagery* bundled with them — we'd need imagery for the same
  tiles from somewhere (Bing Maps' own imagery has licensing restrictions
  on redistribution/ML use that need checking before relying on it; a
  cleaner path is to only use this for AOIs where we can also legally
  source matching imagery, or restrict this to areas already covered by
  SpaceNet-8's own Maxar imagery, where we'd only be adding *extra
  labels* on top of imagery we already have — see §3).

### 2b. OpenStreetMap Overpass API

- **What it is**: Live queryable access to OSM's building/road vector data.
- **Access**: Free, no key, rate-limited (fair-use — heavy queries need a
  self-hosted Overpass instance or a paid tier for production volume).
- **Assessment**: SpaceNet-8's own labels are already partially
  OSM-derived. Pulling fresh OSM data for the *same* AOIs would mostly
  re-derive labels we already effectively have, with the risk of
  introducing label drift (OSM data changes over time; matching it back
  to SpaceNet-8's original imagery timestamp is nontrivial). Lower
  priority than 2a — not recommended as a first move.

### 2c. Copernicus Emergency Management Service (EMS Rapid Mapping)

- **What it is**: Real flood-extent polygons published by the EU's
  Copernicus program for actual disaster response events worldwide —
  the most directly relevant *new flood-specific* ground truth available
  anywhere, since it's the same kind of label SpaceNet-8 itself provides
  (a real flood extent, not a proxy).
- **Access**: Free, registration required
  (`emergency.copernicus.eu/mapping`), data delivered per-activation as
  shapefiles/GeoJSON.
- **What it gives us**: flood extent polygons for many more events than
  SpaceNet-8's two (Germany 2021, Louisiana/Hurricane Ida). Directly
  targets `flooded`, the other collapsing class.
- **Caveat, stated plainly**: the accompanying satellite imagery for each
  EMS activation is **not** bundled with the flood polygons and comes
  from varying sensors/resolutions (often Sentinel-1/2, sometimes
  commercial), not Maxar. Using this data would mean either (a) sourcing
  matching imagery separately per event (real, nontrivial work, and
  imagery availability/licensing varies per activation) or (b) treating
  it as a second, separate dataset the model needs to generalize across
  different imagery characteristics for — a real domain-adaptation
  problem, not a data-volume problem alone.

## 3. Recommended approach, if this is built

**Phase 1 (smaller, lower-risk): Microsoft Building Footprints as
building-only pretraining, restricted to areas SpaceNet-8 already has
imagery for.**

This sidesteps the imagery-licensing question entirely: don't pull new
imagery, just pull MORE/denser building labels for the same 801 tiles'
geographic footprint (Microsoft's building detector may have identified
buildings SpaceNet-8's own GeoJSON missed, or vice versa — either way,
richer labels on imagery we already legitimately have). Concretely:

1. Download the Germany and Louisiana quadkey tiles' building footprint
   GeoJSONL from the Microsoft dataset.
2. Reproject/clip to the same tile boundaries `real_sn8_dataset_full`
   already uses (the georeferencing code in `prepare_real_data.py`
   already does this kind of clipping for SpaceNet-8's own labels —
   reusable, not a rewrite).
3. Compare against SpaceNet-8's existing building masks: where do they
   agree, where do they differ? This alone is informative (data-quality
   insight into the existing labels) before any retraining happens.
4. If it adds meaningfully more building-labeled pixels: rasterize the
   union (or a confidence-weighted combination) into an *additional*
   mask channel or a merged mask, and re-run the existing training
   pipeline against the enriched labels — no architecture change needed,
   this is a data-quality lever, not a model-change lever.

**Phase 2 (larger, higher-risk, not recommended yet): Copernicus EMS for
new flood events**, only if Phase 1 doesn't move the `building` needle
and the `flooded` collapse is still unresolved after this project's other
levers (a separate flood head — `docs/MANUAL.md` §12.14 item 1 — is
still the more direct fix for `flooded` specifically, and should likely
be tried before reaching for an entirely new, differently-sourced
dataset).

## 4. Effort estimate, honestly

- Phase 1: a few hours of real work — a new download/clip script similar
  in shape to `prepare_real_data.py`, plus a comparison/merge step. Low
  architectural risk (no model or training-loop changes required).
- Phase 2: substantially larger — new imagery sourcing per event, likely
  new preprocessing for a different sensor's radiometry, and probably a
  fine-tune-then-adapt strategy rather than a straight merge. Not
  scoped in detail here because it depends on Phase 1's outcome and
  whether the separate-flood-head architecture change happens first.

## 5. When to actually implement this — trigger conditions

Not now, and not automatically after any fixed number of epochs. Revisit
this plan when **one of these becomes true**:

1. **v9 (augmentation) and any immediate follow-up loss/sampling tuning
   both still collapse `building`/`flooded` by a stable pattern** (i.e.
   the in-hand-data levers are exhausted, not just "the first attempt
   didn't instantly fix it").
2. **The separate-flood-head architecture change (`docs/MANUAL.md`
   §12.14 item 1) has been tried and building/flooded collapse persists
   even with a decoupled output.** That result would specifically argue
   for "the model needs more/different data," not "the architecture is
   forcing a bad tradeoff" — the two explanations point at different
   fixes, and this plan is the right response to the data explanation,
   not the architecture one.
3. **Someone explicitly asks for it** — this document exists so that
   request can be acted on quickly with a real plan already in hand,
   not started from zero.

Until one of those is true, the honest, evidence-based use of further
compute is continuing to work the in-hand-data levers (this is what v9 is
doing right now) and the architecture-level fix, not reaching for a new
dataset as a first resort.
