"""
Downloads and rasterizes a real, labeled slice of SpaceNet-8 (the public,
UNSIGNED-access `spacenet-dataset` S3 bucket -- no AWS credentials needed)
into the layout `SpaceNet8Dataset` (dataset.py) expects.

This is real data, not synthetic: real pre/post aerial tiles from real
SpaceNet-8 AOIs (Germany, Louisiana-East -- see AVAILABLE_AOIS), with real
OpenStreetMap-derived building/road labels and a real `flooded: "yes"` flag
SpaceNet-8's own annotators set per feature. It is NOT the full SpaceNet-8
benchmark and pulling every tile from every AOI is still a much larger
download than a quick run needs -- see docs/MANUAL.md for what any given
run size does and doesn't prove.

Louisiana-West_Test_Public also exists in the bucket but ships NO public
annotations (it's SN-8's actual blind competition test set) -- it can't be
used here for labeled training or evaluation, only for unlabeled inference.

Georeferencing: SpaceNet-8's GeoTIFFs carry plain EPSG:4326 tiepoint +
pixel-scale tags (no rotation/skew), so a lon/lat -> pixel transform can be
done in pure Python from PIL's raw TIFF tags -- no GDAL/rasterio needed.
Roads are GeoJSON LineStrings, rasterized as a fixed-width line (roughly
matching real road width at this imagery's ~0.3-0.4m/px resolution);
buildings are Polygons, rasterized filled. Pixels covered by ANY feature
with `flooded == "yes"` are written as the "flooded" class regardless of
whether they came from a building or road feature underneath.

Usage:
    python prepare_real_data.py --n-tiles 24 --out-dir real_sn8_dataset
    python prepare_real_data.py --aoi Louisiana-East_Training_Public --n-tiles 100 \
        --out-dir real_sn8_dataset --append   # add fresh, different-geography tiles
    python prepare_real_data.py --aoi all --n-tiles 300 --out-dir real_sn8_dataset_big
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image, ImageDraw
from PIL.TiffTags import TAGS

BUCKET = "spacenet-dataset"
TILE_PX = 256  # resize every tile to this for the model

# All AOIs found under spacenet/SN8_floods/ with real, public labels (the
# bucket also has Louisiana-West_Test_Public, but that one ships imagery
# with NO public annotations -- it's SN-8's actual blind competition test
# set, so it can't be used for labeled evaluation here).
AVAILABLE_AOIS = ["Germany_Training_Public", "Louisiana-East_Training_Public"]


def geo_transform(tif_path: Path):
    """Returns a function (lon, lat) -> (col, row) pixel coordinates, built
    from the plain EPSG:4326 tiepoint + pixel-scale GeoTIFF tags SN-8 uses
    (no rotation term needed for these tiles)."""
    img = Image.open(tif_path)
    tags = img.tag_v2
    tiepoint = tags[33922]  # (I, J, K, X, Y, Z)
    scale = tags[33550]     # (scaleX, scaleY, scaleZ)
    origin_lon, origin_lat = tiepoint[3], tiepoint[4]
    scale_x, scale_y = scale[0], scale[1]
    width, height = img.size

    def to_pixel(lon: float, lat: float) -> tuple[float, float]:
        col = (lon - origin_lon) / scale_x
        row = (origin_lat - lat) / scale_y
        return col, row

    return to_pixel, width, height


def _ring_to_pixels(ring, to_pixel) -> list[tuple[float, float]]:
    """Converts a GeoJSON coordinate ring/line to pixel points.

    BUG THIS FIXES (compatibility, not yet observed in SN-8's own Germany AOI
    but real in general GeoJSON, which allows an optional 3rd/4th element --
    elevation, a measure): `for lon, lat in ring` unpacks each point
    positionally and raises `ValueError: too many values to unpack` the
    moment a point has more than 2 coordinates. Taking only the first two
    elements makes this robust to any GeoJSON, not just the exact shape of
    the tiles this repo has downloaded so far.
    """
    return [to_pixel(pt[0], pt[1]) for pt in ring]


def rasterize_mask(geojson_path: Path, to_pixel, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    flood_mask = Image.new("L", (width, height), 0)
    flood_draw = ImageDraw.Draw(flood_mask)

    with open(geojson_path) as f:
        data = json.load(f)

    def _draw_polygon_with_holes(rings, fill_value):
        """GeoJSON Polygon rings: the first is the exterior boundary, any
        rest are interior holes to be cut OUT of it.

        BUG THIS FIXES: the old code drew every ring with the same fill,
        which fills a hole in solid rather than cutting it out -- rare in
        this specific OSM-derived building data (3 of 17,474 polygons
        checked across the real, downloaded dataset actually have a second
        ring), but real and reproducible, not hypothetical. Draw the
        exterior filled, then punch each interior ring back to background.
        """
        if not rings:
            return
        exterior = _ring_to_pixels(rings[0], to_pixel)
        if len(exterior) < 3:
            return
        draw.polygon(exterior, fill=fill_value)
        if is_flooded:
            flood_draw.polygon(exterior, fill=1)
        for hole in rings[1:]:
            hole_pts = _ring_to_pixels(hole, to_pixel)
            if len(hole_pts) < 3:
                continue
            draw.polygon(hole_pts, fill=0)
            if is_flooded:
                flood_draw.polygon(hole_pts, fill=0)

    for feature in data["features"]:
        props = feature["properties"]
        geom = feature["geometry"]
        is_building = props.get("building") is not None
        is_road = props.get("highway") is not None
        is_flooded = props.get("flooded") == "yes"
        if not is_building and not is_road:
            continue
        class_value = 1 if is_building else 2

        if geom["type"] == "Polygon":
            _draw_polygon_with_holes(geom["coordinates"], class_value)
        elif geom["type"] in ("LineString",):
            pts = _ring_to_pixels(geom["coordinates"], to_pixel)
            if len(pts) < 2:
                continue  # a degenerate line -- nothing to draw
            draw.line(pts, fill=2, width=10)
            if is_flooded:
                flood_draw.line(pts, fill=1, width=10)
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                _draw_polygon_with_holes(poly, class_value)

    mask_arr = np.array(mask)
    flood_arr = np.array(flood_mask)
    mask_arr[flood_arr == 1] = 3  # flooded overrides building(1)/road(2)
    return mask_arr


def _tile_id_from_filename(fname: str) -> str:
    # "<catalogid>_0_15_63.tif" -> "0_15_63"
    return "_".join(fname.split("_")[1:]).rsplit(".", 1)[0]


def _list_all(s3, prefix: str) -> list[dict]:
    """list_objects_v2 truncates at 1000 keys per call -- BUG THIS FIXES:
    the original version of this function called it once, unpaginated.
    Germany's AOI (~400 pre/post files) stayed under that limit by
    coincidence, so nothing looked wrong; Louisiana-East (599 tiles, ~2+
    files each) is large enough to plausibly exceed 1000, which would have
    silently dropped tiles past the first page with no error -- they'd just
    never appear in pre_by_tile/post_by_tile and get quietly skipped.
    get_paginator handles any number of keys correctly."""
    paginator = s3.get_paginator("list_objects_v2")
    results = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        results.extend(page.get("Contents", []))
    return results


def collect_aoi_tiles(s3, aoi: str, n_tiles: int, max_workers: int = 16):
    """Lists and scores one AOI's tiles by flood-feature count. Returns a
    list of (namespaced_tile_id, ann_key, pre_key, post_key, n_flooded,
    n_features) tuples, at most n_tiles long, biased toward including every
    flooded tile before filling the rest with dry ones."""
    base = f"spacenet/SN8_floods/{aoi}/"

    print(f"[{aoi}] Listing annotation tiles...")
    ann_keys = [o["Key"] for o in _list_all(s3, base + "annotations/")]
    if not ann_keys:
        print(f"[{aoi}] No public annotations here (likely a blind test AOI) -- skipping.")
        return []

    print(f"[{aoi}] Listing pre/post imagery...")
    pre_by_tile = {}
    for o in _list_all(s3, base + "PRE-event/"):
        fname = o["Key"].rsplit("/", 1)[-1]
        pre_by_tile[_tile_id_from_filename(fname)] = o["Key"]

    post_by_tile: dict[str, str] = {}
    for o in _list_all(s3, base + "POST-event/"):
        fname = o["Key"].rsplit("/", 1)[-1]
        post_by_tile.setdefault(_tile_id_from_filename(fname), o["Key"])  # first match only

    # BUG THIS FIXES (performance, not correctness): scanning every
    # annotation file's flood content used to be one s3.get_object per file,
    # sequentially -- 599 blocking round trips for Louisiana-East, with NO
    # progress printed during the wait, which looked exactly like a hang the
    # first time this ran. A thread pool overlaps the network wait time
    # across many requests at once (S3 reads, not CPU work, so the GIL isn't
    # a limiter here) and reports progress as it goes.
    print(f"[{aoi}] Scanning {len(ann_keys)} annotation files for flood content "
          f"({max_workers} concurrent requests)...")
    relevant_keys = [
        key for key in ann_keys
        if key.rsplit("/", 1)[-1].replace(".geojson", "") in pre_by_tile
        and key.rsplit("/", 1)[-1].replace(".geojson", "") in post_by_tile
    ]

    def _scan_one(key: str):
        tile_id = key.rsplit("/", 1)[-1].replace(".geojson", "")
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        n_flooded = sum(1 for f in data["features"] if f["properties"].get("flooded") == "yes")
        return tile_id, key, n_flooded, len(data["features"])

    scored = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_one, key) for key in relevant_keys]
        for i, fut in enumerate(as_completed(futures), 1):
            tile_id, key, n_flooded, n_features = fut.result()
            # Namespace by AOI so tile ids from different AOIs (which reuse
            # the same small grid-index naming, e.g. both have a "0_15_63")
            # never collide when combined into one dataset directory.
            namespaced_id = f"{aoi}__{tile_id}"
            scored.append((namespaced_id, key, pre_by_tile[tile_id], post_by_tile[tile_id],
                            n_flooded, n_features))
            if i % 50 == 0 or i == len(relevant_keys):
                print(f"[{aoi}]   scanned {i}/{len(relevant_keys)}")

    scored.sort(key=lambda x: -x[4])
    n_flood_tiles = n_tiles // 2
    flooded = [s for s in scored if s[4] > 0]
    dry = [s for s in scored if s[4] == 0]
    selected = flooded[:n_flood_tiles] + dry[: n_tiles - min(n_flood_tiles, len(flooded))]
    selected = selected[:n_tiles]
    print(f"[{aoi}] Selected {len(selected)} tiles "
          f"({sum(1 for s in selected if s[4] > 0)} with flood labels, "
          f"{sum(1 for s in selected if s[4] == 0)} without).")
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tiles", type=int, default=24, help="Tiles PER AOI, not total.")
    p.add_argument("--out-dir", type=str, default="real_sn8_dataset")
    p.add_argument(
        "--aoi", type=str, default="Germany_Training_Public",
        help=f"Comma-separated AOI name(s), or 'all'. Available: {', '.join(AVAILABLE_AOIS)}",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Add to an existing index.json in --out-dir instead of overwriting it "
             "(skips tiles whose files already exist). Use this to combine AOIs across "
             "separate runs without re-downloading what you already have.",
    )
    args = p.parse_args()

    aois = AVAILABLE_AOIS if args.aoi == "all" else [a.strip() for a in args.aoi.split(",")]
    unknown = [a for a in aois if a not in AVAILABLE_AOIS]
    if unknown:
        raise SystemExit(f"Unknown AOI(s): {unknown}. Available: {AVAILABLE_AOIS}")

    out_dir = Path(args.out_dir)
    (out_dir / "pre").mkdir(parents=True, exist_ok=True)
    (out_dir / "post").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    index = []
    existing_ids = set()
    index_path = out_dir / "index.json"
    if args.append and index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        existing_ids = {e["tile_id"] for e in index}
        print(f"Appending to existing index.json -- {len(index)} tiles already present.")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for aoi in aois:
        for tile_id, ann_key, pre_key, post_key, n_flooded, n_features in collect_aoi_tiles(
            s3, aoi, args.n_tiles
        ):
            if tile_id in existing_ids:
                print(f"  {tile_id}: already in index.json, skipping")
                continue

            pre_local = out_dir / "pre" / f"{tile_id}.tif"
            post_local = out_dir / "post" / f"{tile_id}.tif"
            ann_local = out_dir / f"{tile_id}.geojson"

            # Download this tile's 3 files (pre/post imagery + annotation)
            # concurrently instead of one after another -- each is a
            # separate network round trip with no shared state between them,
            # so there's nothing serial about them except that the old code
            # waited for each before starting the next. ~3x fewer round
            # trips' worth of wall-clock time spent waiting per tile.
            downloads = []
            if not pre_local.exists():
                downloads.append((pre_key, pre_local))
            if not post_local.exists():
                downloads.append((post_key, post_local))
            if not ann_local.exists():
                downloads.append((ann_key, ann_local))
            if downloads:
                with ThreadPoolExecutor(max_workers=len(downloads)) as pool:
                    futures = [pool.submit(s3.download_file, BUCKET, key, str(local))
                               for key, local in downloads]
                    for fut in as_completed(futures):
                        fut.result()  # re-raise any download error here, not silently

            to_pixel, width, height = geo_transform(pre_local)
            mask_arr = rasterize_mask(ann_local, to_pixel, width, height)

            pre_img = Image.open(pre_local).convert("RGB").resize((TILE_PX, TILE_PX))
            post_img = Image.open(post_local).convert("RGB").resize((TILE_PX, TILE_PX))
            mask_img = Image.fromarray(mask_arr.astype(np.uint8)).resize((TILE_PX, TILE_PX), Image.NEAREST)

            pre_png, post_png, mask_png = f"pre/{tile_id}.png", f"post/{tile_id}.png", f"mask/{tile_id}.png"
            pre_img.save(out_dir / pre_png)
            post_img.save(out_dir / post_png)
            mask_img.save(out_dir / mask_png)

            class_counts = {int(c): int((mask_arr == c).sum()) for c in np.unique(mask_arr)}
            index.append({"pre": pre_png, "post": post_png, "mask": mask_png,
                           "tile_id": tile_id, "aoi": aoi, "n_flooded_features": n_flooded,
                           "class_pixel_counts": class_counts})
            existing_ids.add(tile_id)
            print(f"  {tile_id}: {n_flooded}/{n_features} flooded features, "
                  f"mask classes {class_counts}")

            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)  # write after every tile, not just at the end --
                # a long multi-AOI pull that gets interrupted (network, memory, Ctrl+C) keeps
                # everything downloaded so far usable, instead of losing it to an end-of-run write.

    print(f"\nWrote {len(index)} tiles total to {out_dir}/ (index.json), "
          f"across AOI(s): {', '.join(sorted(set(e.get('aoi', '?') for e in index)))}.")


if __name__ == "__main__":
    main()
