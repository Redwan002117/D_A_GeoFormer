"""
Downloads and rasterizes a real, labeled slice of SpaceNet-8 (the public,
UNSIGNED-access `spacenet-dataset` S3 bucket -- no AWS credentials needed)
into the layout `SpaceNet8Dataset` (dataset.py) expects.

This is real data, not synthetic: real pre/post aerial tiles from the
2021 Germany flood AOI, with real OpenStreetMap-derived building/road
labels and a real `flooded: "yes"` flag SpaceNet-8's own annotators set
per feature. It is NOT the full SpaceNet-8 training set (202 tiles exist
in this one AOI alone; SN-8 has multiple AOIs) and this download only
pulls a small slice of it -- see docs/MANUAL.md for what that does and
doesn't prove.

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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image, ImageDraw
from PIL.TiffTags import TAGS

BUCKET = "spacenet-dataset"
BASE = "spacenet/SN8_floods/Germany_Training_Public/"
TILE_PX = 256  # resize every tile to this for the model


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


def rasterize_mask(geojson_path: Path, to_pixel, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    flood_mask = Image.new("L", (width, height), 0)
    flood_draw = ImageDraw.Draw(flood_mask)

    with open(geojson_path) as f:
        data = json.load(f)

    for feature in data["features"]:
        props = feature["properties"]
        geom = feature["geometry"]
        is_building = props.get("building") is not None
        is_road = props.get("highway") is not None
        is_flooded = props.get("flooded") == "yes"
        if not is_building and not is_road:
            continue

        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                pts = [to_pixel(lon, lat) for lon, lat in ring]
                draw.polygon(pts, fill=1 if is_building else 2)
                if is_flooded:
                    flood_draw.polygon(pts, fill=1)
        elif geom["type"] in ("LineString",):
            pts = [to_pixel(lon, lat) for lon, lat in geom["coordinates"]]
            draw.line(pts, fill=2, width=10)
            if is_flooded:
                flood_draw.line(pts, fill=1, width=10)
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    pts = [to_pixel(lon, lat) for lon, lat in ring]
                    draw.polygon(pts, fill=1 if is_building else 2)
                    if is_flooded:
                        flood_draw.polygon(pts, fill=1)

    mask_arr = np.array(mask)
    flood_arr = np.array(flood_mask)
    mask_arr[flood_arr == 1] = 3  # flooded overrides building(1)/road(2)
    return mask_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tiles", type=int, default=24)
    p.add_argument("--out-dir", type=str, default="real_sn8_dataset")
    p.add_argument("--prefer-flooded", action="store_true", default=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "pre").mkdir(parents=True, exist_ok=True)
    (out_dir / "post").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("Listing annotation tiles...")
    ann_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=BASE + "annotations/")
    ann_keys = [o["Key"] for o in ann_resp["Contents"]]

    print("Listing pre/post imagery...")
    pre_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=BASE + "PRE-event/")
    pre_keys = {k["Key"].split("_", 1)[1].rsplit(".", 1)[0]: k["Key"] for k in
                [{"Key": o["Key"]} for o in pre_resp["Contents"]]}
    # tile_id from "<catalogid>_0_15_63.tif" -> "0_15_63"
    pre_by_tile = {}
    for o in pre_resp["Contents"]:
        fname = o["Key"].rsplit("/", 1)[-1]
        tile_id = "_".join(fname.split("_")[1:]).rsplit(".", 1)[0]
        pre_by_tile[tile_id] = o["Key"]

    post_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=BASE + "POST-event/")
    post_by_tile: dict[str, str] = {}
    for o in post_resp["Contents"]:
        fname = o["Key"].rsplit("/", 1)[-1]
        tile_id = "_".join(fname.split("_")[1:]).rsplit(".", 1)[0]
        post_by_tile.setdefault(tile_id, o["Key"])  # first match only

    # Rank tiles by how many flooded features they contain, to build a
    # class-balanced-ish small sample (mix of flooded and dry tiles).
    print(f"Scanning {len(ann_keys)} annotation files for flood content...")
    scored = []
    for key in ann_keys:
        tile_id = key.rsplit("/", 1)[-1].replace(".geojson", "")
        if tile_id not in pre_by_tile or tile_id not in post_by_tile:
            continue
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        n_flooded = sum(1 for f in data["features"] if f["properties"].get("flooded") == "yes")
        scored.append((tile_id, key, n_flooded, len(data["features"])))

    scored.sort(key=lambda x: -x[2])
    n_flood_tiles = args.n_tiles // 2
    selected = scored[:n_flood_tiles] + [s for s in scored if s[2] == 0][: args.n_tiles - n_flood_tiles]
    selected = selected[: args.n_tiles]
    print(f"Selected {len(selected)} tiles "
          f"({sum(1 for s in selected if s[2] > 0)} with flood labels, "
          f"{sum(1 for s in selected if s[2] == 0)} without).")

    index = []
    for tile_id, ann_key, n_flooded, n_features in selected:
        pre_key, post_key = pre_by_tile[tile_id], post_by_tile[tile_id]
        pre_local = out_dir / "pre" / f"{tile_id}.tif"
        post_local = out_dir / "post" / f"{tile_id}.tif"
        ann_local = out_dir / f"{tile_id}.geojson"

        if not pre_local.exists():
            s3.download_file(BUCKET, pre_key, str(pre_local))
        if not post_local.exists():
            s3.download_file(BUCKET, post_key, str(post_local))
        if not ann_local.exists():
            s3.download_file(BUCKET, ann_key, str(ann_local))

        to_pixel, width, height = geo_transform(pre_local)
        mask_arr = rasterize_mask(ann_local, to_pixel, width, height)

        pre_img = Image.open(pre_local).convert("RGB").resize((TILE_PX, TILE_PX))
        post_img = Image.open(post_local).convert("RGB").resize((TILE_PX, TILE_PX))
        mask_img = Image.fromarray(mask_arr.astype(np.uint8)).resize((TILE_PX, TILE_PX), Image.NEAREST)

        pre_png = f"pre/{tile_id}.png"
        post_png = f"post/{tile_id}.png"
        mask_png = f"mask/{tile_id}.png"
        pre_img.save(out_dir / pre_png)
        post_img.save(out_dir / post_png)
        mask_img.save(out_dir / mask_png)

        class_counts = {int(c): int((mask_arr == c).sum()) for c in np.unique(mask_arr)}
        index.append({"pre": pre_png, "post": post_png, "mask": mask_png,
                       "tile_id": tile_id, "n_flooded_features": n_flooded,
                       "class_pixel_counts": class_counts})
        print(f"  {tile_id}: {n_flooded}/{n_features} flooded features, "
              f"mask classes {class_counts}")

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote {len(index)} tiles to {out_dir}/ (index.json).")


if __name__ == "__main__":
    main()
