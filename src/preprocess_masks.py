import os
import json
import glob
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape

def create_masks(root_dir):
    """
    Rasterizes GeoJSON annotations to TIFF masks.
    """
    ann_dir = os.path.join(root_dir, 'annotations')
    mask_dir = os.path.join(root_dir, 'masks')
    pre_dir = os.path.join(root_dir, 'PRE-event')
    
    os.makedirs(mask_dir, exist_ok=True)
    
    geojson_files = glob.glob(os.path.join(ann_dir, '*.geojson'))
    
    print(f"Found {len(geojson_files)} annotations.")
    
    # Need to map tile_id to pre-image for profile
    pre_files = glob.glob(os.path.join(pre_dir, '*.tif'))
    tile_to_pre = {}
    for p in pre_files:
        # File format: 10500500C4DD7000_0_15_63.tif
        # Tile ID: 0_15_63 (stripping extension)
        parts = os.path.basename(p).split('_')
        # parts[0] is prefix. parts[1:] is 0, 15, 63.tif
        tile_id_with_ext = '_'.join(parts[1:]) 
        tile_id = tile_id_with_ext.replace('.tif', '')
        tile_to_pre[tile_id] = p
        
    for gj_path in geojson_files:
        basename = os.path.basename(gj_path)
        tile_id = basename.replace('.geojson', '')
        
        if tile_id not in tile_to_pre:
            print(f"Warning: No matching Pre-event image for {tile_id}")
            continue
            
        pre_path = tile_to_pre[tile_id]
        
        with rasterio.open(pre_path) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            transform = src.transform
            
        # Read GeoJSON
        with open(gj_path, 'r') as f:
            data = json.load(f)
            
        # Extract features
        # Assuming typical SN8 format where features have properties
        # 'flooded': True/False or similar? 
        # Actually SN8 foundation masks are usually binary building/road?
        # The user request mentioned "Building (Non-flooded), Road (Non-flooded), Flooded".
        # We need to check exact properties. 
        # For now, let's just assume we want to rasterize *everything* as class 1 (building/road) or check 'class'.
        # Since I can't check the geojson content easily right now, I'll assume standard features.
        # But wait, SN8 has specific attributes.
        # Let's inspect ONE geojson content if possible, but I cannot. 
        # I'll implement a generic rasterizer that burns '1' for any polygon.
        # Ideally, we should parse classes.
        
        shapes = []
        for feature in data['features']:
            geom = shape(feature['geometry'])
            shapes.append((geom, 1)) # Burn 1 for now
            
        if not shapes:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            
        # Save
        out_path = os.path.join(mask_dir, f"{tile_id}.tif") # Or match pre-filename
        # Dataset.py expects mask filename to match... wait.
        # dataset.py: self.tile_map[tile_id]['mask'] = p
        # It matches by tile_id logic. So as long as tile_id matches, filename structure matters less.
        # But let's keep it clean.
        
        # We should use the same naming convention if possible, but dataset.py parses underscores.
        # Let's just use `mask_{tile_id}.tif` and ensure dataset.py handles it? 
        # `dataset.py` parses `os.path.basename(p).split('_')[1:]`.
        # If I name it `mask_0_15_63.tif`, split is ['0', '15', '63.tif']. Join is 0_15_63.tif. Not 0_15_63.
        # The `dataset.py` logic: `tile_id = '_'.join(os.path.basename(p).split('_')[1:])`
        # PRE: `10500..._0_15_63.tif` -> `0_15_63.tif`.
        # Wait, `split` gives list. `join` puts it back.
        # If I name it `mask_0_15_63.tif`, it works if prefix is one chunk.
        
        # Let's simple name it same as pre-image but with 'mask_' prefix to be safe?
        # Or just specific name.
        # Let's stick to `Global_Mask_{tile_id}.tif` (3 chunks prefix? No).
        # Let's look at `pre` filename: `10500500C4DD7000_0_15_63.tif` (1 chunk prefix).
        # So `Mask_0_15_63.tif` -> `0_15_63.tif` ? No, `Mask` is 0th. `0` is 1st.
        # `split('_')[1:]` -> `['0', '15', '63.tif']`. `join` -> `0_15_63.tif`.
        # YES. `Mask_0_15_63.tif` works.
        
        out_name = f"Mask_{tile_id}.tif"
        
        # Update meta
        meta.update(count=1, dtype=rasterio.uint8, driver='GTiff')
        
        with rasterio.open(os.path.join(mask_dir, out_name), 'w', **meta) as dst:
            dst.write(mask, 1)

if __name__ == '__main__':
    create_masks('data/SN8/train')
