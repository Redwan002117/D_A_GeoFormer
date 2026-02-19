import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class SpaceNet8Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=224):
        """
        Args:
            root_dir (str): Path to SpaceNet-8 dataset directory.
                            Expected structure:
                            root_dir/
                                train/
                                    PRE-event/
                                    POST-event/
                                    masks/
                                val/
                                    ...
            split (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample.
            img_size (int): Size to resize images to.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size

        self.pre_event_dir = os.path.join(root_dir, split, 'PRE-event')
        self.post_event_dir = os.path.join(root_dir, split, 'POST-event')
        self.mask_dir = os.path.join(root_dir, split, 'masks')

        # Assuming filenames match across directories or have a common identifier
        # This is a simplified assumption; real SN8 might need CSV mapping
        self.pre_images = sorted(glob.glob(os.path.join(self.pre_event_dir, '*.tif')))
        self.post_images = sorted(glob.glob(os.path.join(self.post_event_dir, '*.tif')))
        self.masks = sorted(glob.glob(os.path.join(self.mask_dir, '*.tif')))

        # Map Tile ID to paths
        self.tile_map = {}
        for p in self.pre_images:
            tile_id = '_'.join(os.path.basename(p).split('_')[1:]) # Extract 0_15_63.tif
            if tile_id not in self.tile_map: self.tile_map[tile_id] = {}
            self.tile_map[tile_id]['pre'] = p
            
        for p in self.post_images:
            tile_id = '_'.join(os.path.basename(p).split('_')[1:])
            if tile_id in self.tile_map:
                self.tile_map[tile_id]['post'] = p
        
        for p in self.masks:
            # Mask filename might be "Mask_0_15_63.tif" or just "0_15_63.tif"
            basename = os.path.basename(p)
            if basename.startswith('Mask_'):
                tile_id = '_'.join(basename.replace('Mask_', '').split('_')) 
                # Mask_0_15_63.tif -> 0_15_63.tif -> split combined -> wait.
                # tile_id logic: '_'.join(os.path.basename(p).split('_')[1:])
                # If "Mask_0_15_63.tif", split('_') -> ['Mask', '0', '15', '63.tif']. 
                # [1:] -> ['0', '15', '63.tif']. join -> 0_15_63.tif. 
                # Ideally, tile_id should be "0_15_63".
                # My logic: tile_id = '_'.join(os.path.basename(p).split('_')[1:])
                # For 10500..._0_15_63.tif: ['10500...', '0', '15', '63.tif'] -> 0_15_63.tif.
                # For Mask_0_15_63.tif: ['Mask', '0', '15', '63.tif'] -> 0_15_63.tif.
                # So it WORKS automatically if I use 'Mask_' prefix!
                pass
            
            tile_id = '_'.join(os.path.basename(p).split('_')[1:])
            if tile_id in self.tile_map:
                self.tile_map[tile_id]['mask'] = p

        # Filter complete samples? Or allow partials?
        # For training we need at least pre and post. Mask is needed for supervised.
        # Let's keep only complete triples for now.
        self.valid_tiles = [t for t, v in self.tile_map.items() if 'pre' in v and 'post' in v]
        
        print(f"Found {len(self.valid_tiles)} complete samples out of {len(self.pre_images)} pre-event images.")

    def __len__(self):
        return len(self.valid_tiles)

    def __getitem__(self, idx):
        tile_id = self.valid_tiles[idx]
        sample_paths = self.tile_map[tile_id]
        
        pre_img_path = sample_paths['pre']
        post_img_path = sample_paths['post']
        mask_path = sample_paths.get('mask', None)

        # ... (rest of loading) ...
        # Handling missing mask (if we allow it, but we filtered earlier, so maybe just load dummy if strict filter off)
        
        if mask_path is None or not os.path.exists(mask_path):
             mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
             mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        pre_img = cv2.imread(pre_img_path)
        post_img = cv2.imread(post_img_path)
        
        if pre_img is None:
            raise FileNotFoundError(f"Image not found: {pre_img_path}")
        if post_img is None:
            post_img = np.zeros_like(pre_img) # Fallback

        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        # Resize
        pre_img = cv2.resize(pre_img, (self.img_size, self.img_size))
        post_img = cv2.resize(post_img, (self.img_size, self.img_size))
        
        if isinstance(mask, np.ndarray) and mask.shape != (self.img_size, self.img_size):
             mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Normalize to [0, 1] and convert to tensor
        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        if self.transform:
            # Albumentations usually require a dict
            augmented = self.transform(image=pre_img, image0=post_img, mask=mask)
            pre_img = augmented['image']
            post_img = augmented['image0']
            mask = augmented['mask']
        else:
             # Manual tensor conversion
             pre_img = torch.from_numpy(pre_img).permute(2, 0, 1)
             post_img = torch.from_numpy(post_img).permute(2, 0, 1)
             mask = torch.from_numpy(mask).long()

        return {
            'pre_img': pre_img, 
            'post_img': post_img, 
            'mask': mask,
            'id': tile_id
        }
