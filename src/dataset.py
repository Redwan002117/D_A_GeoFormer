import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class SpaceNet8Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
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

        # Basic consistency check
        if len(self.pre_images) != len(self.post_images):
            print(f"Warning: Number of pre-event ({len(self.pre_images)}) and post-event ({len(self.post_images)}) images do not match.")

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_img_path = self.pre_images[idx]
        # Find corresponding post image and mask
        # Assuming identical filenames for simplicity, in reality might differ slightly
        basename = os.path.basename(pre_img_path)
        post_img_path = os.path.join(self.post_event_dir, basename)
        mask_path = os.path.join(self.mask_dir, basename)

        # START: Dummy data generation for testing if files don't exist
        if not os.path.exists(post_img_path):
             post_img_path = pre_img_path # Fallback 
        if not os.path.exists(mask_path):
             # Return dummy mask if not found
             mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
             mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # END: Dummy data handling

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
            'id': basename
        }
