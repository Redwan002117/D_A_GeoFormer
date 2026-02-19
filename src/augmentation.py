import numpy as np
import cv2
import random

class CopyPasteAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample, source_sample=None):
        """
        Args:
            sample (dict): Target image/mask dict {'pre_img': ..., 'post_img': ..., 'mask': ...}
            source_sample (dict): Source image/mask dict to copy FROM.
        """
        if random.random() > self.prob or source_sample is None:
            return sample

        # Unpack
        target_pre = sample['pre_img']
        target_post = sample['post_img']
        target_mask = sample['mask']

        source_pre = source_sample['pre_img']
        source_post = source_sample['post_img']
        source_mask = source_sample['mask']

        # Extract flooded objects from source (assuming class 3 is flood, or specific classes)
        # For SN8, let's assume mask values: 0=bg, 1=building, 2=road, 3=flood
        # We want to paste 'flood' (3) or 'flooded building' regions.
        
        flood_mask = (source_mask == 3).astype(np.uint8)
        
        # If no flood in source, skip
        if np.sum(flood_mask) == 0:
            return sample

        # Dilate mask slightly to capture edges
        kernel = np.ones((3,3), np.uint8)
        paste_mask = cv2.dilate(flood_mask, kernel, iterations=1)

        # Paste source pixels onto target where paste_mask is 1
        # We paste onto both Pre and Post (or just Post? Usually geometry exists in Pre, damage in Post)
        # Detailed logic: if we paste a flooded building, we should paste valid pixels from Source Pre/Post
        
        # Convert to boolean for indexing
        mask_bool = paste_mask > 0

        # Copy pixels
        target_pre[mask_bool] = source_pre[mask_bool]
        target_post[mask_bool] = source_post[mask_bool]
        target_mask[mask_bool] = source_mask[mask_bool] # Update label

        return {
            'pre_img': target_pre,
            'post_img': target_post,
            'mask': target_mask,
            'id': sample['id']
        }
