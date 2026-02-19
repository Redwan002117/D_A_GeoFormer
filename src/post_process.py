import numpy as np
from skimage.morphology import skeletonize
import cv2

def skeletonize_road_mask(mask):
    """
    Args:
        mask (np.ndarray): Binary mask of roads.
    Returns:
        np.ndarray: Skeletonized mask.
    """
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8)

def bridge_gaps(mask, max_gap=5):
    """
    Simple morphological closing to bridge small gaps.
    For more complex "Global Attention" bridging, use model features.
    Here we implement a morphological approximation.
    """
    kernel = np.ones((max_gap, max_gap), np.uint8)
    bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return bridged

def post_process_prediction(pred_mask, road_class_idx=2):
    """
    Args:
        pred_mask (np.ndarray): (H, W) class indices.
        road_class_idx (int): Index of road class.
    """
    road_mask = (pred_mask == road_class_idx).astype(np.uint8)
    
    # 1. Bridge gaps
    bridged = bridge_gaps(road_mask)
    
    # 2. Skeletonize (optional, depending on requirement)
    # The requirement says "ensure predicted roads are continuous... bridge that gap".
    # Skeletonization reduces it to 1-pixel width, which might be too thin for segmentation metrics
    # but good for topology.
    # We will return the bridged mask primarily.
    
    # Update prediction
    pred_mask[bridged > 0] = road_class_idx
    
    return pred_mask
