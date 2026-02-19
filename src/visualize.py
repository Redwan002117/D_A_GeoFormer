import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model import DualAxisGeoFormer
from dataset import SpaceNet8Dataset
from torch.utils.data import DataLoader

def visualize_inference(model_path, data_dir, output_dir='results', num_samples=5):
    """
    Runs inference on a few samples and saves visualization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    model = DualAxisGeoFormer(num_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Model checkpoint not found. Running with random weights for demo.")
    
    model.eval()
    
    # Load Dataset
    dataset = SpaceNet8Dataset(data_dir, split='train') # Use train for now as we split manually or just demo
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pre_img = batch['pre_img'].to(device)
            post_img = batch['post_img'].to(device)
            mask = batch['mask'].to(device) # (B, H, W)
            tile_id = batch['id'][0]
            
            # Forward
            output = model(pre_img, post_img) # (B, C, H, W)
            pred = torch.argmax(output, dim=1) # (B, H, W)
            
            # Convert to numpy for plotting
            # Denormalize images (assuming [0,1])
            pre_np = pre_img[0].permute(1, 2, 0).cpu().numpy()
            post_np = post_img[0].permute(1, 2, 0).cpu().numpy()
            mask_np = mask[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            
            # Plot
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(pre_np)
            ax[0].set_title("Pre-Event")
            ax[0].axis('off')
            
            ax[1].imshow(post_np)
            ax[1].set_title("Post-Event")
            ax[1].axis('off')
            
            ax[2].imshow(mask_np, cmap='jet', vmin=0, vmax=3)
            ax[2].set_title("Ground Truth")
            ax[2].axis('off')
            
            ax[3].imshow(pred_np, cmap='jet', vmin=0, vmax=3)
            ax[3].set_title("Prediction")
            ax[3].axis('off')
            
            plt.suptitle(f"Tile: {tile_id}")
            save_path = os.path.join(output_dir, f"vis_{tile_id}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to {save_path}")
            
            count += 1
            if count >= num_samples:
                break

if __name__ == '__main__':
    # Assuming checkpoint exists after training
    visualize_inference('best_model.pth', 'data/SN8', num_samples=5)
