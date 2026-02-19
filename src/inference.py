import torch
import cv2
import time
import argparse
import os
import numpy as np
from model import DualAxisGeoFormer
import matplotlib.pyplot as plt

def load_image(path, size=224):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # Normalize (approximate ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    return torch.tensor(img).unsqueeze(0) # Add batch dim

def run_realtime_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # 1. Load Model
    print("Loading model...")
    model = DualAxisGeoFormer(num_classes=4).to(device)
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model_path}")
        return

    model.eval()

    # 2. Prepare Dummy Data or Real Data (for timing)
    # We use random tensors to strictly measure model latency without I/O overhead first
    dummy_pre = torch.randn(1, 3, 224, 224).to(device)
    dummy_post = torch.randn(1, 3, 224, 224).to(device)

    # 3. Warmup
    print("Warming up (2 iters)...")
    with torch.no_grad():
        for i in range(2):
            print(f"Warmup {i+1}/2")
            _ = model(dummy_pre, dummy_post)

    # 4. Latency Test
    print("Measuring latency (5 iters)...")
    timings = []
    with torch.no_grad():
        for i in range(5):
            start = time.time()
            _ = model(dummy_pre, dummy_post)
            # Synchronize if CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            dt = end - start
            timings.append(dt)
            print(f"Iter {i+1}/5: {dt*1000:.2f} ms")

    avg_time = np.mean(timings)
    fps = 1.0 / avg_time
    print(f"\n--- Performance stats ({device.type.upper()}) ---")
    print(f"Average Latency: {avg_time*1000:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    
    # 5. Real Prediction (if paths provided)
    if args.pre_path and args.post_path:
        print(f"\nProcessing real sample: {args.pre_path}")
        pre_img = load_image(args.pre_path).to(device)
        post_img = load_image(args.post_path).to(device)
        
        start = time.time()
        with torch.no_grad():
            output = model(pre_img, post_img)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        proc_time = time.time() - start
        print(f"Single sample processing time: {proc_time*1000:.2f} ms")

        # Save Visualization
        os.makedirs(args.output_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Pre-Event")
        plt.imshow(cv2.cvtColor(cv2.imread(args.pre_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Post-Event")
        plt.imshow(cv2.cvtColor(cv2.imread(args.post_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred, cmap='jet', vmin=0, vmax=3)
        plt.axis('off')
        
        save_path = os.path.join(args.output_dir, "realtime_pred.png")
        plt.savefig(save_path)
        print(f"Prediction saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--pre_path', type=str, help='Path to pre-event image')
    parser.add_argument('--post_path', type=str, help='Path to post-event image')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    # If no paths provided, try to find a sample from data folder
    if not args.pre_path:
        # Simple auto-discovery for demo
        import glob
        pre_files = glob.glob("data/SN8/Germany_Training_Public/PRE-event/*.tif")
        if pre_files:
            file_pre = pre_files[0]
            # Construct post file path assuming standard SN8 structure
            # PRE: 10500500C4DD7000_0_15_63.tif -> POST: 105001001A0FFC00_0_15_63.tif
            # This mapping is non-trivial without the dataset dict, so we might just grab a random post file for 'demo' structural check
            # OR better, use the logic from dataset.py
            
            # Let's try to find a matching pair roughly
            tile_id = '_'.join(os.path.basename(file_pre).split('_')[1:])
            post_files = glob.glob(f"data/SN8/Germany_Training_Public/POST-event/*{tile_id}")
            if post_files:
                args.pre_path = file_pre
                args.post_path = post_files[0]
                print(f"Auto-selected sample pair: {tile_id}")

    run_realtime_inference(args)
