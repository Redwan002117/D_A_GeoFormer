# Dual-Axis GeoFormer for Flood Segmentation

## Project Overview
This project implements a **Dual-Axis GeoFormer** model for multiclass flood segmentation using the **SpaceNet-8** dataset. It leverages a **Siamese MaxViT** encoder to process bi-temporal satellite imagery (Pre-event and Post-event) to identify flood extent and road networks.

The model addresses the "Scale-Connectivity" trade-off by combining:
- **Global Context**: Via MaxViT's Grid Attention (important for long road connectivity).
- **Local Details**: Via MaxViT's Block Attention (important for building footprints).
- **Change Detection**: Via a Difference Module that fuses features from pre- and post-event images.

## Features
- **Bi-Temporal Input**: Takes pairs of images (Before and After flood).
- **Siamese Encoder**: Shared weights for extracting features from both timepoints.
- **Tversky Loss**: Optimized for imbalanced classes (flood pixels are rare).
- **Visualization**: Tools to inspect model predictions side-by-side with ground truth.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Redwan002117/D_A_GeoFormer.git
   cd D_A_GeoFormer
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `torch`, `torchvision`, `timm`, `opencv-python`, `rasterio` (or `tifffile`), `matplotlib`, `boto3`.*

## Data Preparation

The project is designed for the **SpaceNet-8** dataset.

1. **Download Data**:
   Use the included script to download a sample of the dataset from AWS S3 (requires valid AWS credentials or uses public bucket if configured):
   ```bash
   python src/download_data.py
   ```
   *This will download pre-event, post-event images, and GeoJSON annotations to `data/SN8`.*

2. **Preprocess Masks**:
   Convert the vector GeoJSON annotations into raster segmentation masks:
   ```bash
   python src/preprocess_masks.py
   ```
   *Generates TIF masks in `data/SN8/Germany_Training_Public/annotations/masks/`.*

## Training the Model

To train the model from scratch:

```bash
python src/train.py --epochs 10 --batch_size 2 --save_all
```

**Arguments:**
- `--data_dir`: Path to dataset (default: `data/SN8`).
- `--epochs`: Number of training epochs (default: 10).
- `--batch_size`: Batch size (reduce to 1 or 2 if running on CPU/Low VRAM).
- `--lr`: Learning rate.
- `--save_all`: Save a checkpoint at every epoch.

The script automatically saves the model with the lowest validation loss as `best_model.pth`.

## Testing & Inference

To test the model and visualize how it determines flood regions:

```bash
python src/visualize.py
```

This script:
1. Loads the trained `best_model.pth`.
2. Runs inference on samples from the validation set.
3. Saves visualization images to the `results/` folder.
4. Images show: **Pre-Event** | **Post-Event** | **Ground Truth** | **Prediction**.

## How It Works

1. **Input**: The model receives two 224x224 images: one from before the flood and one from after.
2. **Feature Extraction**: The Siamese MaxViT encoder extracts hierarchical features from both images.
3. **Fusion**: A Difference Module calculates the feature difference `(Post - Pre)` to highlight changes.
4. **Decoding**: The decoder upsamples these features to produce a pixel-level classification mask.
5. **Output**: A 4-channel mask representing:
   - Channel 0: Background
   - Channel 1: Building
   - Channel 2: Road
   - Channel 3: Flood

## Folder Structure
```
D_A_GeoFormer/
├── data/               # Dataset storage
├── results/            # Visualization outputs
├── src/
│   ├── dataset.py      # PyTorch Dataset class
│   ├── model.py        # Model architecture
│   ├── train.py        # Training loop
│   ├── visualize.py    # Inference script
│   └── ...
├── best_model.pth      # Best trained checkpoint
└── requirements.txt    # Python dependencies
```
