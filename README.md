# Dual-Axis GeoFormer: Exploiting Global-Local Dependencies for Multiclass Flood Segmentation

## Thesis Overview
**Title:** "Dual-Axis GeoFormer: Exploiting Global-Local Dependencies for Multiclass Flood Segmentation of Narrow Infrastructure via MaxViT"

This project addresses the **"Scale-Connectivity" trade-off** in satellite imagery analysis, specifically for the **SpaceNet-8 (SN-8)** dataset.

### The Problem: Disconnected Roads & Tiny Buildings
In flood segmentation, standard models fail to capture the disparate scales of infrastructure:
*   **Roads**: Long, thin, curvilinear structures that require global context to maintain connectivity (e.g., a flooded road segment should be connected to the river 1km away).
*   **Buildings**: Small, dense objects that require sharp local attention.

Standard CNNs lack the receptive field for global context, while standard Transformers (swin-like) with local windows break road connectivity.

### The Solution: MaxViT & Dual-Axis Attention
We employ **MaxViT (Multi-Axis Vision Transformer)** as the backbone:
*   **Grid Attention**: Captures global flood patterns by establishing sparse connections across the entire image.
*   **Block Attention**: Preserves local high-frequency details for precise building segmentation.

## Implementation Roadmap

### Phase 1: Bi-Temporal Data Architecture
Since SN-8 provides Pre-event and Post-event images, we use a **Siamese MaxViT Encoder**.
*   **Input**: Pre-event (infrastructure) and Post-event (damage) images.
*   **Fusion**: A **Difference Module** computes `(Post - Pre)` features to highlight changes (flood water) while preserving identity features from the Pre-event image.

### Phase 2: Model Construction
*   **Backbone**: MaxViT-Base (ImageNet-21k pretrained).
*   **Decoder**: Custom **MaxViT-UNet** decoder with skip connections.
*   **Geo-Head**: 4-Channel Output:
    *   0: Background
    *   1: Building (Non-flooded)
    *   2: Road (Non-flooded)
    *   3: Flooded (combined)

### Phase 3: Training Strategy (The "Disaster" Curriculum)
*   **Loss**: **Tversky Loss** to heavily penalize False Negatives (missing floods).
*   **Augmentation**: **Copy-Paste Augmentation** to artificially increase the frequency of flooded buildings in the training set.

### Phase 4: Infrastructure-Aware Post-Processing
*   **Skeletonization**: Uses morphological operations to bridge gaps in predicted road masks, ensuring network connectivity.

## Real-Time Inference
The main task is to predict flood extent in real-time to minimize damage. We provide a dedicated inference script for this purpose.

### Running Inference
To run single-sample inference (simulating a real-time feed) and measure latency:

```bash
python src/inference.py --pre_path path/to/pre_event.tif --post_path path/to/post_event.tif
```

If no paths are provided, the script attempts to auto-discover a sample pair from the data directory.

**Performance:**
On a standard CPU, the model achieves approximately **0.4 FPS** (~2.5 seconds per sample). On a GPU, this would be significantly faster (real-time).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Redwan002117/D_A_GeoFormer.git
    cd D_A_GeoFormer
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

**1. Data Preparation:**
```bash
python src/download_data.py
python src/preprocess_masks.py
```

**2. Training:**
```bash
python src/train.py --epochs 10 --batch_size 2
```

**3. Visualization & Testing:**
```bash
python src/visualize.py
```
This generates side-by-side comparisons of Pre, Post, Ground Truth, and Prediction in the `results/` folder.
