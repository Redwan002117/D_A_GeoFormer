# Dual-Axis GeoFormer for SpaceNet-8

This project implements the "Dual-Axis GeoFormer" for multiclass flood segmentation using MaxViT and a Siamese encoder approach.

## Prerequisites

- **Python 3.8+**
- **CUDA-enabled GPU** (recommended for training)
- **SpaceNet-8 Dataset**

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data:**
    If you have AWS credentials or the AWS CLI configured, you can try downloading a sample:
    ```bash
    python src/download_data.py
    ```
    Alternatively, download the SpaceNet-8 dataset manually from [SpaceNet.ai](https://spacenet.ai/sn8-challenge/) and place it in `data/SN8`.

## Project Structure

- `src/dataset.py`: Bi-temporal data loader.
- `src/model.py`: Dual-Axis GeoFormer architecture (MaxViT + U-Net).
- `src/loss.py`: Tversky Loss implementation.
- `src/augmentation.py`: Copy-Paste augmentation.
- `src/post_process.py`: Road skeletonization and gap bridging.
- `src/train.py`: Main training loop.

## Usage

To start training:
```bash
python src/train.py --data_dir data/SN8 --epochs 50 --batch_size 4
```

## Implementation Details

- **Backbone**: `maxvit_base_tf_224.in1k` (via `timm`).
- **Loss**: Tversky Loss (alpha=0.3, beta=0.7).
- **Post-Processing**: Morphological closing to bridge gaps in road predictions.
