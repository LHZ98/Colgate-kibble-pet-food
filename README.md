# Kibble X-ray Microscope Data Segmentation

This project is for the segmentation of kibble x-ray microscope data from Colgate. It includes multiple baseline models for segmentation and can be used for comparison in other projects.

## Introduction

This repository contains code for training and evaluating segmentation models on kibble x-ray microscope data. The project includes various baseline models and a proposed method with cross-scale transformer architecture.

## Folder Structure

```
├── train.py              # Training script for proposed models
├── train_baselines.py    # Training script for baseline models
├── evaluate.py           # Evaluation utilities
├── eval_baseline.py      # Baseline evaluation script
├── Get_pred_result.py    # Prediction and result generation script
├── unet/                 # Proposed UNet models with transformers
│   ├── unet_model.py
│   ├── unet_model_atten.py
│   ├── unet_cross_scale_transformer.py
│   └── ...
├── othermodels/          # Baseline model implementations
│   ├── UNet.py
│   ├── UNet_2Plus.py
│   ├── UNet_3Plus.py
│   ├── SwinUNet.py
│   ├── UNetFormer.py
│   └── ...
├── utils/                # Utility functions
│   ├── data_loading.py
│   ├── dice_score.py
│   ├── focal_loss.py
│   └── ...
└── super_resolution/     # Super-resolution module
    ├── pipeline.py       # Unified pipeline script
    ├── detection/        # YOLOv8 detection module
    ├── preprocessing/    # Cropping and padding module
    └── metasr/           # Meta-SR implementation
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- opencv-python
- Pillow
- wandb (for logging)
- tqdm
- ultralytics (for super-resolution detection)
- scikit-image (for super-resolution)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Colgate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train a model, define your input data paths in the training script and run:

**Proposed model:**
```bash
python train.py --dir_save './checkpoints/your_directory/' --epoch 500 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNet_trans'
```

**Baseline models:**
```bash
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_UNet/' --epoch 500 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNet'
```

### Available Models

- `UNet`: Standard U-Net
- `UNet2Plus`: U-Net 2+
- `UNet3Plus`: U-Net 3+
- `UNetFormer`: U-NetFormer
- `SwinUNet`: Swin Transformer U-Net
- `AttentionUNet`: Attention U-Net
- `TransUNet`: TransUNet
- `UNet_atten`: U-Net with attention
- `UNet_trans`: Proposed cross-scale transformer U-Net
- `UNet_transformer_nospatial`: Ablation variant without spatial attention
- `UNet_transformer_noself`: Ablation variant without self-attention

### Evaluation

For evaluation, use the evaluation scripts:
```bash
python evaluate.py
python eval_baseline.py
```

### Prediction

To generate predictions:
```bash
python Get_pred_result.py <epoch> <dataname> <netname>
```

### Super-Resolution

The project includes a complete super-resolution pipeline for preprocessing kibble images before segmentation. See [super_resolution/README.md](super_resolution/README.md) for details.

Quick start:
```bash
python super_resolution/pipeline.py \
    --input_image_dir ./data/images \
    --input_mask_dir ./data/masks \
    --yolo_model_path ./weights/kibble_detector.pt \
    --output_dir ./output
```

## Data Organization

The data should be organized as follows:
```
data/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Model Weights and Data

Trained model weights and datasets are available on Google Drive:

**Download Link:** [Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-globalnav-goto)

The following resources are available:

- **Pre-trained Models:**
  - Proposed method checkpoint
  - Ablation experiment checkpoints (no_self, no_spatial)
  - Baseline model checkpoints (UNet, UNet2Plus, UNet3Plus, UNetFormer, SwinUNet)

- **Datasets:**
  - Training and validation datasets for kibble X-ray images
  - Preprocessed data for segmentation tasks

Please download the necessary files from Google Drive and organize them according to the folder structure mentioned above.

## Example Scripts

**Training the proposed model:**
```bash
python train.py --dir_save './checkpoints_2024_atten_multi/' --epoch 500 --amp --scale 0.25 --classes 4 --batch 4
```

**Training baseline models:**
```bash
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_UNet/' --epoch 500 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNet'
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_UNet3Plus/' --epoch 500 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNet3Plus'
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_UNet2Plus/' --epoch 100 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNet2Plus'
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_SwinUNet/' --epoch 100 --amp --scale 0.25 --classes 4 --batch 4 --model 'SwinUNet'
python train_baselines.py --dir_save './checkpoints_baselines/checkpoints_UNetFormer/' --epoch 100 --amp --scale 0.25 --classes 4 --batch 4 --model 'UNetFormer'
```

## Environment

- Training: conda environment 'cyolo'
- Testing: virtual kernel in jupyter notebook 'robustlearning'

## License

## Citation

