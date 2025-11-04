# Super-Resolution Module for Kibble Pet Food

This module provides a complete pipeline for super-resolution processing of kibble pet food X-ray images. The pipeline includes object detection, region cropping, padding, and Meta-SR super-resolution.

## Overview

The super-resolution pipeline consists of four main steps:

1. **Detection (Step 2)**: Use YOLOv8 to detect kibble regions in low-resolution images
2. **Cropping (Step 3)**: Extract detected regions from full images and masks
3. **Padding (Step 4.5)**: Pad cropped regions to square format for consistent processing
4. **Super-Resolution (Step 5)**: Apply Meta-SR for arbitrary-scale super-resolution

## Quick Start

```bash
python super_resolution/pipeline.py \
    --input_image_dir ./data/images \
    --input_mask_dir ./data/masks \
    --yolo_model_path ./weights/kibble_detector.pt \
    --output_dir ./output
```

For detailed documentation, see [super_resolution/README.md](super_resolution/README.md).

