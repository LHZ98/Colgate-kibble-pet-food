"""
Unified Pipeline for Kibble Pet Food Super-Resolution

This script provides a complete pipeline for super-resolution processing:
1. Detect kibble regions using YOLOv8
2. Crop detected regions
3. Pad regions to square format
4. Apply Meta-SR super-resolution
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from super_resolution.detection import YOLODetector
from super_resolution.preprocessing import RegionCropper, SquarePadder


def run_full_pipeline(
    input_image_dir,
    input_mask_dir=None,
    yolo_model_path=None,
    yolo_labels_dir=None,
    output_dir='./output',
    conf_threshold=0.5,
    crop_padding=3,
    pad_mode='max_dimension',
    pad_target_size=None,
    skip_detection=False,
    skip_crop=False,
    skip_pad=False,
    skip_sr=False
):
    """
    Run the complete super-resolution pipeline.
    
    Args:
        input_image_dir: Directory containing input images
        input_mask_dir: Directory containing mask images (optional)
        yolo_model_path: Path to YOLOv8 model weights
        yolo_labels_dir: Directory containing YOLO detection labels (if skip_detection=False)
        output_dir: Base output directory
        conf_threshold: Detection confidence threshold
        crop_padding: Padding around detected regions
        pad_mode: Padding mode ('max_dimension' or 'fixed_size')
        pad_target_size: Target size for fixed_size padding
        skip_detection: Skip detection step
        skip_crop: Skip cropping step
        skip_pad: Skip padding step
        skip_sr: Skip super-resolution step
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Detection
    if not skip_detection:
        print("=" * 60)
        print("Step 1: Detecting kibble regions with YOLOv8")
        print("=" * 60)
        
        detector = YOLODetector(model_path=yolo_model_path)
        detector.load_model()
        
        detection_output = output_dir / 'detection'
        results = detector.detect_batch(
            image_dir=input_image_dir,
            output_dir=detection_output,
            conf_threshold=conf_threshold
        )
        
        yolo_labels_dir = detection_output / 'labels'
        print(f"Detection completed. Labels saved to {yolo_labels_dir}")
    else:
        print("Skipping detection step...")
        if yolo_labels_dir is None:
            raise ValueError("yolo_labels_dir required when skip_detection=True")
        yolo_labels_dir = Path(yolo_labels_dir)
    
    # Step 2: Cropping
    if not skip_crop:
        print("\n" + "=" * 60)
        print("Step 2: Cropping detected regions")
        print("=" * 60)
        
        cropper = RegionCropper(padding=crop_padding)
        
        cropped_image_dir = output_dir / 'cropped' / 'images'
        cropped_mask_dir = output_dir / 'cropped' / 'masks'
        
        cropper.crop_from_detections(
            label_dir=yolo_labels_dir,
            image_dir=input_image_dir,
            mask_dir=input_mask_dir if input_mask_dir else input_image_dir,
            output_image_dir=cropped_image_dir,
            output_mask_dir=cropped_mask_dir
        )
        
        print(f"Cropping completed. Results saved to {cropped_image_dir}")
    else:
        print("Skipping cropping step...")
        cropped_image_dir = Path(input_image_dir)
    
    # Step 3: Padding
    if not skip_pad:
        print("\n" + "=" * 60)
        print("Step 3: Padding regions to square format")
        print("=" * 60)
        
        padder = SquarePadder(mode=pad_mode)
        
        padded_image_dir = output_dir / 'padded' / 'images'
        padded_mask_dir = output_dir / 'padded' / 'masks'
        
        padder.process_directory(
            input_dir=cropped_image_dir,
            output_dir=padded_image_dir,
            target_size=pad_target_size,
            is_mask=False
        )
        
        if input_mask_dir and not skip_crop:
            padder.process_directory(
                input_dir=cropped_mask_dir,
                output_dir=padded_mask_dir,
                target_size=pad_target_size,
                is_mask=True
            )
        
        print(f"Padding completed. Results saved to {padded_image_dir}")
    else:
        print("Skipping padding step...")
        padded_image_dir = cropped_image_dir
    
    # Step 4: Super-Resolution
    if not skip_sr:
        print("\n" + "=" * 60)
        print("Step 4: Applying Meta-SR super-resolution")
        print("=" * 60)
        
        print("To run Meta-SR, use the metasr module:")
        print(f"  cd super_resolution/metasr")
        print(f"  python main.py --data_test <dataset_name> --pre_train <model_path> --test_only")
        print(f"\nInput directory: {padded_image_dir}")
    else:
        print("Skipping super-resolution step...")
    
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete super-resolution pipeline for kibble pet food'
    )
    
    parser.add_argument('--input_image_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--input_mask_dir', type=str, default=None,
                       help='Directory containing mask images (optional)')
    parser.add_argument('--yolo_model_path', type=str, default=None,
                       help='Path to YOLOv8 model weights (.pt file)')
    parser.add_argument('--yolo_labels_dir', type=str, default=None,
                       help='Directory containing YOLO labels (if detection already done)')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Base output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--crop_padding', type=int, default=3,
                       help='Padding around detected regions')
    parser.add_argument('--pad_mode', type=str, default='max_dimension',
                       choices=['max_dimension', 'fixed_size'],
                       help='Padding mode')
    parser.add_argument('--pad_target_size', type=int, default=None,
                       help='Target size for fixed_size padding')
    
    # Skip options
    parser.add_argument('--skip_detection', action='store_true',
                       help='Skip detection step')
    parser.add_argument('--skip_crop', action='store_true',
                       help='Skip cropping step')
    parser.add_argument('--skip_pad', action='store_true',
                       help='Skip padding step')
    parser.add_argument('--skip_sr', action='store_true',
                       help='Skip super-resolution step')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        input_image_dir=args.input_image_dir,
        input_mask_dir=args.input_mask_dir,
        yolo_model_path=args.yolo_model_path,
        yolo_labels_dir=args.yolo_labels_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        crop_padding=args.crop_padding,
        pad_mode=args.pad_mode,
        pad_target_size=args.pad_target_size,
        skip_detection=args.skip_detection,
        skip_crop=args.skip_crop,
        skip_pad=args.skip_pad,
        skip_sr=args.skip_sr
    )


if __name__ == '__main__':
    main()

