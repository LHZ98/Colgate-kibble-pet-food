"""
Step 3: Crop Images and Masks Based on Detection Results

This module handles cropping detected kibble regions from full images
and corresponding masks for kibble pet food X-ray data.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np


class RegionCropper:
    """
    Crop detected regions from images and masks based on YOLO detection results.
    """
    
    def __init__(self, padding=3):
        """
        Initialize cropper.
        
        Args:
            padding: Padding around detected bounding boxes (in pixels)
        """
        self.padding = padding
    
    def parse_yolo_label(self, label_path, img_width, img_height):
        """
        Parse YOLO format label file and convert to pixel coordinates.
        
        Args:
            label_path: Path to YOLO label file (.txt)
            img_width: Width of the image
            img_height: Height of the image
            
        Returns:
            List of bounding boxes in format [(xmin, ymin, xmax, ymax), ...]
        """
        boxes = []
        
        if not os.path.exists(label_path):
            return boxes
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            xmin = x_center - width / 2
            xmax = x_center + width / 2
            ymin = y_center - height / 2
            ymax = y_center + height / 2
            
            boxes.append((xmin, ymin, xmax, ymax))
        
        return boxes
    
    def crop_region(self, image, bbox, padding=0):
        """
        Crop a region from image based on bounding box.
        
        Args:
            image: PIL Image object
            bbox: Bounding box tuple (xmin, ymin, xmax, ymax)
            padding: Additional padding around bbox
            
        Returns:
            Cropped PIL Image
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Add padding
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(image.width, xmax + padding)
        ymax = min(image.height, ymax + padding)
        
        cropped = image.crop((xmin, ymin, xmax, ymax))
        return cropped
    
    def crop_from_detections(self, 
                           label_dir, 
                           image_dir, 
                           mask_dir,
                           output_image_dir,
                           output_mask_dir,
                           mask_suffix='_Segmentation.png'):
        """
        Crop regions from images and masks based on YOLO detection labels.
        
        Args:
            label_dir: Directory containing YOLO label files (.txt)
            image_dir: Directory containing input images
            mask_dir: Directory containing mask images
            output_image_dir: Directory to save cropped images
            output_mask_dir: Directory to save cropped masks
            mask_suffix: Suffix pattern for mask files
        """
        label_dir = Path(label_dir)
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        output_image_dir = Path(output_image_dir)
        output_mask_dir = Path(output_mask_dir)
        
        # Create output directories
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all label files
        label_files = sorted(list(label_dir.glob('*.txt')))
        
        for label_file in label_files:
            # Get corresponding image and mask
            base_name = label_file.stem
            
            # Find image file
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            image_path = None
            for ext in image_extensions:
                potential_path = image_dir / f"{base_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
                potential_path = image_dir / f"{base_name}{ext.upper()}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"Warning: Image not found for {base_name}")
                continue
            
            # Find mask file - for kibble pet food, try common naming patterns
            mask_path = mask_dir / f"{base_name}{mask_suffix}"
            if not mask_path.exists():
                mask_path = mask_dir / f"{base_name}_mask.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{base_name}.png"
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {base_name}")
                continue
            
            # Load image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('L')
            
            # Parse detection labels
            boxes = self.parse_yolo_label(label_file, image.width, image.height)
            
            if len(boxes) == 0:
                print(f"No detections found in {label_file.name}")
                continue
            
            # Crop each detected region
            for idx, bbox in enumerate(boxes):
                image_crop = self.crop_region(image, bbox, padding=self.padding)
                mask_crop = self.crop_region(mask, bbox, padding=self.padding)
                
                # Save cropped images
                image_save_name = output_image_dir / f"{base_name}_{idx+1}.png"
                mask_save_name = output_mask_dir / f"{base_name}_{idx+1}.png"
                
                image_crop.save(image_save_name)
                mask_crop.save(mask_save_name)
                
                print(f"Saved cropped region {idx+1} for {base_name}")
        
        print(f"Cropping completed. Saved to {output_image_dir} and {output_mask_dir}")

