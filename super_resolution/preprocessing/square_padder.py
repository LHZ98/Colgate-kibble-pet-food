"""
Step 4.5: Pad Images to Square Format

This module handles padding cropped regions to square format for consistent
super-resolution processing of kibble pet food regions.
"""

import os
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2
import glob


def padding(img, expected_size):
    """
    Pad image to expected size (square).
    
    Args:
        img: PIL Image object
        expected_size: Target size (will create square of this size)
        
    Returns:
        Padded PIL Image
    """
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    """
    Resize image maintaining aspect ratio and pad to square.
    
    Args:
        img: PIL Image object
        expected_size: Target size tuple (width, height)
        
    Returns:
        Resized and padded PIL Image
    """
    img.thumbnail((expected_size[0], expected_size[1]), Image.Resampling.LANCZOS)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding_tuple = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding_tuple)


class SquarePadder:
    """
    Pad images to square format for super-resolution processing.
    """
    
    def __init__(self, mode='max_dimension'):
        """
        Initialize padder.
        
        Args:
            mode: Padding mode - 'max_dimension' (pad to max dimension) 
                  or 'fixed_size' (pad to fixed size)
        """
        self.mode = mode
    
    def pad_to_square(self, image, target_size=None):
        """
        Pad image to square format.
        
        Args:
            image: PIL Image object
            target_size: Target square size (if None, uses max dimension)
            
        Returns:
            Padded square PIL Image
        """
        if self.mode == 'max_dimension':
            size = max(image.size)
            return resize_with_padding(image, (size, size))
        elif self.mode == 'fixed_size':
            if target_size is None:
                raise ValueError("target_size required for fixed_size mode")
            return resize_with_padding(image, (target_size, target_size))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def process_directory(self, input_dir, output_dir, target_size=None, is_mask=False):
        """
        Process all images in a directory, padding them to square.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save padded images
            target_size: Target square size (if None, uses max dimension per image)
            is_mask: Whether processing mask images (for grayscale handling)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(f'*{ext}')))
            image_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        size_list = []
        
        for img_path in sorted(image_files):
            print(f"Processing {img_path.name}...")
            
            # Load image
            if is_mask:
                img = Image.open(img_path).convert('L')
            else:
                img = Image.open(img_path).convert('RGB')
            
            original_size = img.size
            size_list.append(max(original_size))
            
            # Pad to square
            if target_size is None:
                # Use max dimension of this image
                max_dim = max(original_size)
                padded_img = self.pad_to_square(img, target_size=max_dim)
            else:
                # Use fixed target size
                padded_img = self.pad_to_square(img, target_size=target_size)
            
            # Save padded image
            output_path = output_dir / img_path.name
            
            if is_mask:
                # Save as grayscale
                img_array = np.array(padded_img)
                img_array = img_array.reshape(img_array.shape + (1,))
                cv2.imwrite(str(output_path), img_array)
            else:
                padded_img.save(output_path)
            
            print(f"  Original: {original_size} -> Padded: {padded_img.size}")
        
        if size_list:
            avg_size = sum(size_list) / len(size_list)
            print(f"\nAverage max dimension: {avg_size:.2f}")
        
        print(f"Padded images saved to {output_dir}")

