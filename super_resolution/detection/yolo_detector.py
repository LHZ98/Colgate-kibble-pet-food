"""
Step 2: YOLOv8 Detection for Kibble Pet Food

This module handles object detection using YOLOv8 to detect kibble regions
in low-resolution X-ray images.
"""

import os
from pathlib import Path
import torch


class YOLODetector:
    """
    YOLOv8-based detector for kibble pet food regions in X-ray images.
    """
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to trained YOLOv8 model weights (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        
    def load_model(self, model_path=None):
        """
        Load YOLOv8 model from checkpoint.
        
        Args:
            model_path: Path to model weights. If None, uses self.model_path
        """
        try:
            from ultralytics import YOLO
            
            if model_path is None:
                model_path = self.model_path
                
            if model_path is None:
                raise ValueError("Model path not provided")
                
            self.model = YOLO(model_path)
            print(f"Model loaded from {model_path}")
            
        except ImportError:
            raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def detect(self, image_path, conf_threshold=0.5, save_crops=True, save_txt=True):
        """
        Detect kibble regions in image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detection
            save_crops: Whether to save cropped regions
            save_txt: Whether to save detection results as text files
            
        Returns:
            Detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save_crop=save_crops,
            save_txt=save_txt,
            hide_labels=True,
            line_thickness=0
        )
        
        return results
    
    def detect_batch(self, image_dir, output_dir=None, conf_threshold=0.5):
        """
        Detect kibble regions in a batch of images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save detection results
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Dictionary mapping image names to detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_dir = Path(image_dir)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results_dict = {}
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        for img_path in sorted(image_files):
            print(f"Processing {img_path.name}...")
            results = self.detect(
                str(img_path),
                conf_threshold=conf_threshold,
                save_crops=True,
                save_txt=True
            )
            results_dict[img_path.name] = results
        
        return results_dict


def train_yolo_model(data_yaml_path, epochs=500, batch_size=4, imgsz=256, output_dir='./runs/detect'):
    """
    Train YOLOv8 model for kibble detection.
    
    Args:
        data_yaml_path: Path to YOLO dataset YAML file
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Input image size
        output_dir: Directory to save training outputs
        
    Returns:
        Path to best model weights
    """
    try:
        from ultralytics import YOLO
        
        # Load pre-trained YOLOv8 model
        model = YOLO('yolov8m.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=output_dir,
            name='kibble_detection'
        )
        
        best_model_path = Path(output_dir) / 'kibble_detection' / 'weights' / 'best.pt'
        return str(best_model_path)
        
    except ImportError:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")

