"""
Detection module for kibble pet food regions.
"""

from .yolo_detector import YOLODetector, train_yolo_model

__all__ = ['YOLODetector', 'train_yolo_model']

