"""
Super-Resolution Module for Kibble Pet Food X-ray Images

This module provides a complete pipeline for super-resolution processing:
1. Object detection using YOLOv8
2. Cropping detected regions
3. Padding to square format
4. Meta-SR super-resolution

Adapted from the adaptive SR pipeline for kibble pet food segmentation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

__version__ = "1.0.0"
__author__ = "Colgate Kibble Segmentation Team"

