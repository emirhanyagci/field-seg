"""Configuration settings for field segmentation model."""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model training and inference configuration."""
    # Model settings
    model_size: str = "n"  # n, s, m, l, x (nano, small, medium, large, xlarge)
    task: str = "segment"  # segmentation task
    
    # Training settings
    epochs: int = 100
    imgsz: int = 640
    batch_size: int = 16
    workers: int = 8
    device: Optional[str] = None  # None for auto, 'cpu', 'cuda', 'mps', etc.
    
    # Data settings
    data_yaml: str = "field-seg.yaml"
    
    # Optimization settings
    patience: int = 50  # Early stopping patience
    save_period: int = 10  # Save checkpoint every N epochs
    
    # Output settings
    project: str = "runs"
    name: str = "field_segmentation"
    
    # Augmentation settings
    hsv_h: float = 0.015  # Image HSV-Hue augmentation
    hsv_s: float = 0.7  # Image HSV-Saturation augmentation
    hsv_v: float = 0.4  # Image HSV-Value augmentation
    degrees: float = 0.0  # Image rotation (+/- deg)
    translate: float = 0.1  # Image translation (+/- fraction)
    scale: float = 0.5  # Image scale (+/- gain)
    flipud: float = 0.0  # Image flip up-down (probability)
    fliplr: float = 0.5  # Image flip left-right (probability)
    mosaic: float = 1.0  # Image mosaic (probability)
    mixup: float = 0.0  # Image mixup (probability)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Model path
    model_path: str = "best.pt"  # Path to trained model
    
    # Image settings
    imgsz: int = 640
    conf_threshold: float = 0.25  # Confidence threshold
    iou_threshold: float = 0.45  # IoU threshold for NMS
    
    # Output settings
    save_dir: str = "predictions"
    save_txt: bool = False  # Save predictions as text files
    save_conf: bool = True  # Save confidence scores
    show_labels: bool = True
    show_conf: bool = True
    line_width: int = 2
    
    # Visualization settings
    save_mask: bool = True  # Save segmentation masks
    mask_alpha: float = 0.5  # Mask transparency


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

