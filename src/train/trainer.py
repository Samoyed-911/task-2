"""
Ship Detection Trainer

This module provides the training pipeline for ship detection using YOLOv8-OBB
(Oriented Bounding Box) model. It handles:
- Model initialization and configuration
- Training with configurable hyperparameters
- Checkpoint management
- Training monitoring and logging

YOLOv8-OBB is chosen for this task because:
1. Native support for oriented bounding boxes (essential for rotated ship detection)
2. State-of-the-art performance on aerial/satellite imagery
3. Efficient training and inference
4. Well-documented and maintained
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import yaml
import torch
from ultralytics import YOLO


class ShipDetectionTrainer:
    """
    Trainer class for ship detection using YOLOv8-OBB.
    
    This class wraps the Ultralytics YOLO training pipeline with
    customized settings for satellite imagery ship detection.
    """
    
    # Available YOLOv8-OBB model variants
    MODEL_VARIANTS = {
        'nano': 'yolov8n-obb.pt',
        'small': 'yolov8s-obb.pt',
        'medium': 'yolov8m-obb.pt',
        'large': 'yolov8l-obb.pt',
        'xlarge': 'yolov8x-obb.pt'
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_variant: str = 'small',
        device: str = 'auto'
    ):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to training configuration YAML file
            model_variant: Model size variant (nano/small/medium/large/xlarge)
            device: Device to use ('auto', 'cpu', '0', '0,1', etc.)
        """
        self.config = self._load_config(config_path)
        self.model_variant = model_variant
        self.device = device
        self.model = None
        self.results = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration from YAML file."""
        default_config = {
            'model': {
                'base': 'yolov8s-obb',
                'num_classes': 1,
                'imgsz': 640
            },
            'training': {
                'epochs': 100,
                'batch': 16,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'iou': 0.7,
                'conf': 0.001,
                'max_det': 300,
                'amp': True,
                'workers': 8,
                'patience': 50,
                'save_period': 10
            },
            'augmentation': {
                'fliplr': 0.5,
                'flipud': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 180,
                'translate': 0.1,
                'scale': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0
            },
            'output': {
                'project': 'outputs/models',
                'name': 'ship_detection'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            # Merge with default config
            for key in loaded_config:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(loaded_config[key])
                else:
                    default_config[key] = loaded_config[key]
        
        return default_config
    
    def _get_model_path(self) -> str:
        """Get the model path based on variant or config."""
        # Check if config specifies a custom model path
        model_base = self.config.get('model', {}).get('base', '')
        
        if model_base and Path(model_base).exists():
            return model_base
        
        # Use variant mapping
        if self.model_variant in self.MODEL_VARIANTS:
            return self.MODEL_VARIANTS[self.model_variant]
        
        # Default to small
        return self.MODEL_VARIANTS['small']
    
    def initialize_model(self, pretrained: bool = True) -> YOLO:
        """
        Initialize the YOLOv8-OBB model.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Initialized YOLO model
        """
        model_path = self._get_model_path()
        
        print(f"Initializing model: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = YOLO(model_path)
        
        return self.model
    
    def train(
        self,
        data_yaml: str,
        resume: bool = False,
        checkpoint: Optional[str] = None
    ) -> Any:
        """
        Train the model.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            resume: Whether to resume from last checkpoint
            checkpoint: Path to specific checkpoint to resume from
            
        Returns:
            Training results
        """
        if self.model is None:
            self.initialize_model()
        
        # Extract training parameters
        train_cfg = self.config.get('training', {})
        aug_cfg = self.config.get('augmentation', {})
        output_cfg = self.config.get('output', {})
        model_cfg = self.config.get('model', {})
        
        # Build training arguments
        train_args = {
            # Data
            'data': data_yaml,
            'imgsz': model_cfg.get('imgsz', 640),
            
            # Training
            'epochs': train_cfg.get('epochs', 100),
            'batch': train_cfg.get('batch', 16),
            'optimizer': train_cfg.get('optimizer', 'AdamW'),
            'lr0': train_cfg.get('lr0', 0.001),
            'lrf': train_cfg.get('lrf', 0.01),
            'momentum': train_cfg.get('momentum', 0.937),
            'weight_decay': train_cfg.get('weight_decay', 0.0005),
            'warmup_epochs': train_cfg.get('warmup_epochs', 3),
            'warmup_momentum': train_cfg.get('warmup_momentum', 0.8),
            'warmup_bias_lr': train_cfg.get('warmup_bias_lr', 0.1),
            'patience': train_cfg.get('patience', 50),
            'save_period': train_cfg.get('save_period', 10),
            
            # Detection
            'iou': train_cfg.get('iou', 0.7),
            'max_det': train_cfg.get('max_det', 300),
            
            # Augmentation
            'fliplr': aug_cfg.get('fliplr', 0.5),
            'flipud': aug_cfg.get('flipud', 0.5),
            'hsv_h': aug_cfg.get('hsv_h', 0.015),
            'hsv_s': aug_cfg.get('hsv_s', 0.7),
            'hsv_v': aug_cfg.get('hsv_v', 0.4),
            'degrees': aug_cfg.get('degrees', 180),
            'translate': aug_cfg.get('translate', 0.1),
            'scale': aug_cfg.get('scale', 0.5),
            'mosaic': aug_cfg.get('mosaic', 1.0),
            'mixup': aug_cfg.get('mixup', 0.0),
            
            # Output
            'project': output_cfg.get('project', 'outputs/models'),
            'name': output_cfg.get('name', 'ship_detection'),
            'exist_ok': True,
            'plots': output_cfg.get('plots', True),
            'verbose': output_cfg.get('verbose', True),
            
            # Device
            'device': self.device if self.device != 'auto' else None,
            'amp': train_cfg.get('amp', True),
            'workers': train_cfg.get('workers', 8),
            
            # Resume
            'resume': resume
        }
        
        # Handle checkpoint resume
        if checkpoint and Path(checkpoint).exists():
            self.model = YOLO(checkpoint)
            train_args['resume'] = True
        
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Dataset: {data_yaml}")
        print(f"Model: {self._get_model_path()}")
        print(f"Epochs: {train_args['epochs']}")
        print(f"Batch size: {train_args['batch']}")
        print(f"Image size: {train_args['imgsz']}")
        print(f"Learning rate: {train_args['lr0']}")
        print(f"Optimizer: {train_args['optimizer']}")
        print(f"Device: {train_args['device'] or 'auto'}")
        print("="*60 + "\n")
        
        # Start training
        self.results = self.model.train(**train_args)
        
        return self.results
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the path to the best model weights."""
        if self.results is None:
            return None
        
        output_cfg = self.config.get('output', {})
        project = output_cfg.get('project', 'outputs/models')
        name = output_cfg.get('name', 'ship_detection')
        
        best_path = Path(project) / name / 'weights' / 'best.pt'
        if best_path.exists():
            return str(best_path)
        
        return None
    
    def export_model(
        self,
        format: str = 'onnx',
        weights_path: Optional[str] = None
    ) -> str:
        """
        Export the trained model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            weights_path: Path to weights file (uses best if not specified)
            
        Returns:
            Path to exported model
        """
        if weights_path is None:
            weights_path = self.get_best_model_path()
        
        if weights_path is None:
            raise ValueError("No model weights available for export")
        
        model = YOLO(weights_path)
        export_path = model.export(format=format)
        
        print(f"Model exported to: {export_path}")
        return export_path


def get_training_parameters_description() -> str:
    """
    Get a detailed description of all training parameters.
    
    Returns:
        Formatted string with parameter descriptions
    """
    description = """
    =================================================================
    YOLOv8-OBB Training Parameters Description
    =================================================================
    
    MODEL PARAMETERS:
    -----------------
    base: Base model architecture
        Options: yolov8n-obb (nano), yolov8s-obb (small), yolov8m-obb (medium),
                 yolov8l-obb (large), yolov8x-obb (xlarge)
        Larger models = better accuracy but slower training/inference
        
    num_classes: Number of object classes to detect
        For ship-only detection: 1
        
    imgsz: Input image size (must be multiple of 32)
        Common values: 640, 1024, 1280
        Larger size = better small object detection but more memory
    
    TRAINING HYPERPARAMETERS:
    -------------------------
    epochs: Number of training epochs
        Recommended: 100-300 for good convergence
        
    batch: Batch size
        Depends on GPU memory. RTX 3090 (24GB): 16-32
        Larger batch = more stable gradients but more memory
        
    optimizer: Optimization algorithm
        Options: SGD, Adam, AdamW
        AdamW recommended for transformer-based models
        
    lr0: Initial learning rate
        Typical range: 0.001-0.01
        Higher = faster convergence but may overshoot
        
    lrf: Final learning rate factor (lr0 * lrf = final LR)
        Typical: 0.01 (learning rate decays to 1% of initial)
        
    momentum: SGD momentum / Adam beta1
        Typical: 0.937
        Higher = more gradient history used
        
    weight_decay: L2 regularization coefficient
        Typical: 0.0005
        Prevents overfitting
        
    warmup_epochs: Number of warmup epochs
        Typical: 3
        Gradually increases learning rate at start
        
    patience: Early stopping patience
        Training stops if no improvement for N epochs
        
    DETECTION PARAMETERS:
    ---------------------
    iou: IoU threshold for NMS
        Higher = stricter overlap filtering
        
    conf: Confidence threshold for predictions
        Lower during training (0.001), higher during inference
        
    max_det: Maximum detections per image
        Increase for images with many objects
    
    AUGMENTATION PARAMETERS:
    ------------------------
    fliplr: Horizontal flip probability (0-1)
    flipud: Vertical flip probability (0-1)
        Use both for aerial/satellite imagery
        
    hsv_h, hsv_s, hsv_v: HSV color space augmentation
        Helps with varying lighting conditions
        
    degrees: Random rotation range (+/- degrees)
        Use 180 for aerial imagery (objects can be any orientation)
        
    translate: Random translation fraction
    scale: Random scale factor
    
    mosaic: Mosaic augmentation probability
        Combines 4 images into 1 for better small object detection
        
    mixup: MixUp augmentation probability
        Blends two images together
    
    RESOLUTION ADAPTATION:
    ----------------------
    source_resolution: Training data resolution (m/pixel)
        SODA-A: approximately 0.5-1.0 m/pixel
        
    target_resolution: Inference data resolution (m/pixel)
        PlanetScope: 3.0 m/pixel
        
    scale_factor: Ratio for resolution adaptation
        = source_resolution / target_resolution
    
    =================================================================
    """
    return description


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ship detection model")
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="Path to training configuration"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--model", type=str, default="small",
        choices=['nano', 'small', 'medium', 'large', 'xlarge'],
        help="Model variant"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto/cpu/0/0,1)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="Print parameter descriptions and exit"
    )
    
    args = parser.parse_args()
    
    if args.describe:
        print(get_training_parameters_description())
        exit(0)
    
    trainer = ShipDetectionTrainer(
        config_path=args.config,
        model_variant=args.model,
        device=args.device
    )
    
    trainer.train(
        data_yaml=args.data,
        resume=args.resume,
        checkpoint=args.checkpoint
    )
    
    print(f"\nBest model saved at: {trainer.get_best_model_path()}")
