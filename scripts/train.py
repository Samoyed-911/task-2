#!/usr/bin/env python3
"""
Script to train ship detection model using YOLOv8-OBB.

This script provides:
- Model initialization and training
- Configurable hyperparameters via YAML config
- Training progress monitoring
- Checkpoint management

Usage:
    python scripts/train.py \
        --config config/train_config.yaml \
        --data data/processed/dataset.yaml \
        --model small \
        --device 0

    # Resume training
    python scripts/train.py \
        --config config/train_config.yaml \
        --data data/processed/dataset.yaml \
        --resume \
        --checkpoint outputs/models/ship_detection/weights/last.pt
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.trainer import ShipDetectionTrainer, get_training_parameters_description


def main():
    parser = argparse.ArgumentParser(
        description="Train ship detection model with YOLOv8-OBB"
    )
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--model", type=str, default="small",
        choices=['nano', 'small', 'medium', 'large', 'xlarge'],
        help="Model variant (default: small)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'auto', 'cpu', '0', '0,1' for multi-GPU"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--describe_params", action="store_true",
        help="Print detailed parameter descriptions and exit"
    )
    
    args = parser.parse_args()
    
    # Print parameter descriptions if requested
    if args.describe_params:
        print(get_training_parameters_description())
        return
    
    print("="*60)
    print("Ship Detection Model Training")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Dataset: {args.data}")
    print(f"  Model variant: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Resume: {args.resume}")
    
    # Initialize trainer
    trainer = ShipDetectionTrainer(
        config_path=args.config,
        model_variant=args.model,
        device=args.device
    )
    
    # Start training
    trainer.train(
        data_yaml=args.data,
        resume=args.resume,
        checkpoint=args.checkpoint
    )
    
    # Print results
    best_model = trainer.get_best_model_path()
    if best_model:
        print(f"\nTraining completed!")
        print(f"Best model saved at: {best_model}")


if __name__ == "__main__":
    main()
