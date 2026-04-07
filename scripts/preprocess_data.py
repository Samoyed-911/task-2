#!/usr/bin/env python3
"""
Script to preprocess SODA-A dataset for YOLO-OBB training.

This script converts SODA-A dataset from DOTA format to YOLO-OBB format:
- Converts annotation format
- Filters ship-related classes
- Creates train/val splits
- Generates dataset YAML configuration

Usage:
    python scripts/preprocess_data.py \
        --source /path/to/soda-a \
        --output /path/to/processed \
        --classes ship \
        --train_ratio 0.8
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import SODAAPreprocessor, ResolutionAdapter


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SODA-A dataset for YOLO-OBB training"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to original SODA-A dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for processed dataset output"
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Classes to include (default: ship only). "
             "Use 'all' for all SODA-A classes."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Ratio of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--include_empty", action="store_true",
        help="Include images with no target annotations"
    )
    parser.add_argument(
        "--adapt_resolution", action="store_true",
        help="Apply resolution adaptation for PlanetScope 3m imagery"
    )
    parser.add_argument(
        "--source_resolution", type=float, default=0.8,
        help="Source data resolution in m/pixel (default: 0.8)"
    )
    parser.add_argument(
        "--target_resolution", type=float, default=3.0,
        help="Target resolution in m/pixel (default: 3.0 for PlanetScope)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SODA-A Dataset Preprocessor")
    print("="*60)
    
    # Handle 'all' classes option
    classes = args.classes
    if classes and 'all' in classes:
        classes = [
            "airplane", "helicopter", "small-vehicle", "large-vehicle",
            "ship", "container", "storage-tank", "swimming-pool", "windmill"
        ]
    
    # Calculate scale factor if resolution adaptation is enabled
    scale_factor = 1.0
    if args.adapt_resolution:
        adapter = ResolutionAdapter(
            source_resolution=args.source_resolution,
            target_resolution=args.target_resolution
        )
        scale_factor = adapter.get_scale_factor()
        print(f"\nResolution adaptation enabled:")
        print(f"  Source resolution: {args.source_resolution} m/pixel")
        print(f"  Target resolution: {args.target_resolution} m/pixel")
        print(f"  Scale factor: {scale_factor:.3f}")
    
    # Initialize preprocessor
    preprocessor = SODAAPreprocessor(
        source_root=args.source,
        output_root=args.output,
        classes=classes,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"\nSource: {args.source}")
    print(f"Output: {args.output}")
    print(f"Classes: {preprocessor.classes}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Include empty images: {args.include_empty}")
    
    # Run preprocessing
    preprocessor.process_dataset(
        include_empty=args.include_empty,
        scale_factor=scale_factor,
        verbose=True
    )
    
    print(f"\nDataset prepared at: {args.output}")
    print(f"Dataset YAML: {args.output}/dataset.yaml")


if __name__ == "__main__":
    main()
