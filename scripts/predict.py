#!/usr/bin/env python3
"""
Script to run ship detection prediction on satellite imagery.

This script supports:
- Large GeoTIFF images (PlanetScope 3m resolution)
- Regular image formats (jpg, png, etc.)
- Sliding window processing for large images
- GeoJSON output with OBB polygons in EPSG:4326

Usage:
    # Predict on GeoTIFF (outputs GeoJSON)
    python scripts/predict.py \
        --model outputs/models/ship_detection/weights/best.pt \
        --input /path/to/satellite_image.tif \
        --output predictions/results.geojson

    # Predict on regular image (outputs visualization)
    python scripts/predict.py \
        --model outputs/models/ship_detection/weights/best.pt \
        --input /path/to/image.jpg \
        --output predictions/result.jpg
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict.predictor import ShipPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Ship detection prediction for satellite imagery"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input image (GeoTIFF or regular image)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for output (GeoJSON for GeoTIFF, image for others)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--tile_size", type=int, default=640,
        help="Tile size for sliding window (default: 640)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.25,
        help="Tile overlap ratio 0-1 (default: 0.25)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Ship Detection Prediction")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Tile size: {args.tile_size}")
    print(f"  Tile overlap: {args.overlap}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = ShipPredictor(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tile_size=args.tile_size,
        tile_overlap=args.overlap
    )
    
    # Check input file type
    input_path = Path(args.input)
    
    if input_path.suffix.lower() in ['.tif', '.tiff']:
        # Process as GeoTIFF - output GeoJSON
        print(f"\nProcessing GeoTIFF image...")
        stats = predictor.predict_geotiff(
            args.input,
            args.output,
            verbose=not args.quiet
        )
        print(f"\nPrediction completed!")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Output: {args.output}")
    else:
        # Process as regular image - output visualization
        print(f"\nProcessing regular image...")
        detections, _ = predictor.predict_image(
            args.input,
            args.output,
            save_visualization=True
        )
        print(f"\nPrediction completed!")
        print(f"  Total detections: {len(detections)}")
        print(f"  Output: {args.output}")
        
        # Print detection details
        if detections and not args.quiet:
            print(f"\nDetection details:")
            for i, det in enumerate(detections[:10]):  # Show first 10
                print(f"  {i+1}. {det.class_name}: {det.confidence:.3f}")
            if len(detections) > 10:
                print(f"  ... and {len(detections) - 10} more")


if __name__ == "__main__":
    main()
