#!/usr/bin/env python3
"""
Script to evaluate trained ship detection model.

This script provides:
- Model evaluation on validation/test sets
- Comprehensive metrics (mAP, Precision, Recall, F1)
- Confusion matrix visualization
- Inference speed benchmarking
- Results export to JSON

Usage:
    python scripts/evaluate.py \
        --model outputs/models/ship_detection/weights/best.pt \
        --data data/processed/dataset.yaml \
        --output outputs/evaluation_results.json
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate.evaluator import ShipDetectionEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ship detection model performance"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights (.pt file)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML configuration"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=['val', 'test'],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for output JSON results"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference speed benchmark"
    )
    parser.add_argument(
        "--benchmark_image", type=str, default=None,
        help="Image path for benchmarking"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Ship Detection Model Evaluation")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Split: {args.split}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    
    # Initialize evaluator
    evaluator = ShipDetectionEvaluator(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(split=args.split, verbose=True)
    
    # Save results if output path provided
    if args.output:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.export_results(args.output)
        
        # Save visualizations
        evaluator.plot_confusion_matrix(
            output_path=str(output_dir / 'confusion_matrix.png')
        )
        evaluator.plot_precision_recall_curve(
            output_path=str(output_dir / 'pr_curve.png')
        )
    
    # Run benchmark if requested
    if args.benchmark and args.benchmark_image:
        print("\n--- Inference Speed Benchmark ---")
        evaluator.benchmark_inference_speed(
            test_images=[args.benchmark_image],
            num_runs=100,
            warmup_runs=10
        )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
