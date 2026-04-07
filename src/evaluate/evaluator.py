"""
Ship Detection Evaluator

This module provides comprehensive evaluation functionality for ship detection
models, including:
- Standard metrics (mAP, Precision, Recall, F1-score)
- Per-class performance analysis
- Confusion matrix generation
- Visualization of results

Evaluation is based on the COCO evaluation protocol with OBB (Oriented Bounding Box)
specific IoU calculations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ShipDetectionEvaluator:
    """
    Evaluator for ship detection model performance.
    
    This class provides comprehensive evaluation metrics and visualizations
    for YOLOv8-OBB models trained on ship detection tasks.
    """
    
    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model weights (.pt file)
            data_yaml: Path to dataset YAML configuration
            device: Device to use for inference
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.model = None
        self.results = None
        self.class_names = None
        
    def load_model(self) -> YOLO:
        """Load the trained model."""
        self.model = YOLO(self.model_path)
        
        # Get class names from model
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        
        return self.model
    
    def evaluate(
        self,
        split: str = 'val',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset split.
        
        Args:
            split: Dataset split to evaluate ('val' or 'test')
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            self.load_model()
        
        print(f"\nEvaluating model on {split} split...")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        
        # Run validation
        self.results = self.model.val(
            data=self.data_yaml,
            split=split,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device if self.device != 'auto' else None,
            verbose=verbose
        )
        
        # Extract metrics
        metrics = self._extract_metrics()
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _extract_metrics(self) -> Dict[str, Any]:
        """Extract metrics from validation results."""
        if self.results is None:
            return {}
        
        metrics = {
            'box': {},
            'per_class': {},
            'confusion_matrix': None
        }
        
        # Overall box metrics
        if hasattr(self.results, 'box'):
            box = self.results.box
            metrics['box'] = {
                'mAP50': float(box.map50) if hasattr(box, 'map50') else None,
                'mAP50-95': float(box.map) if hasattr(box, 'map') else None,
                'precision': float(box.mp) if hasattr(box, 'mp') else None,
                'recall': float(box.mr) if hasattr(box, 'mr') else None,
            }
            
            # Calculate F1 score
            if metrics['box']['precision'] and metrics['box']['recall']:
                p, r = metrics['box']['precision'], metrics['box']['recall']
                metrics['box']['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        # Per-class metrics
        if hasattr(self.results, 'box') and hasattr(self.results.box, 'ap_class_index'):
            ap_class_index = self.results.box.ap_class_index
            ap50 = self.results.box.ap50
            
            if self.class_names:
                for i, class_idx in enumerate(ap_class_index):
                    class_name = self.class_names.get(int(class_idx), f'class_{class_idx}')
                    metrics['per_class'][class_name] = {
                        'AP50': float(ap50[i]) if i < len(ap50) else None
                    }
        
        # Confusion matrix
        if hasattr(self.results, 'confusion_matrix'):
            cm = self.results.confusion_matrix
            if hasattr(cm, 'matrix'):
                metrics['confusion_matrix'] = cm.matrix.tolist()
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print formatted evaluation metrics."""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        
        print("\n--- Overall Metrics ---")
        if 'box' in metrics:
            box = metrics['box']
            print(f"  mAP@50:      {box.get('mAP50', 'N/A'):.4f}" 
                  if box.get('mAP50') else "  mAP@50:      N/A")
            print(f"  mAP@50-95:   {box.get('mAP50-95', 'N/A'):.4f}"
                  if box.get('mAP50-95') else "  mAP@50-95:   N/A")
            print(f"  Precision:   {box.get('precision', 'N/A'):.4f}"
                  if box.get('precision') else "  Precision:   N/A")
            print(f"  Recall:      {box.get('recall', 'N/A'):.4f}"
                  if box.get('recall') else "  Recall:      N/A")
            print(f"  F1-Score:    {box.get('f1', 'N/A'):.4f}"
                  if box.get('f1') else "  F1-Score:    N/A")
        
        if metrics.get('per_class'):
            print("\n--- Per-Class AP@50 ---")
            for class_name, class_metrics in metrics['per_class'].items():
                ap50 = class_metrics.get('AP50')
                print(f"  {class_name}: {ap50:.4f}" if ap50 else f"  {class_name}: N/A")
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot and save the confusion matrix.
        
        Args:
            output_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if self.results is None:
            print("No evaluation results. Run evaluate() first.")
            return
        
        if not hasattr(self.results, 'confusion_matrix'):
            print("No confusion matrix available.")
            return
        
        cm = self.results.confusion_matrix
        if not hasattr(cm, 'matrix'):
            print("Confusion matrix data not available.")
            return
        
        matrix = cm.matrix
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get class names
        if self.class_names:
            labels = [self.class_names.get(i, f'class_{i}') 
                     for i in range(len(matrix) - 1)]  # -1 for background
            labels.append('background')
        else:
            labels = [f'class_{i}' for i in range(len(matrix))]
        
        # Plot heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot precision-recall curve.
        
        Args:
            output_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if self.results is None:
            print("No evaluation results. Run evaluate() first.")
            return
        
        # The PR curve is typically saved by YOLO during validation
        # Check if it exists in the results directory
        if hasattr(self.results, 'save_dir'):
            pr_curve_path = Path(self.results.save_dir) / 'PR_curve.png'
            if pr_curve_path.exists():
                print(f"PR curve available at: {pr_curve_path}")
                if output_path:
                    import shutil
                    shutil.copy(pr_curve_path, output_path)
                    print(f"Copied to: {output_path}")
                return
        
        print("PR curve not available from validation results.")
    
    def benchmark_inference_speed(
        self,
        test_images: List[str],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            test_images: List of paths to test images
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Dictionary with timing statistics
        """
        if self.model is None:
            self.load_model()
        
        import time
        
        # Ensure we have at least one image
        if not test_images:
            print("No test images provided.")
            return {}
        
        # Use first image for benchmarking
        test_image = test_images[0]
        
        print(f"Benchmarking inference speed...")
        print(f"Test image: {test_image}")
        print(f"Warmup runs: {warmup_runs}")
        print(f"Timing runs: {num_runs}")
        
        # Warmup
        for _ in range(warmup_runs):
            self.model.predict(test_image, verbose=False)
        
        # Timing runs
        times = []
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start = time.perf_counter()
            self.model.predict(test_image, verbose=False)
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        
        results = {
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'fps': float(1.0 / np.mean(times))
        }
        
        print("\n--- Inference Speed ---")
        print(f"  Mean: {results['mean_time_ms']:.2f} ms")
        print(f"  Std:  {results['std_time_ms']:.2f} ms")
        print(f"  Min:  {results['min_time_ms']:.2f} ms")
        print(f"  Max:  {results['max_time_ms']:.2f} ms")
        print(f"  FPS:  {results['fps']:.2f}")
        
        return results
    
    def export_results(self, output_path: str):
        """
        Export evaluation results to JSON file.
        
        Args:
            output_path: Path for output JSON file
        """
        if self.results is None:
            print("No evaluation results. Run evaluate() first.")
            return
        
        metrics = self._extract_metrics()
        
        export_data = {
            'model_path': str(self.model_path),
            'data_yaml': str(self.data_yaml),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'metrics': metrics
        }
        
        # Remove non-serializable items
        if 'confusion_matrix' in export_data['metrics']:
            if export_data['metrics']['confusion_matrix'] is not None:
                export_data['metrics']['confusion_matrix'] = [
                    [float(x) for x in row] 
                    for row in export_data['metrics']['confusion_matrix']
                ]
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {output_path}")


def calculate_obb_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two oriented bounding boxes.
    
    Args:
        box1: First OBB as [x1,y1,x2,y2,x3,y3,x4,y4]
        box2: Second OBB as [x1,y1,x2,y2,x3,y3,x4,y4]
        
    Returns:
        IoU value
    """
    try:
        from shapely.geometry import Polygon
        
        poly1 = Polygon([
            (box1[0], box1[1]),
            (box1[2], box1[3]),
            (box1[4], box1[5]),
            (box1[6], box1[7])
        ])
        
        poly2 = Polygon([
            (box2[0], box2[1]),
            (box2[2], box2[3]),
            (box2[4], box2[5]),
            (box2[6], box2[7])
        ])
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
        
    except Exception:
        return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ship detection model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=['val', 'test'],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON"
    )
    
    args = parser.parse_args()
    
    evaluator = ShipDetectionEvaluator(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    metrics = evaluator.evaluate(split=args.split)
    
    if args.output:
        evaluator.export_results(args.output)
        
        # Save plots
        output_dir = Path(args.output).parent
        evaluator.plot_confusion_matrix(
            output_path=str(output_dir / 'confusion_matrix.png')
        )
