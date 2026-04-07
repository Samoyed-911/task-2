# Ship Detection from Satellite Imagery

A complete pipeline for training, evaluating, and deploying ship detection models on satellite imagery using the SODA-A dataset and YOLOv8-OBB (Oriented Bounding Box).

## Table of Contents

1. [Overview](#overview)
2. [Dataset Information](#dataset-information)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Prediction](#prediction)
8. [Configuration Parameters](#configuration-parameters)
9. [Output Formats](#output-formats)
10. [Resolution Adaptation](#resolution-adaptation)

---

## Overview

This project implements a ship detection system designed for satellite imagery, specifically targeting PlanetScope 3m resolution images. The system uses:

- **Model**: YOLOv8-OBB (Oriented Bounding Box) for rotated object detection
- **Framework**: PyTorch + Ultralytics
- **Dataset**: SODA-A (Small Object Detection for Aerial images)

### Key Features

- Oriented bounding box detection (essential for rotated ships)
- Sliding window processing for large GeoTIFF images
- Geographic coordinate transformation (EPSG:4326)
- GeoJSON output with OBB polygon geometries
- Resolution adaptation between training data and inference data

---

## Dataset Information

### SODA-A Dataset

The SODA-A (Small Object Detection for Aerial images) dataset is used for training:

- **Source**: [Kaggle - SODA-A Dataset](https://www.kaggle.com/datasets/shubham6147/soda-a-small-object-detection-dataset-aerial)
- **Format**: DOTA-style oriented bounding box annotations
- **Resolution**: Approximately 0.5-1.0 meters per pixel (aerial imagery)

### SODA-A Classes

| Class | Description |
|-------|-------------|
| airplane | Fixed-wing aircraft |
| helicopter | Rotary-wing aircraft |
| small-vehicle | Cars, motorcycles, etc. |
| large-vehicle | Trucks, buses, etc. |
| **ship** | Vessels (primary target) |
| container | Shipping containers |
| storage-tank | Storage facilities |
| swimming-pool | Pools |
| windmill | Wind turbines |

### Ship Class Statistics (varies by dataset version)

For ship detection, the relevant class is `ship`. Typical statistics:
- Ship objects vary in size from small boats (20m) to large vessels (400m)
- Ships appear at various orientations (0-360 degrees)
- Resolution affects apparent ship size in pixels

### Annotation Format

SODA-A uses DOTA format (oriented bounding boxes):
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

Where (x1,y1), (x2,y2), (x3,y3), (x4,y4) are the four corners of the rotated bounding box.

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for dataset and models

### Setup

```bash
# Clone or create project directory
cd vega-star-task2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `torch>=2.0.0` - Deep learning framework
- `ultralytics>=8.1.0` - YOLOv8 implementation
- `rasterio>=1.3.0` - GeoTIFF processing
- `shapely>=2.0.0` - Geometric operations
- `pyproj>=3.6.0` - Coordinate transformations
- `geojson>=3.0.0` - GeoJSON handling

---

## Data Preprocessing

### Step 1: Analyze Dataset

First, analyze the SODA-A dataset to understand its structure:

```bash
python scripts/analyze_dataset.py \
    --data_root /path/to/soda-a \
    --output outputs/dataset_stats.json
```

This will output:
- Total image and object counts
- Class distribution statistics
- Ship-specific statistics
- Estimated image resolution
- Bounding box size distributions

### Step 2: Preprocess Data

Convert SODA-A to YOLO-OBB format:

```bash
# Ship-only detection (recommended)
python scripts/preprocess_data.py \
    --source /path/to/soda-a \
    --output data/processed \
    --classes ship \
    --train_ratio 0.8 \
    --seed 42

# Multi-class detection
python scripts/preprocess_data.py \
    --source /path/to/soda-a \
    --output data/processed \
    --classes ship airplane large-vehicle \
    --train_ratio 0.8
```

### Preprocessing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Path to SODA-A dataset | Required |
| `--output` | Output directory for processed data | Required |
| `--classes` | Classes to include | ship |
| `--train_ratio` | Training data ratio | 0.8 |
| `--include_empty` | Include images without targets | False |
| `--adapt_resolution` | Apply resolution scaling | False |
| `--seed` | Random seed | 42 |

### Output Structure

```
data/processed/
    train/
        images/
        labels/
    val/
        images/
        labels/
    dataset.yaml
```

---

## Training

### Basic Training

```bash
python scripts/train.py \
    --config config/train_config.yaml \
    --data data/processed/dataset.yaml \
    --model small \
    --device 0
```

### Training with Custom Configuration

```bash
python scripts/train.py \
    --config config/train_config.yaml \
    --data data/processed/dataset.yaml \
    --model medium \
    --device 0,1  # Multi-GPU
```

### Resume Training

```bash
python scripts/train.py \
    --config config/train_config.yaml \
    --data data/processed/dataset.yaml \
    --resume \
    --checkpoint outputs/models/ship_detection/weights/last.pt
```

### Model Variants

| Variant | Model | Parameters | Speed | Accuracy |
|---------|-------|------------|-------|----------|
| nano | yolov8n-obb | 3.2M | Fastest | Lowest |
| small | yolov8s-obb | 11.2M | Fast | Good |
| medium | yolov8m-obb | 25.9M | Medium | Better |
| large | yolov8l-obb | 43.7M | Slow | High |
| xlarge | yolov8x-obb | 68.2M | Slowest | Highest |

**Recommendation**: Start with `small` for initial experiments, use `medium` or `large` for production.

---

## Evaluation

### Run Evaluation

```bash
python scripts/evaluate.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --data data/processed/dataset.yaml \
    --split val \
    --output outputs/evaluation_results.json
```

### Evaluation Metrics

The evaluation provides:

- **mAP@50**: Mean Average Precision at IoU=0.5
- **mAP@50-95**: Mean AP averaged over IoU thresholds 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Inference Speed Benchmark

```bash
python scripts/evaluate.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --data data/processed/dataset.yaml \
    --benchmark \
    --benchmark_image path/to/test_image.jpg
```

---

## Prediction

### Predict on GeoTIFF (Satellite Imagery)

For PlanetScope 3m resolution GeoTIFF images:

```bash
python scripts/predict.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --input /path/to/satellite_image.tif \
    --output predictions/ships.geojson \
    --conf 0.25 \
    --tile_size 640 \
    --overlap 0.25
```

### Predict on Regular Images

```bash
python scripts/predict.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --input /path/to/image.jpg \
    --output predictions/result.jpg
```

### Prediction Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to model weights | Required |
| `--input` | Input image path | Required |
| `--output` | Output path (GeoJSON or image) | Required |
| `--conf` | Confidence threshold | 0.25 |
| `--iou` | IoU threshold for NMS | 0.45 |
| `--tile_size` | Sliding window tile size | 640 |
| `--overlap` | Tile overlap ratio (0-1) | 0.25 |
| `--device` | Inference device | auto |

### Sliding Window Processing

For large satellite images, the predictor uses a sliding window approach:

1. Image is divided into overlapping tiles
2. Each tile is processed independently
3. Detections are merged using NMS
4. Coordinates are transformed to geographic CRS

---

## Configuration Parameters

### Training Configuration (config/train_config.yaml)

```yaml
# Model Configuration
model:
  base: "yolov8s-obb"    # Base model architecture
  num_classes: 1          # Number of detection classes
  imgsz: 640              # Input image size (multiple of 32)

# Training Hyperparameters
training:
  epochs: 100             # Number of training epochs
  batch: 16               # Batch size (adjust for GPU memory)
  optimizer: "AdamW"      # Optimizer: SGD, Adam, AdamW
  lr0: 0.001              # Initial learning rate
  lrf: 0.01               # Final LR factor (final_lr = lr0 * lrf)
  momentum: 0.937         # SGD momentum / Adam beta1
  weight_decay: 0.0005    # L2 regularization
  warmup_epochs: 3        # Learning rate warmup
  patience: 50            # Early stopping patience
  amp: true               # Automatic Mixed Precision

# Data Augmentation
augmentation:
  fliplr: 0.5             # Horizontal flip probability
  flipud: 0.5             # Vertical flip probability
  hsv_h: 0.015            # HSV-Hue augmentation
  hsv_s: 0.7              # HSV-Saturation augmentation
  hsv_v: 0.4              # HSV-Value augmentation
  degrees: 180            # Random rotation (important for ships)
  translate: 0.1          # Random translation
  scale: 0.5              # Random scale
  mosaic: 1.0             # Mosaic augmentation
```

### Parameter Descriptions

To see detailed parameter descriptions:

```bash
python scripts/train.py --describe_params
```

---

## Output Formats

### GeoJSON Output Structure

The prediction pipeline outputs GeoJSON with the following structure:

```json
{
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "EPSG:4326"
    }
  },
  "features": [
    {
      "type": "Feature",
      "id": "ship_0001",
      "properties": {
        "confidence": 0.9523,
        "class": "ship"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [longitude1, latitude1],
            [longitude2, latitude2],
            [longitude3, latitude3],
            [longitude4, latitude4],
            [longitude1, latitude1]
          ]
        ]
      }
    }
  ]
}
```

### Output Fields

| Field | Description |
|-------|-------------|
| `id` | Unique detection identifier |
| `confidence` | Detection confidence score (0-1) |
| `class` | Object class name |
| `geometry` | OBB polygon in EPSG:4326 |

---

## Resolution Adaptation

### Understanding Resolution Differences

| Data Source | Resolution | Notes |
|-------------|------------|-------|
| SODA-A | 0.5-1.0 m/pixel | Training data (aerial) |
| PlanetScope | 3.0 m/pixel | Inference target |

### Implications

- Ships appear **smaller** in 3m imagery compared to training data
- Small ships may be harder to detect
- Model may need fine-tuning for optimal performance

### Adaptation Strategies

1. **Training augmentation**: Apply scale augmentation to simulate lower resolution
2. **Tile size adjustment**: Use larger tiles to capture more context
3. **Confidence tuning**: Adjust thresholds based on validation performance

### Enable Resolution Adaptation

```bash
python scripts/preprocess_data.py \
    --source /path/to/soda-a \
    --output data/processed_adapted \
    --adapt_resolution \
    --source_resolution 0.8 \
    --target_resolution 3.0
```

---

## Project Structure

```
vega-star-task2/
    README.md                 # This file
    requirements.txt          # Python dependencies
    config/
        train_config.yaml     # Training configuration
    src/
        __init__.py
        data/
            __init__.py
            dataset_analysis.py   # Dataset analysis module
            preprocess.py         # Data preprocessing module
        train/
            __init__.py
            trainer.py            # Training pipeline
        evaluate/
            __init__.py
            evaluator.py          # Evaluation module
        predict/
            __init__.py
            predictor.py          # Prediction pipeline
    scripts/
        analyze_dataset.py    # Dataset analysis script
        preprocess_data.py    # Preprocessing script
        train.py              # Training script
        evaluate.py           # Evaluation script
        predict.py            # Prediction script
    outputs/
        models/               # Trained model weights
        predictions/          # Prediction results
        logs/                 # Training logs
```

---

## Quick Start

### Complete Pipeline Example

```bash
# 1. Analyze dataset
python scripts/analyze_dataset.py \
    --data_root /path/to/soda-a \
    --output outputs/stats.json

# 2. Preprocess data
python scripts/preprocess_data.py \
    --source /path/to/soda-a \
    --output data/processed \
    --classes ship

# 3. Train model
python scripts/train.py \
    --data data/processed/dataset.yaml \
    --model small \
    --device 0

# 4. Evaluate model
python scripts/evaluate.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --data data/processed/dataset.yaml \
    --output outputs/eval_results.json

# 5. Run prediction
python scripts/predict.py \
    --model outputs/models/ship_detection/weights/best.pt \
    --input satellite_image.tif \
    --output predictions/ships.geojson
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **No detections**: Lower confidence threshold
3. **GeoTIFF read error**: Ensure rasterio is properly installed
4. **Coordinate issues**: Verify input CRS matches expectations

### GPU Memory Requirements

| Model | Batch Size | GPU Memory |
|-------|------------|------------|
| nano | 32 | 8GB |
| small | 16 | 12GB |
| medium | 8 | 16GB |
| large | 4 | 24GB |

---

## License

This project is provided for educational and research purposes.

## References

- [SODA-A Dataset](https://www.kaggle.com/datasets/shubham6147/soda-a-small-object-detection-dataset-aerial)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DOTA Dataset Format](https://captain-whu.github.io/DOTA/)
