"""
Ship Detection Predictor for Large GeoTIFF Images

This module provides the prediction pipeline for ship detection on large
satellite GeoTIFF images, specifically designed for PlanetScope 3m imagery.

Key features:
- Sliding window approach for large image processing
- Coordinate transformation from pixel to geographic (EPSG:4326)
- Non-Maximum Suppression (NMS) for overlapping detections
- GeoJSON output with OBB (Oriented Bounding Box) polygons
- Memory-efficient processing for large raster files

Output GeoJSON Format:
{
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    "features": [
        {
            "type": "Feature",
            "id": "ship_001",
            "properties": {
                "confidence": 0.95,
                "class": "ship"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4], [lon1,lat1]]]
            }
        }
    ]
}
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import rowcol, xy
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. GeoTIFF support limited.")

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from ultralytics import YOLO


@dataclass
class Detection:
    """Data class for a single detection."""
    bbox: np.ndarray  # OBB coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
    confidence: float
    class_id: int
    class_name: str
    
    def to_polygon(self) -> List[Tuple[float, float]]:
        """Convert OBB to polygon coordinates."""
        return [
            (self.bbox[0], self.bbox[1]),
            (self.bbox[2], self.bbox[3]),
            (self.bbox[4], self.bbox[5]),
            (self.bbox[6], self.bbox[7]),
            (self.bbox[0], self.bbox[1])  # Close the polygon
        ]


class ShipPredictor:
    """
    Predictor for ship detection on large GeoTIFF satellite images.
    
    This class handles:
    - Large image processing with sliding windows
    - Coordinate transformation to geographic CRS
    - NMS for overlapping detections
    - GeoJSON output generation
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tile_size: int = 640,
        tile_overlap: float = 0.25
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained YOLOv8-OBB model weights
            device: Device for inference ('auto', 'cpu', '0', etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            tile_size: Size of sliding window tiles (pixels)
            tile_overlap: Overlap ratio between tiles (0-1)
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        self.model = None
        self.class_names = None
        
    def load_model(self) -> YOLO:
        """Load the trained model."""
        print(f"Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = {0: 'ship'}
        
        print(f"Model loaded. Classes: {self.class_names}")
        return self.model
    
    def _generate_tiles(
        self,
        image_width: int,
        image_height: int
    ) -> Generator[Tuple[int, int, int, int], None, None]:
        """
        Generate sliding window tile coordinates.
        
        Args:
            image_width: Total image width
            image_height: Total image height
            
        Yields:
            Tuple of (x_start, y_start, tile_width, tile_height)
        """
        stride = int(self.tile_size * (1 - self.tile_overlap))
        
        for y in range(0, image_height, stride):
            for x in range(0, image_width, stride):
                # Calculate tile dimensions (handle edge tiles)
                tile_w = min(self.tile_size, image_width - x)
                tile_h = min(self.tile_size, image_height - y)
                
                yield x, y, tile_w, tile_h
    
    def _process_tile(
        self,
        tile_image: np.ndarray,
        x_offset: int,
        y_offset: int
    ) -> List[Detection]:
        """
        Process a single tile and return detections.
        
        Args:
            tile_image: Tile image array (H, W, C)
            x_offset: X offset in original image
            y_offset: Y offset in original image
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Run inference
        results = self.model.predict(
            tile_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        # Check for OBB results
        if hasattr(result, 'obb') and result.obb is not None:
            obbs = result.obb
            
            if obbs.xyxyxyxy is not None and len(obbs.xyxyxyxy) > 0:
                boxes = obbs.xyxyxyxy.cpu().numpy()
                confs = obbs.conf.cpu().numpy()
                classes = obbs.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    # Flatten OBB coordinates and add offset
                    bbox = boxes[i].flatten()
                    # Add offset to convert to full image coordinates
                    bbox[0::2] += x_offset  # x coordinates
                    bbox[1::2] += y_offset  # y coordinates
                    
                    class_id = int(classes[i])
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=float(confs[i]),
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(detection)
        
        return detections
    
    def _nms_obb(
        self,
        detections: List[Detection],
        iou_threshold: float = None
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to OBB detections.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        if not HAS_SHAPELY:
            # Fallback: simple confidence-based filtering
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[:100]  # Keep top 100
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Create Shapely polygons
        polygons = []
        for det in detections:
            try:
                poly = Polygon([
                    (det.bbox[0], det.bbox[1]),
                    (det.bbox[2], det.bbox[3]),
                    (det.bbox[4], det.bbox[5]),
                    (det.bbox[6], det.bbox[7])
                ])
                if poly.is_valid:
                    polygons.append(poly)
                else:
                    # Try to fix invalid polygon
                    poly = poly.buffer(0)
                    polygons.append(poly)
            except Exception:
                polygons.append(None)
        
        # NMS
        keep = []
        suppressed = set()
        
        for i, det in enumerate(detections):
            if i in suppressed:
                continue
            
            keep.append(det)
            
            if polygons[i] is None:
                continue
            
            for j in range(i + 1, len(detections)):
                if j in suppressed or polygons[j] is None:
                    continue
                
                # Calculate IoU
                try:
                    intersection = polygons[i].intersection(polygons[j]).area
                    union = polygons[i].union(polygons[j]).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > iou_threshold:
                        suppressed.add(j)
                except Exception:
                    continue
        
        return keep
    
    def _pixel_to_geo(
        self,
        pixel_coords: np.ndarray,
        transform: Any,
        src_crs: Any
    ) -> np.ndarray:
        """
        Convert pixel coordinates to geographic coordinates (EPSG:4326).
        
        Args:
            pixel_coords: Array of pixel coordinates [x1,y1,x2,y2,...]
            transform: Rasterio affine transform
            src_crs: Source CRS
            
        Returns:
            Array of geographic coordinates [lon1,lat1,lon2,lat2,...]
        """
        if not HAS_RASTERIO:
            return pixel_coords
        
        geo_coords = np.zeros_like(pixel_coords)
        
        # Create transformer if needed
        if HAS_PYPROJ and src_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(
                src_crs, 
                CRS.from_epsg(4326), 
                always_xy=True
            )
            needs_transform = True
        else:
            needs_transform = False
        
        # Convert each coordinate pair
        for i in range(0, len(pixel_coords), 2):
            px, py = pixel_coords[i], pixel_coords[i+1]
            
            # Pixel to projected coordinates
            x, y = xy(transform, int(py), int(px))
            
            # Project to EPSG:4326 if needed
            if needs_transform:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            
            geo_coords[i] = lon
            geo_coords[i+1] = lat
        
        return geo_coords
    
    def predict_geotiff(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run prediction on a GeoTIFF image and save results as GeoJSON.
        
        Args:
            input_path: Path to input GeoTIFF file
            output_path: Path for output GeoJSON file
            verbose: Whether to print progress
            
        Returns:
            Dictionary with prediction statistics
        """
        if self.model is None:
            self.load_model()
        
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for GeoTIFF processing")
        
        print(f"\nProcessing: {input_path}")
        
        stats = {
            'input_file': input_path,
            'output_file': output_path,
            'total_detections': 0,
            'processing_time': None
        }
        
        start_time = datetime.now()
        all_detections = []
        
        # Open GeoTIFF
        with rasterio.open(input_path) as src:
            image_width = src.width
            image_height = src.height
            transform = src.transform
            src_crs = src.crs
            
            print(f"Image size: {image_width} x {image_height}")
            print(f"CRS: {src_crs}")
            
            # Generate tiles
            tiles = list(self._generate_tiles(image_width, image_height))
            
            if verbose:
                print(f"Processing {len(tiles)} tiles...")
            
            # Process each tile
            for x, y, w, h in tqdm(tiles, desc="Processing tiles", 
                                    disable=not verbose):
                # Read tile
                window = Window(x, y, w, h)
                tile_data = src.read(window=window)
                
                # Handle different band configurations
                if tile_data.shape[0] == 1:
                    # Single band - convert to RGB
                    tile_image = np.stack([tile_data[0]] * 3, axis=-1)
                elif tile_data.shape[0] >= 3:
                    # Multi-band - use first 3 bands as RGB
                    tile_image = np.transpose(tile_data[:3], (1, 2, 0))
                else:
                    # 2 bands - use first band as grayscale
                    tile_image = np.stack([tile_data[0]] * 3, axis=-1)
                
                # Normalize to 0-255 if needed
                if tile_image.dtype != np.uint8:
                    tile_min = tile_image.min()
                    tile_max = tile_image.max()
                    if tile_max > tile_min:
                        tile_image = ((tile_image - tile_min) / 
                                     (tile_max - tile_min) * 255).astype(np.uint8)
                    else:
                        tile_image = np.zeros_like(tile_image, dtype=np.uint8)
                
                # Process tile
                tile_detections = self._process_tile(tile_image, x, y)
                all_detections.extend(tile_detections)
        
        # Apply NMS to all detections
        if verbose:
            print(f"Applying NMS to {len(all_detections)} detections...")
        
        final_detections = self._nms_obb(all_detections)
        
        # Convert to GeoJSON
        if verbose:
            print(f"Converting {len(final_detections)} detections to GeoJSON...")
        
        geojson = self._create_geojson(
            final_detections, 
            transform, 
            src_crs
        )
        
        # Save GeoJSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        # Update stats
        stats['total_detections'] = len(final_detections)
        stats['processing_time'] = str(datetime.now() - start_time)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Processing time: {stats['processing_time']}")
        
        return stats
    
    def _create_geojson(
        self,
        detections: List[Detection],
        transform: Any,
        src_crs: Any
    ) -> Dict[str, Any]:
        """
        Create GeoJSON FeatureCollection from detections.
        
        Args:
            detections: List of Detection objects
            transform: Rasterio affine transform
            src_crs: Source CRS
            
        Returns:
            GeoJSON dictionary
        """
        features = []
        
        for i, det in enumerate(detections):
            # Convert pixel coordinates to geographic
            geo_coords = self._pixel_to_geo(det.bbox, transform, src_crs)
            
            # Create polygon coordinates (GeoJSON format: [lon, lat])
            polygon_coords = [
                [geo_coords[0], geo_coords[1]],
                [geo_coords[2], geo_coords[3]],
                [geo_coords[4], geo_coords[5]],
                [geo_coords[6], geo_coords[7]],
                [geo_coords[0], geo_coords[1]]  # Close polygon
            ]
            
            feature = {
                "type": "Feature",
                "id": f"ship_{i+1:04d}",
                "properties": {
                    "confidence": round(det.confidence, 4),
                    "class": det.class_name
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                }
            }
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "EPSG:4326"
                }
            },
            "features": features
        }
        
        return geojson
    
    def predict_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        save_visualization: bool = True
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Run prediction on a regular image file (non-GeoTIFF).
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            save_visualization: Whether to save visualization
            
        Returns:
            Tuple of (detections list, visualization image)
        """
        if self.model is None:
            self.load_model()
        
        # Load image
        image = np.array(Image.open(image_path))
        
        # Check if image is large enough to need tiling
        if image.shape[0] > self.tile_size * 2 or image.shape[1] > self.tile_size * 2:
            # Use tiling approach
            all_detections = []
            tiles = list(self._generate_tiles(image.shape[1], image.shape[0]))
            
            for x, y, w, h in tqdm(tiles, desc="Processing tiles"):
                tile = image[y:y+h, x:x+w]
                tile_detections = self._process_tile(tile, x, y)
                all_detections.extend(tile_detections)
            
            detections = self._nms_obb(all_detections)
        else:
            # Direct inference
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'obb') and result.obb is not None:
                    obbs = result.obb
                    if obbs.xyxyxyxy is not None:
                        boxes = obbs.xyxyxyxy.cpu().numpy()
                        confs = obbs.conf.cpu().numpy()
                        classes = obbs.cls.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            detection = Detection(
                                bbox=boxes[i].flatten(),
                                confidence=float(confs[i]),
                                class_id=int(classes[i]),
                                class_name=self.class_names.get(
                                    int(classes[i]), 'ship'
                                )
                            )
                            detections.append(detection)
        
        # Create visualization
        vis_image = None
        if save_visualization and output_path:
            vis_image = self._draw_detections(image.copy(), detections)
            Image.fromarray(vis_image).save(output_path)
            print(f"Visualization saved to: {output_path}")
        
        return detections, vis_image
    
    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image
            detections: List of detections
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        import cv2
        
        for det in detections:
            # Get polygon points
            points = np.array([
                [det.bbox[0], det.bbox[1]],
                [det.bbox[2], det.bbox[3]],
                [det.bbox[4], det.bbox[5]],
                [det.bbox[6], det.bbox[7]]
            ], dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(image, [points], True, color, thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(
                image, label,
                (int(det.bbox[0]), int(det.bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return image


def main():
    """Main function for command-line usage."""
    import argparse
    
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
        help="Path for output file (GeoJSON for GeoTIFF, image for others)"
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
        help="Tile overlap ratio (default: 0.25)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
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
        # Process as GeoTIFF
        predictor.predict_geotiff(args.input, args.output)
    else:
        # Process as regular image
        detections, _ = predictor.predict_image(
            args.input, 
            args.output,
            save_visualization=True
        )
        print(f"Total detections: {len(detections)}")


if __name__ == "__main__":
    main()
