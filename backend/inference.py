"""
Climbing Hold Detection - Inference Script
==========================================

This script loads a trained Faster R-CNN model and performs inference
on climbing wall images to detect holds.

Usage:
    python inference.py --model path/to/model.pth --image path/to/image.jpg
    python inference.py --model path/to/model.pth --video path/to/video.mp4
    python inference.py --model path/to/model.pth --webcam
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
import argparse
from pathlib import Path
import time


class ClimbingHoldDetector:
    """Wrapper class for climbing hold detection model."""
    
    def __init__(self, model_path, confidence_threshold=0.5, device=None, max_display_size=(1100, 800)):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained model (.pth file)
            confidence_threshold: Minimum confidence for detections (0-1)
            device: 'cuda', 'cpu', or None (auto-detect)
            max_display_size: Maximum (width, height) for display, None to disable scaling
        """
        self.confidence_threshold = confidence_threshold
        self.max_display_size = max_display_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Confidence threshold: {confidence_threshold}")
        if max_display_size:
            print(f"Max display size: {max_display_size[0]}x{max_display_size[1]}")
    
    def _load_model(self, model_path):
        """Load the trained Faster R-CNN model."""
        # Create model architecture
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 2  # background + climbing hold
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            torch.Tensor (3, H, W) normalized to [0, 1]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return image_tensor
    
    def detect(self, image):
        """
        Detect climbing holds in an image.
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            List of detections, each as dict with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - center: (cx, cy)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Filter by confidence
        scores = predictions['scores'].cpu().numpy()
        boxes = predictions['boxes'].cpu().numpy()
        
        detections = []
        for score, box in zip(scores, boxes):
            if score >= self.confidence_threshold:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'center': (float(cx), float(cy))
                })
        
        return detections
    
    def visualize_detections(self, image, detections, show_boxes=True, show_centers=True):
        """
        Draw detections on image.
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            detections: List of detection dicts from detect()
            show_boxes: Whether to draw bounding boxes
            show_centers: Whether to draw center points
            
        Returns:
            Image with detections drawn
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            cx, cy = [int(v) for v in det['center']]
            
            # Color based on confidence (green = high, yellow = medium, red = low)
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            if show_boxes:
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            if show_centers:
                cv2.circle(vis_image, (cx, cy), 5, color, -1)
                cv2.circle(vis_image, (cx, cy), 8, color, 2)
            
            # Draw confidence score
            label = f"{conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(vis_image, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw count
        count_text = f"Holds detected: {len(detections)}"
        cv2.putText(vis_image, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def scale_for_display(self, image):
        """
        Scale image to fit display while maintaining aspect ratio.
        
        Args:
            image: numpy array (H, W, 3)
            
        Returns:
            Scaled image and scale factor
        """
        if self.max_display_size is None:
            return image, 1.0
        
        h, w = image.shape[:2]
        max_w, max_h = self.max_display_size
        
        # Calculate scale factor to fit within max size
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
        
        # No scaling needed if already fits
        if scale >= 1.0:
            return image, 1.0
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return scaled_image, scale


def process_image(detector, image_path, output_path=None, show=True):
    """Process a single image."""
    image_path = "data/img/" + image_path
    print(f"\nProcessing image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect
    start_time = time.time()
    detections = detector.detect(image)
    inference_time = time.time() - start_time
    
    print(f"Found {len(detections)} holds in {inference_time:.3f}s")
    
    # Print detections
    for i, det in enumerate(detections, 1):
        cx, cy = det['center']
        conf = det['confidence']
        print(f"  Hold {i}: center=({cx:.1f}, {cy:.1f}), confidence={conf:.3f}")
    
    # Visualize
    vis_image = detector.visualize_detections(image, detections)
    
    # Save (always save at full resolution)
    if output_path:
        cv2.imwrite(str(output_path), vis_image)
        print(f"Saved visualization to: {output_path}")
    
    # Show (scaled for display)
    if show:
        display_image, scale = detector.scale_for_display(vis_image)
        if scale < 1.0:
            print(f"Scaling for display: {scale:.2f}x ({display_image.shape[1]}x{display_image.shape[0]})")
        
        cv2.imshow("Climbing Hold Detection", display_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections, vis_image



def batch_process_directory(detector, input_dir, output_dir=None):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_detections = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Error: Could not load image")
            continue
        
        # Detect
        detections = detector.detect(image)
        all_detections.append((image_file.name, len(detections)))
        
        print(f"  Found {len(detections)} holds")
        
        # Save visualization
        if output_dir:
            vis_image = detector.visualize_detections(image, detections)
            output_file = output_path / f"{image_file.stem}_detected{image_file.suffix}"
            cv2.imwrite(str(output_file), vis_image)
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    for filename, count in all_detections:
        print(f"{filename}: {count} holds")
    print(f"\nTotal images: {len(all_detections)}")
    print(f"Average holds per image: {sum(c for _, c in all_detections) / len(all_detections):.1f}")
    if output_dir:
        print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Climbing Hold Detection - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --model model.pth --image wall.jpg
  
  # Save output
  python inference.py --model model.pth --image wall.jpg --output result.jpg
  
  # Process video
  python inference.py --model model.pth --video climb.mp4 --output output.mp4
  
  # Webcam
  python inference.py --model model.pth --webcam
  
  # Batch process directory
  python inference.py --model model.pth --directory ./images --output ./results
  
  # Adjust confidence threshold
  python inference.py --model model.pth --image wall.jpg --confidence 0.7
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--image', type=str,
                       help='Path to input image')
    parser.add_argument('--directory', type=str,
                       help='Process all images in directory')
    parser.add_argument('--output', type=str,
                       help='Path to save output (file or directory)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1, default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display results (useful for batch processing)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Check that at least one input is specified
    if not any([args.image, args.directory]):
        parser.error("Must specify one of: --image, or --directory")
    
    # Initialize detector
    detector = ClimbingHoldDetector(
        model_path="models/" + args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Process based on input type
    if args.image:
        process_image(detector, args.image, args.output, show=not args.no_display)
    
    elif args.directory:
        batch_process_directory(detector, args.directory, args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()