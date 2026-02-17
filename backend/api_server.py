"""
Flask API Server for Climbing Hold Detection
============================================

Simple REST API that accepts images and returns detected climbing holds.
Perfect for rapid prototyping and MVP development.

Usage:
    python api_server.py --model path/to/model.pth --port 5000

Requirements:
    pip install flask flask-cors pillow --break-system-packages
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from PIL import Image
import io
import base64
import argparse
from pathlib import Path


app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web

# Global model variable
detector = None


class ClimbingHoldDetector:
    """Lightweight detector for API use."""
    
    def __init__(self, model_path, confidence_threshold=0.5, device=None):
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print("Model loaded successfully!")
    
    def _load_model(self, model_path):
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def detect(self, image_array):
        """
        Detect holds in image.
        
        Args:
            image_array: numpy array (H, W, 3) in RGB format
            
        Returns:
            List of detections with bbox, confidence, center
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
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
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    },
                    'confidence': float(score),
                    'center': {
                        'x': float(cx),
                        'y': float(cy)
                    }
                })
        
        return detections


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'device': str(detector.device) if detector else None
    })


@app.route('/detect', methods=['POST'])
def detect_holds():
    """
    Main detection endpoint.
    
    Accepts:
        - multipart/form-data with 'image' file
        - JSON with 'image' as base64 string
    
    Returns:
        JSON with detected holds and metadata
    """
    try:
        # Parse image from request
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        elif request.is_json and 'image' in request.json:
            # Base64 encoded
            base64_str = request.json['image']
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]  # Remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert to RGB numpy array
        image_rgb = np.array(image.convert('RGB'))
        
        # Get original dimensions
        original_height, original_width = image_rgb.shape[:2]
        
        # Detect holds
        detections = detector.detect(image_rgb)
        
        # Prepare response
        response = {
            'success': True,
            'holds': detections,
            'metadata': {
                'count': len(detections),
                'image_width': original_width,
                'image_height': original_height,
                'confidence_threshold': detector.confidence_threshold
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect_batch', methods=['POST'])
def detect_holds_batch():
    """
    Batch detection endpoint for multiple images.
    
    Accepts JSON with array of base64 images.
    """
    try:
        if not request.is_json or 'images' not in request.json:
            return jsonify({'error': 'Expected JSON with "images" array'}), 400
        
        images_data = request.json['images']
        results = []
        
        for idx, img_data in enumerate(images_data):
            try:
                # Decode base64
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                image_bytes = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(image_bytes))
                image_rgb = np.array(image.convert('RGB'))
                
                # Detect
                detections = detector.detect(image_rgb)
                
                results.append({
                    'success': True,
                    'index': idx,
                    'holds': detections,
                    'count': len(detections)
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(images_data)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    """Update confidence threshold dynamically."""
    try:
        if not request.is_json or 'threshold' not in request.json:
            return jsonify({'error': 'Expected JSON with "threshold" field'}), 400
        
        threshold = float(request.json['threshold'])
        if not 0 <= threshold <= 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        detector.confidence_threshold = threshold
        
        return jsonify({
            'success': True,
            'confidence_threshold': threshold
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    parser = argparse.ArgumentParser(description='Climbing Hold Detection API Server')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    global detector
    detector = ClimbingHoldDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    print("\n" + "="*50)
    print("Climbing Hold Detection API Server")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {detector.device}")
    print("\nEndpoints:")
    print(f"  GET  /health - Health check")
    print(f"  POST /detect - Detect holds in single image")
    print(f"  POST /detect_batch - Detect holds in multiple images")
    print(f"  POST /update_confidence - Update confidence threshold")
    print("="*50 + "\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()