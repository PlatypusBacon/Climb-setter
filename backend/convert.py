"""
Convert Faster R-CNN PyTorch Model for Mobile Deployment
========================================================

This script converts the trained PyTorch Faster R-CNN model to formats
suitable for mobile deployment. Since direct TFLite conversion is not
possible for Faster R-CNN, we use ONNX as an intermediate format.

Note: Faster R-CNN is complex and may have compatibility issues.
This script provides the best-effort conversion approach.

Requirements:
    pip install onnx onnxruntime torch torchvision --break-system-packages
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import onnx
from onnx import shape_inference
import argparse
from pathlib import Path
import numpy as np
import cv2


def load_pytorch_model(model_path, device='cpu'):
    """Load the trained PyTorch Faster R-CNN model."""
    print(f"Loading PyTorch model from {model_path}...")
    
    # Create model architecture
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # background + hold
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    print("✓ PyTorch model loaded successfully")
    return model


def export_to_onnx(model, output_path, input_size=(416, 416)):
    """
    Export PyTorch model to ONNX format.
    
    Note: Full Faster R-CNN export is tricky. This exports the backbone + RPN.
    For full pipeline, you may need to use PyTorch Mobile instead.
    """
    print(f"\nExporting to ONNX (input size: {input_size})...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Set model to eval and disable gradient
    model.eval()
    
    try:
        # Export with dynamic axes for flexibility
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # Use opset 11 for better compatibility
            do_constant_folding=True,
            input_names=['image'],
            output_names=['boxes', 'labels', 'scores'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'boxes': {0: 'batch_size', 1: 'num_detections'},
                'labels': {0: 'batch_size', 1: 'num_detections'},
                'scores': {0: 'batch_size', 1: 'num_detections'},
            }
        )
        
        print(f"✓ ONNX model saved to {output_path}")
        
        # Verify the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        print("\nNote: Faster R-CNN has complex control flow that may not convert to ONNX.")
        print("Consider using PyTorch Mobile instead (see alternative approach below).")
        return False


def export_to_torchscript(model, output_path, input_size=(416, 416)):
    """
    Export to TorchScript format - this is more reliable for Faster R-CNN.
    TorchScript can be used with PyTorch Mobile on Android/iOS.
    """
    print(f"\nExporting to TorchScript (input size: {input_size})...")
    
    try:
        # Create example input
        example = [torch.rand(3, input_size[0], input_size[1])]
        
        # Trace the model
        model.eval()
        traced_script_module = torch.jit.trace(model, example)
        
        # Optimize for mobile
        traced_script_module_optimized = torch.jit.optimize_for_inference(
            traced_script_module
        )
        
        # Save
        traced_script_module_optimized.save(output_path)
        
        print(f"✓ TorchScript model saved to {output_path}")
        
        # Get file size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        return False


def test_onnx_inference(onnx_path, test_image_path, input_size=(416, 416)):
    """Test ONNX model inference."""
    print(f"\nTesting ONNX inference...")
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        
        # Load and preprocess image
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        
        print("✓ ONNX inference successful")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX inference test failed: {e}")
        return False


def test_torchscript_inference(torchscript_path, test_image_path, input_size=(416, 416)):
    """Test TorchScript model inference."""
    print(f"\nTesting TorchScript inference...")
    
    try:
        # Load model
        model = torch.jit.load(torchscript_path)
        model.eval()
        
        # Load and preprocess image
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW
        
        # Run inference
        with torch.no_grad():
            output = model([img])
        
        print("✓ TorchScript inference successful")
        
        if isinstance(output, list) and len(output) > 0:
            result = output[0]
            if 'boxes' in result:
                print(f"  Detected {len(result['boxes'])} boxes")
            if 'scores' in result:
                print(f"  Confidence range: [{result['scores'].min():.3f}, {result['scores'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ TorchScript inference test failed: {e}")
        return False


def create_mobile_config(output_dir):
    """Create configuration file for mobile deployment."""
    config = {
        "model_type": "faster_rcnn",
        "input_size": [416, 416],
        "num_classes": 2,
        "class_names": ["background", "hold"],
        "confidence_threshold": 0.5,
        "nms_threshold": 0.5,
        "preprocessing": {
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "scale": 255.0
        }
    }
    
    import json
    config_path = Path(output_dir) / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Mobile config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Faster R-CNN PyTorch model for mobile deployment'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--output-dir', type=str, default='./mobile_models',
                       help='Output directory for converted models')
    parser.add_argument('--input-size', type=int, nargs=2, default=[416, 416],
                       help='Input size (width height)')
    parser.add_argument('--test-image', type=str,
                       help='Test image path for verification')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'],
                       default='both', help='Export format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Faster R-CNN Mobile Conversion Pipeline")
    print("="*60)
    
    # Load model
    model = load_pytorch_model(args.model)
    
    input_size = tuple(args.input_size)
    
    # Export to requested formats
    success = {}
    
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / 'model.onnx'
        success['onnx'] = export_to_onnx(model, str(onnx_path), input_size)
        
        if success['onnx'] and args.test_image:
            test_onnx_inference(str(onnx_path), args.test_image, input_size)
    
    if args.format in ['torchscript', 'both']:
        torchscript_path = output_dir / 'model.pt'
        success['torchscript'] = export_to_torchscript(model, str(torchscript_path), input_size)
        
        if success['torchscript'] and args.test_image:
            test_torchscript_inference(str(torchscript_path), args.test_image, input_size)
    
    # Create config file
    create_mobile_config(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    for format_name, status in success.items():
        status_str = "✓ SUCCESS" if status else "✗ FAILED"
        print(f"{format_name.upper()}: {status_str}")
    
    if success.get('torchscript', False):
        print("\n✓ RECOMMENDED: Use TorchScript (.pt) with PyTorch Mobile")
        print(f"  Model: {output_dir / 'model.pt'}")
        print(f"  Config: {output_dir / 'model_config.json'}")
        print("\nNext steps:")
        print("  1. Add pytorch_lite to your Flutter project")
        print("  2. Copy model.pt to assets/")
        print("  3. Use the Flutter integration code provided")
    
    if success.get('onnx', False):
        print("\n✓ ALTERNATIVE: Use ONNX (.onnx) with ONNX Runtime")
        print(f"  Model: {output_dir / 'model.onnx'}")
        print("  Note: May have compatibility issues with complex operations")
    
    if not any(success.values()):
        print("\n✗ All conversions failed")
        print("\nRecommendation: Use the API server approach instead")
        print("Or consider training a simpler model for mobile deployment")
    
    print("="*60)


if __name__ == '__main__':
    main()