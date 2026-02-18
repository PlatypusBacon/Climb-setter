"""
Climbing Hold Detection - Inference Script
==========================================

Supports both PyTorch (.pth) Faster R-CNN models and TensorFlow (.keras / .tflite)
models trained with the training script.

Model type is inferred automatically from the file extension:
  .pth      → PyTorch Faster R-CNN (requires torch, torchvision)
  .keras    → TensorFlow/Keras detector (requires tensorflow)
  .tflite   → TensorFlow Lite (requires tensorflow, runs on CPU only)

Usage:
    python inference.py --model path/to/model.pth    --image path/to/image.jpg
    python inference.py --model path/to/model.keras  --image path/to/image.jpg
    python inference.py --model path/to/model.tflite --image path/to/image.jpg
    python inference.py --model path/to/model.pth    --video path/to/video.mp4
    python inference.py --model path/to/model.pth    --webcam
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time


# ──────────────────────────────────────────────
# Backend loaders
# ──────────────────────────────────────────────

def _load_pytorch_model(model_path, device):
    """Load a Faster R-CNN .pth checkpoint."""
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_keras_model(model_path):
    """Load a .keras (or .h5) model."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return model


def _load_tflite_model(model_path):
    """Load a .tflite model and return a ready interpreter."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


# ──────────────────────────────────────────────
# Per-backend inference helpers
# ──────────────────────────────────────────────

def _infer_pytorch(model, image_bgr, device):
    """
    Run PyTorch Faster R-CNN inference.

    Returns:
        boxes  : np.ndarray (N, 4)  [x1, y1, x2, y2] in pixel coords
        scores : np.ndarray (N,)
    """
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).to(device)

    with torch.no_grad():
        preds = model([tensor])[0]

    boxes  = preds['boxes'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    return boxes, scores


def _decode_anchor_output(pred_raw, img_h, img_w, num_anchors=6, num_classes=1,
                           conf_threshold=0.5):
    """
    Decode a raw SSD/EfficientDet scale output tensor into boxes + scores.

    pred_raw shape: (1, gH, gW, num_anchors*(5+num_classes))
    Returns boxes in pixel [x1,y1,x2,y2], scores array.
    """
    gH, gW = pred_raw.shape[1], pred_raw.shape[2]
    pred = pred_raw[0].reshape(gH, gW, num_anchors, 5 + num_classes)

    # Decode sigmoid outputs
    pred_xy  = 1 / (1 + np.exp(-pred[..., 0:2]))   # (gH, gW, A, 2)  cx,cy offset in cell
    pred_wh  = pred[..., 2:4]                        # log-scale w,h
    pred_obj = 1 / (1 + np.exp(-pred[..., 4]))       # (gH, gW, A)

    # Build grid offsets
    gx = np.arange(gW, dtype=np.float32)
    gy = np.arange(gH, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)            # (gH, gW)
    grid_x = grid_x[..., np.newaxis]                 # (gH, gW, 1)
    grid_y = grid_y[..., np.newaxis]

    cx = (pred_xy[..., 0] + grid_x) / gW            # normalised [0,1]
    cy = (pred_xy[..., 1] + grid_y) / gH
    w  = np.exp(pred_wh[..., 0]) / gW
    h  = np.exp(pred_wh[..., 1]) / gH

    # Filter by objectness
    mask = pred_obj > conf_threshold
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    cx_f = cx[mask] * img_w
    cy_f = cy[mask] * img_h
    w_f  = w[mask]  * img_w
    h_f  = h[mask]  * img_h

    x1 = cx_f - w_f / 2
    y1 = cy_f - h_f / 2
    x2 = cx_f + w_f / 2
    y2 = cy_f + h_f / 2

    boxes  = np.stack([x1, y1, x2, y2], axis=-1)
    scores = pred_obj[mask].astype(np.float32)
    return boxes, scores


def _decode_centernet_output(outputs, img_h, img_w, conf_threshold=0.15):
    """
    Decode CenterNet dict output {'heatmap', 'wh', 'offset'} into boxes + scores.
    All tensors are numpy arrays at this point.
    """
    heatmap = outputs['heatmap'][0]    # (oH, oW, C)
    wh      = outputs['wh'][0]         # (oH, oW, 2)
    offset  = outputs['offset'][0]     # (oH, oW, 2)

    oH, oW = heatmap.shape[:2]
    conf_map = heatmap[..., 0]         # single-class

    # Simple local-max peak extraction (3x3 max-pool NMS)
    from scipy.ndimage import maximum_filter
    peaks = (conf_map == maximum_filter(conf_map, size=3)) & (conf_map > conf_threshold)
    ys, xs = np.where(peaks)

    if len(ys) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    scores = conf_map[ys, xs]

    # Decode centres (add sub-pixel offset)
    cx = (xs + offset[ys, xs, 0]) / oW * img_w
    cy = (ys + offset[ys, xs, 1]) / oH * img_h
    bw = wh[ys, xs, 0] * img_w
    bh = wh[ys, xs, 1] * img_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    return boxes, scores.astype(np.float32)


def _infer_keras(model, image_bgr, conf_threshold=0.5, input_size=(320, 320)):
    """
    Run inference with a .keras model (SSD, EfficientDet, or CenterNet).

    Detects model type by inspecting output signature:
      - dict with 'heatmap' key → CenterNet
      - list/tuple of tensors    → SSD or EfficientDet

    Returns:
        boxes  : np.ndarray (N, 4)  [x1, y1, x2, y2] in pixel coords
        scores : np.ndarray (N,)
    """
    import tensorflow as tf
    print("OUTPUT TYPE:", type(outputs))

    if isinstance(outputs, dict):
        for k, v in outputs.items():
            print(k, v.shape)
    else:
        if isinstance(outputs, (list, tuple)):
            for i, v in enumerate(outputs):
                print(f"output[{i}]:", v.shape)
        else:
            print("single output:", outputs.shape)

    img_h, img_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, input_size).astype(np.float32) / 255.0
    img_t = tf.convert_to_tensor(image_resized[np.newaxis])

    outputs = model(img_t, training=False)

    # ── CenterNet ────────────────────────────────────────────────────────
    if isinstance(outputs, dict) and 'heatmap' in outputs:
        np_outputs = {k: v.numpy() for k, v in outputs.items()}
        # Scale to original image size for coordinate decoding
        boxes, scores = _decode_centernet_output(np_outputs, img_h, img_w, conf_threshold)

    # ── SSD / EfficientDet ───────────────────────────────────────────────
    else:
        scale_outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        all_boxes, all_scores = [], []
        for scale_pred in scale_outputs:
            b, s = _decode_anchor_output(
                scale_pred.numpy(), img_h, img_w,
                conf_threshold=conf_threshold
            )
            all_boxes.append(b)
            all_scores.append(s)

        if all_boxes:
            boxes  = np.concatenate(all_boxes,  axis=0)
            scores = np.concatenate(all_scores, axis=0)
        else:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

    # Apply NMS to remove duplicate detections across scales
    if len(boxes) > 0:
        boxes, scores = _nms(boxes, scores, iou_threshold=0.25)

    return boxes, scores


def _infer_tflite(interpreter, image_bgr, conf_threshold=0.5):
    """
    Run inference with a .tflite interpreter.

    Handles both CenterNet (dict-like named outputs) and anchor-based models.

    Returns:
        boxes  : np.ndarray (N, 4)  [x1, y1, x2, y2] in pixel coords
        scores : np.ndarray (N,)
    """
    img_h, img_w = image_bgr.shape[:2]

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize to model's expected input shape
    _, in_h, in_w, _ = input_details[0]['shape']
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (in_w, in_h)).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], image_resized[np.newaxis])
    interpreter.invoke()

    # Collect all outputs keyed by name
    raw = {d['name']: interpreter.get_tensor(d['index']) for d in output_details}

    # ── CenterNet outputs ────────────────────────────────────────────────
    # DEBUG: print outputs
    for k, v in raw.items():
        print("TFLite output:", k, v.shape)

    # Detect CenterNet by number + shape of outputs
    if len(raw) == 3:
        tensors = list(raw.values())

        # Sort by last dimension so we map correctly:
        # heatmap = channels (1)
        # wh      = 2
        # offset  = 2
        tensors = sorted(tensors, key=lambda x: x.shape[-1])

        # Expect shapes like:
        # (1, oH, oW, 1), (1, oH, oW, 2), (1, oH, oW, 2)
        heatmap = tensors[0]
        wh      = tensors[1]
        offset  = tensors[2]

        outputs = {
            'heatmap': heatmap,
            'wh': wh,
            'offset': offset
        }
        


        boxes, scores = _decode_centernet_output(outputs, img_h, img_w, conf_threshold)
        hm = outputs['heatmap']   # or correct index
        print("Heatmap min/max:", hm.min(), hm.max())
    else:
        # fallback to anchor decoder
        all_boxes, all_scores = [], []
        for tensor in raw.values():
            if tensor.ndim == 4:
                b, s = _decode_anchor_output(
                    tensor, img_h, img_w,
                    conf_threshold=conf_threshold
                )
                all_boxes.append(b)
                all_scores.append(s)

        if all_boxes:
            boxes  = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
        else:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

    if len(boxes) > 0:
        boxes, scores = _nms(boxes, scores, iou_threshold=0.25)

    return boxes, scores


def _nms(boxes, scores, iou_threshold=0.25):
    """Simple greedy NMS. Returns filtered (boxes, scores)."""
    if len(boxes) == 0:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    keep  = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        rest  = order[1:]
        inter = _intersection_area_np(boxes[i], boxes[rest])
        area_i    = (boxes[i, 2]    - boxes[i, 0])    * (boxes[i, 3]    - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-7)
        order = rest[iou < iou_threshold]

    return boxes[keep], scores[keep]


def _intersection_area_np(box, boxes):
    """Intersection area of one box vs an array of boxes. All [x1,y1,x2,y2]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    return np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)


# ──────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────

class ClimbingHoldDetector:
    """
    Unified wrapper for climbing hold detection.

    Automatically selects PyTorch or TensorFlow backend based on
    the model file extension:
      .pth            → PyTorch Faster R-CNN
      .keras / .h5    → TensorFlow Keras (SSD / EfficientDet / CenterNet)
      .tflite         → TensorFlow Lite
    """

    def __init__(self, model_path, confidence_threshold=0.8,
                 device=None, max_display_size=(1100, 800),
                 input_size=(320, 320)):
        """
        Args:
            model_path           : Path to model file (.pth / .keras / .tflite)
            confidence_threshold : Minimum detection confidence (0–1)
            device               : 'cuda' | 'cpu' | None (auto, PyTorch only)
            max_display_size     : (width, height) cap for on-screen display
            input_size           : (width, height) to resize images to before
                                   feeding into Keras / TFLite models.
                                   Ignored for PyTorch (uses native image size).
        """
        self.confidence_threshold = confidence_threshold
        self.max_display_size     = max_display_size
        self.input_size           = input_size
        self.model_path           = Path(model_path)
        self.backend              = self._detect_backend()

        # Resolve device for PyTorch
        if self.backend == 'pytorch':
            import torch
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = None   # TF manages its own device placement

        print(f"Backend:    {self.backend}")
        if self.device:
            print(f"Device:     {self.device}")
        print(f"Threshold:  {confidence_threshold}")
        if max_display_size:
            print(f"Max display: {max_display_size[0]}x{max_display_size[1]}")

        self.model = self._load_model()
        print(f"Model loaded: {model_path}\n")

    # ── Setup ──────────────────────────────────────────────────────────────

    def _detect_backend(self):
        suffix = self.model_path.suffix.lower()
        if suffix == '.pth':
            return 'pytorch'
        elif suffix in ('.keras', '.h5'):
            return 'keras'
        elif suffix == '.tflite':
            return 'tflite'
        else:
            raise ValueError(
                f"Unrecognised model extension '{suffix}'. "
                "Expected .pth, .keras, .h5, or .tflite."
            )

    def _load_model(self):
        if self.backend == 'pytorch':
            return _load_pytorch_model(str(self.model_path), self.device)
        elif self.backend == 'keras':
            return _load_keras_model(str(self.model_path))
        else:
            return _load_tflite_model(str(self.model_path))

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, image):
        """
        Detect climbing holds in an image.

        Args:
            image : numpy array (H, W, 3) in BGR format (OpenCV native)

        Returns:
            List of dicts, each with:
                'bbox'       : [x1, y1, x2, y2]  pixel coordinates
                'confidence' : float
                'center'     : (cx, cy)  pixel coordinates
        """
        if self.backend == 'pytorch':
            boxes, scores = _infer_pytorch(self.model, image, self.device)
        elif self.backend == 'keras':
            boxes, scores = _infer_keras(
                self.model, image,
                conf_threshold=self.confidence_threshold,
                input_size=self.input_size
            )
        else:
            boxes, scores = _infer_tflite(
                self.model, image,
                conf_threshold=self.confidence_threshold
            )

        detections = []
        for box, score in zip(boxes, scores):
            # PyTorch already filtered by threshold inside _infer_pytorch;
            # apply it here uniformly for TF backends too.
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box
            detections.append({
                'bbox':       [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(score),
                'center':     (float((x1 + x2) / 2), float((y1 + y2) / 2)),
            })

        return detections

    def visualize_detections(self, image, detections,
                             show_boxes=True, show_centers=True):
        """Draw detections on image. Returns annotated copy."""
        vis = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cx, cy = int(det['center'][0]), int(det['center'][1])
            conf   = det['confidence']

            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.6 else (0, 165, 255)

            if show_boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if show_centers:
                cv2.circle(vis, (cx, cy), 5, color, -1)
                cv2.circle(vis, (cx, cy), 8, color, 2)

            label = f"{conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
            cv2.putText(vis, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(vis, f"Holds detected: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis

    def scale_for_display(self, image):
        """Downscale image to fit max_display_size, preserving aspect ratio."""
        if self.max_display_size is None:
            return image, 1.0

        h, w   = image.shape[:2]
        max_w, max_h = self.max_display_size
        scale  = min(max_w / w, max_h / h, 1.0)

        if scale >= 1.0:
            return image, 1.0

        scaled = cv2.resize(image, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
        return scaled, scale


# ──────────────────────────────────────────────
# Processing functions
# ──────────────────────────────────────────────

def process_image(detector, image_path, output_path=None, show=True):
    """Process a single image file."""
    full_path = Path("data/img") / image_path
    print(f"\nProcessing: {full_path}")

    image = cv2.imread(str(full_path))
    if image is None:
        print(f"Error: could not load {full_path}")
        return None, None

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    start = time.time()
    detections = detector.detect(image)
    elapsed = time.time() - start

    print(f"Found {len(detections)} holds in {elapsed:.3f}s  "
          f"[backend: {detector.backend}]")
    for i, det in enumerate(detections, 1):
        cx, cy = det['center']
        print(f"  Hold {i}: center=({cx:.1f}, {cy:.1f}), confidence={det['confidence']:.3f}")

    vis = detector.visualize_detections(image, detections)

    if output_path:
        cv2.imwrite(str(output_path), vis)
        print(f"Saved to: {output_path}")

    if show:
        display, scale = detector.scale_for_display(vis)
        if scale < 1.0:
            print(f"Display scale: {scale:.2f}x ({display.shape[1]}x{display.shape[0]})")
        cv2.imshow("Climbing Hold Detection", display)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections, vis


def batch_process_directory(detector, input_dir, output_dir=None):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for ext in extensions
        for f in list(input_path.glob(f"*{ext}")) + list(input_path.glob(f"*{ext.upper()}"))
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images  [backend: {detector.backend}]")

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {image_file.name}")
        image = cv2.imread(str(image_file))
        if image is None:
            print("  Error: could not load image")
            continue

        detections = detector.detect(image)
        results.append((image_file.name, len(detections)))
        print(f"  Found {len(detections)} holds")

        if out_path:
            vis = detector.visualize_detections(image, detections)
            cv2.imwrite(str(out_path / f"{image_file.stem}_detected{image_file.suffix}"), vis)

    print("\n" + "=" * 50)
    print("BATCH SUMMARY")
    print("=" * 50)
    for filename, count in results:
        print(f"  {filename}: {count} holds")
    if results:
        avg = sum(c for _, c in results) / len(results)
        print(f"\n  Total: {len(results)} images  |  Avg holds/image: {avg:.1f}")
    if output_dir:
        print(f"  Results saved to: {output_dir}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Climbing Hold Detection — supports .pth, .keras, and .tflite models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types (auto-detected from extension):
  .pth     PyTorch Faster R-CNN
  .keras   TensorFlow/Keras  (SSD MobileNet / EfficientDet / CenterNet)
  .tflite  TensorFlow Lite   (exported from .keras via --convert)

Examples:
  python inference.py --model model.pth    --image wall.jpg
  python inference.py --model model.keras  --image wall.jpg
  python inference.py --model model.tflite --image wall.jpg --no-display --output result.jpg
  python inference.py --model model.pth    --directory ./images --output ./results
  python inference.py --model model.keras  --confidence 0.6 --input-size 416 416
        """
    )

    parser.add_argument('--model',      type=str, required=True,
                        help='Path to model file (.pth, .keras, or .tflite)')
    parser.add_argument('--image',      type=str,
                        help='Path to input image')
    parser.add_argument('--directory',  type=str,
                        help='Process all images in this directory')
    parser.add_argument('--output',     type=str,
                        help='Output file or directory for visualisations')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not show results on screen')
    parser.add_argument('--device',     type=str, choices=['cuda', 'cpu'],
                        help='PyTorch device (default: auto). Ignored for .keras/.tflite.')
    parser.add_argument('--input-size', type=int, nargs=2, default=[320, 320],
                        metavar=('W', 'H'),
                        help='Input resolution for Keras/TFLite models (default: 320 320). '
                             'Use the same size the model was trained with.')

    args = parser.parse_args()

    if not args.image and not args.directory:
        parser.error("Specify --image or --directory")

    detector = ClimbingHoldDetector(
        model_path=Path("models") / args.model,
        confidence_threshold=args.confidence,
        device=args.device,
        input_size=tuple(args.input_size),
    )

    if args.image:
        process_image(detector, args.image, args.output, show=not args.no_display)
    elif args.directory:
        batch_process_directory(detector, args.directory, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()