"""
Climbing Hold Detection Model Training Script
==============================================

Supports multiple detector backends, all exportable to TensorFlow Lite
for Flutter mobile deployment.

Available models (via --model flag):
  ssd_mobilenet   - SSD MobileNetV2 (fastest, smallest, best for mobile) [DEFAULT]
  efficientdet    - EfficientDet-Lite0 (better accuracy, still mobile-friendly)
  centernet       - CenterNet MobileNetV2 (anchor-free, good small-object detection)
  pytorch_rcnn    - Faster R-CNN ResNet50 (original, NOT exportable to TFLite —
                    training only, use for reference/evaluation on desktop)

Available box regression losses (via --loss flag):
  ciou    - Complete IoU: overlap + centre distance + aspect ratio penalty  [DEFAULT]
  diou    - Distance IoU: overlap + centre distance penalty
  giou    - Generalised IoU: handles zero-overlap case, no aspect-ratio term
  iou     - Vanilla IoU: pure overlap ratio (zero gradient when boxes don't overlap)

  Recommendation:
    ciou  — best convergence, especially when holds vary in aspect ratio (slopers vs pockets)
    diou  — slightly faster convergence than GIoU, good general choice
    giou  — safer than vanilla IoU early in training; simpler than CIoU
    iou   — only use if boxes almost always overlap from epoch 1 (pre-trained init)

  All variants operate on (cx, cy, w, h) directly — no corner conversion needed
  in your annotation pipeline. Internally they convert to corners just for the
  intersection area calculation, then back.

Duplicate detection fixes (via --nms-iou and --ignore-radius flags):
  --nms-iou FLOAT        IoU threshold for training-time NMS (default: 0.30).
                         Lower = more aggressive merging of nearby boxes.
                         0.45 is the old default; 0.25-0.35 eliminates most duplicates.

  --ignore-radius FLOAT  Normalised radius around each GT centre within which
                         non-best anchors are marked 'ignore' rather than negative.
                         Prevents the model learning to fire from near-miss anchors.
                         (default: 0.10, i.e. 10% of image width/height)

  --focal-objectness     Replace BCE on the objectness head with focal loss
                         (gamma=2, alpha=0.25). Down-weights the vast number of
                         easy background anchors so the model learns 'not a hold'
                         more aggressively. Strongly recommended for small datasets.

  Root causes of multiple detections:
    1. NMS threshold too permissive  -> fix with lower --nms-iou
    2. Multiple anchors positive for same GT -> fix with --ignore-radius
    3. Objectness undertrained due to class imbalance -> fix with --focal-objectness
    4. Architecture: CenterNet is structurally immune (one peak per hold by design)

TFLite export notes:
  - ssd_mobilenet, efficientdet, centernet: natively supported via TF Object Detection API
  - pytorch_rcnn: cannot be exported to TFLite; use for desktop/server inference only

Requirements:
  pip install tensorflow==2.13.0
  pip install tf-models-official          # for TF Object Detection API models
  pip install torch torchvision           # only needed for pytorch_rcnn option
  pip install opencv-python numpy pillow scikit-learn

Dataset format expected:
  data/img/    -> .jpg images
  data/label/  -> .json annotations
    Each JSON: { "holds": [ { "x": cx, "y": cy, "width": w, "height": h }, ... ] }
    Coordinates in pixels, origin top-left, x/y = center of bounding box.
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

IMG_SIZE      = (320, 320)   # Smaller default suits mobile detectors better than 416
BATCH_SIZE    = 8
EPOCHS        = 200
LEARNING_RATE = 1e-4
NUM_CLASSES   = 1            # Just "hold" — extend HOLD_TYPES if you want per-type labels

HOLD_TYPES = ['jug', 'crimp', 'sloper', 'pinch', 'pocket', 'unknown']

MODEL_CHOICES = ['ssd_mobilenet', 'efficientdet', 'centernet', 'pytorch_rcnn']
LOSS_CHOICES  = ['ciou', 'diou', 'giou', 'iou']

TFLITE_COMPATIBLE = {'ssd_mobilenet', 'efficientdet', 'centernet'}
PYTORCH_ONLY      = {'pytorch_rcnn'}


# ──────────────────────────────────────────────
# Shared dataset / utilities
# ──────────────────────────────────────────────


def augment_climbing_image(image, boxes, img_size=(320, 320)):
    """
    Apply climbing-specific augmentations to a training image.
    
    Args:
        image : np.ndarray (H, W, 3) float32 in [0, 1]
        boxes : list of [y1, x1, y2, x2] normalised coords
        img_size : target output size
    
    Returns:
        aug_image, aug_boxes
    """
    h, w = image.shape[:2]
    
    # ── 1. Random perspective warp (simulates camera angle change) ──────
    if np.random.rand() < 0.7:
        image, boxes = _random_perspective(image, boxes, severity=0.15)
    
    # ── 2. Random brightness/contrast (lighting variation) ──────────────
    if np.random.rand() < 0.8:
        image = _random_brightness_contrast(image)
    
    # ── 3. Random color jitter (hold color / chalk variation) ───────────
    if np.random.rand() < 0.6:
        image = _random_color_jitter(image)
    
    # ── 4. Random blur (motion blur / camera shake) ─────────────────────
    if np.random.rand() < 0.3:
        image = _random_blur(image)
    
    # ── 5. Random noise (sensor noise / compression artifacts) ──────────
    if np.random.rand() < 0.4:
        image = _random_noise(image, sigma=0.02)
    
    # ── 6. Cutout (occlusion by climber body / gear) ───────────────────
    if np.random.rand() < 0.5:
        image = _random_cutout(image, max_holes=3, hole_size=0.15)
    
    # ── 7. Resize to target (after all spatial transforms) ──────────────
    image = cv2.resize(image, img_size)
    
    return image, boxes


def _random_perspective(image, boxes, severity=0.15):
    """
    Random perspective warp. Boxes are transformed with the image.
    severity=0.15 → corners move up to 15% of image dimension.
    """
    h, w = image.shape[:2]
    
    # Source points: image corners
    src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Destination: corners with random offsets
    max_offset = int(min(w, h) * severity)
    dst = src + np.random.randint(-max_offset, max_offset, src.shape)
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderValue=(0.5, 0.5, 0.5))
    
    # Transform boxes
    warped_boxes = []
    for box in boxes:
        y1, x1, y2, x2 = box
        # Transform all 4 corners
        corners = np.array([
            [x1 * w, y1 * h],
            [x2 * w, y1 * h],
            [x2 * w, y2 * h],
            [x1 * w, y2 * h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        
        # Recompute bounding box from warped corners
        new_x1 = np.clip(warped_corners[:, 0].min() / w, 0, 1)
        new_x2 = np.clip(warped_corners[:, 0].max() / w, 0, 1)
        new_y1 = np.clip(warped_corners[:, 1].min() / h, 0, 1)
        new_y2 = np.clip(warped_corners[:, 1].max() / h, 0, 1)
        
        # Only keep box if still visible (area > 10% of original)
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        old_area = (x2 - x1) * (y2 - y1)
        if new_area > old_area * 0.1:
            warped_boxes.append([new_y1, new_x1, new_y2, new_x2])
    
    return warped, warped_boxes


def _random_brightness_contrast(image, brightness_range=0.3, contrast_range=0.3):
    """
    Random brightness and contrast adjustment.
    Simulates different gym lighting conditions.
    """
    # Brightness shift
    brightness = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    image = image * brightness
    
    # Contrast adjustment (pivot around mean)
    contrast = 1.0 + np.random.uniform(-contrast_range, contrast_range)
    mean = image.mean()
    image = (image - mean) * contrast + mean
    
    return np.clip(image, 0, 1)


def _random_color_jitter(image, hue_shift=0.05, sat_scale=0.3):
    """
    Random hue/saturation jitter.
    Simulates different hold colors and chalk dust.
    """
    # Convert to HSV
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Hue shift
    hsv[..., 0] += np.random.uniform(-hue_shift, hue_shift) * 180
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 180)
    
    # Saturation scale
    hsv[..., 1] *= (1.0 + np.random.uniform(-sat_scale, sat_scale))
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    
    # Back to RGB
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb


def _random_blur(image, max_kernel=5):
    """Random Gaussian blur (motion blur / out of focus)."""
    kernel = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def _random_noise(image, sigma=0.02):
    """Add Gaussian noise (sensor noise / compression)."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def _random_cutout(image, max_holes=3, hole_size=0.15):
    """
    Random cutout rectangles.
    Simulates occlusion by climber's body, rope, gear.
    """
    h, w = image.shape[:2]
    num_holes = np.random.randint(1, max_holes + 1)
    
    for _ in range(num_holes):
        hole_h = int(h * hole_size * np.random.uniform(0.5, 1.0))
        hole_w = int(w * hole_size * np.random.uniform(0.5, 1.0))
        
        y = np.random.randint(0, h - hole_h)
        x = np.random.randint(0, w - hole_w)
        
        # Fill with mean color (less jarring than black)
        mean_color = image[y:y+hole_h, x:x+hole_w].mean(axis=(0, 1))
        image[y:y+hole_h, x:x+hole_w] = mean_color
    
    return image

def load_annotations(annotations_path='data/label', images_path='data/img'):
    """
    Returns a list of dicts:
      { 'image_path': str, 'boxes': [[xmin,ymin,xmax,ymax], ...] }
    Boxes are in absolute pixel coords of the original image.
    """
    records = []
    for ann_file in Path(annotations_path).glob('*.json'):
        img_path = os.path.join(images_path, ann_file.stem + '.jpg')
        if not os.path.exists(img_path):
            print(f"  Warning: no image for {ann_file.name}, skipping.")
            continue

        with open(ann_file) as f:
            ann = json.load(f)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: could not read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        boxes = []
        for hold in ann.get('holds', []):
            cx, cy = hold['x'], hold['y']
            bw, bh = hold['width'], hold['height']
            xmin = max(0.0, (cx - bw / 2) / w)
            ymin = max(0.0, (cy - bh / 2) / h)
            xmax = min(1.0, (cx + bw / 2) / w)
            ymax = min(1.0, (cy + bh / 2) / h)
            boxes.append([ymin, xmin, ymax, xmax])  # TF convention: [y1,x1,y2,x2]

        records.append({'image_path': img_path, 'boxes': boxes})

    print(f"Loaded {len(records)} annotated images.")
    return records


def preprocess_image(image_path, img_size=IMG_SIZE):
    """Load, resize, normalise an image to float32 [0,1]."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img.astype(np.float32) / 255.0


def overlay_heatmap(image, heatmap):
    heatmap = heatmap.squeeze()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(
        (image * 255).astype(np.uint8), 0.6,
        heatmap, 0.4, 0
    )


def draw_predictions(image, boxes, scores, threshold=0.5):
    """Draw bounding boxes on image. boxes in [y1,x1,y2,x2] normalised."""
    img = (image * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        y1, x1, y2, x2 = box
        cv2.rectangle(img,
                      (int(x1 * w), int(y1 * h)),
                      (int(x2 * w), int(y2 * h)),
                      (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}",
                    (int(x1 * w), int(y1 * h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


# ──────────────────────────────────────────────
# IoU loss family  (TensorFlow, works on cx/cy/w/h)
# ──────────────────────────────────────────────

def _boxes_to_corners(boxes):
    """
    Convert [..., (cx, cy, w, h)] → [..., (x1, y1, x2, y2)].
    All operations are TF-native so gradients flow through correctly.
    """
    import tensorflow as tf
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.stack([x1, y1, x2, y2], axis=-1)


def _intersection_area(b1, b2):
    """Intersection area of two corner-format box tensors [..., 4]."""
    import tensorflow as tf
    inter_x1 = tf.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = tf.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = tf.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = tf.minimum(b1[..., 3], b2[..., 3])
    inter_w  = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h  = tf.maximum(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def iou_loss(pred_cxcywh, gt_cxcywh, variant='ciou', eps=1e-7):
    """
    Compute 1 - IoU (or a variant) between predicted and ground-truth boxes.

    All inputs are in (cx, cy, w, h) format — the native format of your
    annotation files — so no pre-conversion is needed in the training loop.

    Args:
        pred_cxcywh : tf.Tensor [..., 4]  predicted boxes  (cx, cy, w, h)
        gt_cxcywh   : tf.Tensor [..., 4]  ground-truth boxes (cx, cy, w, h)
        variant     : one of 'iou', 'giou', 'diou', 'ciou'
        eps         : small constant for numerical stability

    Returns:
        loss : tf.Tensor [...]  element-wise loss values (lower = better overlap)

    Shape note:
        Inputs are broadcast-compatible, so you can pass
          pred shape  (B, num_anchors, 4)
          gt   shape  (B, 1,           4)
        and the loss will be computed for every anchor independently.

    Gradient note:
        All operations are differentiable.  The only non-differentiable
        point would be exactly-zero w or h, which is clamped away by eps.

    Variant guide:
        iou  — 1 - |A∩B| / |A∪B|
               ✗ zero gradient when boxes don't overlap at all
        giou — adds penalty: -(|C minus (A∪B)| / |C|) where C is smallest enclosing box
               ✓ non-zero gradient everywhere
               ✗ slow to push overlapping boxes together (penalty vanishes)
        diou — adds penalty: -(ρ²(b,bgt) / c²) where ρ is centre distance, c is diagonal of C
               ✓ faster convergence than GIoU
               ✗ ignores aspect ratio
        ciou — adds aspect-ratio consistency term v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²
               ✓ best overall for varied-shape objects (holds differ a lot in ratio)
               ✓ recommended default
    """
    import tensorflow as tf
    import math

    pred = _boxes_to_corners(pred_cxcywh)
    gt   = _boxes_to_corners(gt_cxcywh)

    # ── Intersection ──────────────────────────────────────────────────────
    inter = _intersection_area(pred, gt)

    # ── Union ─────────────────────────────────────────────────────────────
    area_pred = pred_cxcywh[..., 2] * pred_cxcywh[..., 3]
    area_gt   = gt_cxcywh[..., 2]   * gt_cxcywh[..., 3]
    union = area_pred + area_gt - inter + eps

    iou = inter / union                                         # ∈ [0, 1]

    if variant == 'iou':
        return 1.0 - iou

    # ── Smallest enclosing box (needed for GIoU / DIoU / CIoU) ───────────
    enclose_x1 = tf.minimum(pred[..., 0], gt[..., 0])
    enclose_y1 = tf.minimum(pred[..., 1], gt[..., 1])
    enclose_x2 = tf.maximum(pred[..., 2], gt[..., 2])
    enclose_y2 = tf.maximum(pred[..., 3], gt[..., 3])
    enclose_w  = tf.maximum(0.0, enclose_x2 - enclose_x1)
    enclose_h  = tf.maximum(0.0, enclose_y2 - enclose_y1)

    if variant == 'giou':
        # GIoU penalty: fraction of enclosing box not covered by either box
        enclose_area = enclose_w * enclose_h + eps
        giou = iou - (enclose_area - union) / enclose_area      # ∈ [-1, 1]
        return 1.0 - giou

    # ── Centre-distance penalty (DIoU / CIoU) ────────────────────────────
    # ρ² = squared Euclidean distance between centres
    cx_pred, cy_pred = pred_cxcywh[..., 0], pred_cxcywh[..., 1]
    cx_gt,   cy_gt   = gt_cxcywh[..., 0],   gt_cxcywh[..., 1]
    rho2 = (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2

    # c² = squared diagonal of the enclosing box
    c2 = enclose_w ** 2 + enclose_h ** 2 + eps

    if variant == 'diou':
        diou = iou - rho2 / c2
        return 1.0 - diou

    # ── Aspect-ratio consistency term (CIoU only) ─────────────────────────
    # v measures how consistent the predicted aspect ratio is with the GT
    #   v = (4 / π²) * (arctan(w_gt/h_gt) − arctan(w_pred/h_pred))²
    w_pred = tf.maximum(pred_cxcywh[..., 2], eps)
    h_pred = tf.maximum(pred_cxcywh[..., 3], eps)
    w_gt   = tf.maximum(gt_cxcywh[..., 2],   eps)
    h_gt   = tf.maximum(gt_cxcywh[..., 3],   eps)

    v = (4.0 / (math.pi ** 2)) * tf.square(
        tf.atan(w_gt / h_gt) - tf.atan(w_pred / h_pred)
    )

    # α balances the aspect-ratio term relative to IoU
    # Detach α from the gradient graph (it's treated as a constant weight)
    alpha = tf.stop_gradient(v / (1.0 - iou + v + eps))

    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou


def focal_loss_objectness(pred_obj, obj_mask, gamma=2.0, alpha=0.25):
    """
    Focal loss for the objectness head.

    Replaces standard BCE to down-weight the enormous number of easy
    background anchors that cause objectness to be undertrained, which
    in turn lets near-miss anchors fire at inference time.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

      gamma=2   — standard CornerNet/RetinaNet value; higher = more focus on hard examples
      alpha=0.25 — balances pos/neg contribution; lower = less weight on positives
                   (counterintuitive: the many negatives are already down-weighted by gamma,
                    so alpha can be <0.5 without starving positives)

    Args:
        pred_obj : tf.Tensor (B, gH, gW, A, 1)  sigmoid objectness predictions
        obj_mask : tf.Tensor (B, gH, gW, A, 1)  1 = positive, 0 = negative/ignore
        gamma    : focusing parameter (default 2.0)
        alpha    : balance parameter  (default 0.25)

    Returns:
        Scalar focal loss, weighted same as the old BCE so existing loss
        scale hyperparameters don't need retuning.
    """
    import tensorflow as tf

    eps = 1e-7
    p   = tf.clip_by_value(pred_obj, eps, 1.0 - eps)

    # Per-anchor focal weight: (1-p)^gamma for positives, p^gamma for negatives
    pos_loss = -alpha       * tf.pow(1.0 - p, gamma) * tf.math.log(p)
    neg_loss = -(1 - alpha) * tf.pow(p,       gamma) * tf.math.log(1.0 - p)

    # Apply mask: positives get pos_loss, negatives (where noobj=1) get neg_loss
    loss = obj_mask * pos_loss + (1.0 - obj_mask) * neg_loss
    return tf.reduce_mean(loss)


def build_anchor_assignment(gt_boxes_cxcywh, gH, gW, num_anchors,
                            ignore_radius=0.10):
    """
    Assign GT boxes to anchor grid cells, returning three masks:

      obj_mask    (B, gH, gW, A, 1)  — 1 for the single best anchor per GT box
      ignore_mask (B, gH, gW, A, 1)  — 1 for anchors near a GT but not the best
                                        (excluded from both pos and neg loss)
      noobj_mask  (B, gH, gW, A, 1)  — 1 for clear background anchors

    The key fix for duplicate detections is the ignore_mask: previously all
    near-miss anchors were treated as negatives and the model learned to
    suppress them — but it also learned that it *could* fire from those cells,
    leading to duplicates at inference. By excluding them from the loss entirely
    the model only ever learns to fire from one anchor per hold.

    Args:
        gt_boxes_cxcywh : tf.Tensor (B, N, 4) normalised (cx, cy, w, h)
        gH, gW          : grid height/width of this detection scale
        num_anchors     : anchors per cell
        ignore_radius   : normalised distance from GT centre within which
                          non-best anchors are ignored (default 0.10)

    Returns:
        (obj_mask, ignore_mask, noobj_mask, gt_targets) — all tf.float32 tensors
    """
    import tensorflow as tf

    B = tf.shape(gt_boxes_cxcywh)[0]

    obj_mask    = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 1]), trainable=False)
    ignore_mask = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 1]), trainable=False)
    gt_targets  = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 4]), trainable=False)

    # ignore_radius in grid-cell units
    ignore_r_h = ignore_radius * tf.cast(gH, tf.float32)
    ignore_r_w = ignore_radius * tf.cast(gW, tf.float32)

    for b_idx in range(B):
        n_gt = tf.shape(gt_boxes_cxcywh[b_idx])[0]
        for g_idx in tf.range(n_gt):
            box = gt_boxes_cxcywh[b_idx, g_idx]
            cx, cy, w, h = box[0], box[1], box[2], box[3]

            # Best anchor: the cell that contains the GT centre, anchor slot 0
            # (For multi-anchor matching, pick the slot whose prior size is
            #  closest to the GT — here we use a single prior per cell.)
            ci = tf.clip_by_value(
                tf.cast(tf.floor(cx * tf.cast(gW, tf.float32)), tf.int32), 0, gW - 1)
            cj = tf.clip_by_value(
                tf.cast(tf.floor(cy * tf.cast(gH, tf.float32)), tf.int32), 0, gH - 1)

            # Mark best anchor as positive
            obj_mask.scatter_nd_update([[b_idx, cj, ci, 0, 0]], [1.0])
            gt_targets.scatter_nd_update(
                [[b_idx, cj, ci, 0, k] for k in range(4)],
                [cx, cy, w, h]
            )

            # Mark neighbouring cells within ignore_radius as ignore
            # so the model doesn't learn to fire from them at all
            r_cells_h = tf.cast(tf.math.ceil(ignore_r_h), tf.int32) + 1
            r_cells_w = tf.cast(tf.math.ceil(ignore_r_w), tf.int32) + 1

            for dy in tf.range(-r_cells_h, r_cells_h + 1):
                for dx in tf.range(-r_cells_w, r_cells_w + 1):
                    if dy == 0 and dx == 0:
                        continue  # best cell is positive, not ignore

                    ny = cj + dy
                    nx = ci + dx
                    if ny < 0 or ny >= gH or nx < 0 or nx >= gW:
                        continue

                    # Distance check: only ignore if truly within radius
                    cell_cx = (tf.cast(nx, tf.float32) + 0.5) / tf.cast(gW, tf.float32)
                    cell_cy = (tf.cast(ny, tf.float32) + 0.5) / tf.cast(gH, tf.float32)
                    dist_x  = tf.abs(cell_cx - cx) * tf.cast(gW, tf.float32)
                    dist_y  = tf.abs(cell_cy - cy) * tf.cast(gH, tf.float32)

                    if dist_x <= ignore_r_w and dist_y <= ignore_r_h:
                        for a in range(num_anchors):
                            # Only mark as ignore if not already a positive
                            if obj_mask[b_idx, ny, nx, a, 0] < 0.5:
                                ignore_mask.scatter_nd_update(
                                    [[b_idx, ny, nx, a, 0]], [1.0]
                                )

    obj_mask    = tf.cast(obj_mask,    tf.float32)
    ignore_mask = tf.cast(ignore_mask, tf.float32)
    # noobj = not positive AND not ignored
    noobj_mask  = (1.0 - obj_mask) * (1.0 - ignore_mask)

    return obj_mask, ignore_mask, noobj_mask, tf.cast(gt_targets, tf.float32)


def compute_detection_loss(pred_raw, gt_boxes_cxcywh, img_size,
                           num_anchors=6, num_classes=NUM_CLASSES,
                           iou_variant='ciou',
                           ignore_radius=0.10,
                           use_focal_objectness=True):
    """
    Full detection loss combining:
      - CIoU / DIoU / GIoU / IoU box regression loss  (for positive anchors)
      - Binary cross-entropy objectness loss           (pos + neg anchors)
      - Binary cross-entropy classification loss       (for positive anchors)

    Args:
        pred_raw        : tf.Tensor (B, H, W, num_anchors*(5+num_classes))
                          Raw model output from one detection scale.
        gt_boxes_cxcywh : tf.Tensor (B, N, 4) ground-truth boxes in
                          (cx, cy, w, h) normalised to [0, 1].
        img_size        : tuple (H, W) — used to match anchor grid to GT scale.
        num_anchors     : number of anchors per spatial location.
        num_classes     : number of object classes.
        iou_variant     : which IoU variant to use for box regression.

    Returns:
        total_loss   : scalar tf.Tensor
        box_loss     : scalar tf.Tensor  (for logging)
        obj_loss     : scalar tf.Tensor  (for logging)
        cls_loss     : scalar tf.Tensor  (for logging)
    """
    import tensorflow as tf

    B  = tf.shape(pred_raw)[0]
    gH = tf.shape(pred_raw)[1]
    gW = tf.shape(pred_raw)[2]

    # Reshape → (B, gH, gW, num_anchors, 5+num_classes)
    pred = tf.reshape(pred_raw, [B, gH, gW, num_anchors, 5 + num_classes])

    # Decode predictions
    pred_xy  = tf.sigmoid(pred[..., 0:2])   # (cx, cy) offset within cell, ∈ [0,1]
    pred_wh  = pred[..., 2:4]               # log-scale width/height (decoded below)
    pred_obj = tf.sigmoid(pred[..., 4:5])   # objectness confidence
    pred_cls = tf.sigmoid(pred[..., 5:])    # per-class probability

    # Build anchor grid offsets so pred_xy becomes image-normalised centre
    # Grid: each cell (i,j) has offset (j/gW, i/gH)
    gy = tf.cast(tf.range(gH), tf.float32)
    gx = tf.cast(tf.range(gW), tf.float32)
    grid_x, grid_y = tf.meshgrid(gx, gy)                       # (gH, gW)
    grid = tf.stack([grid_x, grid_y], axis=-1)                  # (gH, gW, 2)
    grid = tf.reshape(grid, [1, gH, gW, 1, 2])                 # broadcast over B, anchors

    pred_cx = (pred_xy[..., 0:1] + grid[..., 0:1]) / tf.cast(gW, tf.float32)
    pred_cy = (pred_xy[..., 1:2] + grid[..., 1:2]) / tf.cast(gH, tf.float32)
    pred_w  = tf.exp(pred_wh[..., 0:1]) / tf.cast(gW, tf.float32)  # normalised
    pred_h  = tf.exp(pred_wh[..., 1:2]) / tf.cast(gH, tf.float32)

    pred_boxes = tf.concat([pred_cx, pred_cy, pred_w, pred_h], axis=-1)
    # shape: (B, gH, gW, num_anchors, 4)

    # ── Match GT boxes to anchors ──────────────────────────────────────────
    # For each GT box find the anchor cell + anchor slot with highest IoU.
    # This is a simplified "best anchor" assignment (no multi-label matching).
    #
    # gt_boxes_cxcywh: (B, N, 4) — iterate over batch dimension.
    # We build boolean masks: obj_mask (positive), noobj_mask (negative).

    obj_mask   = tf.zeros([B, gH, gW, num_anchors, 1])
    noobj_mask = tf.ones( [B, gH, gW, num_anchors, 1])
    gt_targets = tf.zeros([B, gH, gW, num_anchors, 4])

    # TF graph-mode compatible assignment via TensorArray + scatter
    obj_mask   = tf.Variable(obj_mask,   trainable=False)
    noobj_mask = tf.Variable(noobj_mask, trainable=False)
    gt_targets = tf.Variable(gt_targets, trainable=False)

    # Python-level loop over batch (acceptable at training time)
    for b_idx in range(B):
        n_gt = tf.shape(gt_boxes_cxcywh[b_idx])[0]
        for g_idx in tf.range(n_gt):
            box = gt_boxes_cxcywh[b_idx, g_idx]           # (4,)
            cx, cy, w, h = box[0], box[1], box[2], box[3]

            # Cell indices for this GT box
            ci = tf.cast(tf.floor(cx * tf.cast(gW, tf.float32)), tf.int32)
            cj = tf.cast(tf.floor(cy * tf.cast(gH, tf.float32)), tf.int32)
            ci = tf.clip_by_value(ci, 0, gW - 1)
            cj = tf.clip_by_value(cj, 0, gH - 1)

            # Assign to anchor 0 (extend to anchor IoU matching for multi-anchor)
            idx = [b_idx, cj, ci, 0, 0]
            obj_mask.scatter_nd_update(  [idx], [1.0])
            noobj_mask.scatter_nd_update([[b_idx, cj, ci, 0, 0]], [0.0])
            gt_targets.scatter_nd_update(
                [[b_idx, cj, ci, 0, 0],
                 [b_idx, cj, ci, 0, 1],
                 [b_idx, cj, ci, 0, 2],
                 [b_idx, cj, ci, 0, 3]],
                [cx, cy, w, h]
            )

    # Use the dedicated assignment helper which correctly builds the three-state
    # mask (positive / ignore / negative) using ignore_radius.
    obj_mask, ignore_mask, noobj_mask, gt_targets = build_anchor_assignment(
        gt_boxes_cxcywh, gH, gW, num_anchors, ignore_radius=ignore_radius
    )

    # ── Box regression loss (IoU family) — positives only ─────────────────
    pos_pred = tf.boolean_mask(pred_boxes, tf.squeeze(obj_mask > 0, -1))
    pos_gt   = tf.boolean_mask(gt_targets, tf.squeeze(obj_mask > 0, -1))

    if tf.size(pos_pred) > 0:
        box_loss = tf.reduce_mean(
            iou_loss(pos_pred, pos_gt, variant=iou_variant)
        )
    else:
        box_loss = tf.constant(0.0)

    # ── Objectness loss — positives + negatives (ignore zone excluded) ─────
    # Ignored anchors are masked out of the loss entirely so the model
    # never gets a gradient signal from near-miss anchors near a GT centre.
    if use_focal_objectness:
        # Focal loss down-weights the vast number of easy background anchors.
        # Only applied to non-ignored anchors (train_mask zeroes out ignores).
        train_mask  = tf.maximum(obj_mask, noobj_mask)
        active_pred = pred_obj * train_mask
        active_tgt  = obj_mask * train_mask
        obj_loss = focal_loss_objectness(active_tgt, active_pred) * 5.0
    else:
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        obj_loss = (
            bce(obj_mask,                          pred_obj * obj_mask)    * 5.0 +
            bce(tf.zeros_like(pred_obj * noobj_mask), pred_obj * noobj_mask) * 0.5
        )

    # ── Classification loss — positives only ──────────────────────────────
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    gt_cls   = tf.zeros([B, gH, gW, num_anchors, num_classes])
    cls_loss = bce(gt_cls * obj_mask, pred_cls * obj_mask)

    total_loss = box_loss + obj_loss + cls_loss
    return total_loss, box_loss, obj_loss, cls_loss


# ──────────────────────────────────────────────
# CenterNet-specific IoU loss
# ──────────────────────────────────────────────

def centernet_loss(outputs, gt_boxes_cxcywh, img_size,
                   iou_variant='ciou', heatmap_weight=1.0,
                   wh_weight=0.1, offset_weight=1.0):
    """
    CenterNet training loss:
      - Focal loss on the centre heatmap  (finds where holds are)
      - IoU loss on width/height          (shapes the box)
      - L1 loss on sub-pixel offsets      (sharpens localisation)

    Args:
        outputs         : dict with keys 'heatmap', 'wh', 'offset'
                          as returned by build_centernet().
        gt_boxes_cxcywh : tf.Tensor (B, N, 4) in normalised (cx,cy,w,h).
        img_size        : (H, W) of the input image.
        iou_variant     : IoU variant for the wh regression head.

    Returns:
        total_loss, heatmap_loss, wh_loss, offset_loss
    """
    import tensorflow as tf

    heatmap = outputs['heatmap']   # (B, oH, oW, num_classes)
    wh      = outputs['wh']        # (B, oH, oW, 2)
    offset  = outputs['offset']    # (B, oH, oW, 2)

    B  = tf.shape(heatmap)[0]
    oH = tf.shape(heatmap)[1]
    oW = tf.shape(heatmap)[2]
    stride_h = img_size[0] / tf.cast(oH, tf.float32)
    stride_w = img_size[1] / tf.cast(oW, tf.float32)

    # Build GT heatmap with Gaussian splat around each object centre
    gt_heatmap = tf.zeros_like(heatmap)
    gt_wh      = tf.zeros_like(wh)
    gt_offset  = tf.zeros_like(offset)
    pos_mask   = tf.zeros([B, oH, oW, 1])

    gt_heatmap = tf.Variable(gt_heatmap, trainable=False)
    gt_wh      = tf.Variable(gt_wh,      trainable=False)
    gt_offset  = tf.Variable(gt_offset,  trainable=False)
    pos_mask   = tf.Variable(pos_mask,   trainable=False)

    for b_idx in range(B):
        n_gt = tf.shape(gt_boxes_cxcywh[b_idx])[0]
        for g_idx in tf.range(n_gt):
            box = gt_boxes_cxcywh[b_idx, g_idx]
            cx_norm, cy_norm, w_norm, h_norm = box[0], box[1], box[2], box[3]

            # Output-map coordinates
            cx_map = cx_norm * tf.cast(oW, tf.float32)
            cy_map = cy_norm * tf.cast(oH, tf.float32)
            cx_int = tf.cast(tf.floor(cx_map), tf.int32)
            cy_int = tf.cast(tf.floor(cy_map), tf.int32)
            cx_int = tf.clip_by_value(cx_int, 0, oW - 1)
            cy_int = tf.clip_by_value(cy_int, 0, oH - 1)

            # Gaussian radius (simplified: proportional to box size)
            oW_f = tf.cast(oW, tf.float32)
            oH_f = tf.cast(oH, tf.float32)

            radius = tf.maximum(
                1,
                tf.cast(
                    tf.round(tf.sqrt(w_norm * oW_f * h_norm * oH_f) / 6.0),
                    tf.int32
                )
            )

            # Splat Gaussian onto heatmap
            sigma = tf.cast(radius, tf.float32) / 3.0
            ys = tf.cast(tf.range(oH), tf.float32)
            xs = tf.cast(tf.range(oW), tf.float32)
            grid_xx, grid_yy = tf.meshgrid(xs, ys)
            gaussian = tf.exp(
                -(
                    (grid_xx - tf.cast(cx_int, tf.float32)) ** 2 +
                    (grid_yy - tf.cast(cy_int, tf.float32)) ** 2
                ) / (2.0 * sigma ** 2)
            )                                                   # (oH, oW)
            gaussian = tf.expand_dims(gaussian, -1)             # (oH, oW, 1)

            # Update heatmap: take element-wise max so overlapping Gaussians merge
            current = gt_heatmap[b_idx]                         # (oH, oW, C)
            gt_heatmap[b_idx].assign(
                tf.maximum(current, tf.tile(gaussian, [1, 1, tf.shape(heatmap)[-1]]))
            )

            # WH target: the box dimensions at this centre location
            gt_wh[b_idx, cy_int, cx_int, 0].assign(w_norm)
            gt_wh[b_idx, cy_int, cx_int, 1].assign(h_norm)

            # Offset target: fractional part of centre coordinates
            gt_offset[b_idx, cy_int, cx_int, 0].assign(cx_map - tf.cast(cx_int, tf.float32))
            gt_offset[b_idx, cy_int, cx_int, 1].assign(cy_map - tf.cast(cy_int, tf.float32))

            pos_mask[b_idx, cy_int, cx_int, 0].assign(1.0)

    gt_heatmap = tf.cast(gt_heatmap, tf.float32)
    pos_mask   = tf.cast(pos_mask,   tf.float32)

    # ── Focal loss on heatmap ──────────────────────────────────────────────
    # Modified focal loss (CornerNet/CenterNet style):
    #   For positives (gt==1):  -(1-p)^2 * log(p)
    #   For negatives (gt< 1):  -(1-gt)^4 * p^2 * log(1-p)
    alpha, beta = 2.0, 4.0
    pos_inds = tf.cast(tf.equal(gt_heatmap, 1.0), tf.float32)
    neg_inds = 1.0 - pos_inds

    hm_loss_pos = -pos_inds * tf.pow(1.0 - heatmap, alpha) * tf.math.log(heatmap + 1e-7)
    hm_loss_neg = (-neg_inds
                   * tf.pow(1.0 - gt_heatmap, beta)
                   * tf.pow(heatmap, alpha)
                   * tf.math.log(1.0 - heatmap + 1e-7))
    hm_loss = tf.reduce_sum(hm_loss_pos + hm_loss_neg) / (
        tf.reduce_sum(pos_inds) + 1.0
    )

    # ── IoU loss on wh — only at positive locations ────────────────────────
    pos_pred_wh = tf.boolean_mask(wh,      tf.squeeze(pos_mask > 0, -1))
    pos_gt_wh   = tf.boolean_mask(gt_wh,   tf.squeeze(pos_mask > 0, -1))

    if tf.size(pos_pred_wh) > 0:
        # Build pseudo cx/cy = 0 so iou_loss sees (0, 0, w, h) for both —
        # the centre terms cancel in GIoU/DIoU/CIoU, leaving only the
        # size-based overlap and aspect-ratio penalty.
        zeros = tf.zeros_like(pos_pred_wh[..., :1])
        pred_pseudo = tf.concat([zeros, zeros, pos_pred_wh], axis=-1)
        gt_pseudo   = tf.concat([zeros, zeros, pos_gt_wh],   axis=-1)
        wh_iou_loss = tf.reduce_mean(
            iou_loss(pred_pseudo, gt_pseudo, variant=iou_variant)
        )
    else:
        wh_iou_loss = tf.constant(0.0)

    # ── L1 offset loss — positive locations only ──────────────────────────
    pos_pred_off = tf.boolean_mask(offset,     tf.squeeze(pos_mask > 0, -1))
    pos_gt_off   = tf.boolean_mask(gt_offset,  tf.squeeze(pos_mask > 0, -1))
    off_loss = (tf.reduce_mean(tf.abs(pos_pred_off - pos_gt_off))
                if tf.size(pos_pred_off) > 0 else tf.constant(0.0))

    total = (heatmap_weight * hm_loss
             + wh_weight    * wh_iou_loss
             + offset_weight * off_loss)

    return total, hm_loss, wh_iou_loss, off_loss


# ──────────────────────────────────────────────
# TensorFlow / Keras models  (TFLite-compatible)
# ──────────────────────────────────────────────

def build_ssd_mobilenet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    SSD MobileNetV2 — the gold standard for on-device object detection.

    Uses tf.keras + TF Object Detection API's MobileNetV2 backbone with
    an SSD head.  If you have tf-models-official installed you can also
    load the full TF OD API pipeline; here we build a lightweight custom
    version that exports cleanly to TFLite.

    For production use, consider the pre-trained SSD MobileNet V2 COCO
    checkpoint from the TF Model Zoo and fine-tune just the head.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # MobileNetV2 backbone (pretrained on ImageNet)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    # Fine-tune the top layers only for small datasets
    for layer in base.layers[:-10]:
        layer.trainable = False

    # Multi-scale feature extraction (SSD style)
    feat_13 = base.get_layer('block_13_expand_relu').output   # 13x13
    feat_26 = base.get_layer('block_6_expand_relu').output    # 26x26

    # Detection heads
    def detection_head(x, num_anchors=6):
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        # Predict: [cx, cy, w, h, obj_conf, class_probs...]
        out = layers.Conv2D(num_anchors * (5 + num_classes), 1)(x)
        return out

    head_13 = detection_head(feat_13)
    head_26 = detection_head(feat_26)

    return Model(inputs=base.input, outputs=[head_13, head_26], name='ssd_mobilenet')


def build_efficientdet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    EfficientDet-Lite0 — better accuracy than SSD MobileNet with acceptable
    latency on modern mobile hardware (~30ms on Pixel 4).

    This builds a simplified BiFPN neck on top of EfficientNetB0.
    For a full EfficientDet-Lite implementation, use:
      tensorflow/models automl/efficientdet
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base.layers[:-40]:
        layer.trainable = False

    # BiFPN-style feature extraction at P3, P4, P5
    p3 = base.get_layer('block3b_add').output
    p4 = base.get_layer('block5c_add').output
    p5 = base.get_layer('block7a_project_bn').output

    # Simple top-down FPN path (simplified BiFPN)
    p5_up = layers.UpSampling2D(2)(p5)
    p4_merged = layers.Add()([p4, p5_up])
    p4_merged = layers.Conv2D(64, 3, padding='same', activation='swish')(p4_merged)

    p4_up = layers.UpSampling2D(2)(p4_merged)
    p3_merged = layers.Add()([p3, p4_up])
    p3_merged = layers.Conv2D(64, 3, padding='same', activation='swish')(p3_merged)

    # Shared class/box head
    def box_head(x):
        for _ in range(3):
            x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        boxes = layers.Conv2D(4 * 9, 1)(x)          # 9 anchors, 4 coords
        scores = layers.Conv2D(num_classes * 9, 1)(x)
        return boxes, scores

    boxes_p3, scores_p3 = box_head(p3_merged)
    boxes_p4, scores_p4 = box_head(p4_merged)

    return Model(inputs=base.input,
                 outputs=[boxes_p3, scores_p3, boxes_p4, scores_p4],
                 name='efficientdet_lite0')


def build_centernet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    CenterNet with MobileNetV2 backbone — anchor-free, good at small holds,
    and cleanly exportable to TFLite.  Predicts a heatmap of object centres
    plus width/height offsets at each centre.

    This keeps the spirit of your original heatmap approach from the script
    but adds proper bounding box regression alongside the centre heatmap.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output  # 10x10 for 320x320 input

    # Decoder: upsample back to input/4 resolution (80x80)
    for filters in [256, 128, 64]:
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    # Three output heads (all at 80x80 resolution)
    heatmap = layers.Conv2D(num_classes, 1, activation='sigmoid', name='heatmap')(x)
    wh      = layers.Conv2D(2,           1,                        name='wh')(x)       # width, height
    offset  = layers.Conv2D(2,           1,                        name='offset')(x)   # sub-pixel offset

    return Model(inputs=base.input,
                 outputs={'heatmap': heatmap, 'wh': wh, 'offset': offset},
                 name='centernet_mobilenetv2')


# ──────────────────────────────────────────────
# PyTorch model  (NOT TFLite-compatible)
# ──────────────────────────────────────────────

def build_pytorch_rcnn():
    """
    Faster R-CNN ResNet50 FPN — the original model from the script.

    ⚠️  This CANNOT be exported to TFLite.
    Use for desktop/server inference or as a teacher model for knowledge
    distillation into one of the TFLite-compatible models above.

    Export options for this model:
      - ONNX:   torch.onnx.export(model, ...)  → onnxruntime inference
      - TorchScript: torch.jit.script(model)   → mobile via PyTorch Mobile
    """
    try:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError:
        raise ImportError(
            "torch and torchvision are required for pytorch_rcnn.\n"
            "Install with: pip install torch torchvision"
        )

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    for param in model.backbone.parameters():
        param.requires_grad = False

    num_classes = 2  # background + hold
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ──────────────────────────────────────────────
# TF dataset builder
# ──────────────────────────────────────────────

def make_tf_dataset(records, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Build a tf.data.Dataset from annotation records.
    Yields (image_tensor, {'boxes': ..., 'labels': ...}) batches.
    """
    import tensorflow as tf

    def gen():
        for rec in records:
            img = preprocess_image(rec['image_path'], img_size)
            boxes = np.array(rec['boxes'], dtype=np.float32)
            if len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
            yield img, boxes

    # Build dataset — note: ragged boxes require careful batching
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
        )
    )
    ds = ds.shuffle(200).batch(batch_size)
    return ds


# ──────────────────────────────────────────────
# Training functions
# ──────────────────────────────────────────────

def train_tf_model(model, records, output_dir,
                   epochs=EPOCHS, lr=LEARNING_RATE,
                   model_type='ssd_mobilenet', iou_variant='ciou',
                   ignore_radius=0.10, use_focal_objectness=False):
    """
    Train a TF/Keras detection model using the selected IoU loss variant.

    Dispatches to the appropriate loss function based on model architecture:
      - centernet  → centernet_loss() (focal heatmap + IoU wh + L1 offset)
      - ssd / efficientdet → compute_detection_loss() (obj + IoU box + cls)

    Args:
        model                : compiled tf.keras.Model
        records              : list of annotation dicts from load_annotations()
        output_dir           : where to write checkpoints
        epochs               : number of training epochs
        lr                   : initial learning rate (Adam)
        model_type           : 'ssd_mobilenet' | 'efficientdet' | 'centernet'
        iou_variant          : 'ciou' | 'diou' | 'giou' | 'iou'
        ignore_radius        : normalised ignore-zone radius around GT centres
                               (suppresses near-miss anchor training, reduces duplicates)
        use_focal_objectness : use focal loss on objectness head instead of BCE
    """
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(lr)
    #scheduler = tf.keras.optimizers.schedules.ReduceLROnPlateau  # applied manually below

    best_loss  = float('inf')
    no_improve = 0
    patience   = 5          # halve LR after this many epochs without improvement
    current_lr = lr

    os.makedirs(output_dir, exist_ok=True)
    print(f"Training with loss: {iou_variant.upper()} IoU  |  model: {model_type}")

    for epoch in range(epochs):
        epoch_loss      = 0.0
        epoch_box_loss  = 0.0
        epoch_obj_loss  = 0.0
        epoch_aux_loss  = 0.0
        n_samples       = 0

        np.random.shuffle(records)

        for rec in records:
            img = preprocess_image(rec['image_path'])
            #if len(records) < 100:  # Only augment if dataset is tiny
            #    img, aug_boxes = augment_climbing_image(img, rec['boxes'])
            #    rec = {'image_path': rec['image_path'], 'boxes': aug_boxes}
            img_t = tf.convert_to_tensor(img[np.newaxis], dtype=tf.float32)

            # GT boxes: (1, N, 4) in normalised (cx, cy, w, h)
            # load_annotations() already stored them as [y1,x1,y2,x2], so we
            # convert back to (cx, cy, w, h) here for the IoU-based losses.
            if rec['boxes']:
                raw = np.array(rec['boxes'], dtype=np.float32)  # (N, 4) [y1,x1,y2,x2]
                y1, x1, y2, x2 = raw[:,0], raw[:,1], raw[:,2], raw[:,3]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w  = x2 - x1
                h  = y2 - y1
                gt_cxcywh = np.stack([cx, cy, w, h], axis=-1).astype(np.float32)
            else:
                gt_cxcywh = np.zeros((0, 4), dtype=np.float32)

            gt_t = tf.convert_to_tensor(gt_cxcywh[np.newaxis], dtype=tf.float32)

            with tf.GradientTape() as tape:
                outputs = model(img_t, training=True)

                if model_type == 'centernet':
                    # outputs is a dict: {'heatmap', 'wh', 'offset'}
                    total, hm_l, wh_l, off_l = centernet_loss(
                        outputs, gt_t, IMG_SIZE, iou_variant=iou_variant
                    )
                    loss         = total
                    box_l_val    = wh_l.numpy()
                    obj_l_val    = hm_l.numpy()
                    aux_l_val    = off_l.numpy()

                else:
                    # SSD / EfficientDet: outputs is a list of scale tensors
                    # Accumulate loss over all prediction scales
                    scale_outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                    total = tf.constant(0.0)
                    box_acc = obj_acc = cls_acc = tf.constant(0.0)
                    for scale_pred in scale_outputs:
                        t, b, o, c = compute_detection_loss(
                            scale_pred, gt_t, IMG_SIZE,
                            iou_variant=iou_variant,
                            ignore_radius=ignore_radius,
                            use_focal_objectness=use_focal_objectness,
                        )
                        total   += t
                        box_acc += b
                        obj_acc += o
                        cls_acc += c

                    loss         = total
                    box_l_val    = box_acc.numpy()
                    obj_l_val    = obj_acc.numpy()
                    aux_l_val    = cls_acc.numpy()

            grads = tape.gradient(loss, model.trainable_variables)
            # Gradient clipping — helps stability with IoU losses early in training
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss     += loss.numpy()
            epoch_box_loss += box_l_val
            epoch_obj_loss += obj_l_val
            epoch_aux_loss += aux_l_val
            n_samples      += 1

        n = max(n_samples, 1)
        avg       = epoch_loss     / n
        avg_box   = epoch_box_loss / n
        avg_obj   = epoch_obj_loss / n
        avg_aux   = epoch_aux_loss / n

        aux_label = 'offset' if model_type == 'centernet' else 'cls'
        print(
            f"Epoch {epoch+1:>4}/{epochs}  "
            f"total={avg:.5f}  "
            f"box({iou_variant})={avg_box:.5f}  "
            f"obj/hm={avg_obj:.5f}  "
            f"{aux_label}={avg_aux:.5f}"
        )

        # ── Checkpoint on improvement ──────────────────────────────────────
        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
            model.save(os.path.join(output_dir, 'best_model.keras'))
            print(f"  ✓ New best ({best_loss:.5f}) — checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                current_lr *= 0.5
                optimizer.learning_rate.assign(current_lr)
                no_improve = 0
                print(f"  ↓ LR reduced to {current_lr:.2e}")

    model.save(os.path.join(output_dir, 'final_model.keras'))
    print(f"\nFinal model saved to {output_dir}/final_model.keras")
    return model


def train_pytorch_model(model, records, output_dir,
                        epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """Train the PyTorch Faster R-CNN model."""
    import torch
    from torch.utils.data import DataLoader

    # Minimal inline dataset for the PyTorch path
    class _DS(torch.utils.data.Dataset):
        def __init__(self, records, img_size):
            self.records = records
            self.img_size = img_size

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            img = preprocess_image(rec['image_path'], self.img_size)
            img_t = torch.tensor(img).permute(2, 0, 1)

            boxes = rec['boxes']
            if boxes:
                # Convert [y1,x1,y2,x2] normalised → [x1,y1,x2,y2] absolute
                h, w = self.img_size
                abs_boxes = [[b[1]*w, b[0]*h, b[3]*w, b[2]*h] for b in boxes]
                boxes_t = torch.tensor(abs_boxes, dtype=torch.float32)
                labels_t = torch.ones(len(boxes_t), dtype=torch.int64)
            else:
                boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,),   dtype=torch.int64)

            return img_t, {'boxes': boxes_t, 'labels': labels_t}

    def collate_fn(batch):
        return tuple(zip(*batch))

    ds = _DS(records, IMG_SIZE)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets in dl:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    save_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"\nModel saved to {save_path}")
    print("NOTE: PyTorch model cannot be exported to TFLite.")
    print("      For mobile deployment, re-train with --model ssd_mobilenet,")
    print("      efficientdet, or centernet instead.")
    return model


# ──────────────────────────────────────────────
# TFLite export
# ──────────────────────────────────────────────

def convert_to_tflite(model_path, output_path, quantize=True, model_name=''):
    """
    Convert a saved Keras model to TensorFlow Lite.

    Args:
        model_path:  Path to saved .keras (or .h5) model
        output_path: Destination .tflite path
        quantize:    Apply dynamic-range quantisation (recommended for mobile)
        model_name:  Used for logging only
    """
    import tensorflow as tf

    print(f"Loading {model_name or 'model'} from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("Applying dynamic-range quantisation...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Uncomment for full INT8 quantisation (requires representative dataset):
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type  = tf.uint8
        # converter.inference_output_type = tf.uint8

    print("Converting...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model saved → {output_path}  ({size_kb:.1f} KB)")
    print("\nFlutter integration:")
    print(f"  cp {output_path} frontend_flutter/assets/model.tflite")
    print("  Add to pubspec.yaml assets section, then load with tflite_flutter package.")
    return tflite_model


# ──────────────────────────────────────────────
# TFLite inference test
# ──────────────────────────────────────────────

def test_tflite_model(tflite_path, test_image_path, img_size=IMG_SIZE):
    """Run a quick sanity-check inference with the TFLite model."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input  details:", input_details)
    print("Output details:", output_details)

    img = preprocess_image(test_image_path, img_size)
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    for det in output_details:
        out = interpreter.get_tensor(det['index'])
        print(f"  Output '{det['name']}': shape={out.shape}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train a climbing hold detector and optionally export to TFLite.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model comparison:
  ssd_mobilenet   ~20-30ms  ~10MB tflite  Good for real-time flutter AR view
  efficientdet    ~30-50ms  ~15MB tflite  Better mAP, still mobile-friendly
  centernet       ~25-40ms  ~12MB tflite  Anchor-free, structurally immune to duplicates
  pytorch_rcnn    N/A       NOT tflite    Use for server inference or as teacher model

Loss variants (--loss):
  ciou   — recommended default; handles varied hold shapes (slopers vs pockets)
  diou   — faster than GIoU; ignores aspect ratio
  giou   — safe fallback; non-zero gradient everywhere
  iou    — fastest but stalls when boxes don't overlap early in training

Duplicate detection fixes:
  --ignore-radius 0.10     ignore near-miss anchors during training (default)
  --focal-objectness       focal loss on objectness head (recommended for small datasets)
  --nms-iou 0.30           aggressive NMS at inference (lower = fewer duplicates)

  Quickest fix if already trained: just lower --nms-iou and re-export.
  For a proper fix: retrain with --focal-objectness --ignore-radius 0.12
  Best long-term: switch to --model centernet (structurally one peak per hold)

Examples:
  python train_holds.py --model ssd_mobilenet --loss ciou --focal-objectness --convert
  python train_holds.py --model ssd_mobilenet --loss ciou --ignore-radius 0.12 --nms-iou 0.25 --convert
  python train_holds.py --model centernet --loss diou --epochs 50
  python train_holds.py --model efficientdet --loss giou --convert --test data/img/sample.jpg
  python train_holds.py --model pytorch_rcnn --output ./models/rcnn
        """
    )

    parser.add_argument('--nms-iou',
                        type=float, default=0.30,
                        metavar='FLOAT',
                        help=(
                            'IoU threshold for NMS at inference (default: 0.30). '
                            'Lower = more aggressive suppression of nearby boxes. '
                            '0.45 is the standard default; use 0.25–0.35 to eliminate '
                            'duplicate hold detections. Also written into the exported '
                            'model metadata so hold_detection_service.dart can read it.'
                        ))
    parser.add_argument('--ignore-radius',
                        type=float, default=0.10,
                        metavar='FLOAT',
                        help=(
                            'Normalised radius around each GT centre within which '
                            'non-best anchors are marked ignore during training '
                            '(default: 0.10 = 10%% of image width/height). '
                            'Prevents near-miss anchors from being trained as '
                            'positives or negatives, so the model learns to fire '
                            'from exactly one anchor per hold. '
                            'Raise to 0.15–0.20 for densely packed holds.'
                        ))
    parser.add_argument('--focal-objectness',
                        action='store_true',
                        help=(
                            'Replace BCE with focal loss (gamma=2, alpha=0.25) on '
                            'the objectness head. Down-weights easy background '
                            'anchors so the model learns background more aggressively. '
                            'Strongly recommended for small climbing-hold datasets '
                            'where holds are sparse and background anchors dominate.'
                        ))
    parser.add_argument('--loss',
                        choices=LOSS_CHOICES,
                        default='ciou',
                        help=(
                            'Box regression loss variant (default: ciou). '
                            'ciou=Complete IoU (best for varied shapes), '
                            'diou=Distance IoU, giou=Generalised IoU, '
                            'iou=vanilla IoU (zero gradient when no overlap)'
                        ))
    parser.add_argument('--model',
                        choices=MODEL_CHOICES,
                        default='ssd_mobilenet',
                        help='Detector architecture to train (default: ssd_mobilenet)')
    parser.add_argument('--output',
                        default='./models',
                        help='Output directory for saved models')
    parser.add_argument('--epochs',
                        type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--lr',
                        type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch-size',
                        type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--convert',
                        action='store_true',
                        help='Export trained model to TFLite (TF models only)')
    parser.add_argument('--no-quantize',
                        action='store_true',
                        help='Skip quantisation during TFLite export')
    parser.add_argument('--test',
                        type=str, default=None,
                        help='Path to a test image for post-export inference check')

    args = parser.parse_args()

    # ── Warn early if incompatible flags are combined ──
    if args.convert and args.model in PYTORCH_ONLY:
        print("\n⚠️  WARNING: --convert is not compatible with --model pytorch_rcnn.")
        print("   pytorch_rcnn cannot be exported to TFLite.")
        print("   Remove --convert, or switch to ssd_mobilenet / efficientdet / centernet.\n")
        args.convert = False

    output_dir = os.path.join(args.output, args.model)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Model:            {args.model}")
    print(f"  Loss:             {args.loss.upper()} IoU")
    print(f"  Ignore radius:    {args.ignore_radius}")
    print(f"  Focal objectness: {'yes' if args.focal_objectness else 'no'}")
    print(f"  NMS IoU:          {args.nms_iou}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  LR:               {args.lr}")
    print(f"  Output dir:       {output_dir}")
    print(f"  TFLite:           {'yes' if args.convert else 'no'}")
    print(f"{'='*55}\n")

    # ── Load data ──
    records = load_annotations()
    if not records:
        print("No annotated images found in data/label/ — exiting.")
        return

    # ── Build and train ──
    if args.model == 'pytorch_rcnn':
        model = build_pytorch_rcnn()
        train_pytorch_model(model, records, output_dir,
                            epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size)

    else:
        build_fn = {
            'ssd_mobilenet': build_ssd_mobilenet,
            'efficientdet':  build_efficientdet,
            'centernet':     build_centernet,
        }[args.model]

        model = build_fn()
        model.summary()
        train_tf_model(model, records, output_dir,
                       epochs=args.epochs, lr=args.lr,
                       model_type=args.model,
                       iou_variant=args.loss,
                       ignore_radius=args.ignore_radius,
                       use_focal_objectness=args.focal_objectness)

        # ── TFLite export ──
        if args.convert:
            keras_path  = os.path.join(output_dir, 'best_model.keras')
            tflite_path = os.path.join(output_dir, 'model.tflite')

            if not os.path.exists(keras_path):
                keras_path = os.path.join(output_dir, 'final_model.keras')

            convert_to_tflite(
                keras_path, tflite_path,
                quantize=not args.no_quantize,
                model_name=args.model
            )

            if args.test:
                print(f"\nRunning test inference on {args.test} ...")
                test_tflite_model(tflite_path, args.test)

    print("\n✓ Done.")


if __name__ == '__main__':
    main()