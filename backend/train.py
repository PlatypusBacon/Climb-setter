"""
Climbing Hold Detection Model Training Script
==============================================

Supports multiple detector backends, all exportable to TensorFlow Lite
for Flutter mobile deployment.

Available models (via --model flag):
  ssd_mobilenet   - SSD MobileNetV2 (fastest, smallest, best for mobile) [DEFAULT]
  efficientdet    - EfficientDet-Lite0 (better accuracy, still mobile-friendly)
  centernet       - CenterNet MobileNetV2 (anchor-free, good small-object detection)
  pytorch_rcnn    - Faster R-CNN ResNet50 (NOT exportable to TFLite)

Recommended training commands:
  # CenterNet — best for hold detection (anchor-free, no duplicate detections)
  python train_holds.py --model centernet --loss ciou --epochs 200 --lr 1e-4 --convert

  # SSD MobileNet — fastest inference, good baseline
  python train_holds.py --model ssd_mobilenet --loss ciou --focal-objectness --epochs 200 --convert

Available box regression losses (via --loss flag):
  ciou    - Complete IoU [DEFAULT] — best for varied hold shapes
  diou    - Distance IoU
  giou    - Generalised IoU
  iou     - Vanilla IoU (zero gradient when boxes don't overlap)
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

IMG_SIZE      = (320, 320)
BATCH_SIZE    = 8
EPOCHS        = 200
LEARNING_RATE = 1e-4
NUM_CLASSES   = 1

HOLD_TYPES    = ['jug', 'crimp', 'sloper', 'pinch', 'pocket', 'unknown']
MODEL_CHOICES = ['ssd_mobilenet', 'efficientdet', 'centernet', 'pytorch_rcnn']
LOSS_CHOICES  = ['ciou', 'diou', 'giou', 'iou']
TFLITE_COMPATIBLE = {'ssd_mobilenet', 'efficientdet', 'centernet'}
PYTORCH_ONLY      = {'pytorch_rcnn'}


# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────

def augment_climbing_image(image, boxes, img_size=(320, 320)):
    """
    Apply climbing-specific augmentations.

    Args:
        image    : np.ndarray (H, W, 3) float32 in [0, 1]
        boxes    : list of [y1, x1, y2, x2] normalised coords
        img_size : (W, H) target for cv2.resize

    Returns:
        aug_image (float32 [0,1]), aug_boxes (list of [y1,x1,y2,x2])
    """
    if np.random.rand() < 0.7:
        image, boxes = _random_perspective(image, boxes, severity=0.15)
    if np.random.rand() < 0.8:
        image = _random_brightness_contrast(image)
    if np.random.rand() < 0.6:
        image = _random_color_jitter(image)
    if np.random.rand() < 0.3:
        image = _random_blur(image)
    if np.random.rand() < 0.4:
        image = _random_noise(image, sigma=0.02)
    if np.random.rand() < 0.5:
        image = _random_cutout(image, max_holes=3, hole_size=0.15)

    image = cv2.resize(image, img_size)
    return image, boxes


def _random_perspective(image, boxes, severity=0.15):
    """
    Random perspective warp with correct float32 dst points.

    BUG FIX: np.random.randint returns int64. Adding int64 to the float32
    src array silently promotes the result to int64, which causes:
      cv2.error: Assertion failed src.checkVector(2, CV_32F) == 4
    Fix: cast the offset array explicitly to float32 before adding.
    """
    h, w = image.shape[:2]

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    max_offset = int(min(w, h) * severity)
    # FIX: explicit .astype(np.float32) so src + offset stays float32
    offset = np.random.randint(-max_offset, max_offset + 1,
                               src.shape).astype(np.float32)
    dst = src + offset

    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderValue=(0.5, 0.5, 0.5))

    warped_boxes = []
    for box in boxes:
        y1, x1, y2, x2 = box
        corners = np.array([
            [x1*w, y1*h], [x2*w, y1*h],
            [x2*w, y2*h], [x1*w, y2*h],
        ], dtype=np.float32).reshape(-1, 1, 2)

        wc = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        new_x1 = float(np.clip(wc[:, 0].min() / w, 0, 1))
        new_x2 = float(np.clip(wc[:, 0].max() / w, 0, 1))
        new_y1 = float(np.clip(wc[:, 1].min() / h, 0, 1))
        new_y2 = float(np.clip(wc[:, 1].max() / h, 0, 1))

        old_area = (x2 - x1) * (y2 - y1)
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        if old_area > 0 and new_area > old_area * 0.1:
            warped_boxes.append([new_y1, new_x1, new_y2, new_x2])

    return warped, warped_boxes


def _random_brightness_contrast(image, brightness_range=0.3, contrast_range=0.3):
    brightness = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    image = image * brightness
    contrast = 1.0 + np.random.uniform(-contrast_range, contrast_range)
    mean  = image.mean()
    image = (image - mean) * contrast + mean
    return np.clip(image, 0, 1)


def _random_color_jitter(image, hue_shift=0.05, sat_scale=0.3):
    hsv = cv2.cvtColor((image * 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] += np.random.uniform(-hue_shift, hue_shift) * 180
    hsv[..., 0]  = np.clip(hsv[..., 0], 0, 180)
    hsv[..., 1] *= (1.0 + np.random.uniform(-sat_scale, sat_scale))
    hsv[..., 1]  = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8),
                        cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def _random_blur(image):
    kernel = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def _random_noise(image, sigma=0.02):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def _random_cutout(image, max_holes=3, hole_size=0.15):
    h, w     = image.shape[:2]
    n_holes  = np.random.randint(1, max_holes + 1)
    for _ in range(n_holes):
        hole_h = max(1, int(h * hole_size * np.random.uniform(0.5, 1.0)))
        hole_w = max(1, int(w * hole_size * np.random.uniform(0.5, 1.0)))
        y = np.random.randint(0, max(h - hole_h, 1))
        x = np.random.randint(0, max(w - hole_w, 1))
        mean_color = image[y:y+hole_h, x:x+hole_w].mean(axis=(0, 1))
        image[y:y+hole_h, x:x+hole_w] = mean_color
    return image


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

def load_annotations(annotations_path='data/label', images_path='data/img'):
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
            xmin = max(0.0, (cx - bw/2) / w)
            ymin = max(0.0, (cy - bh/2) / h)
            xmax = min(1.0, (cx + bw/2) / w)
            ymax = min(1.0, (cy + bh/2) / h)
            boxes.append([ymin, xmin, ymax, xmax])

        records.append({'image_path': img_path, 'boxes': boxes})

    print(f"Loaded {len(records)} annotated images.")
    return records


def preprocess_image(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img.astype(np.float32) / 255.0


def overlay_heatmap(image, heatmap):
    heatmap = (heatmap.squeeze() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted((image * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0)


def draw_predictions(image, boxes, scores, threshold=0.5):
    img  = (image * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        y1, x1, y2, x2 = box
        cv2.rectangle(img, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1*w), int(y1*h)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


# ──────────────────────────────────────────────
# IoU loss family
# ──────────────────────────────────────────────

def _boxes_to_corners(boxes):
    import tensorflow as tf
    cx, cy, w, h = boxes[...,0], boxes[...,1], boxes[...,2], boxes[...,3]
    return tf.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=-1)


def _intersection_area(b1, b2):
    import tensorflow as tf
    iw = tf.maximum(0.0, tf.minimum(b1[...,2], b2[...,2]) - tf.maximum(b1[...,0], b2[...,0]))
    ih = tf.maximum(0.0, tf.minimum(b1[...,3], b2[...,3]) - tf.maximum(b1[...,1], b2[...,1]))
    return iw * ih


def iou_loss(pred_cxcywh, gt_cxcywh, variant='ciou', eps=1e-7):
    import tensorflow as tf, math

    pred  = _boxes_to_corners(pred_cxcywh)
    gt    = _boxes_to_corners(gt_cxcywh)
    inter = _intersection_area(pred, gt)
    area_pred = pred_cxcywh[...,2] * pred_cxcywh[...,3]
    area_gt   = gt_cxcywh[...,2]   * gt_cxcywh[...,3]
    union = area_pred + area_gt - inter + eps
    iou   = inter / union

    if variant == 'iou':
        return 1.0 - iou

    enclose_x1 = tf.minimum(pred[...,0], gt[...,0])
    enclose_y1 = tf.minimum(pred[...,1], gt[...,1])
    enclose_x2 = tf.maximum(pred[...,2], gt[...,2])
    enclose_y2 = tf.maximum(pred[...,3], gt[...,3])
    enclose_w  = tf.maximum(0.0, enclose_x2 - enclose_x1)
    enclose_h  = tf.maximum(0.0, enclose_y2 - enclose_y1)

    if variant == 'giou':
        enc_area = enclose_w * enclose_h + eps
        return 1.0 - (iou - (enc_area - union) / enc_area)

    rho2 = ((pred_cxcywh[...,0]-gt_cxcywh[...,0])**2 +
            (pred_cxcywh[...,1]-gt_cxcywh[...,1])**2)
    c2   = enclose_w**2 + enclose_h**2 + eps

    if variant == 'diou':
        return 1.0 - (iou - rho2/c2)

    # ciou
    w_pred = tf.maximum(pred_cxcywh[...,2], eps)
    h_pred = tf.maximum(pred_cxcywh[...,3], eps)
    w_gt   = tf.maximum(gt_cxcywh[...,2],   eps)
    h_gt   = tf.maximum(gt_cxcywh[...,3],   eps)
    v     = (4.0/(math.pi**2)) * tf.square(tf.atan(w_gt/h_gt) - tf.atan(w_pred/h_pred))
    alpha = tf.stop_gradient(v / (1.0 - iou + v + eps))
    return 1.0 - (iou - rho2/c2 - alpha*v)


def focal_loss_objectness(pred_obj, obj_mask, gamma=2.0, alpha=0.25):
    import tensorflow as tf
    eps = 1e-7
    p   = tf.clip_by_value(pred_obj, eps, 1.0-eps)
    loss = (obj_mask      * (-alpha     * tf.pow(1.0-p, gamma) * tf.math.log(p)) +
            (1.0-obj_mask) * (-(1-alpha) * tf.pow(p,     gamma) * tf.math.log(1.0-p)))
    return tf.reduce_mean(loss)


def build_anchor_assignment(gt_boxes_cxcywh, gH, gW, num_anchors, ignore_radius=0.10):
    import tensorflow as tf
    B           = tf.shape(gt_boxes_cxcywh)[0]
    obj_mask    = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 1]), trainable=False)
    ignore_mask = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 1]), trainable=False)
    gt_targets  = tf.Variable(tf.zeros([B, gH, gW, num_anchors, 4]), trainable=False)
    ignore_r_h  = ignore_radius * tf.cast(gH, tf.float32)
    ignore_r_w  = ignore_radius * tf.cast(gW, tf.float32)

    for b_idx in range(B):
        for g_idx in tf.range(tf.shape(gt_boxes_cxcywh[b_idx])[0]):
            box = gt_boxes_cxcywh[b_idx, g_idx]
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            ci = tf.clip_by_value(tf.cast(tf.floor(cx*tf.cast(gW, tf.float32)), tf.int32), 0, gW-1)
            cj = tf.clip_by_value(tf.cast(tf.floor(cy*tf.cast(gH, tf.float32)), tf.int32), 0, gH-1)
            obj_mask.scatter_nd_update([[b_idx, cj, ci, 0, 0]], [1.0])
            gt_targets.scatter_nd_update([[b_idx, cj, ci, 0, k] for k in range(4)], [cx, cy, w, h])
            r_h = tf.cast(tf.math.ceil(ignore_r_h), tf.int32) + 1
            r_w = tf.cast(tf.math.ceil(ignore_r_w), tf.int32) + 1
            for dy in tf.range(-r_h, r_h+1):
                for dx in tf.range(-r_w, r_w+1):
                    if dy == 0 and dx == 0: continue
                    ny, nx = cj+dy, ci+dx
                    if ny < 0 or ny >= gH or nx < 0 or nx >= gW: continue
                    cell_cx = (tf.cast(nx, tf.float32)+0.5)/tf.cast(gW, tf.float32)
                    cell_cy = (tf.cast(ny, tf.float32)+0.5)/tf.cast(gH, tf.float32)
                    if (tf.abs(cell_cx-cx)*tf.cast(gW, tf.float32) <= ignore_r_w and
                        tf.abs(cell_cy-cy)*tf.cast(gH, tf.float32) <= ignore_r_h):
                        for a in range(num_anchors):
                            if obj_mask[b_idx, ny, nx, a, 0] < 0.5:
                                ignore_mask.scatter_nd_update([[b_idx, ny, nx, a, 0]], [1.0])

    obj_mask    = tf.cast(obj_mask,    tf.float32)
    ignore_mask = tf.cast(ignore_mask, tf.float32)
    noobj_mask  = (1.0 - obj_mask) * (1.0 - ignore_mask)
    return obj_mask, ignore_mask, noobj_mask, tf.cast(gt_targets, tf.float32)


def compute_detection_loss(pred_raw, gt_boxes_cxcywh, img_size,
                           num_anchors=6, num_classes=NUM_CLASSES,
                           iou_variant='ciou', ignore_radius=0.10,
                           use_focal_objectness=True):
    import tensorflow as tf
    B  = tf.shape(pred_raw)[0]
    gH = tf.shape(pred_raw)[1]
    gW = tf.shape(pred_raw)[2]

    pred     = tf.reshape(pred_raw, [B, gH, gW, num_anchors, 5+num_classes])
    pred_xy  = tf.sigmoid(pred[..., 0:2])
    pred_wh  = pred[..., 2:4]
    pred_obj = tf.sigmoid(pred[..., 4:5])
    pred_cls = tf.sigmoid(pred[..., 5:])

    gx, gy   = tf.meshgrid(tf.cast(tf.range(gW), tf.float32),
                             tf.cast(tf.range(gH), tf.float32))
    grid     = tf.reshape(tf.stack([gx, gy], axis=-1), [1, gH, gW, 1, 2])
    pred_cx  = (pred_xy[...,0:1] + grid[...,0:1]) / tf.cast(gW, tf.float32)
    pred_cy  = (pred_xy[...,1:2] + grid[...,1:2]) / tf.cast(gH, tf.float32)
    pred_w   = tf.exp(pred_wh[...,0:1]) / tf.cast(gW, tf.float32)
    pred_h   = tf.exp(pred_wh[...,1:2]) / tf.cast(gH, tf.float32)
    pred_boxes = tf.concat([pred_cx, pred_cy, pred_w, pred_h], axis=-1)

    obj_mask, ignore_mask, noobj_mask, gt_targets = build_anchor_assignment(
        gt_boxes_cxcywh, gH, gW, num_anchors, ignore_radius=ignore_radius)

    pos_pred = tf.boolean_mask(pred_boxes, tf.squeeze(obj_mask > 0, -1))
    pos_gt   = tf.boolean_mask(gt_targets, tf.squeeze(obj_mask > 0, -1))
    box_loss = (tf.reduce_mean(iou_loss(pos_pred, pos_gt, variant=iou_variant))
                if tf.size(pos_pred) > 0 else tf.constant(0.0))

    if use_focal_objectness:
        train_mask = tf.maximum(obj_mask, noobj_mask)
        obj_loss   = focal_loss_objectness(pred_obj*train_mask, obj_mask*train_mask) * 5.0
    else:
        bce      = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        obj_loss = (bce(obj_mask, pred_obj*obj_mask) * 5.0 +
                    bce(tf.zeros_like(pred_obj*noobj_mask), pred_obj*noobj_mask) * 0.5)

    bce      = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    gt_cls   = tf.zeros([B, gH, gW, num_anchors, num_classes])
    cls_loss = bce(gt_cls*obj_mask, pred_cls*obj_mask)

    return box_loss+obj_loss+cls_loss, box_loss, obj_loss, cls_loss


# ──────────────────────────────────────────────
# CenterNet loss
# ──────────────────────────────────────────────

def centernet_loss(outputs, gt_boxes_cxcywh, img_size,
                   iou_variant='ciou',
                   heatmap_weight=2.0,   # raised from 1.0 — heatmap must converge first
                   wh_weight=0.5,        # raised from 0.1 — stronger wh signal
                   offset_weight=1.0):
    """
    CenterNet training loss.

    Fixes vs original:
      1. Gaussian radius capped at 4 cells (÷10 instead of ÷6).
         Large radius → diffuse supervision → flat heatmap (max~0.2, seen in debug).
      2. WH loss = 0.5*L1 + 0.5*IoU. Pure IoU gives zero gradient when predicted
         box doesn't overlap GT at all (common early in training).
      3. Offset loss correctly uses off_loss, not wh_iou_loss (was a copy-paste bug).
      4. heatmap_weight=2.0, wh_weight=0.5 (better balance).
    """
    import tensorflow as tf

    heatmap = outputs['heatmap']
    wh      = outputs['wh']
    offset  = outputs['offset']

    B  = tf.shape(heatmap)[0]
    oH = tf.shape(heatmap)[1]
    oW = tf.shape(heatmap)[2]

    gt_heatmap = tf.Variable(tf.zeros_like(heatmap),   trainable=False)
    gt_wh      = tf.Variable(tf.zeros_like(wh),        trainable=False)
    gt_offset  = tf.Variable(tf.zeros_like(offset),    trainable=False)
    pos_mask   = tf.Variable(tf.zeros([B, oH, oW, 1]), trainable=False)

    for b_idx in range(B):
        for g_idx in tf.range(tf.shape(gt_boxes_cxcywh[b_idx])[0]):
            box = gt_boxes_cxcywh[b_idx, g_idx]
            cx_norm, cy_norm, w_norm, h_norm = box[0], box[1], box[2], box[3]

            cx_map = cx_norm * tf.cast(oW, tf.float32)
            cy_map = cy_norm * tf.cast(oH, tf.float32)
            cx_int = tf.clip_by_value(tf.cast(tf.floor(cx_map), tf.int32), 0, oW-1)
            cy_int = tf.clip_by_value(tf.cast(tf.floor(cy_map), tf.int32), 0, oH-1)

            oW_f = tf.cast(oW, tf.float32)
            oH_f = tf.cast(oH, tf.float32)

            # FIX 1: divide by 10 and cap at 4 — prevents diffuse supervision
            radius = tf.minimum(
                tf.maximum(1, tf.cast(
                    tf.round(tf.sqrt(w_norm*oW_f * h_norm*oH_f) / 10.0), tf.int32)),
                4)
            sigma = tf.cast(radius, tf.float32) / 3.0

            ys = tf.cast(tf.range(oH), tf.float32)
            xs = tf.cast(tf.range(oW), tf.float32)
            grid_xx, grid_yy = tf.meshgrid(xs, ys)
            gaussian = tf.expand_dims(tf.exp(
                -((grid_xx - tf.cast(cx_int, tf.float32))**2 +
                  (grid_yy - tf.cast(cy_int, tf.float32))**2) / (2.0*sigma**2)
            ), -1)

            gt_heatmap[b_idx].assign(
                tf.maximum(gt_heatmap[b_idx],
                           tf.tile(gaussian, [1, 1, tf.shape(heatmap)[-1]])))

            gt_wh[b_idx, cy_int, cx_int, 0].assign(w_norm)
            gt_wh[b_idx, cy_int, cx_int, 1].assign(h_norm)
            gt_offset[b_idx, cy_int, cx_int, 0].assign(cx_map - tf.cast(cx_int, tf.float32))
            gt_offset[b_idx, cy_int, cx_int, 1].assign(cy_map - tf.cast(cy_int, tf.float32))
            pos_mask[b_idx, cy_int, cx_int, 0].assign(1.0)

    gt_heatmap = tf.cast(gt_heatmap, tf.float32)
    pos_mask   = tf.cast(pos_mask,   tf.float32)

    # ── Heatmap focal loss (CornerNet style) ──────────────────────────────
    pos_inds = tf.cast(tf.equal(gt_heatmap, 1.0), tf.float32)
    neg_inds = 1.0 - pos_inds
    hm_loss  = tf.reduce_sum(
        -pos_inds * tf.pow(1.0-heatmap, 2.0) * tf.math.log(heatmap + 1e-7)
        -neg_inds * tf.pow(1.0-gt_heatmap, 4.0) * tf.pow(heatmap, 2.0) * tf.math.log(1.0-heatmap + 1e-7)
    ) / (tf.reduce_sum(pos_inds) + 1.0)

    # ── WH loss: L1 + IoU blend ───────────────────────────────────────────
    # FIX 2: L1 component gives gradient even when boxes don't overlap
    pos_pred_wh = tf.boolean_mask(wh,    tf.squeeze(pos_mask > 0, -1))
    pos_gt_wh   = tf.boolean_mask(gt_wh, tf.squeeze(pos_mask > 0, -1))
    if tf.size(pos_pred_wh) > 0:
        wh_l1   = tf.reduce_mean(tf.abs(pos_pred_wh - pos_gt_wh))
        zeros   = tf.zeros_like(pos_pred_wh[..., :1])
        wh_iou  = tf.reduce_mean(iou_loss(
            tf.concat([zeros, zeros, pos_pred_wh], axis=-1),
            tf.concat([zeros, zeros, pos_gt_wh],   axis=-1),
            variant=iou_variant))
        wh_loss = 0.5 * wh_l1 + 0.5 * wh_iou
    else:
        wh_loss = tf.constant(0.0)

    # ── Offset L1 loss ────────────────────────────────────────────────────
    # FIX 3: was accidentally using wh_iou_loss here in the previous version
    pos_pred_off = tf.boolean_mask(offset,    tf.squeeze(pos_mask > 0, -1))
    pos_gt_off   = tf.boolean_mask(gt_offset, tf.squeeze(pos_mask > 0, -1))
    off_loss = (tf.reduce_mean(tf.abs(pos_pred_off - pos_gt_off))
                if tf.size(pos_pred_off) > 0 else tf.constant(0.0))

    total = heatmap_weight*hm_loss + wh_weight*wh_loss + offset_weight*off_loss
    return total, hm_loss, wh_loss, off_loss


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

def build_ssd_mobilenet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    for layer in base.layers[:-10]:
        layer.trainable = False

    feat_13 = base.get_layer('block_13_expand_relu').output
    feat_26 = base.get_layer('block_6_expand_relu').output

    def detection_head(x, num_anchors=6):
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        return layers.Conv2D(num_anchors * (5 + num_classes), 1)(x)

    return Model(inputs=base.input,
                 outputs=[detection_head(feat_13), detection_head(feat_26)],
                 name='ssd_mobilenet')


def build_efficientdet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    for layer in base.layers[:-40]:
        layer.trainable = False

    p3 = base.get_layer('block3b_add').output
    p4 = base.get_layer('block5c_add').output
    p5 = base.get_layer('block7a_project_bn').output

    p5_up     = layers.UpSampling2D(2)(p5)
    p4_merged = layers.Conv2D(64, 3, padding='same', activation='swish')(
                    layers.Add()([p4, p5_up]))
    p3_merged = layers.Conv2D(64, 3, padding='same', activation='swish')(
                    layers.Add()([p3, layers.UpSampling2D(2)(p4_merged)]))

    def box_head(x):
        for _ in range(3):
            x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        return layers.Conv2D(4*9, 1)(x), layers.Conv2D(num_classes*9, 1)(x)

    bp3, sp3 = box_head(p3_merged)
    bp4, sp4 = box_head(p4_merged)
    return Model(inputs=base.input, outputs=[bp3, sp3, bp4, sp4],
                 name='efficientdet_lite0')


def build_centernet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    CenterNet MobileNetV2.
    Uses Conv2DTranspose (not UpSampling2D) so TFLite export avoids the
    INT8/FLOAT32 TRANSPOSE_CONV mismatch error from dynamic-range quantisation.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output   # 10×10 for 320×320 input
    for filters in [256, 128, 64]:
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    heatmap = layers.Conv2D(num_classes, 1, activation='sigmoid', name='heatmap')(x)
    wh      = layers.Conv2D(2,           1,                        name='wh')(x)
    offset  = layers.Conv2D(2,           1,                        name='offset')(x)

    return Model(inputs=base.input,
                 outputs={'heatmap': heatmap, 'wh': wh, 'offset': offset},
                 name='centernet_mobilenetv2')


def build_pytorch_rcnn():
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError:
        raise ImportError("pip install torch torchvision")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


# ──────────────────────────────────────────────
# TF dataset
# ──────────────────────────────────────────────

def make_tf_dataset(records, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    import tensorflow as tf

    def gen():
        for rec in records:
            img   = preprocess_image(rec['image_path'], img_size)
            boxes = np.array(rec['boxes'], dtype=np.float32)
            if len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
            yield img, boxes

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(*img_size, 3), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
        ))
    return ds.shuffle(200).batch(batch_size)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_tf_model(model, records, output_dir,
                   epochs=EPOCHS, lr=LEARNING_RATE,
                   model_type='ssd_mobilenet', iou_variant='ciou',
                   ignore_radius=0.10, use_focal_objectness=False):
    import tensorflow as tf

    optimizer  = tf.keras.optimizers.Adam(lr)
    best_loss  = float('inf')
    no_improve = 0
    patience   = 5
    current_lr = lr

    os.makedirs(output_dir, exist_ok=True)
    print(f"Training  model={model_type}  loss={iou_variant.upper()}  "
          f"epochs={epochs}  lr={lr}")

    for epoch in range(epochs):
        epoch_loss = epoch_box = epoch_obj = epoch_aux = 0.0
        n_samples  = 0
        np.random.shuffle(records)

        for rec in records:
            img   = preprocess_image(rec['image_path'])
            boxes = list(rec['boxes'])   # copy so original record isn't mutated

            # Always augment — essential for generalisation on small datasets.
            img, boxes = augment_climbing_image(img, boxes, img_size=IMG_SIZE)

            img_t = tf.convert_to_tensor(img[np.newaxis], dtype=tf.float32)

            if boxes:
                raw = np.array(boxes, dtype=np.float32)   # [y1,x1,y2,x2]
                y1, x1, y2, x2 = raw[:,0], raw[:,1], raw[:,2], raw[:,3]
                gt_cxcywh = np.stack(
                    [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=-1
                ).astype(np.float32)
            else:
                gt_cxcywh = np.zeros((0, 4), dtype=np.float32)

            gt_t = tf.convert_to_tensor(gt_cxcywh[np.newaxis], dtype=tf.float32)

            with tf.GradientTape() as tape:
                outputs = model(img_t, training=True)
                if model_type == 'centernet':
                    total, hm_l, wh_l, off_l = centernet_loss(
                        outputs, gt_t, IMG_SIZE, iou_variant=iou_variant)
                    loss      = total
                    box_l_val = wh_l.numpy()
                    obj_l_val = hm_l.numpy()
                    aux_l_val = off_l.numpy()
                else:
                    scale_outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                    total = box_acc = obj_acc = cls_acc = tf.constant(0.0)
                    for sp in scale_outs:
                        t, b, o, c = compute_detection_loss(
                            sp, gt_t, IMG_SIZE,
                            iou_variant=iou_variant,
                            ignore_radius=ignore_radius,
                            use_focal_objectness=use_focal_objectness)
                        total += t; box_acc += b; obj_acc += o; cls_acc += c
                    loss      = total
                    box_l_val = box_acc.numpy()
                    obj_l_val = obj_acc.numpy()
                    aux_l_val = cls_acc.numpy()

            grads, _ = tf.clip_by_global_norm(
                tape.gradient(loss, model.trainable_variables), 10.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            epoch_box  += box_l_val
            epoch_obj  += obj_l_val
            epoch_aux  += aux_l_val
            n_samples  += 1

        n       = max(n_samples, 1)
        avg     = epoch_loss / n
        aux_lbl = 'offset' if model_type == 'centernet' else 'cls'
        print(f"Epoch {epoch+1:>4}/{epochs}  total={avg:.5f}  "
              f"box({iou_variant})={epoch_box/n:.5f}  "
              f"hm/obj={epoch_obj/n:.5f}  {aux_lbl}={epoch_aux/n:.5f}")

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
    import torch
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self, r, s): self.r, self.s = r, s
        def __len__(self): return len(self.r)
        def __getitem__(self, i):
            rec   = self.r[i]
            img_t = torch.tensor(preprocess_image(rec['image_path'], self.s)).permute(2, 0, 1)
            if rec['boxes']:
                h, w = self.s
                abs_b = [[b[1]*w, b[0]*h, b[3]*w, b[2]*h] for b in rec['boxes']]
                return img_t, {'boxes':  torch.tensor(abs_b, dtype=torch.float32),
                               'labels': torch.ones(len(abs_b), dtype=torch.int64)}
            return img_t, {'boxes':  torch.zeros((0,4), dtype=torch.float32),
                           'labels': torch.zeros((0,),  dtype=torch.int64)}

    dl     = DataLoader(_DS(records, IMG_SIZE), batch_size=batch_size,
                        shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, targets in dl:
            imgs    = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss    = sum(model(imgs, targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
            total  += loss.item()
        sched.step(total)
        print(f"Epoch {epoch+1}/{epochs}  loss={total:.4f}")

    save_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"\nModel saved to {save_path}")
    return model


# ──────────────────────────────────────────────
# TFLite export
# ──────────────────────────────────────────────

def convert_to_tflite(model_path, output_path, quantize=True, model_name=''):
    import tensorflow as tf

    print(f"Loading {model_name or 'model'} from {model_path} ...")
    model     = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # CenterNet: Conv2DTranspose + dynamic-range quant = INT8 weights / FLOAT32
    # activations → TRANSPOSE_CONV kernel rejects this. Keep FLOAT32 throughout.
    if quantize and model_name != 'centernet':
        print("Applying dynamic-range quantisation...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif model_name == 'centernet':
        print("CenterNet: skipping quantisation to avoid TRANSPOSE_CONV type mismatch.")

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Print exact tensor order — cross-check with Dart log output
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    print("TFLite output tensor order:")
    for d in interp.get_output_details():
        print(f"  [{d['index']}] '{d['name']}'  "
              f"shape={d['shape'].tolist()}  dtype={d['dtype'].__name__}")

    print(f"\nTFLite model → {output_path}  ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model


def test_tflite_model(tflite_path, test_image_path, img_size=IMG_SIZE):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp  = interp.get_input_details()
    outp = interp.get_output_details()
    img  = np.expand_dims(preprocess_image(test_image_path, img_size), 0)
    interp.set_tensor(inp[0]['index'], img)
    interp.invoke()
    for d in outp:
        out = interp.get_tensor(d['index'])
        print(f"  '{d['name']}': shape={out.shape}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train a climbing hold detector and export to TFLite.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended commands:
  python train_holds.py --model centernet --loss ciou --epochs 200 --lr 1e-4 --convert
  python train_holds.py --model ssd_mobilenet --loss ciou --focal-objectness --epochs 200 --convert
        """
    )
    parser.add_argument('--nms-iou',          type=float, default=0.30)
    parser.add_argument('--ignore-radius',    type=float, default=0.10)
    parser.add_argument('--focal-objectness', action='store_true')
    parser.add_argument('--loss',             choices=LOSS_CHOICES, default='ciou')
    parser.add_argument('--model',            choices=MODEL_CHOICES, default='ssd_mobilenet')
    parser.add_argument('--output',           default='./models')
    parser.add_argument('--epochs',           type=int,   default=EPOCHS)
    parser.add_argument('--lr',               type=float, default=LEARNING_RATE)
    parser.add_argument('--batch-size',       type=int,   default=BATCH_SIZE)
    parser.add_argument('--convert',          action='store_true')
    parser.add_argument('--no-quantize',      action='store_true')
    parser.add_argument('--test',             type=str,   default=None)
    args = parser.parse_args()

    if args.convert and args.model in PYTORCH_ONLY:
        print("⚠️  pytorch_rcnn cannot be exported to TFLite. Ignoring --convert.")
        args.convert = False

    output_dir = os.path.join(args.output, args.model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Model:            {args.model}")
    print(f"  Loss:             {args.loss.upper()} IoU")
    print(f"  Ignore radius:    {args.ignore_radius}")
    print(f"  Focal objectness: {'yes' if args.focal_objectness else 'no'}")
    print(f"  NMS IoU:          {args.nms_iou}")
    print(f"  Epochs:           {args.epochs}  LR: {args.lr}")
    print(f"  Output:           {output_dir}")
    print(f"  TFLite export:    {'yes' if args.convert else 'no'}")
    print(f"{'='*55}\n")

    records = load_annotations()
    if not records:
        print("No annotated images found in data/label/ — exiting.")
        return

    if args.model == 'pytorch_rcnn':
        train_pytorch_model(build_pytorch_rcnn(), records, output_dir,
                            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    else:
        build_fn = {'ssd_mobilenet': build_ssd_mobilenet,
                    'efficientdet':  build_efficientdet,
                    'centernet':     build_centernet}[args.model]
        model = build_fn()
        model.summary()
        train_tf_model(model, records, output_dir,
                       epochs=args.epochs, lr=args.lr,
                       model_type=args.model, iou_variant=args.loss,
                       ignore_radius=args.ignore_radius,
                       use_focal_objectness=args.focal_objectness)

        if args.convert:
            keras_path  = os.path.join(output_dir, 'best_model.keras')
            tflite_path = os.path.join(output_dir, 'model.tflite')
            if not os.path.exists(keras_path):
                keras_path = os.path.join(output_dir, 'final_model.keras')
            convert_to_tflite(keras_path, tflite_path,
                               quantize=not args.no_quantize,
                               model_name=args.model)
            if args.test:
                test_tflite_model(tflite_path, args.test)

    print("\n✓ Done.")


if __name__ == '__main__':
    main()