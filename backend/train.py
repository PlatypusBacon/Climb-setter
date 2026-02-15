"""
Climbing Hold Detection Model Training Script
==============================================

This script trains a TensorFlow model to detect climbing holds in images
and exports it to TensorFlow Lite format for mobile deployment.

Requirements:
- tensorflow >= 2.10.0
- numpy
- opencv-python
- pillow
- scikit-learn
"""

import torchvision
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import cv2
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

# Configuration
IMG_SIZE = (416, 416)  # Input image size
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.00001

# Hold types
HOLD_TYPES = ['jug', 'crimp', 'sloper', 'pinch', 'pocket', 'unknown']

import matplotlib.pyplot as plt

def visualize_sample(image, heatmap):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap.squeeze(), cmap='jet')
    plt.colorbar()

    plt.show()

def overlay_heatmap(image, heatmap):
    heatmap = heatmap.squeeze()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        (image * 255).astype(np.uint8),
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay

def draw_predictions(image, holds):
    img = (image * 255).astype(np.uint8).copy()

    for x, y, conf in holds:
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (int(x), int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return img


class ClimbingHoldDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_size=IMG_SIZE, transforms=None, augment=True):
        self.img_size = img_size
        self.images_path = 'data/img'
        self.annotations_path = 'data/label'
        self.transforms = transforms
        self.img_size = img_size
        self.augment = augment
        self.annotation_files = list(Path(self.annotations_path).glob('*.json'))
    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        ann_file = self.annotation_files[idx]

        img_name = ann_file.stem + '.jpg'
        img_path = self.images_path + '/' + img_name

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Resize if needed
        if self.img_size:
            img = cv2.resize(img, self.img_size)
            new_h, new_w = self.img_size
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
        else:
            new_h, new_w = orig_h, orig_w
            scale_x, scale_y = 1.0, 1.0

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW

        # Load annotation
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        boxes = []

        for hold in annotations.get('holds', []):
            # Original format
            x = hold['x']
            y = hold['y']
            w = hold['width']
            h = hold['height']

            # Scale to resized image
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y

            # Convert (center, w, h) -> (xmin, ymin, xmax, ymax)
            xmin = x - w / 2
            ymin = y - h / 2
            xmax = x + w / 2
            ymax = y + h / 2

            boxes.append([xmin, ymin, xmax, ymax])

        # Handle no-box case
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # single class

        target = {
            "boxes": boxes,
            "labels": labels
        }
        if self.augment:
        # Random horizontal flip
            if np.random.rand() > 0.5:
                img = torch.flip(img, [2])  # flip width
                boxes[:, [0, 2]] = img.shape[2] - boxes[:, [2, 0]]
            
            # Random brightness/contrast
            img = img * (0.8 + np.random.rand() * 0.4)
            img = torch.clamp(img, 0, 1)

        return img, target



   
def build_the_overfitter(input_shape=(416, 416, 3)):
    inputs = keras.Input(shape=input_shape)
    # time for literally just three conv layers
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

def build_baby_unet(input_shape=(416, 416, 3)):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x1 = layers.Conv2D(16, 3, padding='same', activation='relu')(x1)
    p1 = layers.MaxPooling2D()(x1)

    x2 = layers.Conv2D(32, 3, padding='same', activation='relu')(p1)
    x2 = layers.Conv2D(32, 3, padding='same', activation='relu')(x2)
    p2 = layers.MaxPooling2D()(x2)

    x3 = layers.Conv2D(64, 3, padding='same', activation='relu')(p2)
    x3 = layers.Conv2D(64, 3, padding='same', activation='relu')(x3)

    # Decoder
    u2 = layers.UpSampling2D()(x3)
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(32, 3, padding='same', activation='relu')(u2)

    u1 = layers.UpSampling2D()(u2)
    u1 = layers.Concatenate()([u1, x1])
    u1 = layers.Conv2D(16, 3, padding='same', activation='relu')(u1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    return keras.Model(inputs, outputs)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(output_dir='./models', use_mobile=True):
    """Main training function."""
    print("Loading dataset...")
    dataset = ClimbingHoldDataset(img_size=(512, 512))
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    
    print("making rcnn")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Freeze backbone initially for small dataset
    for param in model.backbone.parameters():
        param.requires_grad = False
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
        )
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step(total_loss)

        print(f"Epoch {epoch}, Loss: {total_loss}")

    # Save final model
    # Replace model.save() with:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(output_dir, 'final_model.pth'))

    return model, history


def convert_to_tflite(model_path, output_path='model.tflite', quantize=True):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to saved Keras model (.h5)
        output_path: Output path for .tflite file
        quantize: Whether to apply quantization for smaller model size
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying quantization...")
        # Dynamic range quantization - reduces model size significantly
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # For even better optimization, you can use full integer quantization
        # This requires a representative dataset
        # converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
    
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


def test_tflite_model(tflite_path, test_image_path):
    """Test the TFLite model on a sample image."""
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input details:", input_details)
    print("Output details:", output_details)
    
    # Load and preprocess image
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # Get output
    heatmap = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Output shape: {heatmap.shape}")
    print(f"Output range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    return heatmap


def extract_holds_from_heatmap(heatmap, threshold=0.5, min_distance=20):
    """
    Extract individual hold positions from heatmap.
    
    Args:
        heatmap: Output heatmap from model (H, W, 1)
        threshold: Confidence threshold for detection
        min_distance: Minimum distance between holds in pixels
    
    Returns:
        List of (x, y, confidence) tuples
    """
    heatmap = heatmap.squeeze()  # Remove channel dimension
    
    # Apply threshold
    binary = (heatmap > threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    holds = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y = centroids[i]
        confidence = heatmap[int(y), int(x)]
        holds.append((float(x), float(y), float(confidence)))
    
    # Non-maximum suppression to remove close duplicates
    holds = _non_max_suppression(holds, min_distance)
    
    return holds


def _non_max_suppression(holds, min_distance):
    """Remove holds that are too close to each other."""
    if not holds:
        return []
    
    holds = sorted(holds, key=lambda x: x[2], reverse=True)  # Sort by confidence
    kept = []
    
    for hold in holds:
        x, y, conf = hold
        too_close = False
        
        for kept_hold in kept:
            kx, ky, _ = kept_hold
            distance = np.sqrt((x - kx)**2 + (y - ky)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            kept.append(hold)
    
    return kept


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train climbing hold detection model')
    parser.add_argument('--output', type=str, default='./models', help='Output directory')
    parser.add_argument('--convert', action='store_true', help='Convert to TFLite after training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train model
    model, history = train_model(args.output, use_mobile=args.mobile)
    # Convert to TFLite
    if args.convert:
        model_path = os.path.join(args.output, 'best_model.pth')
        tflite_path = os.path.join(args.output, 'model.tflite')
        convert_to_tflite(model_path, tflite_path, quantize=True)
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"TFLite model saved to: {tflite_path}")
        print("Copy this file to: frontend_flutter/assets/model.tflite")
        print("="*50)