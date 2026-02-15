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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = (416, 416)  # Input image size
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# Hold types
HOLD_TYPES = ['jug', 'crimp', 'sloper', 'pinch', 'pocket', 'unknown']


class ClimbingHoldDataset:
    """
    Dataset handler for climbing hold detection.
    
    Expected directory structure:
    dataset/
        images/
            image1.jpg
            image2.jpg
            ...
        annotations/
            image1.json
            image2.json
            ...
    
    Annotation format (JSON):
    {
        "holds": [
            {
                "x": 100,
                "y": 150,
                "width": 50,
                "height": 50,
                "type": "jug",
                "confidence": 1.0
            },
            ...
        ]
    }
    """
    
    def __init__(self, dataset_path, img_size=IMG_SIZE):
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.images_path = self.dataset_path / 'images'
        self.annotations_path = self.dataset_path / 'annotations'
        
    def load_data(self):
        """Load all images and annotations."""
        images = []
        heatmaps = []
        
        annotation_files = list(self.annotations_path.glob('*.json'))
        
        for ann_file in annotation_files:
            img_name = ann_file.stem + '.jpg'
            img_path = self.images_path / img_name
            
            if not img_path.exists():
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_h, self.original_w = img.shape[:2]
            img_resized = cv2.resize(img, self.img_size)
            img_resized = img_resized.astype(np.float32) / 255.0
            
            # Load annotations
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            
            # Create heatmap
            heatmap = self._create_heatmap(annotations, img.shape[:2])
            
            images.append(img)
            heatmaps.append(heatmap)
        
        return np.array(images), np.array(heatmaps)
    
    def _create_heatmap(self, annotations, img_shape):
        h, w = img_shape
        heatmap = np.zeros((h, w, 1), dtype=np.float32)

        for hold in annotations.get('holds', []):
            cx = int(hold['x'] * w / self.original_w)
            cy = int(hold['y'] * h / self.original_h)

            sigma = max(5, int(min(hold['width'], hold['height']) / 4))

            x_grid = np.arange(0, w)
            y_grid = np.arange(0, h)
            x_grid, y_grid = np.meshgrid(x_grid, y_grid)

            gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))

            heatmap[:, :, 0] = np.maximum(heatmap[:, :, 0], gaussian)

        return heatmap
    



def build_model(input_shape=(*IMG_SIZE, 3)):
    """
    Build a U-Net style model for hold detection.
    This architecture is good for detecting multiple objects with precise localization.
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Downsampling)
    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip1 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    skip2 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    skip3 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Block 4
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    skip4 = x
    x = layers.MaxPooling2D(2)(x)
    
    # Bottleneck
    x = layers.Conv2D(1024, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(1024, 3, padding='same', activation='relu')(x)
    
    # Decoder (Upsampling)
    # Block 5
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    
    # Block 6
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Block 7
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    # Block 8
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Output layer - heatmap prediction
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='climbing_hold_detector')
    
    return model


def build_mobilenet_model(input_shape=(*IMG_SIZE, 3)):
    """
    Alternative lighter model using MobileNetV2 backbone.
    Better for mobile deployment - smaller and faster.
    """
    # Use MobileNetV2 as backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = base_model(inputs, training=False)
    
    # Decoder
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='climbing_hold_detector_mobile')
    
    return model


def train_model(dataset_path, output_dir='./models', use_mobile=True):
    """Main training function."""
    
    print("Loading dataset...")
    dataset = ClimbingHoldDataset(dataset_path)
    X, y = dataset.load_data()
    
    print(f"Loaded {len(X)} images")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    if use_mobile:
        print("Building MobileNetV2-based model...")
        model = build_mobilenet_model()
    else:
        print("Building U-Net model...")
        model = build_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
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
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./models', help='Output directory')
    parser.add_argument('--mobile', action='store_true', help='Use MobileNet architecture')
    parser.add_argument('--convert', action='store_true', help='Convert to TFLite after training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train model
    model, history = train_model(args.dataset, args.output, use_mobile=args.mobile)
    
    # Convert to TFLite
    if args.convert:
        model_path = os.path.join(args.output, 'best_model.h5')
        tflite_path = os.path.join(args.output, 'model.tflite')
        convert_to_tflite(model_path, tflite_path, quantize=True)
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"TFLite model saved to: {tflite_path}")
        print("Copy this file to: frontend_flutter/assets/model.tflite")
        print("="*50)