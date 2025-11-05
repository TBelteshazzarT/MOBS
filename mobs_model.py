"""
MOBS v1.0 - Model Classes & Functions
Minecraft Object Boundary System

Contains all the core classes and functions for the ore detection model.
Students can modify this for advanced customization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import xml.etree.ElementTree as ET


class MobsDataset:
    """
    Handles loading and preprocessing of Minecraft ore images and XML annotations
    """

    def __init__(self, images_dir, annotations_dir, img_size=(320, 320), batch_size=4):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_map = {
            1: 'diamond',
            2: 'gold',
            3: 'iron',
            4: 'redstone',
            5: 'coal'
        }
        self.num_classes = len(self.label_map)

    def parse_annotation(self, annotation_path):
        """Parse XML annotation file from LabelImg"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label_name = obj.find('name').text

            # Convert label name to ID
            label_id = None
            for id, name in self.label_map.items():
                if name == label_name:
                    label_id = id
                    break

            if label_id is None:
                continue  # Skip unknown labels

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / img_width  # Normalize
            ymin = float(bndbox.find('ymin').text) / img_height  # Normalize
            xmax = float(bndbox.find('xmax').text) / img_width  # Normalize
            ymax = float(bndbox.find('ymax').text) / img_height  # Normalize

            # Format: [y_min, x_min, y_max, x_max] for TensorFlow
            boxes.append([ymin, xmin, ymax, xmax])
            labels.append(label_id)

        return boxes, labels

    def load_dataset(self):
        """Load all images and annotations into a TensorFlow Dataset"""
        image_paths = []
        all_boxes = []
        all_labels = []

        print("📥 Loading dataset...")

        # Find all image files with corresponding annotations
        for img_file in os.listdir(self.images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(self.images_dir, img_file)
                ann_path = os.path.join(self.annotations_dir, base_name + '.xml')

                if os.path.exists(ann_path):
                    boxes, labels = self.parse_annotation(ann_path)
                    if boxes:  # Only include images with annotations
                        image_paths.append(img_path)
                        all_boxes.append(boxes)
                        all_labels.append(labels)

        print(f"✅ Loaded {len(image_paths)} images with annotations")

        def process_path(img_path, boxes, labels):
            """Process a single image and its annotations"""
            # Load and preprocess image
            image = tf.io.read_file(img_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, self.img_size)
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]

            # Convert to ragged tensors for variable number of objects
            boxes = tf.ragged.constant(boxes)
            labels = tf.ragged.constant(labels)

            return image, {
                'boxes': boxes,
                'labels': labels
            }

        # Create TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, all_boxes, all_labels))
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def train_val_split(self, validation_split=0.2):
        """Split dataset into training and validation sets"""
        full_dataset = self.load_dataset()

        # Calculate split sizes
        total_images = len([f for f in os.listdir(self.images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        train_size = int((1 - validation_split) * total_images)

        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size)

        print(f"📊 Training: {train_size} images, Validation: {total_images - train_size} images")
        return train_dataset, val_dataset


class MobsModel:
    """
    Object detection model for Minecraft ores
    Uses a simple CNN architecture suitable for educational purposes
    """

    def __init__(self, num_classes=5, img_size=(320, 320)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None

    def build_simple_cnn(self):
        """Build a simple CNN model for object detection - great for learning!"""
        # Input layer
        inputs = keras.Input(shape=(*self.img_size, 3))

        # Feature extraction
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)

        # Global features
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output heads
        classification_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        regression_output = layers.Dense(4, activation='sigmoid', name='regression')(x)  # 4 bbox coordinates

        self.model = keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
        return self.model

    def build_advanced_model(self):
        """Build a more advanced model using transfer learning"""
        # Use MobileNetV2 as base for better performance
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model initially
        base_model.trainable = False

        # Add custom heads
        inputs = base_model.input
        features = base_model.output

        # Global features
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output heads
        classification_output = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        regression_output = layers.Dense(4, activation='sigmoid', name='regression')(x)

        self.model = keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate loss functions and metrics"""
        if self.model is None:
            self.build_simple_cnn()  # Default to simple CNN

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'regression': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'regression': 1.0
            },
            metrics={
                'classification': ['accuracy'],
                'regression': ['mse']
            }
        )
        print("✅ Model compiled successfully!")


class MobsTrainingPipeline:
    """
    Handles the complete training process with callbacks and visualization
    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.history = None

    def setup_callbacks(self):
        """Setup training callbacks for better training"""
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath='mobs_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Stop early if no improvement
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when stuck
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        return callbacks

    def train(self, epochs=50):
        """Train the model"""
        print("🚀 Starting training...")

        # Get datasets
        train_ds, val_ds = self.dataset.train_val_split()

        # Setup callbacks
        callbacks = self.setup_callbacks()

        # Train the model
        self.history = self.model.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )

        print("✅ Training completed!")
        return self.history

    def evaluate(self):
        """Evaluate the model on validation data"""
        _, val_ds = self.dataset.train_val_split()
        results = self.model.model.evaluate(val_ds, verbose=0)

        print("📊 Evaluation Results:")
        print(f"   Total Loss: {results[0]:.4f}")
        print(f"   Classification Loss: {results[1]:.4f}")
        print(f"   Regression Loss: {results[2]:.4f}")
        print(f"   Classification Accuracy: {results[3]:.4f}")
        print(f"   Regression MSE: {results[4]:.4f}")

        return results

    def plot_training_history(self):
        """Plot training history to visualize progress"""
        if self.history is None:
            print("No training history available.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot losses
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()

        # Plot classification accuracy
        axes[0, 1].plot(self.history.history['classification_accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_classification_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].legend()

        # Plot classification loss
        axes[1, 0].plot(self.history.history['classification_loss'], label='Training')
        axes[1, 0].plot(self.history.history['val_classification_loss'], label='Validation')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].legend()

        # Plot regression loss
        axes[1, 1].plot(self.history.history['regression_loss'], label='Training')
        axes[1, 1].plot(self.history.history['val_regression_loss'], label='Validation')
        axes[1, 1].set_title('Bounding Box Regression Loss')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('mobs_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📈 Training history plot saved as 'mobs_training_history.png'")


def predict_ores(model, image_path, confidence_threshold=0.5):
    """
    Predict ores in a single image using the trained model
    """
    # Load and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    original_size = image.shape[:2]

    image_resized = tf.image.resize(image, (320, 320))
    image_resized = tf.cast(image_resized, tf.float32) / 255.0
    image_batch = tf.expand_dims(image_resized, 0)

    # Make prediction
    class_probs, bbox_coords = model.predict(image_batch)

    # Process results
    detected_ores = []
    for i, prob in enumerate(class_probs[0]):
        if prob > confidence_threshold:
            ore_type = ['diamond', 'gold', 'iron', 'redstone', 'coal'][i]

            # Convert normalized coordinates back to original image size
            ymin, xmin, ymax, xmax = bbox_coords[0]
            ymin = int(ymin * original_size[0])
            xmin = int(xmin * original_size[1])
            ymax = int(ymax * original_size[0])
            xmax = int(xmax * original_size[1])

            detected_ores.append({
                'ore_type': ore_type,
                'confidence': float(prob),
                'bbox': [xmin, ymin, xmax, ymax]
            })

    return detected_ores