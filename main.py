"""
MOBS v1.0 - Main Training Pipeline
Minecraft Object Boundary System

Clear, easy-to-understand training script for students.
Modify the parameters in the MAIN CONFIGURATION section below.
"""

import tensorflow as tf
from mobs_model import MobsDataset, MobsModel, MobsTrainingPipeline, predict_ores
import os

print("=" * 60)
print("🏗️  MOBS - Minecraft Ore Detection Training")
print("=" * 60)

# ==================== MAIN CONFIGURATION ====================
# 🎯 STUDENTS: MODIFY THESE PARAMETERS TO EXPERIMENT!

# Dataset paths (modify if your folder structure is different)
IMAGES_DIR = "data/images"  # Folder containing Minecraft screenshots
ANNOTATIONS_DIR = "data/annotations"  # Folder containing XML annotation files

# Model parameters
IMAGE_SIZE = (320, 320)  # Image size for training (width, height)
BATCH_SIZE = 4  # Number of images per batch (2, 4, 8, 16)
MODEL_TYPE = "simple"  # "simple" or "advanced" - try both!

# Training parameters
LEARNING_RATE = 0.001  # How fast the model learns (0.1, 0.01, 0.001, 0.0001)
EPOCHS = 50  # How long to train (start with 20-50)
VALIDATION_SPLIT = 0.2  # Percentage of data for validation (0.1-0.3)


# ==================== TRAINING PIPELINE ====================
# 🚀 DON'T MODIFY BELOW UNLESS YOU'RE COMFORTABLE WITH THE CODE

def main():
    print("📋 Configuration Summary:")
    print(f"   Images: {IMAGES_DIR}")
    print(f"   Annotations: {ANNOTATIONS_DIR}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Model Type: {MODEL_TYPE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print()

    # Step 1: Check if data exists
    print("🔍 Checking dataset...")
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ ERROR: Images directory not found: {IMAGES_DIR}")
        print("💡 Please make sure your Minecraft screenshots are in this folder")
        return

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"❌ ERROR: Annotations directory not found: {ANNOTATIONS_DIR}")
        print("💡 Please make sure your XML annotation files are in this folder")
        return

    # Count files
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]

    print(f"📊 Found {len(image_files)} images and {len(annotation_files)} annotations")

    if len(image_files) == 0:
        print("❌ No images found! Please add Minecraft screenshots to the images folder.")
        return

    if len(annotation_files) == 0:
        print("❌ No annotations found! Please use LabelImg to create XML annotation files.")
        return

    # Step 2: Setup dataset
    print("\n📥 Preparing dataset...")
    dataset = MobsDataset(
        images_dir=IMAGES_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Step 3: Build model
    print("🧠 Building model...")
    model = MobsModel(
        num_classes=5,  # diamond, gold, iron, redstone, coal
        img_size=IMAGE_SIZE
    )

    if MODEL_TYPE == "simple":
        model.build_simple_cnn()
        print("✅ Using Simple CNN model (great for learning!)")
    elif MODEL_TYPE == "advanced":
        model.build_advanced_model()
        print("✅ Using Advanced MobileNetV2 model (better performance)")
    else:
        print("❌ Unknown model type. Using simple CNN.")
        model.build_simple_cnn()

    # Step 4: Compile model
    model.compile_model(learning_rate=LEARNING_RATE)

    # Show model architecture
    print("\n📋 Model Architecture:")
    model.model.summary()

    # Step 5: Setup training pipeline
    pipeline = MobsTrainingPipeline(model, dataset)

    # Step 6: Train the model!
    print("\n" + "=" * 50)
    print("🚀 STARTING TRAINING")
    print("=" * 50)

    history = pipeline.train(epochs=EPOCHS)

    # Step 7: Evaluate the model
    print("\n📈 Evaluating model performance...")
    pipeline.evaluate()

    # Step 8: Show training results
    print("\n📊 Generating training plots...")
    pipeline.plot_training_history()

    # Step 9: Save the final model
    model.model.save('mobs_trained_model.keras')
    print("💾 Model saved as 'mobs_trained_model.keras'")

    # Step 10: Test prediction (optional)
    print("\n🎯 Testing prediction on a sample image...")
    if image_files:
        test_image = os.path.join(IMAGES_DIR, image_files[0])
        if os.path.exists(test_image):
            try:
                detections = predict_ores(model.model, test_image)
                print(f"🔍 Found {len(detections)} ores in test image:")
                for detection in detections:
                    print(f"   - {detection['ore_type']}: {detection['confidence']:.1%} confidence")
            except Exception as e:
                print(f"⚠️  Could not test prediction: {e}")
        else:
            print("⚠️  Test image not available for prediction demo")
    else:
        print("⚠️  No images available for prediction demo")

    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\n📁 Your trained files:")
    print("   - mobs_trained_model.keras (your trained model)")
    print("   - mobs_best_model.keras (best model during training)")
    print("   - mobs_training_history.png (training progress)")
    print("\n🎉 Great job! You've trained a Minecraft ore detection model!")


if __name__ == "__main__":
    # Check TensorFlow version
    print(f"🔧 TensorFlow Version: {tf.__version__}")

    # Run the training pipeline
    main()