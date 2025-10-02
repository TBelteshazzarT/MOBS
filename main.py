from preprocessing import Annotator
from config import config
import os


def main():
    """
    Test annotation visualization for MOBS project.

    This script demonstrates how to load existing annotations and visualize them
    to verify annotation quality.
    """
    # Create an instance of Annotator class
    annotator = Annotator(images_dir=config.IMAGES_DIR, annotations_dir=config.ANNOTATIONS_DIR)

    # Select file (without extension)
    filename = 'diamonds-in-cave'  # Use without .jpg extension for XML lookup

    print("🔍 Testing annotation visualization...")
    print(f"Images directory: {annotator.images_dir}")
    print(f"Annotations directory: {annotator.annotations_dir}")

    # Generate correct paths
    img_path = os.path.join(annotator.images_dir, filename + '.jpg')
    ann_path = os.path.join(annotator.annotations_dir, filename + '.xml')

    print(f"Looking for image: {img_path}")
    print(f"Looking for annotations: {ann_path}")

    # Check if files exist
    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        return

    if not os.path.exists(ann_path):
        print(f"❌ Annotation file not found: {ann_path}")
        return

    # Find boxes and labels from annotations
    boxes, labels = annotator.load_annotations(ann_path)
    print(f"📦 Found {len(boxes)} bounding boxes: {labels}")

    if len(boxes) == 0:
        print("❌ No bounding boxes found in annotation file!")
        print("💡 Check that your XML file has <object> tags with <bndbox> data")
        return

    # Create new image with annotations overlay
    print("🎨 Creating visualization...")
    viz_image = annotator.visualize_annotations(
        image_path=img_path,
        annotation_path=ann_path,
        save_path=os.path.join(annotator.annotations_dir, filename + '_viz.png')
    )

    print("✅ Visualization complete! Check the _viz.png file")


if __name__ == '__main__':
    main()