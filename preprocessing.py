# preprocessing.py
import tensorflow as tf
import xml.etree.ElementTree as ET
from config import config
import os
from PIL import Image, ImageDraw, ImageFont
import json


class Annotator:
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.current_image = None
        self.current_boxes = []
        self.current_labels = []

        # Create directories if they don't exist
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Color scheme for different ores
        self.ore_colors = {
            'diamond': '#00BFFF',  # Deep sky blue
            'gold': '#FFD700',  # Gold
            'iron': '#C0C0C0',  # Silver
            'redstone': '#FF0000',  # Red
            'coal': '#36454F'  # Charcoal
        }

    def load_annotations(self, annotation_path):
        """
        Load annotations from XML file
        Returns: (boxes, labels) in standard format [xmin, ymin, xmax, ymax]
        """
        if not os.path.exists(annotation_path):
            print(f"⚠️  Annotation file not found: {annotation_path}")
            return [], []

        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            boxes = []
            labels = []

            # Count objects found
            object_count = 0

            for obj in root.findall('object'):
                object_count += 1
                label = obj.find('name').text
                bndbox = obj.find('bndbox')

                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Keep in standard format [xmin, ymin, xmax, ymax]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

            print(f"📖 Read {object_count} objects from {os.path.basename(annotation_path)}")
            return boxes, labels

        except Exception as e:
            print(f"❌ Error reading annotation file: {e}")
            return [], []

    def visualize_annotations(self, image_path, annotation_path=None, save_path=None):
        """
        Draw bounding boxes on image to visualize annotations
        Great for verifying annotation quality!
        """
        # Load image
        try:
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            print(f"🖼️  Loaded image: {os.path.basename(image_path)} ({image.size[0]}x{image.size[1]})")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None

        # Try to load font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
                print("⚠️  Using default font (arial.ttf not available)")
            except:
                font = None

        # Load annotations if provided
        boxes = []
        labels = []

        if annotation_path and os.path.exists(annotation_path):
            boxes, labels = self.load_annotations(annotation_path)
        else:
            # Look for corresponding XML file
            xml_path = os.path.join(
                self.annotations_dir,
                os.path.splitext(os.path.basename(image_path))[0] + '.xml'
            )
            if os.path.exists(xml_path):
                boxes, labels = self.load_annotations(xml_path)

        print(f"🎯 Drawing {len(boxes)} bounding boxes...")

        # Draw bounding boxes and labels
        for i, (box, label) in enumerate(zip(boxes, labels)):
            xmin, ymin, xmax, ymax = box

            # Get color for this ore type
            color = self.ore_colors.get(label, '#00FF00')  # Default to green

            # Draw bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            # Draw label background
            label_text = f"{label}"

            if font:
                try:
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    # Fallback for older Pillow versions
                    text_width = len(label_text) * 10
                    text_height = 20
            else:
                text_width = len(label_text) * 10
                text_height = 20

            # Draw label background
            draw.rectangle(
                [xmin, ymin - text_height - 5, xmin + text_width + 10, ymin],
                fill=color
            )

            # Draw label text
            if font:
                draw.text(
                    (xmin + 5, ymin - text_height - 2),
                    label_text,
                    fill='white',
                    font=font
                )
            else:
                # Fallback without font
                draw.text(
                    (xmin + 5, ymin - text_height - 2),
                    label_text,
                    fill='white'
                )

            print(f"   📦 Box {i + 1}: {label} at [{xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f}]")

        # Save or show result
        if save_path:
            image.save(save_path)
            print(f"💾 Visualization saved: {save_path}")
        else:
            image.show()
            print("👀 Displaying visualization...")

        return image

    def create_pascal_voc_annotation(self, image_path, boxes, labels, image_size):
        """
        Create Pascal VOC XML annotation file
        boxes: list of [xmin, ymin, xmax, ymax] in standard format
        labels: list of ore types
        image_size: (width, height)
        """
        image_filename = os.path.basename(image_path)
        annotation_path = os.path.join(
            self.annotations_dir,
            os.path.splitext(image_filename)[0] + '.xml'
        )

        # Create XML structure
        annotation = ET.Element('annotation')

        # Folder and filename
        folder = ET.SubElement(annotation, 'folder')
        folder.text = os.path.basename(self.images_dir)

        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_filename

        # Image size
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_size[0])
        height = ET.SubElement(size, 'height')
        height.text = str(image_size[1])
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'  # RGB

        # Add each object (ore)
        for box, label in zip(boxes, labels):
            obj = ET.SubElement(annotation, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = label

            # Bounding box - KEEP ORIGINAL COORDINATES
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(box[0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(box[1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(box[2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(box[3]))

        # Write to file
        tree = ET.ElementTree(annotation)
        tree.write(annotation_path, encoding='utf-8', xml_declaration=True)

        print(f"✅ Annotation saved: {annotation_path}")
        return annotation_path

    def annotate_image(self, image_path, boxes, labels):
        """
        Simple function for annotating image
        image_path: path to screenshot
        boxes: list of [xmin, ymin, xmax, ymax] in standard format
        labels: list of object types
        """
        # Get image size
        with Image.open(image_path) as img:
            image_size = img.size

        # Create annotation
        annotation_path = self.create_pascal_voc_annotation(
            image_path, boxes, labels, image_size
        )

        # Create visualization
        viz_path = annotation_path.replace('.xml', '_viz.png')
        self.visualize_annotations(image_path, annotation_path, viz_path)

        return annotation_path, viz_path

    def debug_annotation_file(self, annotation_path):
        """
        Debug function to see what's in an annotation file
        """
        print(f"\n🔍 Debugging {annotation_path}:")
        if not os.path.exists(annotation_path):
            print("❌ File does not exist")
            return

        with open(annotation_path, 'r') as f:
            content = f.read()
            print(f"File content:\n{content}")