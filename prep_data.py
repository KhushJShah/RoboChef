import os
import json
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
csv_path = r'E:/RoboChef/Annotated/Annotations_3.csv.csv'
dataset_dir = r'E:/RoboChef/dataset'
output_dir = r'E:/RoboChef/spice_detection_dataset_yolo1'
train_img_dir = os.path.join(output_dir, 'images/train')
val_img_dir = os.path.join(output_dir, 'images/val')
train_label_dir = os.path.join(output_dir, 'labels/train')
val_label_dir = os.path.join(output_dir, 'labels/val')

# Create necessary directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Load the CSV file
annotations_df = pd.read_csv(csv_path)

# Define a function to parse region attributes and extract class ID
def parse_region_attributes(region_attributes):
    try:
        attributes = json.loads(region_attributes)
        spice_type = attributes.get('Spices', 'None')

        # Map spice type to class ID
        class_id = {
            'Cinnamon': 0,
            'Clove': 1,
            'Bay Leaf': 2,
            'Black Pepper': 3,
            'Dry Ginger': 4,
            'Star Anise': 5,
            'None': 6
        }.get(spice_type, 6)  # Default to 'None' if not found

        return class_id
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return 6  # Default to 'None' class

# Define a function to apply augmentations
def augment_image(img, bbox):
    augmentations = []
    h, w = img.shape[:2]
    
    # Original image
    augmentations.append((img, bbox))

    # Horizontal flip
    img_flip = cv2.flip(img, 1)
    bbox_flip = bbox.copy()
    bbox_flip[0], bbox_flip[2] = w - bbox[2], w - bbox[0]
    augmentations.append((img_flip, bbox_flip))

    # Rotate 90 degrees
    img_rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    bbox_rot90 = np.array([h - bbox[1], bbox[0], h - bbox[3], bbox[2]])
    augmentations.append((img_rot90, bbox_rot90))

    # Rotate 180 degrees
    img_rot180 = cv2.rotate(img, cv2.ROTATE_180)
    bbox_rot180 = np.array([w - bbox[2], h - bbox[3], w - bbox[0], h - bbox[1]])
    augmentations.append((img_rot180, bbox_rot180))

    # Rotate 270 degrees
    img_rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bbox_rot270 = np.array([bbox[1], w - bbox[2], bbox[3], w - bbox[0]])
    augmentations.append((img_rot270, bbox_rot270))

    return augmentations

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    # Ensure the coordinates are within valid range
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)

    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    bbox_width = (x_max - x_min) / img_w
    bbox_height = (y_max - y_min) / img_h

    # Ensure the coordinates are within the range [0, 1]
    x_center, y_center = min(1, max(0, x_center)), min(1, max(0, y_center))
    bbox_width, bbox_height = min(1, max(0, bbox_width)), min(1, max(0, bbox_height))

    return x_center, y_center, bbox_width, bbox_height

# Process each row in the DataFrame
data = []
for index, row in annotations_df.iterrows():
    image_name = row['filename']
    region_shape_attributes = json.loads(row['region_shape_attributes'])
    region_attributes = row['region_attributes']

    # Extract class ID and bounding box
    class_id = parse_region_attributes(region_attributes)
    if region_shape_attributes['name'] == 'rect':
        x_min = region_shape_attributes['x']
        y_min = region_shape_attributes['y']
        width = region_shape_attributes['width']
        height = region_shape_attributes['height']
        x_max = x_min + width
        y_max = y_min + height
    else:
        print(f"Unsupported shape type: {region_shape_attributes['name']}")
        continue

    # Full image path
    for folder in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, folder, image_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            img_h, img_w = img.shape[:2]
            bbox = np.array([x_min, y_min, x_max, y_max])
            augmentations = augment_image(img, bbox)

            for idx, (aug_img, aug_bbox) in enumerate(augmentations):
                x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(aug_bbox, img_w, img_h)
                yolo_annotation = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}"
                
                aug_img_name = f"{Path(image_name).stem}_aug{idx}.jpg"
                label_name = f"{Path(image_name).stem}_aug{idx}.txt"

                if np.random.rand() < 0.8:
                    img_save_path = os.path.join(train_img_dir, aug_img_name)
                    label_save_path = os.path.join(train_label_dir, label_name)
                else:
                    img_save_path = os.path.join(val_img_dir, aug_img_name)
                    label_save_path = os.path.join(val_label_dir, label_name)

                cv2.imwrite(img_save_path, aug_img)
                with open(label_save_path, 'w') as f:
                    f.write(yolo_annotation)

            # Overwrite original label with new annotation
            label_save_path = os.path.join(dataset_dir, folder, f"{Path(image_name).stem}.txt")
            with open(label_save_path, 'w') as f:
                f.write(yolo_annotation)

            break