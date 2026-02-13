
import json
import os
import shutil
import cv2
import random
from pathlib import Path
from tqdm import tqdm

def convert_iml_to_yolo(iml_dir, output_dir, train_ratio=0.8):
    """
    Converts IML Malaria dataset (JSON annotations) to YOLO format.
    
    Args:
        iml_dir (str): Path to temp_iml_dataset root.
        output_dir (str): Path to dataset root (where images/ and labels/ are).
        train_ratio (float): Fraction of images to use for training.
    """
    
    # Paths
    json_path = os.path.join(iml_dir, "annotations.json")
    images_source_dir = os.path.join(iml_dir, "IML_Malaria")
    
    # Output structure
    images_train_dir = os.path.join(output_dir, "images", "train")
    images_val_dir = os.path.join(output_dir, "images", "val")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    labels_val_dir = os.path.join(output_dir, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Class Mapping
    # 0: parasitized (ring, trophozoite, schizont, gametocyte)
    # 1: uninfected (red blood cell)
    class_map = {
        "red blood cell": 1,
        "ring": 0,
        "trophozoite": 0,
        "schizont": 0,
        "gametocyte": 0,
        "leukocyte": -1 # Ignore WBS if present, or maybe just skip
    }

    print("Loading annotations from {}...".format(json_path))
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Shuffle for random split
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print("Total images: {}".format(len(data)))
    print("Training: {}, Validation: {}".format(len(train_data), len(val_data)))

    def process_batch(batch_data, img_dest_dir, lbl_dest_dir):
        for item in tqdm(batch_data):
            img_name = item['image_name']
            src_img_path = os.path.join(images_source_dir, img_name)
            
            if not os.path.exists(src_img_path):
                print(f"Warning: Image {img_name} not found in {images_source_dir}")
                continue
                
            # Read image to get dimensions
            img = cv2.imread(src_img_path)
            if img is None:
                print(f"Warning: Could not read {src_img_path}")
                continue
                
            height, width, _ = img.shape
            
            # Prepare Label File
            label_filename = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(lbl_dest_dir, label_filename)
            
            # Copy Image
            dst_img_path = os.path.join(img_dest_dir, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Write Labels
            with open(label_path, 'w') as lf:
                for obj in item['objects']:
                    obj_type = obj['type']
                    if obj_type not in class_map:
                        continue
                        
                    class_id = class_map[obj_type]
                    if class_id == -1:
                        continue
                        
                    bbox = obj['bbox']
                    # IML format: x, y, w, h (top-left x, y) or center?
                    # Let's check the json snippet in memory. 
                    # "bbox": {"x": "176", "y": "250", "h": "78", "w": "82"}
                    # The dataset paper/readme usually specifies. Standard JSON often uses Top-Left.
                    # Given the coordinates (e.g. x=176, w=82), it looks like pixel coords.
                    # We will assume x,y is Top-Left.
                    
                    x = float(bbox['x'])
                    y = float(bbox['y'])
                    w = float(bbox['w'])
                    h = float(bbox['h'])
                    
                    # Convert to YOLO (Center Normalized)
                    x_center = (x + w / 2.0) / width
                    y_center = (y + h / 2.0) / height
                    n_w = w / width
                    n_h = h / height
                    
                    # Clamp to 0-1 just in case
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    n_w = max(0, min(1, n_w))
                    n_h = max(0, min(1, n_h))
                    
                    lf.write(f"{class_id} {x_center} {y_center} {n_w} {n_h}\n")

    print("Processing Training Set...")
    process_batch(train_data, images_train_dir, labels_train_dir)
    
    print("Processing Validation Set...")
    process_batch(val_data, images_val_dir, labels_val_dir)
    
    print("IML Conversion Completed!")

if __name__ == "__main__":
    iml_dir = r"c:/Users/talha/.gemini/antigravity/scratch/malaria_detection/temp_iml_dataset"
    output_dir = r"c:/Users/talha/.gemini/antigravity/scratch/malaria_detection/dataset"
    convert_iml_to_yolo(iml_dir, output_dir)
