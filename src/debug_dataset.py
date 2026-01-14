import cv2
import os
import glob
import random

def visualize_split(split_name, num_samples=5):
    img_dir = os.path.join("..", "dataset", "images", split_name)
    lbl_dir = os.path.join("..", "dataset", "labels", split_name)
    output_dir = os.path.join("..", "dataset", "debug_viz", split_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Checking split: {split_name}...")
    
    # Get all images
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    random.shuffle(img_paths)
    
    for i, img_path in enumerate(img_paths[:num_samples]):
        # Read image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Read label
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        lbl_path = os.path.join(lbl_dir, name_no_ext + ".txt")
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                cls = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                
                # Draw box
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = (0, 255, 0) if cls == 1 else (0, 0, 255) # Red for Parasite, Green for Uninfected
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(cls), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            print(f"Warning: Label missing for {basename}")
            cv2.putText(img, "NO LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save debug image
        out_path = os.path.join(output_dir, f"viz_{basename}")
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize_split("train", 5)
    visualize_split("test", 10) # Check more test images since that's likely the issue
