import requests
import zipfile
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import random
import os
from typing import List, Tuple, Dict

class CellCompositor:
    """
    Composes synthetic microscopy slides from single-cell images.
    Generates YOLO-format bounding box annotations.
    """
    def __init__(self, dataset_name='malaria', split='train', canvas_size=(640, 640)):
        self.canvas_size = canvas_size
        # Dataset and cell_images are in root, so ../cell_images
        self.dataset_dir = os.path.join("..", "cell_images")
        self.images_data = []
        
        # Ensure data exists (check relative path)
        if not os.path.exists(self.dataset_dir):
            self.download_and_extract()
        else:
            print("Dataset found, skipping download.")
            
        print("Loading images into memory...")
        self.load_images()
        print(f"Buffer loaded with {len(self.images_data)} cells.")

    def download_and_extract(self):
        url = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
        zip_path = os.path.join("..", "cell_images.zip")
        
        print(f"Downloading dataset from {url}...")
        try:
            # NIH Data server might not need headers, but keeping them is safe.
            headers = {'User-Agent': 'Mozilla/5.0'} 
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to parent dir (root)
                zip_ref.extractall("..")
            print("Extraction complete.")
            # NIH zip extracts to 'cell_images' folder with 'Parasitized' and 'Uninfected' subfolders
        except Exception as e:
            print(f"Download failed: {e}")
            # Peek at file content if it exists
            if os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    print(f"File start: {f.read(100)}")
            raise

    def load_images(self, max_images=2000):
        # Load Parasitized (Class 0)
        parasitized_files = glob.glob(os.path.join(self.dataset_dir, "cell_images", "Parasitized", "*.png"))
        # Check specific path structure. Usually cell_images.zip extracts to cell_images/Parasitized
        # But verify if 'cell_images' repeats.
        if not parasitized_files:
             parasitized_files = glob.glob(os.path.join(self.dataset_dir, "Parasitized", "*.png"))

        uninfected_files = glob.glob(os.path.join(self.dataset_dir, "cell_images", "Uninfected", "*.png"))
        if not uninfected_files:
             uninfected_files = glob.glob(os.path.join(self.dataset_dir, "Uninfected", "*.png"))
        
        # Limit for demo speed / memory
        # Shuffle first to get variety if we limit
        random.shuffle(parasitized_files)
        random.shuffle(uninfected_files)
        
        files_map = {
            0: parasitized_files[:max_images//2], # Parasitized
            1: uninfected_files[:max_images//2]   # Uninfected
        }
        
        for class_id, files in files_map.items():
            for fpath in files:
                img = cv2.imread(fpath)
                if img is not None:
                     self.images_data.append((img, class_id))


    def generate_biological_background(self):
        """
        Generates a synthetic biological background (Giemsa stain style).
        Instead of white, it produces a noisy, light purple/pink background.
        """
        # Base color: Light purple/pink (Giemsa background)
        # BGR: (240, 230, 240) -> slightly noisy
        base_color = np.array([230, 220, 230], dtype=np.uint8)
        
        # Create canvas
        canvas = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8)
        canvas[:] = base_color
        
        # Add Color Noise
        noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add Vignetting (Optical falloff)
        rows, cols = canvas.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2) # Sigma for falloff
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        # Expand dims for broadcasting: (H, W) -> (H, W, 1)
        mask = mask[:, :, np.newaxis]
        
        # Darken corners
        canvas = (canvas * mask).astype(np.uint8)
        
        return canvas

    def random_rotate(self, image):
        """Rotates an image by a random angle (0-360)."""
        angle = random.randint(0, 360)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Calculate new bounding box to avoid clipping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Use REFLECT to avoid "White Corner" artifacts
        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT_101)
        return rotated

    def generate_slide(self, num_cells_range=(10, 30)) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Generates a single synthetic slide.
        Returns:
            image: (H, W, 3) BGR numpy array
            labels: List of [class_id, x_center, y_center, w, h] (normalized)
        """
        # 1. Biological Background
        canvas = self.generate_biological_background()
        
        num_cells = random.randint(*num_cells_range)
        labels = [] # YOLO format: class x_c y_c w h
        
        # Simple painter's algorithm
        for _ in range(num_cells):
            # Pick a random cell
            idx = random.randint(0, len(self.images_data) - 1)
            cell_img, cell_label = self.images_data[idx]
            
            # 2. Random Rotation
            cell_img = self.random_rotate(cell_img)
            
            # Masking (remove black corners from rotation/crop if reflect didn't cover everything, or original crop)
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            # Threshold to find content (cells usually distinct from black/white padding)
            # NIH dataset is cropped cells.
            # We assume non-black/non-white is content.
            # Using simple threshold for mask.
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            h, w, c = cell_img.shape
            
            # Random position (ensure fully inside)
            x_pos = random.randint(0, max(0, self.canvas_size[1] - w))
            y_pos = random.randint(0, max(0, self.canvas_size[0] - h))
            
            if self.canvas_size[1] - w < 0 or self.canvas_size[0] - h < 0:
                continue

            # 3. Blending (Seamless Clone for realistic edges)
            # Center of the cell in the destination image
            center = (x_pos + w // 2, y_pos + h // 2)
            
            # Seamless clone requires mask to be uint8 255
            # Ensure ROI is valid
            try:
                # Use MIXED_CLONE for transparency look or NORMAL_CLONE
                # Poisson blending requires the destination to be larger than source, which we ensured.
                # However, seamlessClone can be slow. 
                # Let's use it. It makes a huge difference.
                # Note: seamlessClone crashes if mask touches border of canvas.
                if x_pos <= 0 or y_pos <= 0 or (x_pos+w) >= self.canvas_size[1] or (y_pos+h) >= self.canvas_size[0]:
                    # Fallback to simple paste if too close to edge (Poisson needs context)
                    roi = canvas[y_pos:y_pos+h, x_pos:x_pos+w]
                    mask_bool = mask > 0
                    roi[mask_bool] = cell_img[mask_bool]
                    canvas[y_pos:y_pos+h, x_pos:x_pos+w] = roi
                else:
                    canvas = cv2.seamlessClone(cell_img, canvas, mask, center, cv2.NORMAL_CLONE)
            except Exception as e:
                # Fallback
                roi = canvas[y_pos:y_pos+h, x_pos:x_pos+w]
                mask_bool = mask > 0
                roi[mask_bool] = cell_img[mask_bool]
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = roi

            
            # Calculate YOLO coordinates (Normalized 0-1)
            # x_center, y_center, width, height
            x_center = (x_pos + w / 2) / self.canvas_size[1]
            y_center = (y_pos + h / 2) / self.canvas_size[0]
            width = w / self.canvas_size[1]
            height = h / self.canvas_size[0]
            
            labels.append([cell_label, x_center, y_center, width, height])
            
        return canvas, labels

    def create_yaml(self, output_dir):
        """Creates the data.yaml file for YOLO training."""
        yaml_content = f"""
path: {os.path.abspath(output_dir)} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test:  # test images (optional)

names:
  0: parasitized
  1: uninfected
"""
        with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content.strip())
        print(f"Created dataset.yaml in {output_dir}")

    def save_dataset(self, output_dir, num_images=100, split_name='train'):
        """
        Generates and saves a dataset in YOLO format.
        output_dir/
          images/
            split_name/
              img0.jpg
          labels/
            split_name/
              img0.txt
        """
        img_dir = os.path.join(output_dir, 'images', split_name)
        lbl_dir = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"Generating {num_images} images for {split_name}...")
        
        for i in range(num_images):
            img, lbls = self.generate_slide()
            
            # Save Image
            img_name = f"{split_name}_{i:05d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_name), img)
            
            # Save Labels
            lbl_name = f"{split_name}_{i:05d}.txt"
            with open(os.path.join(lbl_dir, lbl_name), 'w') as f:
                for lbl in lbls:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                    
        print(f"Done generating {split_name}.")

    def save_real_val_set(self, output_dir, num_images=50):
        """
        Saves a validation set of REAL NIH crops pasted onto a canvas.
        This preserves the pixel-scale of the parasite, matching the trained model.
        """
        img_dir = os.path.join(output_dir, 'images', 'test')
        lbl_dir = os.path.join(output_dir, 'labels', 'test')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"Generating Scaled Real World Validation Set (n={num_images})...")
        
        if not self.images_data:
            self.load_images()
            
        count = 0
        # Canvas size matching training/inference
        CANVAS_H, CANVAS_W = self.canvas_size 
        
        for i, (cell_img, label) in enumerate(self.images_data):
            if count >= num_images:
                break
                
            h, w, c = cell_img.shape
            
            # Skip if cell is somehow larger than canvas (unlikely)
            if h > CANVAS_H or w > CANVAS_W:
                continue

            # 1. Create Blank Canvas (using the biological background function is best, 
            # but a grey/purple constant is fine for validation to avoid noise confusion)
            canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 230 # Light grey-purple
            
            # 2. Center the real cell on the canvas
            x_off = (CANVAS_W - w) // 2
            y_off = (CANVAS_H - h) // 2
            
            canvas[y_off:y_off+h, x_off:x_off+w] = cell_img
            
            # 3. Save Image
            label_str = "parasitized" if label == 0 else "uninfected"
            fname = f"real_{label_str}_{i:04d}.jpg"
            cv2.imwrite(os.path.join(img_dir, fname), canvas)
            
            # 4. Create Label (Normalized relative to 640x640 canvas)
            # Center x, Center y, Width, Height
            cx = (x_off + w / 2) / CANVAS_W
            cy = (y_off + h / 2) / CANVAS_H
            nw = w / CANVAS_W
            nh = h / CANVAS_H
            
            with open(os.path.join(lbl_dir, fname.replace('.jpg', '.txt')), 'w') as f:
                f.write(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
            count += 1
            
        print(f"Done. Real images preserved at native scale on {CANVAS_W}x{CANVAS_H} canvas.")

if __name__ == "__main__":
    # Test execution
    compositor = CellCompositor()
    OUTPUT_DIR = os.path.join("..", "dataset")
    
    # Generate Training/Val (Synthetic)
    # Using ThreadPoolExecutor for speed inside the loop is tricky, passing to ProcessPool better
    # But simple linear generation is safer for this prototype unless requested.
    compositor.save_dataset(OUTPUT_DIR, num_images=100, split_name='train')
    compositor.save_dataset(OUTPUT_DIR, num_images=20, split_name='val')
    
    # Generate Real Test Set (Ground Truth)
    compositor.save_real_val_set(OUTPUT_DIR, num_images=50) # Real crops
    
    compositor.create_yaml(OUTPUT_DIR)
    print(f"Test generation complete. Check '{OUTPUT_DIR}' folder.")
