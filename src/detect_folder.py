import os
import cv2
import sys
from ultralytics import YOLO
import glob

def detect_folder(input_folder, output_folder="output_results", model_path=None):
    """
    Runs YOLOv11 inference on all images in input_folder and saves them to output_folder.
    """
    if model_path is None:
         model_path = os.path.join("..", "malaria_yolo", "run1", "weights", "best.pt")
         
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
        
    print(f"Found {len(files)} images in {input_folder}")
    
    for fpath in files:
        filename = os.path.basename(fpath)
        print(f"Processing {filename}...")
        
        # Inference
        results = model(fpath)
        
        # Visualize and Save
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, im_array)
            
    print(f"Processing complete. Results saved to {output_folder}")

def find_best_model():
    """Finds the most recent/best model in the malaria_yolo project directory."""
    base_dir = os.path.join("..", "malaria_yolo")
    if not os.path.exists(base_dir):
        return None
    
    # Check runs: run1, run2, run3_final ...
    # We want the 'weights/best.pt' from the latest run folder
    runs = sorted([d for d in os.listdir(base_dir) if d.startswith('run')], reverse=True)
    
    for run in runs:
        candidate = os.path.join(base_dir, run, "weights", "best.pt")
        if os.path.exists(candidate):
            return candidate
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_folder.py <path_to_images_folder>")
        print("Example: python detect_folder.py C:/Users/talha/Downloads/my_lab_data")
        # Try to find model anyway to show we can
        model_path = find_best_model()
        if model_path:
             print(f"(Defaulting model to: {model_path})")
    else:
        target_folder = sys.argv[1]
        model_path = find_best_model()
        if not model_path:
             print("Error: Could not find any trained model in ../malaria_yolo/")
             sys.exit(1)
             
        print(f"Using dynamically found model: {model_path}")
        detect_folder(target_folder, model_path=model_path)
