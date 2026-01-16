from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
# from ultralytics import YOLO # SAHI wraps this
import uvicorn
import numpy as np
import cv2
import io
import os
from PIL import Image

# SAHI Imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = FastAPI(title="Malaria Detection API")

# Load model (lazy loading on startup)
detection_model = None

def find_best_model():
    """Finds the most recent/best model in the malaria_yolo project directory."""
    base_dir = os.path.join("..", "malaria_yolo")
    if not os.path.exists(base_dir):
        # Fallback to current directory for dev
        if os.path.exists("yolo11m.pt"): return "yolo11m.pt"
        if os.path.exists("yolo11n.pt"): return "yolo11n.pt"
        return None
    
    # Check runs: run1, run2, run3_final ...
    # We want the 'weights/best.pt' from the latest run folder
    runs = sorted([d for d in os.listdir(base_dir) if d.startswith('run')], reverse=True)
    
    for run in runs:
        candidate = os.path.join(base_dir, run, "weights", "best.pt")
        if os.path.exists(candidate):
            return candidate
    return None

@app.on_event("startup")
def load_model():
    global detection_model
    try:
        model_path = find_best_model()
        if model_path:
            print(f"Loading SAHI wrapper for model: {model_path}")
            detection_model = AutoDetectionModel(
                model_type="yolov8", # SAHI support for v8/v11 is via 'yolov8'
                model_path=model_path,
                confidence_threshold=0.25,
                device="cpu" # or 'cuda'
            )
            print("Model loaded successfully.")
        else:
             print("WARNING: No trained model found. Waiting for training...")
    except Exception as e:
        print(f"CRITICAL ERROR: System cannot start safely.")
        print(e)
        # Don't raise, let it start so we can fix it if needed? No, fail hard usually better.
        # But for dev, we might want to stay up.
        pass

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if detection_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    # SAHI Inference (Async/Non-blocking)
    # slice_height/width 640 is standard YOLO input
    # overlap_height_ratio 0.2 is generic good starting point
    
    result = await run_in_threadpool(
        get_sliced_prediction,
        image_np=img_np,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    object_prediction_list = result.object_prediction_list
    
    detections = []
    for pred in object_prediction_list:
        # SAHI bbox is [minx, miny, maxx, maxy]
        bbox = pred.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        
        # Convert to Center-XYWH for frontend compatibility (or keep xyxy and update frontend? 
        # App.py expects: x, y, w, h which it thinks is Center?
        # Let's check App.py again.
        # Main.py OLD: [gx, gy, lw, lh] -> returned as "bbox": [cx, cy, w, h]
        # App.py: x, y, w, h = det['bbox']
        # start = (x - w/2) -> implies X is Center.
        
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        
        detections.append({
            "bbox": [cx, cy, w, h], 
            "confidence": pred.score.value,
            "class": pred.category.id
        })
            
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
