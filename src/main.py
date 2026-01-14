from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2
import io
import os
from PIL import Image

app = FastAPI(title="Malaria Detection API")

# Load model (lazy loading on startup)
model = None

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

@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = find_best_model()
        if model_path:
            model = YOLO(model_path)
            print(f"Loaded tailored model from {model_path}")
        else:
             raise FileNotFoundError("No custom trained model found in malaria_yolo/")
    except Exception as e:
        print(f"CRITICAL ERROR: System cannot start safely.")
        print(e)
        raise e

import torch
from torchvision.ops import nms

def non_max_suppression_fast(boxes, overlapThresh):
    """
    Standard Non-Maximum Suppression (NMS) using Torchvision (GPU/CPU optimized).
    boxes: list of [x, y, w, h, score, class_id] (all float/int)
    """
    if len(boxes) == 0:
        return []

    # Convert to Tensor
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    
    if boxes_tensor.numel() == 0:
        return []
        
    # xywh -> xyxy
    x = boxes_tensor[:, 0]
    y = boxes_tensor[:, 1]
    w = boxes_tensor[:, 2]
    h = boxes_tensor[:, 3]
    scores = boxes_tensor[:, 4]
    class_ids = boxes_tensor[:, 5]
    
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    
    # Stack for NMS: (N, 4)
    boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=1)
    
    # Torchvision NMS is class-agnostic by default.
    # To make it Class-Aware, we offset boxes by class_id * max_coordinate
    # This effectively separates classes in coordinate space.
    max_coordinate = boxes_xyxy.max() + 5000
    offsets = class_ids * max_coordinate
    boxes_for_nms = boxes_xyxy + offsets[:, None]
    
    keep_indices = nms(boxes_for_nms, scores, overlapThresh)
    
    # Return matched boxes
    final_picks = boxes_tensor[keep_indices].tolist()
    return final_picks

def slice_image(image_np, tile_size=640, overlap=0.25): # Increased overlap for safety
    """
    Slices a large image into tiles with overlap.
    Returns: list of (tile, x_offset, y_offset)
    """
    h, w, _ = image_np.shape
    step = int(tile_size * (1 - overlap))
    tiles = []
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate coords
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size) # Adjust back if at edge
            y1 = max(0, y2 - tile_size)
            
            tile = image_np[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))
            
    return tiles

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    
    h, w, c = img_np.shape
    raw_detections = []
    
    # Check if image is large (e.g., > 1000px) -> Use Tiling
    if h > 1000 or w > 1000:
        print(f"Large image detected ({w}x{h}). Using Tiling Inference.")
        tiles = slice_image(img_np)
        
        for tile, x_off, y_off in tiles:
            results = model(tile, verbose=False)
            for result in results:
                for box in result.boxes:
                    # LOCAL bbox (center x, center y, w, h)
                    lx, ly, lw, lh = box.xywh.tolist()[0]
                    
                    # GLOBAL bbox
                    # Center simply shifts by offset
                    gx = lx + x_off
                    gy = ly + y_off
                    
                    raw_detections.append([gx, gy, lw, lh, float(box.conf), int(box.cls)])
    else:
        # Standard Inference
        results = model(img_np)
        for result in results:
            for box in result.boxes:
                raw_detections.append([*box.xywh.tolist()[0], float(box.conf), int(box.cls)])
                
    # Apply Global NMS to merged detections
    # IoU threshold 0.3 to remove overlapping duplicates from tiles
    final_boxes = non_max_suppression_fast(raw_detections, 0.3)
    
    detections = []
    for box in final_boxes:
        detections.append({
            "bbox": [box[0], box[1], box[2], box[3]], # cx, cy, w, h
            "confidence": box[4],
            "class": int(box[5])
        })
            
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
