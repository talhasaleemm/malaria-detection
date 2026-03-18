"""
FastAPI Backend for Malaria Parasite Detection.
Provides a high-performance inference API using SAHI (Slicing Aided Hyper Inference)
over a custom-trained YOLOv11 model. Highly robust for production environments.
"""

import os
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# SAHI configuration
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global model instance
detection_model = None

# --- Pydantic Models ---

class BoundingBox(BaseModel):
    cx: float = Field(..., description="Center X coordinate")
    cy: float = Field(..., description="Center Y coordinate")
    w: float = Field(..., description="Width of the bounding box")
    h: float = Field(..., description="Height of the bounding box")

class Detection(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [cx, cy, w, h]")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    class_id: int = Field(..., alias="class", description="Predicted class integer ID")
    
    class Config:
        populate_by_name = True

class PredictResponse(BaseModel):
    detections: List[Detection] = Field(..., description="List of detected parasites")

# --- Application Logic ---

def find_best_model() -> Optional[str]:
    """Finds the most recent/best model in the local directory structure."""
    candidate_paths = ["yolo11m.pt", "yolo11n.pt"]
    
    # Check parent directory for previous run folder models if applicable
    base_dir = os.path.join("..", "malaria_yolo")
    if os.path.exists(base_dir):
        runs = sorted([d for d in os.listdir(base_dir) if d.startswith('run')], reverse=True)
        for run in runs:
            candidate = os.path.join(base_dir, run, "weights", "best.pt")
            if os.path.exists(candidate):
                return candidate
                
    # Fallback to current directory defaults
    for path in candidate_paths:
        if os.path.exists(path):
            return path
            
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event manager for FastAPI execution flow."""
    global detection_model
    model_path = find_best_model()
    
    if model_path:
        logger.info(f"Loading SAHI detection wrapper for model: {model_path}")
        try:
            detection_model = AutoDetectionModel(
                model_type="yolov8", # SAHI support for v8/v11 utilizes the "yolov8" designation
                model_path=model_path,
                confidence_threshold=0.25,
                device="cpu" # Support for "cuda" should be toggled per env
            )
            logger.info("YOLO Model loaded successfully via SAHI.")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    else:
        logger.warning("No trained YOLO model found (.pt file). Startup continues in degraded mode.")
        
    yield
    # Cleanup resources on shutdown
    logger.info("Shutting down model inference service.")
    detection_model = None

app = FastAPI(
    title="Malaria AI Detection Pipeline",
    description="Enterprise API for micro-object inference on gigapixel blood smear slides.",
    version="1.0.0",
    lifespan=lifespan
)

# Standard Security Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Validates API and AI Model health status for orchestration layers like Kubernetes."""
    is_ready = detection_model is not None
    return {"status": "healthy" if is_ready else "degraded", "model_loaded": is_ready}

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_slide(file: UploadFile = File(...)):
    """
    Analyzes an uploaded microscopic slide image to locate and classify malaria parasites.
    Utilizes SAHI integration for detecting extremely small objects over large image resolutions.
    """
    if detection_model is None:
        logger.error("Inference requested but AI model is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI detection model is not loaded yet."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception as e:
        logger.warning(f"Failed to process uploaded image format: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file provided.")
    
    try:
        # Offload CPU-bound inference to threadpool to prevent event loop blocking
        result = await run_in_threadpool(
            get_sliced_prediction,
            image_np=img_np,
            detection_model=detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        detections = []
        for pred in result.object_prediction_list:
            x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
            
            # Convert to standard Center-XY Format
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            
            detections.append(
                Detection(
                    bbox=[cx, cy, w, h],
                    confidence=pred.score.value,
                    class_id=pred.category.id
                )
            )
            
        logger.info(f"Inference complete: found {len(detections)} targets.")
        return PredictResponse(detections=detections)
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference application error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
