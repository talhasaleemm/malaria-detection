from ultralytics import YOLO
import os

def train_model():
    # Load a model
    model = YOLO("yolo11m.pt")  # load a pretrained model (Medium - Stronger than Nano)

    # Train the model
    # We assume 'dataset/dataset.yaml' exists in root, so ../dataset/dataset.yaml
    dataset_yaml = os.path.abspath(os.path.join("..", "dataset", "dataset.yaml"))
    
    print(f"Starting training using config: {dataset_yaml}")
    
    results = model.train(
        data=dataset_yaml,
        epochs=100, # Increased for FULL convergence
        imgsz=640,
        batch=8,
        project=os.path.abspath(os.path.join("..", "malaria_yolo")),
        name="run5_production", # Final Production Run
        exist_ok=True,
        # Augmentations 
        degrees=180.0, 
        translate=0.1, 
        scale=0.1,    
        fliplr=0.5,
        flipud=0.5, 
        close_mosaic=10, # IMPORTANT: Disable mosaic for the last 10 epochs
        mixup=0.0     
    )
    
    print("Training Complete.")
    print(f"Best model saved at: {model.trainer.best}")

if __name__ == "__main__":
    train_model()
