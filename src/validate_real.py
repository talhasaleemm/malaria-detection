from ultralytics import YOLO
import os
import sys

def validate_real():
    # 1. Load Production Model
    # Try finding run5 first, else fall back to dynamic
    model_path = os.path.join("..", "malaria_yolo", "run5_production", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print("Warning: Specific run5 model not found at {}".format(model_path))
        # Try dynamic find
        base_dir = os.path.join("..", "malaria_yolo")
        runs = sorted([d for d in os.listdir(base_dir) if d.startswith('run')], reverse=True)
        if runs:
            model_path = os.path.join(base_dir, runs[0], "weights", "best.pt")
            print("Using latest model: {}".format(model_path))
        else:
            print("Error: No models found.")
            return

    print("Loading model: {}".format(model_path))
    model = YOLO(model_path)
    
    # 2. Update yaml to ensure 'test' path is correct
    yaml_path = os.path.join("..", "dataset", "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        print("Error: dataset.yaml not found. Did data_pipeline.py run?")
        return

    # Read yaml
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # data_pipeline might leave "test: " empty or commented
    # We need "test: images/test"
    # Let's just forcefully update/write it to be sure
    if "test: images/test" not in content:
        # Check if we can just append or replace
        lines = content.splitlines()
        new_lines = []
        found_test = False
        for line in lines:
            if line.strip().startswith("test:"):
                new_lines.append("test: images/test")
                found_test = True
            else:
                new_lines.append(line)
        
        if not found_test:
            new_lines.append("test: images/test")
            
        with open(yaml_path, 'w') as f:
            f.write("\n".join(new_lines))
        print("Updated dataset.yaml to point 'test' to 'images/test'.")

    print("Validating on REAL WORLD data (images/test) using split='test'...")
    
    # 3. Run Validation
    try:
        metrics = model.val(
            data=os.path.abspath(yaml_path),
            split='test',
            project=os.path.abspath(os.path.join("..", "malaria_yolo")),
            name="val_real_world",
            plots=True
        )
        
        print("\n" + "="*40)
        print("REAL WORLD VALIDATION RESULTS")
        print("="*40)
        print("mAP50-95 : {:.4f}".format(metrics.box.map))
        print("mAP50    : {:.4f}".format(metrics.box.map50))
        print("Recall   : {:.4f}".format(metrics.box.r.mean())) # Mean recall
        print("Precision: {:.4f}".format(metrics.box.p.mean()))
        print("="*40 + "\n")
        
    except Exception as e:
        print("Validation failed: {}".format(e))

if __name__ == "__main__":
    validate_real()
