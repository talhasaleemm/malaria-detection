
import cv2
import os
import glob
import numpy as np

def create_pan_zoom_video(image_dir, output_path, duration_per_image=3, fps=30):
    """
    Creates a video by panning across large microscopic images.
    """
    images = glob.glob(os.path.join(image_dir, "*.JPG"))
    if not images:
        images = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not images:
        print("No images found in", image_dir)
        return

    # Take first 5 images for demo
    images = images[:5]
    
    writer = None
    width, height = 640, 640  # Standard YOLO/App size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {len(images)} images...")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        
        h_img, w_img, _ = img.shape
        
        # Define a path for the "camera"
        # Let's pan from top-left to bottom-right
        total_frames = duration_per_image * fps
        
        start_x, start_y = 0, 0
        end_x = max(0, w_img - width)
        end_y = max(0, h_img - height)
        
        for i in range(total_frames):
            alpha = i / float(total_frames)
            
            curr_x = int(start_x * (1 - alpha) + end_x * alpha)
            curr_y = int(start_y * (1 - alpha) + end_y * alpha)
            
            # Crop
            frame = img[curr_y:curr_y+height, curr_x:curr_x+width]
            
            # Resize if the image was smaller than target (unlikely for IML)
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            writer.write(frame)
            
    writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    iml_dir = r"c:/Users/talha/.gemini/antigravity/scratch/malaria_detection/temp_iml_dataset/IML_Malaria"
    out_video = r"c:/Users/talha/.gemini/antigravity/scratch/malaria_detection/demo_video.mp4"
    create_pan_zoom_video(iml_dir, out_video)
