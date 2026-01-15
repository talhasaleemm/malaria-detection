import cv2
import os
from PIL import Image

def make_gif(input_path, output_path, target_width=480, fps=10):
    cap = cv2.VideoCapture(input_path)
    frames = []
    
    frame_count = 0
    step = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            # Resize
            height = int(frame.shape[0] * (target_width / frame.shape[1]))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((target_width, height), Image.Resampling.LANCZOS)
            # Quantize to reduce size
            pil_img = pil_img.quantize(colors=128, method=2)
            frames.append(pil_img)
            
        frame_count += 1
        
    cap.release()
    
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=int(1000/fps),
            loop=0
        )
        print("GIF saved: {} ({:.2f} MB)".format(output_path, os.path.getsize(output_path)/1024/1024))

def compress_video(input_path, output_path, target_width=None, target_fps=15):
    if not os.path.exists(input_path):
        print("Error: {} not found.".format(input_path))
        return

    cap = cv2.VideoCapture(input_path)
    
    # Get original properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Original: {}x{} @ {}fps, {} frames".format(width, height, fps, total_frames))
    
    # Calculate new dimensions
    if target_width:
        scale = target_width / width
        new_width = target_width
        new_height = int(height * scale)
    else:
        new_width = width
        new_height = height
        
    # Ensure even dimensions for some codecs
    if new_width % 2 != 0: new_width -= 1
    if new_height % 2 != 0: new_height -= 1
    
    print("Target: {}x{} @ {}fps".format(new_width, new_height, target_fps))

    # Try standard codecs. H264 is best for web but might depend on system availability.
    # mp4v is safer but less efficient. 
    # 'avc1' is H.264
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))
    
    # If avc1 fails to initialize (size 0), fallback to mp4v
    if not out.isOpened():
        print("Fallback to mp4v codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))

    frame_count = 0
    # Process frames
    # We want to drop frames to match target FPS
    step = max(1, int(fps / target_fps))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Skip frames to reduce FPS approximately (simple sampling)
        if frame_count % step != 0:
            continue
            
        # Resize
        if new_width != width:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        out.write(frame)
        
        if frame_count % 100 == 0:
            print("Processed {}/{} frames...".format(frame_count, total_frames))

    cap.release()
    out.release()
    print("Compression Complete: {}".format(output_path))
    print("Final Size: {:.2f} MB".format(os.path.getsize(output_path) / 1024 / 1024))

if __name__ == "__main__":
    input_video = "../assets/demo-malaria.mp4"
    output_video = "../assets/demo-malaria-optimized.mp4"
    output_gif = "../assets/demo-malaria.gif"
    
    # 1. Aggressive Video Compression (480p, 12fps) -> Target < 3MB
    compress_video(input_video, output_video, target_width=480, target_fps=12)
    
    # 2. Convert to GIF for README (480p, 8fps)
    make_gif(output_video, output_gif, target_width=480, fps=8)
