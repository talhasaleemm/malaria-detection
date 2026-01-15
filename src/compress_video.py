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

def compress_video_imageio(input_path, output_path, target_width=480, target_fps=10):
    import imageio
    
    if not os.path.exists(input_path):
        print("Error: {} not found.".format(input_path))
        return

    print("Reading video: {}".format(input_path))
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    print("Original Meta: {}".format(meta))
    
    # Create writer with H.264 codec (libx264) which is standard for web/github
    # fps=target_fps
    # quality=5 (variable bitrate, roughly crf 28ish) or use ffmpeg_params
    writer = imageio.get_writer(output_path, fps=target_fps, codec='libx264', quality=5, pixelformat='yuv420p')

    for i, frame in enumerate(reader):
        # Frame dropping to match fps roughly
        # If original is 30 and target is 10, keep every 3rd
        orig_fps = meta['fps']
        step = max(1, int(orig_fps / target_fps))
        
        if i % step != 0:
            continue

        # Resize using PIL
        pil_img = Image.fromarray(frame)
        
        # Calc height
        if target_width:
             w_percent = (target_width / float(pil_img.size[0]))
             h_size = int((float(pil_img.size[1]) * float(w_percent)))
             # Ensure even dims
             if target_width % 2 != 0: target_width -= 1
             if h_size % 2 != 0: h_size -= 1
             
             pil_img = pil_img.resize((target_width, h_size), Image.Resampling.LANCZOS)
        
        writer.append_data(np.array(pil_img))
        
        if i % 100 == 0:
            print("Processing frame {}...".format(i))

    writer.close()
    print("Compressed (H.264) saved: {}".format(output_path))
    print("Final Size: {:.2f} MB".format(os.path.getsize(output_path) / 1024 / 1024))

if __name__ == "__main__":
    import numpy as np # imageio needs numpy arrays
    input_video = "../assets/demo-malaria.mp4"
    output_video = "../assets/demo-malaria-optimized.mp4"
    output_gif = "../assets/demo-malaria.gif"
    
    # 1. H.264 Compression using ImageIO (Guaranteed GitHub compatibility)
    compress_video_imageio(input_video, output_video, target_width=480, target_fps=10)
    
    # 2. GIF is assumed good from previous run, but we can regen if needed.
    # Let's keep the previous GIF logic if we want, or just stick to the video fix.
    # We'll just run video fix.

