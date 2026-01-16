import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np

# API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Malaria Object Detection", page_icon="🦟", layout="wide")

import tempfile

st.title("🦟 Malaria Parasite Detection (YOLOv11)")
st.markdown("Professional Grade Object Detection System")

tab1, tab2 = st.tabs(["Image Inference", "Video Inference"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Microscopy Slide")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        # Confidence Threshold Slider
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
        if st.button('Analyze Slide'):
            with st.spinner('Detecting parasites...'):
                try:
                    # Prepare payload
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    
                    # Send to API
                    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        detections = results.get("detections", [])
                        
                        # Draw boxes
                        img_np = np.array(image)
                        
                        count_parasitized = 0
                        
                        for det in detections:
                            cls = det['class']
                            conf = det['confidence']
                            x, y, w, h = det['bbox']
                            
                            # Filter by confidence slider
                            if conf < confidence_threshold:
                                continue
                            
                            start_point = (int(x - w/2), int(y - h/2))
                            end_point = (int(x + w/2), int(y + h/2))
                            
                            if cls == 0: # Parasitized
                                color = (255, 0, 0) # Red
                                label = f"Parasitized {conf:.2f}"
                                count_parasitized += 1
                            else:
                                color = (0, 255, 0) # Green
                                label = f"Uninfected {conf:.2f}"
                                
                            cv2.rectangle(img_np, start_point, end_point, color, 2)
                            cv2.putText(img_np, label, (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        with col2:
                            st.header("Detection Results")
                            st.image(img_np, caption=f'Processed Image ({count_parasitized} Parasites Detected)', use_column_width=True)
                            st.success(f"Analysis Complete. Found {count_parasitized} Parasitized Cells.")
                            
                    else:
                        st.error(f"Error from API: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}. Is the FastAPI backend running?")

with tab2:
    st.header("Upload Microscopic Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Send to API (Frame by Frame - POC)
            # Optimization: Skip frames or resize
            try:
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
                # Use longer timeout for video frames if needed
                response = requests.post(API_URL, files=files, timeout=5)
                
                if response.status_code == 200:
                        results = response.json()
                        detections = results.get("detections", [])
                        
                        for det in detections:
                            cls = det['class']
                            conf = det['confidence']
                            x, y, w, h = det['bbox']
                            
                            # Fixed threshold for video to avoid flickering or add slider here too
                            if conf < 0.25: continue
                                
                            start_point = (int(x - w/2), int(y - h/2))
                            end_point = (int(x + w/2), int(y + h/2))
                            
                            color = (255, 0, 0) if cls == 0 else (0, 255, 0)
                            cv2.rectangle(frame_rgb, start_point, end_point, color, 2)
            except:
                pass # Skip errors for smooth playback
                
            stframe.image(frame_rgb, caption='Real-time Analysis')
            
        vf.release()
