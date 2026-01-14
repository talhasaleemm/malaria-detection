import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np

# API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Malaria Object Detection", page_icon="🦟", layout="wide")

st.title("🦟 Malaria Parasite Detection (YOLOv11)")
st.markdown("Professional Grade Object Detection System")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Microscopy Slide")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

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
                        
                        # YOLO format is normalized center x,y w,h ? 
                        # Wait, ultralytics result.boxes.xywh is usually absolute pixels.
                        # Referencing main.py: `box.xywh.tolist()[0]`
                        # Ultralytics returns Absolute Pixels for xywh.
                        
                        start_point = (int(x - w/2), int(y - h/2))
                        end_point = (int(x + w/2), int(y + h/2))
                        
                        # Colors: Green for Uninfected (1), Red for Parasitized (0)
                        # Depends on dataset mapping. In data_pipeline.py: 
                        # 0: parasitized, 1: uninfected.
                        
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
