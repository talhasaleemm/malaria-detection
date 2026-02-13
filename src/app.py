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
    st.header("🎥 Video Detection & Download")
    st.markdown("Upload a microscopy video to detect parasites frame-by-frame and download the annotated result.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        # Video processing settings
        video_confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            key="video_conf"
        )
        
        # Model selection
        model_options = {
            "YOLOv11 Medium (Recommended)": "yolo11m.pt",
            "YOLOv11 Nano (Faster)": "yolo11n.pt"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        model_path = model_options[selected_model]
    
    if uploaded_video is not None:
        # Read the uploaded video bytes once (to avoid stream consumption issues)
        uploaded_video_bytes = uploaded_video.read()
        
        with col1:
            st.video(uploaded_video_bytes)
            st.caption("Preview: Original Uploaded Video")
        
        # Process button
        if st.button("🔬 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes depending on video length."):
                try:
                    # Import video processor
                    from video_processor import VideoProcessor
                    
                    # Save uploaded video to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                        tmp_input.write(uploaded_video_bytes)
                        input_path = tmp_input.name
                    
                    # Create output path
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name
                    
                    # Initialize video processor
                    processor = VideoProcessor(model_path, video_confidence_threshold)
                    
                    # Get video info first
                    video_info = processor.extract_video_info(input_path)
                    st.info(f"📹 Video Info: {video_info['frame_count']} frames, {video_info['fps']} FPS, {video_info['duration_seconds']:.1f}s duration")
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def update_progress(current, total):
                        progress = current / total
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing frame {current}/{total} ({progress*100:.1f}%)")
                    
                    # Process video
                    stats = processor.process_video(input_path, output_path, progress_callback=update_progress)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    progress_text.empty()
                    
                    # Show results in col2
                    with col2:
                        st.success("✅ Video Processing Complete!")
                        
                        # Display statistics
                        st.metric("Total Parasites Detected", stats['total_parasites'])
                        st.metric("Average Parasites/Frame", f"{stats['avg_parasites_per_frame']:.2f}")
                        st.metric("Total Frames Processed", stats['total_frames'])
                        
                        # Show processed video
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        st.video(video_bytes)
                        st.caption("Preview: Processed Video with Detections")
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=video_bytes,
                            file_name=f"malaria_detected_{uploaded_video.name}",
                            mime="video/mp4",
                            type="primary"
                        )
                    
                    # Clean up temp files
                    import os
                    try:
                        os.unlink(input_path)
                        os.unlink(output_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
