"""
Streamlit Frontend for the Malaria Detection System.
Provides a modern, responsive UI for clinical researchers.
"""

import io
import os
import tempfile
import traceback
from typing import Dict, Any, List

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# System constants
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
APP_TITLE = "Malaria Object Detection Pipeline"

st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")

def draw_detections(image: Image.Image, detections: List[Dict[str, Any]], conf_threshold: float) -> tuple[np.ndarray, int]:
    """Draws bounding boxes over detected parasites based on model confidence."""
    img_np = np.array(image)
    count_parasitized = 0
    
    for det in detections:
        conf = det.get('confidence', 0.0)
        if conf < conf_threshold:
            continue
            
        cls = det.get('class', 0)
        cx, cy, w, h = det['bbox']
        
        start_point = (int(cx - w/2), int(cy - h/2))
        end_point = (int(cx + w/2), int(cy + h/2))
        
        if cls == 0:  # Parasitized
            color = (255, 65, 54)  # Vibrant Red
            label = f"Parasitized {conf:.2f}"
            count_parasitized += 1
        else:
            color = (46, 204, 64)  # Muted Green
            label = f"Uninfected {conf:.2f}"
            
        cv2.rectangle(img_np, start_point, end_point, color, 2)
        cv2.putText(
            img_np, label, (start_point[0], max(0, start_point[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        
    return img_np, count_parasitized

def process_static_slide() -> None:
    """Manages the UI layout and API invocations for static image processing."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Microscopy Slide Ingestion")
        uploaded_file = st.file_uploader("Upload high-resolution clinical slide...", type=["jpg", "png", "jpeg"])
        confidence_threshold = st.slider(
            "Detection Confidence Level", 
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="Filters out bounding boxes below this confidence to minimize false positives."
        )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Invalid image format provided.")
            return

        with col1:
            st.image(image, caption='Clinical Slide Input', use_column_width=True)
            analyze_btn = st.button('Run AI Analysis', type="primary")
            
        if analyze_btn:
            with st.spinner('Orchestrating AI Inference...'):
                try:
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    
                    response = requests.post(API_URL, files={'file': ('image.jpg', img_bytes, 'image/jpeg')})
                    response.raise_for_status()
                    
                    results = response.json()
                    detections = results.get("detections", [])
                    
                    annotated_img, count_parasitized = draw_detections(image, detections, confidence_threshold)
                    
                    with col2:
                        st.subheader("Inference Results")
                        st.image(annotated_img, caption='Object Detection Output', use_column_width=True)
                        st.success(f"Diagnostics Complete: Identified **{count_parasitized}** Parasitized Cells.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Microservice Connection Error: Ensure API at `{API_URL}` is running. Details: {e}")
                except Exception as e:
                    st.error(f"Application Error: {str(e)}")

def process_video_stream() -> None:
    """Manages the UI layout for processing continuous video frame streams."""
    st.subheader("🎥 Real-time Video Stream Diagnostics")
    st.markdown("Upload microscopy scans (video format) to perform temporal parasite tracking and counting.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_video = st.file_uploader("Upload clinical scan routine...", type=["mp4", "avi", "mov"])
        video_conf = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="vid_conf")
        
        model_options = {
            "YOLOv11 Medium (High Accuracy)": "yolo11m.pt",
            "YOLOv11 Nano (Ultra Fast)": "yolo11n.pt"
        }
        selected_model = st.selectbox("Select Inference Backbone", list(model_options.keys()))
        model_path = model_options[selected_model]
    
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        
        with col1:
            st.video(video_bytes)
            
        if st.button("🧪 Launch Video Processing", type="primary"):
            with st.spinner("Analyzing micro-frames. This utilizes heavy computational resources..."):
                try:
                    from video_processor import VideoProcessor
                    
                    # Store temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                        tmp_input.write(video_bytes)
                        input_path = tmp_input.name
                        
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name
                    
                    processor = VideoProcessor(model_path, video_conf)
                    video_info = processor.extract_video_info(input_path)
                    st.info(f"Video Meta: {video_info['frame_count']} frames | {video_info['fps']} FPS")
                    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def ui_callback(current: int, total: int):
                        progress_bar.progress(current / total)
                        progress_text.text(f"Inferencing: Frame {current}/{total}")
                        
                    stats = processor.process_video(input_path, output_path, progress_callback=ui_callback)
                    
                    progress_bar.empty()
                    progress_text.empty()
                    
                    with col2:
                        st.success("✅ Analysis Complete!")
                        st.metric("Total Parasites Tracked", stats['total_parasites'])
                        st.metric("Avg Parasitemia/Frame", f"{stats['avg_parasites_per_frame']:.2f}")
                        
                        with open(output_path, 'rb') as f:
                            out_video = f.read()
                        st.video(out_video)
                        st.download_button(
                            "⬇️ Export Clinical Video Asset", data=out_video,
                            file_name=f"diagnostics_{uploaded_video.name}", mime="video/mp4", type="primary"
                        )
                    
                    os.unlink(input_path)
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"Inference Pipeline Failure: {str(e)}")
                    st.code(traceback.format_exc())

def main() -> None:
    st.title("🧬 Next-Gen Malaria Detection Platform")
    st.markdown(
        """
        Powered by **YOLOv11** & **SAHI** (Slicing Aided Hyper Inference). 
        Engineered for precision clinical diagnostics with sub-cellular localization mechanics.
        """
    )
    
    tab_img, tab_vid = st.tabs(["Static Image Diagnostics", "Continuous Video Scans"])
    
    with tab_img:
        process_static_slide()
        
    with tab_vid:
        process_video_stream()

if __name__ == "__main__":
    main()
