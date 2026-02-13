"""
Video Processing Module for Malaria Detection
Handles frame-by-frame video processing with YOLO detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Callable, Optional, Tuple
import os


class VideoProcessor:
    """
    Processes videos frame-by-frame for malaria parasite detection
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Initialize the video processor
        
        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def extract_video_info(self, video_path: str) -> dict:
        """
        Extract metadata from video file
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with video metadata (fps, frame_count, width, height, codec)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_seconds': frame_count / fps if fps > 0 else 0
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process a single frame with YOLO detection and draw bounding boxes
        
        Args:
            frame: Input frame (BGR format from cv2)
            
        Returns:
            Tuple of (annotated_frame, parasite_count)
        """
        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(frame_rgb, verbose=False)
        
        # Draw bounding boxes on original frame
        parasite_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Filter by confidence threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Draw bounding box
                if cls == 0:  # Parasitized
                    color = (0, 0, 255)  # Red in BGR
                    label = f"Parasite {conf:.2f}"
                    parasite_count += 1
                else:  # Uninfected
                    color = (0, 255, 0)  # Green in BGR
                    label = f"Uninfected {conf:.2f}"
                
                # Draw rectangle and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, parasite_count
    
    def process_video(
        self, 
        input_path: str, 
        output_path: str, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Process entire video: extract frames, detect parasites, save output video
        
        Args:
            input_path: Path to input video file
            output_path: Path to save processed video
            progress_callback: Optional callback function(current_frame, total_frames)
            
        Returns:
            Dictionary with processing statistics
        """
        # Get video metadata
        video_info = self.extract_video_info(input_path)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Define codec and create VideoWriter
        # Use mp4v codec for MP4 files (most compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info['fps'],
            (video_info['width'], video_info['height'])
        )
        
        # Processing statistics
        total_frames = video_info['frame_count']
        current_frame = 0
        total_parasites = 0
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame with detection
            annotated_frame, parasite_count = self.process_frame(frame)
            
            # Write to output video
            out.write(annotated_frame)
            
            # Update statistics
            current_frame += 1
            total_parasites += parasite_count
            
            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(current_frame, total_frames)
        
        # Release resources
        cap.release()
        out.release()
        
        # Return statistics
        return {
            'total_frames': current_frame,
            'total_parasites': total_parasites,
            'avg_parasites_per_frame': total_parasites / current_frame if current_frame > 0 else 0,
            'fps': video_info['fps'],
            'duration_seconds': video_info['duration_seconds'],
            'output_path': output_path
        }


def process_video_simple(
    input_path: str,
    output_path: str,
    model_path: str,
    confidence_threshold: float = 0.25
) -> dict:
    """
    Convenience function to process a video in one call
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        model_path: Path to YOLO model weights
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Processing statistics dictionary
    """
    processor = VideoProcessor(model_path, confidence_threshold)
    return processor.process_video(input_path, output_path)
