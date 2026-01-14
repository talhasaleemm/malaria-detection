# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Grouping them to reduce layers
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    ultralytics \
    fastapi \
    uvicorn \
    python-multipart \
    streamlit \
    requests \
    pillow \
    torch \
    torchvision --index-url https://download.pytorch.org/whl/cpu
    # Note: Using CPU torch to keep image size small. 
    # For GPU, use the official pytorch/pytorch base image.

# Copy the entire project
COPY . /app

# Create necessary directories
RUN mkdir -p dataset/images/train dataset/images/val dataset/images/test \
    dataset/labels/train dataset/labels/val dataset/labels/test \
    malaria_yolo

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# By default, we don't run a command, we let docker-compose handle it
CMD ["bash"]
