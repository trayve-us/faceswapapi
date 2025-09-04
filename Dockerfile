# Use Python 3.9 slim image for better performance
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Clone CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download CodeFormer pretrained models
RUN cd /app/CodeFormer && \
    mkdir -p weights/CodeFormer weights/facelib && \
    wget -O weights/CodeFormer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    wget -O weights/facelib/detection_Resnet50_Final.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -O weights/facelib/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth

# Copy FastAPI application
COPY . .

# Update Python path for CodeFormer imports
ENV PYTHONPATH="/app:/app/CodeFormer:/app/CodeFormer/basicsr:${PYTHONPATH}"

# Environment variables for headless OpenCV operation
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=
ENV MPLBACKEND=Agg

# Expose port (Render will set the PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
