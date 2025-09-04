FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Clone CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download CodeFormer models
RUN cd /app/CodeFormer && \
    mkdir -p weights/CodeFormer weights/facelib && \
    wget -O weights/CodeFormer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    wget -O weights/facelib/detection_Resnet50_Final.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -O weights/facelib/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth

# Copy application
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:/app/CodeFormer:/app/CodeFormer/basicsr"
ENV OPENCV_IO_ENABLE_OPENEXR=0

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
