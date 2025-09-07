FROM python:3.9.18-slim

WORKDIR /app

# Install minimal system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git /app/CodeFormer

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==23.2.1 setuptools==68.0.0 wheel==0.41.0
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Download CodeFormer models
RUN cd /app/CodeFormer && \
    mkdir -p weights/CodeFormer weights/facelib && \
    wget -O weights/CodeFormer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    wget -O weights/facelib/detection_Resnet50_Final.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -O weights/facelib/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth

# Fix missing basicsr.version module that CodeFormer dependencies expect
RUN echo '"""BasicSR version information"""' > /app/CodeFormer/basicsr/version.py && \
    echo '' >> /app/CodeFormer/basicsr/version.py && \
    echo '__version__ = "1.3.2"' >> /app/CodeFormer/basicsr/version.py && \
    echo '__gitsha__ = "unknown"' >> /app/CodeFormer/basicsr/version.py

# Copy application
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:/app/CodeFormer:/app/CodeFormer/basicsr"
ENV OPENCV_IO_ENABLE_OPENEXR=0

# DigitalOcean typically uses port 8080, but support both 8000 and 8080
EXPOSE 8000
EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
