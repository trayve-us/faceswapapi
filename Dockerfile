# Multi-stage build for CodeFormer Face Swap API
FROM python:3.10-slim

# Install system dependencies (minimal for headless deployment)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless OpenCV operation
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=1048576000
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=""
ENV MPLBACKEND=Agg

WORKDIR /app

# Clone CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version first - using stable compatible versions
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install CodeFormer's basicsr in development mode for proper module setup
WORKDIR /app/CodeFormer
RUN pip install -e .
WORKDIR /app

# Ensure we only have headless OpenCV (remove any conflicts)
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

# Copy application code and scripts
COPY main.py .
COPY download_models.sh .

# Make script executable
RUN chmod +x download_models.sh

# Download models during build
RUN ./download_models.sh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Render will set PORT environment variable)
EXPOSE $PORT

# Start command (JSON format recommended)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
