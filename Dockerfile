# Multi-stage build for CodeFormer Face Swap API
FROM python:3.9-slim

# Install system dependencies (minimal for headless deployment)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone CodeFormer repository
RUN git clone https://github.com/sczhou/CodeFormer.git

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create model download script
RUN echo '#!/bin/bash\n\
cd CodeFormer\n\
python -c "from facelib.utils.face_restoration_helper import FaceRestoreHelper; helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=\"retinaface_resnet50\", save_ext=\"png\", use_parse=True, device=\"cpu\"); print(\"Models downloaded successfully\")"\n\
echo "Model download completed"' > download_models.sh

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
