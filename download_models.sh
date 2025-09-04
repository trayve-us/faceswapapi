#!/bin/bash

# Set environment variables for headless operation
export QT_QPA_PLATFORM=offscreen
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting model download for CodeFormer..."

# Change to CodeFormer directory
cd CodeFormer

# Add paths to Python
export PYTHONPATH="${PYTHONPATH}:.:./basicsr"

echo "Testing imports..."

# Test OpenCV
python -c "import cv2; print('✅ OpenCV imported successfully')" || { echo "❌ OpenCV import failed"; }

# Test PyTorch
python -c "import torch; print('✅ PyTorch imported successfully')" || { echo "❌ PyTorch import failed"; }

# Test lpips
python -c "import lpips; print('✅ lpips imported successfully')" || { echo "❌ lpips import failed"; }

# Test basicsr
python -c "import basicsr; print('✅ basicsr imported successfully')" || { echo "❌ basicsr import failed"; }

echo "Attempting to initialize FaceRestoreHelper..."

# Try to initialize FaceRestoreHelper and download models
python -c "
import sys
sys.path.append('.')
sys.path.append('./basicsr')

try:
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    print('✅ FaceRestoreHelper imported successfully')
    
    helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=False,
        device='cpu'
    )
    print('✅ FaceRestoreHelper initialized successfully')
    print('✅ Models downloaded successfully')
    
except Exception as e:
    print(f'❌ FaceRestoreHelper initialization failed: {e}')
    import traceback
    traceback.print_exc()
"

echo "Model download script completed"
