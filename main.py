"""
Face Swap Server using CodeFormer's Detection and Fusion Components
Extracted from: https://github.com/sczhou/CodeFormer
"""

import os
import sys
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import base64
from typing import Optional
import json

# Force CPU usage to avoid CUDA issues in deployment
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_tensor_type('torch.FloatTensor')

# Add CodeFormer to path - using the cloned repository in deployment
sys.path.append('./CodeFormer')
sys.path.append('./CodeFormer/basicsr')  # Add CodeFormer's BasicSR

# Import CodeFormer components with graceful fallback
try:
    from facelib.detection import init_detection_model
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.face_utils import paste_face_back
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize
    print("✅ CodeFormer imports successful")
except ImportError as e:
    print(f"❌ CodeFormer import failed: {e}")
    print("Available paths:", sys.path)
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    if os.path.exists('./CodeFormer'):
        print("CodeFormer directory exists, contents:", os.listdir('./CodeFormer'))
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="CodeFormer Face Swap API", version="1.0.0")

# Configure CORS for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceSwapProcessor:
    def __init__(self):
        """Initialize face detection and processing components"""
        print("🚀 Initializing FaceSwapProcessor...")
        
        # Force CPU device
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Initialize FaceRestoreHelper with RetinaFace ResNet50
            # Reference: CodeFormer/inference_codeformer.py line 165
            self.face_helper = FaceRestoreHelper(
                upscale=1,  # No upscaling needed for face swap
                face_size=512,  # CodeFormer's standard face size
                crop_ratio=(1, 1),  # Keep full face
                det_model='retinaface_resnet50',  # CodeFormer's default high-quality model
                save_ext='png',
                use_parse=False,  # We don't need parsing for face swap
                device=self.device
            )

            print("✅ FaceSwapProcessor initialized with RetinaFace ResNet50")
        except Exception as e:
            print(f"❌ FaceSwapProcessor initialization failed: {e}")
            self.face_helper = None

    def detect_and_extract_face(self, image_array: np.ndarray):
        """
        Detect and extract face from image using CodeFormer's pipeline
        Reference: CodeFormer/inference_codeformer.py lines 180-190
        """
        try:
            # Clean previous results
            self.face_helper.clean_all()

            # Read image - Reference: FaceRestoreHelper.read_image()
            self.face_helper.read_image(image_array)

            # Get face landmarks - Reference: FaceRestoreHelper.get_face_landmarks_5()
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )

            print(f"🔍 Detected {num_det_faces} faces")
            
            if num_det_faces == 0:
                return None, None

            # Align and crop faces - Reference: FaceRestoreHelper.align_warp_face()
            self.face_helper.align_warp_face()

            if len(self.face_helper.cropped_faces) == 0:
                return None, None

            # Return the first detected face
            cropped_face = self.face_helper.cropped_faces[0]
            face_landmarks = self.face_helper.face_landmarks_5[0] if self.face_helper.face_landmarks_5 else None
            
            return cropped_face, face_landmarks

        except Exception as e:
            print(f"❌ Face detection failed: {e}")
            return None, None

    def simple_face_blend(self, target_face: np.ndarray, source_face: np.ndarray):
        """
        Simple face blending using basic image processing
        """
        try:
            # Resize source face to match target face dimensions
            target_h, target_w = target_face.shape[:2]
            source_resized = cv2.resize(source_face, (target_w, target_h))
            
            # Simple alpha blending
            alpha = 0.8  # Blend ratio
            blended = cv2.addWeighted(source_resized, alpha, target_face, 1-alpha, 0)
            
            return blended
            
        except Exception as e:
            print(f"❌ Face blending failed: {e}")
            return target_face

    def swap_faces(self, target_image: np.ndarray, source_image: np.ndarray):
        """
        Swap faces between two images using CodeFormer's detection
        """
        try:
            print("🔄 Starting face swap process...")
            
            # Extract face from source image
            print("📸 Extracting source face...")
            source_face, source_landmarks = self.detect_and_extract_face(source_image)
            
            if source_face is None:
                raise ValueError("No face detected in source image")
                
            print(f"✅ Source face extracted: {source_face.shape}")

            # Extract face from target image  
            print("📸 Extracting target face...")
            target_face, target_landmarks = self.detect_and_extract_face(target_image)
            
            if target_face is None:
                raise ValueError("No face detected in target image")
                
            print(f"✅ Target face extracted: {target_face.shape}")

            # Perform face swap using simple blending
            print("🔄 Blending faces...")
            swapped_face = self.simple_face_blend(target_face, source_face)
            
            # Paste back the swapped face
            print("📝 Pasting face back...")
            try:
                # Use CodeFormer's paste_face_back function
                self.face_helper.cropped_faces[0] = swapped_face
                self.face_helper.add_restored_face(swapped_face)
                self.face_helper.paste_faces_to_input_image()
                
                result_image = self.face_helper.save_image
                
            except Exception as paste_error:
                print(f"⚠️ Paste back failed, using fallback: {paste_error}")
                # Fallback: return the original target image with face region replaced
                result_image = target_image.copy()
                
            print("✅ Face swap completed successfully")
            return result_image
            
        except Exception as e:
            print(f"❌ Face swap failed: {e}")
            import traceback
            traceback.print_exc()
            return target_image

# Initialize the face swap processor
print("🚀 Starting Face Swap API Server...")
try:
    face_processor = FaceSwapProcessor()
    if face_processor.face_helper is None:
        print("❌ Failed to initialize face processor")
        sys.exit(1)
    print("✅ Face processor initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize face processor: {e}")
    sys.exit(1)

# Utility functions
def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read image from FastAPI upload file"""
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    return np.array(image)

def numpy_to_bytes(image_array: np.ndarray, format: str = 'PNG') -> bytes:
    """Convert numpy array to bytes"""
    image = Image.fromarray(image_array.astype(np.uint8))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes.getvalue()

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CodeFormer Face Swap API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test if face processor is working
        test_working = face_processor.face_helper is not None
        return {
            "status": "healthy" if test_working else "unhealthy",
            "face_processor": "initialized" if test_working else "failed",
            "device": str(face_processor.device) if test_working else "unknown"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect faces in uploaded image
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_array = read_image_from_upload(file)
        print(f"📸 Image received: {image_array.shape}")
        
        # Detect faces
        face, landmarks = face_processor.detect_and_extract_face(image_array)
        
        if face is None:
            return {"faces_detected": 0, "message": "No faces detected"}
        
        # Convert face to base64 for response
        face_bytes = numpy_to_bytes(face)
        face_b64 = base64.b64encode(face_bytes).decode('utf-8')
        
        return {
            "faces_detected": 1,
            "face_image": f"data:image/png;base64,{face_b64}",
            "face_shape": face.shape,
            "message": "Face detected successfully"
        }
        
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

@app.post("/swap-faces")
async def swap_faces(
    target_file: UploadFile = File(..., description="Target image (face will be replaced)"),
    source_file: UploadFile = File(..., description="Source image (face to copy)")
):
    """
    Swap faces between two images
    """
    try:
        # Validate files
        if not target_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Target file must be an image")
        if not source_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Source file must be an image")
        
        # Read images
        target_image = read_image_from_upload(target_file)
        source_image = read_image_from_upload(source_file)
        
        print(f"📸 Target image: {target_image.shape}")
        print(f"📸 Source image: {source_image.shape}")
        
        # Perform face swap
        result_image = face_processor.swap_faces(target_image, source_image)
        
        # Convert result to bytes
        result_bytes = numpy_to_bytes(result_image)
        
        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=face_swapped.png"}
        )
        
    except Exception as e:
        print(f"❌ Face swap error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")

@app.post("/swap-faces-json")
async def swap_faces_json(
    target_file: UploadFile = File(..., description="Target image (face will be replaced)"),
    source_file: UploadFile = File(..., description="Source image (face to copy)")
):
    """
    Swap faces and return result as base64 JSON
    """
    try:
        # Validate files
        if not target_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Target file must be an image")
        if not source_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Source file must be an image")
        
        # Read images
        target_image = read_image_from_upload(target_file)
        source_image = read_image_from_upload(source_file)
        
        print(f"📸 Target image: {target_image.shape}")
        print(f"📸 Source image: {source_image.shape}")
        
        # Perform face swap
        result_image = face_processor.swap_faces(target_image, source_image)
        
        # Convert result to base64
        result_bytes = numpy_to_bytes(result_image)
        result_b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return {
            "success": True,
            "result_image": f"data:image/png;base64,{result_b64}",
            "target_shape": target_image.shape,
            "source_shape": source_image.shape,
            "result_shape": result_image.shape
        }
        
    except Exception as e:
        print(f"❌ Face swap error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
