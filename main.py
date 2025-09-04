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

# Add CodeFormer to path - referencing the cloned repository
sys.path.append('/app/CodeFormer')
sys.path.append('/app/CodeFormer/basicsr')

# Import CodeFormer components with graceful fallback
try:
    from facelib.detection import init_detection_model
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.face_utils import paste_face_back
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize
    from basicsr.archs.codeformer_arch import CodeFormer
    CODEFORMER_AVAILABLE = True
    print("✅ CodeFormer components imported successfully")
except ImportError as e:
    print(f"⚠️  CodeFormer import failed: {e}")
    CODEFORMER_AVAILABLE = False

# Try BasicSR with fallback - use CodeFormer's BasicSR
try:
    from basicsr.utils.misc import get_device
    print("✅ Using CodeFormer's BasicSR")
except ImportError:
    print("⚠️  BasicSR import failed, using torch device detection")
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(title="CodeFormer Face Swap API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceSwapProcessor:
    """
    Face Swap Processor using CodeFormer's exact algorithms
    Reference: CodeFormer/inference_codeformer.py and FaceRestoreHelper
    """

    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")

        if not CODEFORMER_AVAILABLE:
            print("⚠️  CodeFormer not available - using fallback mode")
            self.face_helper = None
            return

        try:
            # Initialize face detection model - using CodeFormer's default
            # Reference: CodeFormer/facelib/detection/__init__.py
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,  # Fixed parameter name
                face_size=512,
                crop_ratio=(1, 1),
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

            # Get face landmarks - Reference: CodeFormer/inference_codeformer.py line 183
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )

            if num_det_faces == 0:
                return None, None, "No face detected"

            # Align and warp face - Reference: CodeFormer/inference_codeformer.py line 186
            self.face_helper.align_warp_face()

            # Get the first detected face and its transformation matrix
            extracted_face = self.face_helper.cropped_faces[0]
            affine_matrix = self.face_helper.affine_matrices[0]

            return extracted_face, affine_matrix, f"Face detected successfully. Total faces: {num_det_faces}"

        except Exception as e:
            return None, None, f"Error in face detection: {str(e)}"

    def swap_faces(self, target_image: np.ndarray, source_face: np.ndarray, affine_matrix: np.ndarray):
        """
        Swap face into target image using CodeFormer's fusion algorithm
        Reference: CodeFormer/facelib/utils/face_utils.py paste_face_back function
        """
        try:
            # Calculate inverse affine transformation
            # Reference: CodeFormer/facelib/utils/face_restoration_helper.py line 352
            inverse_affine = cv2.invertAffineTransform(affine_matrix)

            # Use CodeFormer's paste_face_back function for seamless blending
            # Reference: CodeFormer/facelib/utils/face_utils.py lines 190-208
            # This function takes exactly 3 parameters: (img, face, inverse_affine)
            result_image = paste_face_back(target_image, source_face, inverse_affine)

            # Convert to uint8 for proper image format
            result_image = np.clip(result_image, 0, 255).astype(np.uint8)

            return result_image, "Face swap completed successfully"

        except Exception as e:
            return None, f"Error in face swapping: {str(e)}"

    def complete_face_swap(self, source_image, target_image):
        """
        Complete face swap: extract face from source, replace face in target, enhance with CodeFormer
        """
        try:
            # Step 1: Extract face from source image
            self.face_helper.clean_all()
            self.face_helper.read_image(source_image)

            # Detect and extract source face
            num_source_faces = self.face_helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
            if num_source_faces == 0:
                return None, "No faces detected in source image"

            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                return None, "Failed to extract face from source image"

            source_face = self.face_helper.cropped_faces[0]
            print(f"Extracted source face: {source_face.shape}")

            # Step 2: Process target image and replace face
            self.face_helper.clean_all()
            self.face_helper.read_image(target_image)

            # Detect faces in target image
            num_target_faces = self.face_helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
            if num_target_faces == 0:
                return None, "No faces detected in target image"

            print(f"Detected {num_target_faces} faces in target image")

            # Align and warp target faces
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                return None, "Failed to extract face from target image"

            # Step 3: Replace target face with source face (resize to match)
            target_face_shape = self.face_helper.cropped_faces[0].shape
            source_face_resized = cv2.resize(source_face, (target_face_shape[1], target_face_shape[0]))

            # Replace the first target face with resized source face
            self.face_helper.cropped_faces[0] = source_face_resized
            print(f"Replaced target face with source face: {source_face_resized.shape}")

            # Step 4: Enhance all faces with CodeFormer
            enhanced_faces = []
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                # Convert to tensor
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

                try:
                    with torch.no_grad():
                        output = self.net(cropped_face_t, w=0.5, adain=True)[0]
                        enhanced_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f"CodeFormer enhancement failed for face {idx}: {error}")
                    enhanced_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                enhanced_face = enhanced_face.astype('uint8')
                enhanced_faces.append(enhanced_face)
                self.face_helper.add_restored_face(enhanced_face, cropped_face)

            print(f"Enhanced {len(enhanced_faces)} faces with CodeFormer")

            # Step 5: Calculate inverse affine transformations and paste back
            self.face_helper.get_inverse_affine(None)

            # Paste enhanced faces back to target image
            result_image = self.face_helper.paste_faces_to_input_image()

            if result_image is None:
                return None, "Failed to paste enhanced faces back to target image"

            print(f"Final result shape: {result_image.shape}")
            return result_image, "Face swap completed successfully with CodeFormer enhancement"

        except Exception as e:
            print(f"Complete face swap error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Face swap failed: {str(e)}"

# Global processor instance
processor = FaceSwapProcessor()

def image_to_array(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to numpy array (BGR format for OpenCV)"""
    try:
        # Read image data
        image_data = upload_file.file.read()

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array and BGR format (OpenCV standard)
        image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def array_to_response(image_array: np.ndarray) -> StreamingResponse:
    """Convert numpy array to HTTP response"""
    try:
        # Convert BGR to RGB for proper display
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Save to bytes
        img_io = io.BytesIO()
        pil_image.save(img_io, format='PNG')
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating response: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "CodeFormer Face Swap API",
        "version": "1.0.0",
        "device": str(processor.device),
        "model": "RetinaFace ResNet50"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "codeformer_available": CODEFORMER_AVAILABLE,
        "face_helper_initialized": processor.face_helper is not None,
        "device": str(processor.device),
        "model": "RetinaFace ResNet50" if processor.face_helper else "None"
    }

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect and extract face from uploaded image
    Based on CodeFormer's face detection pipeline
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Convert to numpy array
        image_array = image_to_array(file)

        # Extract face using CodeFormer's algorithm
        extracted_face, affine_matrix, message = processor.detect_and_extract_face(image_array)

        if extracted_face is None:
            raise HTTPException(status_code=400, detail=message)

        # Convert extracted face to base64 for JSON response
        face_rgb = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_io = io.BytesIO()
        face_pil.save(face_io, format='PNG')
        face_base64 = base64.b64encode(face_io.getvalue()).decode()

        # Convert affine matrix to list for JSON serialization
        affine_list = affine_matrix.tolist()

        return {
            "success": True,
            "message": message,
            "extracted_face": face_base64,
            "affine_matrix": affine_list,
            "face_size": extracted_face.shape[:2]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/swap-faces")
async def swap_faces(
    target_image: UploadFile = File(...),
    source_face: str = None,
    affine_matrix: str = None
):
    """
    Swap faces using CodeFormer's fusion algorithm
    Based on paste_face_back function
    """
    if not target_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Target image must be an image file")

    if not source_face or not affine_matrix:
        raise HTTPException(status_code=400, detail="Source face and affine matrix required")

    try:
        # Convert target image to array
        target_array = image_to_array(target_image)

        # Decode source face from base64
        face_data = base64.b64decode(source_face)
        face_pil = Image.open(io.BytesIO(face_data))
        face_array = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)

        # Parse affine matrix
        affine_data = json.loads(affine_matrix)
        affine_array = np.array(affine_data, dtype=np.float32)

        # Perform face swap using CodeFormer's algorithm
        result_image, message = processor.swap_faces(target_array, face_array, affine_array)

        if result_image is None:
            raise HTTPException(status_code=400, detail=message)

        # Return result image
        return array_to_response(result_image)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/complete-face-swap")
async def complete_face_swap(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    """
    Complete face swap pipeline: detect face in source, swap into target, enhance with CodeFormer
    Uses proper CodeFormer workflow for best quality results
    """
    if not source_image.content_type.startswith('image/') or not target_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Both files must be images")

    try:
        # Convert uploaded files to numpy arrays
        source_array = image_to_array(source_image)
        target_array = image_to_array(target_image)

        # Use complete face swap workflow
        result_image, message = processor.complete_face_swap(source_array, target_array)

        if result_image is None:
            raise HTTPException(status_code=400, detail=message)

        # Return final result
        return array_to_response(result_image)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting CodeFormer Face Swap Server...")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
