# FastAPI Face Swap API with runtime dependency loading
import os
import sys

# CRITICAL: Add CodeFormer paths FIRST before any other imports
# This ensures we use the local CodeFormer BasicSR instead of pip versions
sys.path.insert(0, './CodeFormer')
sys.path.insert(0, './CodeFormer/basicsr')
print("✅ CodeFormer paths added to sys.path")

import json
import base64
import io
import subprocess
from PIL import Image
from typing import Optional, Tuple
import tempfile
import urllib.request
import urllib.error

# SIMPLIFIED FIX: Basic typing compatibility for older ML libraries
# Only apply minimal patches since CodeFormer paths are set correctly
print("🔧 Applying minimal typing compatibility patches...")

# Basic built-in type subscriptability for Python 3.9+
for builtin_type in [list, dict, tuple, set]:
    if not hasattr(builtin_type, '__class_getitem__'):
        builtin_type.__class_getitem__ = classmethod(lambda cls, item: cls)

print("✅ Minimal typing compatibility applied")

# Import numpy normally like in the local server version
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize CodeFormer availability flag (will be set after runtime installation)
CODEFORMER_AVAILABLE = False

def install_runtime_dependencies():
    """Install heavy ML dependencies at runtime"""
    dependencies = [
        "numpy==1.23.5",  # Specific version that works with Python 3.9+ and cv2
        "torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu",
        "torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu",
        "opencv-python==4.8.1.78",  # Specific version compatible with numpy
        # Skip basicsr and facexlib - they're included locally in CodeFormer repo
        "lpips==0.1.4",
        "pyyaml==6.0.1",
        "tqdm==4.66.1",
        "addict",        # Required by CodeFormer
        "scikit-image"   # Required by CodeFormer
    ]

    print("🔄 Installing runtime dependencies...")
    for dep in dependencies:
        try:
            print(f"📦 Installing {dep.split('==')[0]}")
            # Install with no-cache and reduced memory usage
            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-build-isolation"] + dep.split()
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {dep}: {e}")
            # Continue with other dependencies instead of failing completely
            continue
    print("✅ Runtime dependencies installation completed")
    return True

def setup_codeformer_runtime():
    """Download and setup CodeFormer at runtime"""
    if not os.path.exists('./CodeFormer'):
        print("🔄 Cloning CodeFormer repository...")
        try:
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/sczhou/CodeFormer.git"
            ])
            print("✅ CodeFormer repository cloned")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone CodeFormer: {e}")
            return False

    # Add to Python path
    sys.path.insert(0, './CodeFormer')
    return True

def download_model_if_needed(url: str, path: str) -> bool:
    """Download model file if it doesn't exist"""
    if os.path.exists(path):
        print(f"✅ Model already exists: {path}")
        return True

    try:
        print(f"📥 Downloading model: {os.path.basename(path)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"✅ Downloaded: {path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {path}: {e}")
        return False

def ensure_models_downloaded():
    """Ensure all required models are downloaded"""
    models = [
        {
            'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            'path': './CodeFormer/weights/CodeFormer/codeformer.pth'
        },
        {
            'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
            'path': './CodeFormer/weights/facelib/detection_Resnet50_Final.pth'
        }
    ]

    for model in models:
        if not download_model_if_needed(model['url'], model['path']):
            return False
    return True

def initialize_runtime_environment():
    """Initialize the complete runtime environment"""
    print("🚀 Initializing runtime environment...")

    # Step 1: Install dependencies
    if not install_runtime_dependencies():
        print("❌ Failed to install dependencies")
        return False

    # Step 2: Setup CodeFormer
    if not setup_codeformer_runtime():
        print("❌ Failed to setup CodeFormer")
        return False

    # Step 3: Download models
    if not ensure_models_downloaded():
        print("❌ Failed to download models")
        return False

    print("✅ Runtime environment ready")
    return True

# Initialize FastAPI app first (before heavy initialization)
app = FastAPI(
    title="CodeFormer Face Swap API",
    description="Face swapping API using CodeFormer and RetinaFace",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for runtime setup
RUNTIME_READY = False
CODEFORMER_AVAILABLE = False
INITIALIZATION_IN_PROGRESS = False

import threading
import asyncio

def initialize_runtime_in_background():
    """Initialize runtime environment in background thread"""
    global RUNTIME_READY, CODEFORMER_AVAILABLE, INITIALIZATION_IN_PROGRESS

    INITIALIZATION_IN_PROGRESS = True
    print("🚀 Starting background runtime environment initialization...")

    try:
        # Try to import dependencies first (they might already be installed)
        try:
            import torch
            import cv2
            print("✅ Core dependencies already available")
            dependencies_available = True
        except ImportError:
            print("📦 Core dependencies not found, installing...")
            dependencies_available = False

        # Initialize runtime environment only if needed
        if not dependencies_available:
            RUNTIME_READY = initialize_runtime_environment()
        else:
            # Dependencies available, just setup CodeFormer
            if not setup_codeformer_runtime():
                print("❌ Failed to setup CodeFormer")
                RUNTIME_READY = False
            elif not ensure_models_downloaded():
                print("❌ Failed to download models")
                RUNTIME_READY = False
            else:
                print("✅ Runtime environment ready")
                RUNTIME_READY = True

        # Check if CodeFormer is available and import accordingly
        if RUNTIME_READY:
            try:
                # CRITICAL: Import cv2 FIRST to avoid numpy._DTypeMeta error
                import cv2
                print("✅ OpenCV imported successfully")
                
                # Import torch after cv2
                import torch
                print("✅ PyTorch imported successfully")
                
                # Now import CodeFormer components after cv2/torch are loaded
                from facelib.detection import init_detection_model
                from facelib.utils.face_restoration_helper import FaceRestoreHelper
                from facelib.utils.face_utils import paste_face_back
                from basicsr.utils import img2tensor, tensor2img
                from torchvision.transforms.functional import normalize
                from basicsr.archs.codeformer_arch import CodeFormer
                
                # Try BasicSR with fallback
                try:
                    from basicsr.utils.misc import get_device
                    print("✅ Using CodeFormer's BasicSR")
                except ImportError:
                    print("⚠️ BasicSR import failed, using torch device detection")
                    def get_device():
                        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                CODEFORMER_AVAILABLE = True
                print("✅ CodeFormer components imported successfully after runtime setup")

            except ImportError as e:
                print(f"⚠️ CodeFormer import failed after runtime setup: {e}")
                print("Running in limited mode without CodeFormer enhancement")
                CODEFORMER_AVAILABLE = False
        else:
            print("⚠️ Runtime environment setup failed")

    except Exception as e:
        print(f"❌ Background initialization failed: {e}")
        RUNTIME_READY = False

    finally:
        INITIALIZATION_IN_PROGRESS = False
        print(f"🔄 Background initialization completed. Runtime ready: {RUNTIME_READY}")

@app.on_event("startup")
async def startup_event():
    """Start background initialization without blocking"""
    print("🚀 Starting non-blocking runtime initialization...")

    # Start initialization in background thread
    init_thread = threading.Thread(target=initialize_runtime_in_background, daemon=True)
    init_thread.start()

    print("✅ Background initialization started. App is ready to serve health checks.")

def get_device():
    """Get the appropriate device for processing"""
    try:
        import torch
        # Force CPU for deployment
        return torch.device('cpu')
    except ImportError:
        # Fallback if torch not available
        return 'cpu'

class FaceSwapProcessor:
    def __init__(self):
        """Initialize face detection and processing components"""
        print("🚀 Initializing FaceSwapProcessor with CodeFormer...")

        self.device = get_device()
        print(f"Using device: {self.device}")

        if not CODEFORMER_AVAILABLE:
            print("❌ CodeFormer not available - cannot initialize")
            self.face_helper = None
            self.net = None
            return

        try:
            # Initialize FaceRestoreHelper with RetinaFace ResNet50
            # Reference: CodeFormer/inference_codeformer.py line 165
            self.face_helper = FaceRestoreHelper(
                upscale=1,
                face_size=512,  # CodeFormer's standard face size
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',  # CodeFormer's default high-quality model
                save_ext='png',
                use_parse=False,  # We don't need parsing for face swap
                device=self.device
            )

            # Initialize CodeFormer network
            try:
                self.net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                    connect_list=['32', '64', '128', '256']).to(self.device)
                # Load pretrained weights - Heroku paths
                checkpoint_path = './CodeFormer/weights/CodeFormer/codeformer.pth'
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.net.load_state_dict(checkpoint['params_ema'])
                self.net.eval()
                print("✅ CodeFormer network loaded successfully")
            except Exception as e:
                print(f"⚠️  CodeFormer network loading failed: {e}")
                self.net = None

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

# Global processor instance - will be initialized in background
processor = None

def get_processor():
    """Get the processor instance, initialize if needed"""
    global processor
    if processor is None and RUNTIME_READY and CODEFORMER_AVAILABLE:
        try:
            processor = FaceSwapProcessor()
            print("✅ FaceSwapProcessor initialized")
        except Exception as e:
            print(f"❌ Failed to initialize FaceSwapProcessor: {e}")
    return processor

def image_to_array(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to numpy array (BGR format for OpenCV)"""
    try:
        # Import cv2 if available
        try:
            import cv2
        except ImportError:
            raise HTTPException(status_code=503, detail="OpenCV not available. Runtime initialization in progress.")

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
        # Import cv2 if available
        try:
            import cv2
        except ImportError:
            raise HTTPException(status_code=503, detail="OpenCV not available. Runtime initialization in progress.")

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
        "status": "ready" if RUNTIME_READY else "initializing" if INITIALIZATION_IN_PROGRESS else "starting",
        "runtime_ready": RUNTIME_READY,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - always returns healthy status for container health checks"""
    status = "healthy"

    if INITIALIZATION_IN_PROGRESS:
        runtime_status = "initializing"
    elif RUNTIME_READY:
        runtime_status = "ready"
    else:
        runtime_status = "starting"

    return {
        "status": status,  # Always healthy for container health checks
        "runtime_status": runtime_status,
        "runtime_ready": RUNTIME_READY,
        "codeformer_available": CODEFORMER_AVAILABLE,
        "initialization_in_progress": INITIALIZATION_IN_PROGRESS,
        "message": "Service is healthy. " + (
            "Runtime environment ready for face swapping" if RUNTIME_READY else
            "Runtime environment initializing in background..." if INITIALIZATION_IN_PROGRESS else
            "Runtime environment starting..."
        )
    }

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect and extract face from uploaded image
    Based on CodeFormer's face detection pipeline
    """
    if not RUNTIME_READY:
        raise HTTPException(
            status_code=503,
            detail="Runtime environment not ready. Please wait for initialization to complete."
        )

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Get processor instance
        proc = get_processor()
        if proc is None:
            raise HTTPException(status_code=503, detail="Face processing system not ready")

        # Convert to numpy array
        image_array = image_to_array(file)

        # Extract face using CodeFormer's algorithm
        extracted_face, affine_matrix, message = proc.detect_and_extract_face(image_array)

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
    if not RUNTIME_READY:
        raise HTTPException(
            status_code=503,
            detail="Runtime environment not ready. Please wait for initialization to complete."
        )

    if not target_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Target image must be an image file")

    if not source_face or not affine_matrix:
        raise HTTPException(status_code=400, detail="Source face and affine matrix required")

    try:
        # Get processor instance
        proc = get_processor()
        if proc is None:
            raise HTTPException(status_code=503, detail="Face processing system not ready")

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
        result_image, message = proc.swap_faces(target_array, face_array, affine_array)

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
    if not RUNTIME_READY:
        raise HTTPException(
            status_code=503,
            detail="Runtime environment not ready. Please wait for initialization to complete."
        )

    if not source_image.content_type.startswith('image/') or not target_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Both files must be images")

    try:
        # Get processor instance
        proc = get_processor()
        if proc is None:
            raise HTTPException(status_code=503, detail="Face processing system not ready")

        # Convert uploaded files to numpy arrays
        source_array = image_to_array(source_image)
        target_array = image_to_array(target_image)

        # Use complete face swap workflow
        result_image, message = proc.complete_face_swap(source_array, target_array)

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
