# CodeFormer Face Swap API - Render Deployment

This is a production-ready FastAPI server that provides face swapping functionality using CodeFormer's state-of-the-art face restoration models.

## 🚀 Live API

**Endpoints:**

- `GET /health` - Health check
- `POST /complete-face-swap` - Complete face swap (main endpoint)
- `GET /docs` - Interactive API documentation

## 🛠️ Deployment

### Prerequisites

- GitHub account
- Render account (render.com)

### Deploy to Render

1. **Push this repository to GitHub**
2. **Create new Web Service on Render:**

   - Connect GitHub repository
   - Runtime: `Docker`
   - Instance Type: `Pro` (4GB RAM minimum)
   - Health Check Path: `/health`

3. **Environment Variables:**
   ```
   PORT=10000 (automatically set by Render)
   PYTHON_VERSION=3.9
   ```

### Build Process

- Docker builds include CodeFormer repository clone
- Models are downloaded during build (~10-15 minutes first time)
- Subsequent builds use Docker layer caching

## 📝 Usage

### JavaScript/TypeScript

```javascript
const swapFaces = async (sourceFile, targetFile) => {
  const formData = new FormData();
  formData.append("source_image", sourceFile);
  formData.append("target_image", targetFile);

  const response = await fetch("YOUR_RENDER_URL/complete-face-swap", {
    method: "POST",
    body: formData,
  });

  return response.blob(); // Returns processed image
};
```

### curl

```bash
curl -X POST "YOUR_RENDER_URL/complete-face-swap" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg" \
  --output result.jpg
```

## 🔧 Technical Details

- **Runtime:** Python 3.9 on CPU (optimized for Render)
- **Models:** RetinaFace ResNet50 for face detection
- **Processing:** CodeFormer face enhancement + seamless blending
- **Output:** High-quality face swapped images

## 💰 Cost Estimation (Render)

- **Instance:** Pro plan $85/month (4GB RAM)
- **Bandwidth:** First 100GB free
- **Processing:** ~2-5 seconds per image on CPU

## 🔒 Security

- Non-root Docker container
- No persistent storage of uploaded images
- CORS enabled for web applications
- Health checks for reliability
