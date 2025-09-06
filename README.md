# CodeFormer Face Swap API - Heroku Deployment

A high-quality face swapping API using CodeFormer, optimized for Heroku deployment.

## Features

- High-quality face detection using RetinaFace ResNet50
- Face swapping with CodeFormer enhancement
- FastAPI web interface with automatic docs
- Heroku-optimized deployment

## Quick Deploy to Heroku

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## Manual Deployment

1. **Install Heroku CLI** and login:

   ```bash
   heroku login
   ```

2. **Create Heroku app**:

   ```bash
   heroku create your-faceswap-api
   ```

3. **Deploy**:

   ```bash
   git push heroku main
   ```

4. **Scale up** (if needed):
   ```bash
   heroku ps:scale web=1
   ```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /extract-face` - Extract face from image
- `POST /swap-faces` - Swap faces between images
- `POST /complete-face-swap` - Complete face swap with enhancement

## API Documentation

Once deployed, visit `https://your-app.herokuapp.com/docs` for interactive API documentation.

const response = await fetch("YOUR_RENDER_URL/complete-face-swap", {
method: "POST",
body: formData,
});

return response.blob(); // Returns processed image
};

````

### curl

```bash
curl -X POST "YOUR_RENDER_URL/complete-face-swap" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg" \
  --output result.jpg
````

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
