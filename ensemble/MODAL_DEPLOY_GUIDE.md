# AI Text Detector - Modal Deployment Guide

## Quick Start (3 Steps)

### 1. Install & Authenticate Modal
```bash
pip install modal
modal token new
```

### 2. Navigate to Ensemble Directory
```bash
cd /home/lightdesk/Projects/AI-Text/ensemble
```

### 3. Deploy to Modal
```bash
modal deploy modal_app.py
```

After deployment, you'll see output like:
```
✓ Created AITextDetector.health => https://blackpool25--ai-text-detector-health.modal.run
✓ Created AITextDetector.detect_endpoint => https://blackpool25--ai-text-detector-detect-endpoint.modal.run
✓ Created AITextDetector.whatsapp_endpoint => https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run
✓ Created AITextDetector.batch_endpoint => https://blackpool25--ai-text-detector-batch-endpoint.modal.run
```

### Test Your Deployment

```bash
# Test local before deploy
modal serve modal_app.py

# Test deployed endpoints
curl -X POST https://blackpool25--ai-text-detector-detect-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze here...", "format": "web"}'
```

---

## API Endpoints

### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600,
  "requests_handled": 150
}
```

### 2. Main Detection Endpoint
```
POST /detect-endpoint
```

**Request:**
```json
{
  "text": "Text to analyze...",
  "format": "web",  // "web" | "whatsapp" | "raw"
  "include_breakdown": true
}
```

**Response (format: "web"):**
```json
{
  "success": true,
  "timestamp": "2024-12-03T10:30:00",
  "result": {
    "prediction": "AI",
    "is_ai": true,
    "is_human": false,
    "confidence": 0.87,
    "confidence_percent": "87.0%",
    "ensemble_score": 0.75,
    "agreement": "3/3",
    "suggested_action": "REJECT"
  },
  "breakdown": {
    "detective": {"prediction": "AI", "score": 0.85, "confidence": 0.7},
    "binoculars": {"prediction": "AI", "score": 0.72, "confidence": 0.6},
    "fast_detect": {"prediction": "AI", "score": 0.68, "confidence": 0.5}
  }
}
```

### 3. WhatsApp Endpoint
```
POST /whatsapp-endpoint
```

**Request:**
```json
{"text": "Text to analyze..."}
```

**Response:**
```json
{
  "success": true,
  "message": "🤖 *AI Detection Result*\n\n📊 *Verdict:* AI-Generated\n🟢 *Confidence:* Very High (87%)\n...",
  "prediction": "AI",
  "confidence": 0.87
}
```

### 4. Batch Detection
```
POST /batch-endpoint
```

**Request:**
```json
{
  "texts": ["Text 1...", "Text 2...", "Text 3..."]
}
```

**Response:**
```json
{
  "success": true,
  "results": [...],
  "summary": {
    "total": 3,
    "ai_detected": 2,
    "human_detected": 1,
    "uncertain": 0
  }
}
```

---

## Cost Optimization

### GPU Selection
- **T4** (default): $0.59/hr - Good balance of cost/performance
- **A10G**: $1.10/hr - 2x faster inference
- **L4**: $0.80/hr - Better than T4, cheaper than A10G

### Tips to Reduce Costs
1. **Container idle timeout** is set to 5 minutes - containers shut down when not in use
2. **Cold start** takes ~30-60 seconds; keep endpoints warm for production
3. Use **batch endpoint** for multiple texts to share container time
4. **Truncate long texts** (max 10,000 chars) to reduce processing time

---

## Monitoring

### View Logs
```bash
modal app logs ai-text-detector
```

### Check Usage
```bash
modal app list
```

### Stop Deployment
```bash
modal app stop ai-text-detector
```

---

## Updating the Deployment

After making code changes:
```bash
cd /home/lightdesk/Projects/AI-Text/ensemble
modal deploy modal_app.py
```

The new version will replace the old one with zero downtime.
