# AI Text Detection API Documentation

## Base URL
After deploying to Modal, you'll get a URL like: `https://your-username--ai-text-detector.modal.run`

---

## Endpoints

### 1. Health Check (GET)
Quick endpoint to verify the service is running.

**Endpoint:** `GET /health`

**Request:**
```bash
curl https://your-username--ai-text-detector.modal.run/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "3.4.0",
  "features": "Deepfake-based model, domain-adaptive, provenance analysis",
  "model": "Deepfake_best.pth",
  "database": "deepfake_full",
  "accuracy": "~90%",
  "detectors": ["detective", "binoculars", "fast_detect"],
  "provenance_analyzers": ["unicode", "dash", "repetition", "web_search"]
}
```

---

### 2. Main Detection Endpoint (POST)
Main endpoint for AI text detection with flexible formatting options.

**Endpoint:** `POST /detect_endpoint`

**Request Body:**
```json
{
  "text": "The text you want to analyze for AI detection",
  "format": "web",              // Optional: "web" (default) or "whatsapp"
  "include_breakdown": true,    // Optional: true (default) or false
  "enable_provenance": false    // Optional: true or false (default)
}
```

**Parameters:**
- `text` (required, string): The text to analyze. Minimum 20 characters. Maximum 10,000 characters (will be truncated if longer).
- `format` (optional, string): 
  - `"web"` (default): Returns structured JSON response
  - `"whatsapp"`: Returns formatted message string for WhatsApp
- `include_breakdown` (optional, boolean): 
  - `true` (default): Includes detailed breakdown of each detector
  - `false`: Returns only main prediction
- `enable_provenance` (optional, boolean):
  - `false` (default): Disables provenance analysis (faster)
  - `true`: Enables provenance analysis (unicode, dash, repetition, web search)

**Example Request (cURL):**
```bash
curl -X POST https://your-username--ai-text-detector.modal.run/detect_endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has revolutionized various industries by providing innovative solutions to complex problems.",
    "format": "web",
    "include_breakdown": true,
    "enable_provenance": false
  }'
```

**Example Request (Python):**
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/detect_endpoint"
payload = {
    "text": "Artificial intelligence has revolutionized various industries by providing innovative solutions to complex problems.",
    "format": "web",
    "include_breakdown": True,
    "enable_provenance": False
}

response = requests.post(url, json=payload)
result = response.json()
print(result)
```

**Response (format="web"):**
```json
{
  "success": true,
  "timestamp": "2024-12-04T18:30:00.123456Z",
  "version": "3.0.0",
  "features": "Domain-adaptive thresholding, confidence-gated voting",
  "result": {
    "prediction": "AI",
    "is_ai": true,
    "is_human": false,
    "is_uncertain": false,
    "confidence": 0.8500,
    "confidence_percent": "85.0%",
    "ensemble_score": 0.7234,
    "agreement": "2/3",
    "suggested_action": "REJECT",
    "domain": "INFORMAL",
    "domain_adjusted": false,
    "burstiness": 0.4523
  },
  "breakdown": {
    "detective": {
      "prediction": "AI",
      "score": 0.9000,
      "confidence": 0.8000,
      "details": {
        "ai_votes": 9,
        "human_votes": 1,
        "k": 10,
        "method": "KNN style clustering"
      }
    },
    "binoculars": {
      "prediction": "AI",
      "score": 0.6500,
      "confidence": 0.7000,
      "details": {
        "raw_score": 0.6234,
        "threshold": 0.71,
        "method": "Cross-model perplexity divergence"
      }
    },
    "fast_detect": {
      "prediction": "Human",
      "score": 0.3000,
      "confidence": 0.4000,
      "details": {
        "raw_curvature": 2.1234,
        "n_tokens": 45,
        "token_factor": 0.45,
        "scoring_model": "gpt2",
        "sampling_model": "gpt2-medium",
        "method": "Conditional probability curvature (Fast-DetectGPT)"
      }
    }
  }
}
```

**Response Fields:**
- `success` (boolean): Whether the request was successful
- `timestamp` (string): ISO timestamp of the request
- `version` (string): API version
- `result.prediction` (string): Main prediction - `"AI"`, `"Human"`, or `"UNCERTAIN"`
- `result.is_ai` (boolean): True if prediction is AI
- `result.is_human` (boolean): True if prediction is Human
- `result.is_uncertain` (boolean): True if prediction is UNCERTAIN
- `result.confidence` (float): Confidence score 0.0-1.0
- `result.confidence_percent` (string): Confidence as percentage
- `result.ensemble_score` (float): Weighted ensemble score
- `result.agreement` (string): Detector agreement (e.g., "2/3" means 2 out of 3 detectors agree)
- `result.suggested_action` (string): `"ACCEPT"`, `"REJECT"`, or `"REVIEW"`
- `result.domain` (string): Text domain - `"FORMAL"`, `"INFORMAL"`, or `"TECHNICAL"`
- `result.domain_adjusted` (boolean): Whether domain-adaptive thresholds were applied
- `result.burstiness` (float): Text burstiness score (higher = more human-like)
- `breakdown` (object): Detailed results from each detector (if `include_breakdown=true`)

**Error Response:**
```json
{
  "success": false,
  "error": "Text too short",
  "detail": "Minimum 20 characters required"
}
```

---

### 3. WhatsApp Endpoint (POST)
Simplified endpoint that returns WhatsApp-formatted messages.

**Endpoint:** `POST /whatsapp_endpoint`

**Request Body:**
```json
{
  "text": "The text you want to analyze"
}
```
OR
```json
{
  "message": "The text you want to analyze"
}
```

**Parameters:**
- `text` or `message` (required, string): The text to analyze
- `enable_provenance` (optional, boolean): Enable provenance analysis (default: false)

**Example Request (cURL):**
```bash
curl -X POST https://your-username--ai-text-detector.modal.run/whatsapp_endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has revolutionized various industries by providing innovative solutions to complex problems.",
    "enable_provenance": false
  }'
```

**Example Request (Python):**
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/whatsapp_endpoint"
payload = {
    "text": "Artificial intelligence has revolutionized various industries...",
    "enable_provenance": False
}

response = requests.post(url, json=payload)
result = response.json()
print(result["message"])  # Formatted WhatsApp message
```

**Response:**
```json
{
  "success": true,
  "message": "📊 *AI Text Analysis Complete* (v3.2)\n\n🤖 *AI-Generated Content Detected*\n\n🟢 *Confidence:* 85% (Very High)\n📈 *Model Agreement:* 2/3\n🎯 *Ensemble Score:* 0.72\n\n🔍 *Text Style:* 💬 Casual/Informal\n📊 *Burstiness:* 0.45 (Low (Uniform))\n\n─────────────────\n📋 *Detector Breakdown:*\n\n🎨 *DeTeCtive* (Style Analysis)\n   🤖 AI • Score: 90.0%\n   📊 Votes: 9 AI / 1 Human\n\n🔬 *Binoculars* (Stats Analysis)\n   🤖 AI • Score: 65.0%\n   📏 Raw Score: 0.623\n\n📈 *Fast-GPT* (Curve Analysis)\n   👤 Human • Score: 30.0%\n   📉 Curvature: 2.12 (→ Mixed)\n\n─────────────────\n\n⛔ *Recommendation:* This text appears to be AI-generated\n\n💡 Suggested Action: *REJECT*\n\n_Type *start* to analyze more content_",
  "prediction": "AI",
  "confidence": 0.85,
  "is_ai": true,
  "is_human": false,
  "is_uncertain": false,
  "suggested_action": "REJECT",
  "agreement": "2/3",
  "ensemble_score": 0.7234,
  "domain": "INFORMAL",
  "domain_adjusted": false,
  "burstiness": 0.4523,
  "breakdown": {
    "detective": {
      "prediction": "AI",
      "score": 0.9,
      "confidence": 0.8
    },
    "binoculars": {
      "prediction": "AI",
      "score": 0.65,
      "confidence": 0.7
    },
    "fast_detect": {
      "prediction": "Human",
      "score": 0.3,
      "confidence": 0.4
    }
  },
  "provenance_enabled": false
}
```

---

## Complete Examples

### Example 1: Basic Detection (Web Format)
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/detect_endpoint"
payload = {
    "text": "This is a sample text that I want to check if it was written by AI or a human."
}

response = requests.post(url, json=payload)
data = response.json()

if data["success"]:
    result = data["result"]
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_percent']}")
    print(f"Is AI: {result['is_ai']}")
    print(f"Suggested Action: {result['suggested_action']}")
else:
    print(f"Error: {data['error']}")
```

### Example 2: With Full Breakdown
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/detect_endpoint"
payload = {
    "text": "Your text here...",
    "format": "web",
    "include_breakdown": True,
    "enable_provenance": False
}

response = requests.post(url, json=payload)
data = response.json()

if data["success"]:
    # Main result
    result = data["result"]
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence_percent']}")
    print(f"Agreement: {result['agreement']}")
    
    # Detector breakdown
    if "breakdown" in data:
        for detector_name, detector_result in data["breakdown"].items():
            print(f"\n{detector_name}:")
            print(f"  Prediction: {detector_result['prediction']}")
            print(f"  Score: {detector_result['score']}")
            print(f"  Confidence: {detector_result['confidence']}")
```

### Example 3: WhatsApp Format
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/whatsapp_endpoint"
payload = {
    "text": "Your text here...",
    "enable_provenance": False
}

response = requests.post(url, json=payload)
data = response.json()

if data["success"]:
    # For WhatsApp, use the formatted message
    print(data["message"])
    
    # Or access structured data
    print(f"Prediction: {data['prediction']}")
    print(f"Confidence: {data['confidence']}")
```

### Example 4: With Provenance Analysis
```python
import requests

url = "https://your-username--ai-text-detector.modal.run/detect_endpoint"
payload = {
    "text": "Your text here...",
    "format": "web",
    "include_breakdown": True,
    "enable_provenance": True  # Enable provenance analysis
}

response = requests.post(url, json=payload)
data = response.json()

if data["success"] and "provenance" in data:
    prov = data["provenance"]
    
    # Unicode analysis
    if prov.get("unicode"):
        unicode_data = prov["unicode"]
        print(f"Invisible chars: {unicode_data.get('invisible_chars_count', 0)}")
        print(f"Smart quotes: {unicode_data.get('smart_quotes_count', 0)}")
        print(f"Emojis: {unicode_data.get('emoji_count', 0)}")
    
    # Dash analysis
    if prov.get("dash"):
        dash_data = prov["dash"]
        print(f"Primary dash: {dash_data.get('primary_dash')}")
        print(f"Em-dash overuse: {dash_data.get('em_dash_overuse', False)}")
    
    # Web search (if enabled)
    if prov.get("web_search"):
        web_data = prov["web_search"]
        print(f"Exact matches: {web_data.get('exact_quote_matches', 0)}")
        print(f"Likely copied: {web_data.get('is_likely_copied', False)}")
```

---

## Response Values Explained

### Prediction Values
- `"AI"`: Text is likely AI-generated
- `"Human"`: Text is likely human-written
- `"UNCERTAIN"`: Cannot determine with sufficient confidence

### Suggested Actions
- `"ACCEPT"`: High confidence human text - safe to accept
- `"REJECT"`: High confidence AI text - should be rejected
- `"REVIEW"`: Uncertain or low confidence - requires manual review

### Domain Values
- `"FORMAL"`: Formal/academic text (Wikipedia, research papers)
- `"INFORMAL"`: Casual/conversational text
- `"TECHNICAL"`: Technical documentation or code-related text

### Confidence Levels
- **0.75-1.0**: Very High confidence
- **0.70-0.74**: High confidence
- **0.55-0.69**: Moderate confidence
- **0.0-0.54**: Low confidence

---

## Notes

1. **Text Length**: Minimum 20 characters, maximum 10,000 characters (will be truncated)
2. **Timeout**: 180 seconds per request
3. **Provenance Analysis**: Requires additional processing time. Set `enable_provenance: true` only when needed.
4. **Web Search**: Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` environment variables in Modal secrets
5. **Model**: Uses Deepfake_best.pth with deepfake_full database (20K samples)
6. **Accuracy**: ~88-90% on HC3 dataset

---

## Error Handling

Always check the `success` field in the response:

```python
response = requests.post(url, json=payload)
data = response.json()

if not data.get("success"):
    print(f"Error: {data.get('error', 'Unknown error')}")
    if "traceback" in data:
        print(f"Traceback: {data['traceback']}")
else:
    # Process successful response
    result = data["result"]
    ...
```

