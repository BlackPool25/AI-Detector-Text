# 🌐 Website Integration Guide

Complete guide to integrate the AI Text Detector into your website.

---

## API Endpoint

After Modal deployment:
```
https://blackpool25--ai-text-detector-detect-endpoint.modal.run
```

---

## Basic Integration

### JavaScript (Fetch API)

```javascript
const DETECTOR_URL = 'https://blackpool25--ai-text-detector-detect-endpoint.modal.run';

async function detectAI(text) {
    const response = await fetch(DETECTOR_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            format: 'web',
            include_breakdown: true
        })
    });
    return response.json();
}

// Usage
const result = await detectAI("Text to analyze...");
console.log(result);
```

### Response Format

```json
{
    "success": true,
    "timestamp": "2024-12-03T10:30:00.000Z",
    "result": {
        "prediction": "AI",
        "is_ai": true,
        "is_human": false,
        "is_uncertain": false,
        "confidence": 0.92,
        "confidence_percent": "92.0%",
        "ensemble_score": 0.85,
        "agreement": "3/3",
        "suggested_action": "REJECT",
        "action_color": "#dc3545"
    },
    "breakdown": {
        "detective": {
            "prediction": "AI",
            "score": 0.95,
            "confidence": 0.9,
            "method": "KNN style clustering"
        },
        "binoculars": {
            "prediction": "AI", 
            "score": 0.78,
            "confidence": 0.7,
            "method": "Cross-model perplexity divergence"
        },
        "fast_detect": {
            "prediction": "AI",
            "score": 0.82,
            "confidence": 0.8,
            "method": "Conditional probability curvature"
        }
    },
    "truncated": false
}
```

---

## React Integration

### Component

```jsx
import React, { useState } from 'react';
import './AIDetector.css';

const DETECTOR_URL = 'https://blackpool25--ai-text-detector-detect-endpoint.modal.run';

function AIDetector() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleDetect = async () => {
        if (text.length < 20) {
            setError('Please enter at least 20 characters');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(DETECTOR_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    format: 'web',
                    include_breakdown: true
                })
            });

            const data = await response.json();
            
            if (data.success) {
                setResult(data);
            } else {
                setError(data.error || 'Detection failed');
            }
        } catch (err) {
            setError('Service unavailable. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="ai-detector">
            <h2>🔍 AI Text Detector</h2>
            
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste text to analyze..."
                rows={6}
            />
            
            <button 
                onClick={handleDetect} 
                disabled={loading || text.length < 20}
            >
                {loading ? 'Analyzing...' : 'Detect AI'}
            </button>

            {error && <div className="error">{error}</div>}

            {result && (
                <div className="result" style={{ borderColor: result.result.action_color }}>
                    <div className="verdict">
                        <span className="icon">
                            {result.result.is_ai ? '🤖' : '👤'}
                        </span>
                        <span className="text">
                            {result.result.prediction === 'AI' ? 'AI-Generated' : 
                             result.result.prediction === 'Human' ? 'Human-Written' : 
                             'Uncertain'}
                        </span>
                    </div>
                    
                    <div className="confidence">
                        <strong>Confidence:</strong> {result.result.confidence_percent}
                    </div>
                    
                    <div className="agreement">
                        <strong>Agreement:</strong> {result.result.agreement}
                    </div>

                    {result.breakdown && (
                        <div className="breakdown">
                            <h4>Detector Breakdown</h4>
                            {Object.entries(result.breakdown).map(([name, det]) => (
                                <div key={name} className="detector">
                                    <span className="name">{name}</span>
                                    <span className="score">{(det.score * 100).toFixed(0)}%</span>
                                    <span className="pred">{det.prediction}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    <div 
                        className="action"
                        style={{ backgroundColor: result.result.action_color }}
                    >
                        {result.result.suggested_action === 'REJECT' ? '⛔ Likely AI' :
                         result.result.suggested_action === 'REVIEW' ? '⚠️ Review Needed' :
                         '✅ Likely Authentic'}
                    </div>
                </div>
            )}
        </div>
    );
}

export default AIDetector;
```

### CSS

```css
.ai-detector {
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.ai-detector textarea {
    width: 100%;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 14px;
    resize: vertical;
}

.ai-detector button {
    width: 100%;
    padding: 12px;
    margin-top: 12px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
}

.ai-detector button:disabled {
    background: #ccc;
    cursor: not-allowed;
}

.ai-detector .error {
    color: #dc3545;
    padding: 12px;
    margin-top: 12px;
    background: #fff5f5;
    border-radius: 8px;
}

.ai-detector .result {
    margin-top: 20px;
    padding: 20px;
    border: 3px solid;
    border-radius: 12px;
    background: #fafafa;
}

.ai-detector .verdict {
    font-size: 24px;
    text-align: center;
    margin-bottom: 16px;
}

.ai-detector .verdict .icon {
    font-size: 48px;
    display: block;
}

.ai-detector .breakdown {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
}

.ai-detector .detector {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
}

.ai-detector .action {
    text-align: center;
    padding: 12px;
    margin-top: 16px;
    border-radius: 8px;
    color: white;
    font-weight: bold;
}
```

---

## Vue.js Integration

```vue
<template>
  <div class="ai-detector">
    <h2>🔍 AI Text Detector</h2>
    
    <textarea v-model="text" placeholder="Paste text to analyze..." rows="6"></textarea>
    
    <button @click="detect" :disabled="loading || text.length < 20">
      {{ loading ? 'Analyzing...' : 'Detect AI' }}
    </button>

    <div v-if="error" class="error">{{ error }}</div>

    <div v-if="result" class="result" :style="{ borderColor: result.result.action_color }">
      <div class="verdict">
        {{ result.result.is_ai ? '🤖 AI-Generated' : '👤 Human-Written' }}
      </div>
      <div>Confidence: {{ result.result.confidence_percent }}</div>
      <div>Agreement: {{ result.result.agreement }}</div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      text: '',
      result: null,
      loading: false,
      error: null
    }
  },
  methods: {
    async detect() {
      this.loading = true;
      this.error = null;
      
      try {
        const res = await fetch('https://blackpool25--ai-text-detector-detect-endpoint.modal.run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: this.text,
            format: 'web',
            include_breakdown: true
          })
        });
        
        this.result = await res.json();
      } catch (err) {
        this.error = 'Service unavailable';
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>
```

---

## Plain HTML/JavaScript

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Text Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        textarea { width: 100%; height: 150px; padding: 10px; margin-bottom: 10px; }
        button { width: 100%; padding: 15px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:disabled { background: #ccc; }
        .result { margin-top: 20px; padding: 20px; border: 2px solid #ddd; border-radius: 8px; }
        .ai { border-color: #dc3545; background: #fff5f5; }
        .human { border-color: #28a745; background: #f5fff5; }
    </style>
</head>
<body>
    <h1>🔍 AI Text Detector</h1>
    
    <textarea id="text" placeholder="Paste text here..."></textarea>
    <button id="detect" onclick="detectAI()">Detect AI</button>
    
    <div id="result" class="result" style="display:none"></div>

    <script>
        const API = 'https://blackpool25--ai-text-detector-detect-endpoint.modal.run';
        
        async function detectAI() {
            const text = document.getElementById('text').value;
            const btn = document.getElementById('detect');
            const result = document.getElementById('result');
            
            if (text.length < 20) {
                alert('Need at least 20 characters');
                return;
            }
            
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const res = await fetch(API, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, format: 'web' })
                });
                
                const data = await res.json();
                
                if (data.success) {
                    const r = data.result;
                    result.className = `result ${r.is_ai ? 'ai' : 'human'}`;
                    result.innerHTML = `
                        <h2>${r.is_ai ? '🤖 AI-Generated' : '👤 Human-Written'}</h2>
                        <p><strong>Confidence:</strong> ${r.confidence_percent}</p>
                        <p><strong>Agreement:</strong> ${r.agreement}</p>
                    `;
                    result.style.display = 'block';
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Detect AI';
            }
        }
    </script>
</body>
</html>
```

---

## Backend Integration (Python Flask)

```python
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)
DETECTOR_URL = 'https://blackpool25--ai-text-detector-detect-endpoint.modal.run'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    text = request.json.get('text', '')
    
    if len(text) < 20:
        return jsonify({'success': False, 'error': 'Text too short'}), 400
    
    try:
        response = requests.post(DETECTOR_URL, json={
            'text': text,
            'format': 'web',
            'include_breakdown': True
        }, timeout=30)
        
        return jsonify(response.json())
    except requests.Timeout:
        return jsonify({'success': False, 'error': 'Request timeout'}), 504
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Node.js/Express Backend

```javascript
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const DETECTOR_URL = 'https://blackpool25--ai-text-detector-detect-endpoint.modal.run';

app.post('/api/detect', async (req, res) => {
    const { text } = req.body;
    
    if (!text || text.length < 20) {
        return res.status(400).json({ success: false, error: 'Text too short' });
    }
    
    try {
        const response = await axios.post(DETECTOR_URL, {
            text,
            format: 'web',
            include_breakdown: true
        }, { timeout: 30000 });
        
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

---

## CORS Configuration

If calling directly from frontend, Modal handles CORS automatically. For your own backend, add:

```python
# Flask
from flask_cors import CORS
CORS(app)
```

```javascript
// Express
const cors = require('cors');
app.use(cors());
```

---

## Error Handling Best Practices

```javascript
async function detectWithRetry(text, maxRetries = 2) {
    for (let i = 0; i <= maxRetries; i++) {
        try {
            const response = await fetch(DETECTOR_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, format: 'web' }),
                signal: AbortSignal.timeout(30000)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            if (i === maxRetries) throw error;
            await new Promise(r => setTimeout(r, 1000 * (i + 1)));
        }
    }
}
```

---

## Loading States

```jsx
function LoadingIndicator() {
    return (
        <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing text with 3 AI detectors...</p>
            <ul>
                <li>🔍 DeTeCtive: Checking writing style...</li>
                <li>📊 Binoculars: Analyzing statistics...</li>
                <li>📈 Fast-DetectGPT: Computing probability curves...</li>
            </ul>
        </div>
    );
}
```

---

## Performance Tips

1. **Debounce** real-time analysis
2. **Cache** recent results
3. **Truncate** text over 10,000 chars
4. **Show loading** - analysis takes 1-3 seconds

```javascript
// Debounce function
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const debouncedDetect = debounce(detectAI, 500);
```
