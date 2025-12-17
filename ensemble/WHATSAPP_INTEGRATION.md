# 🤖 WhatsApp Bot Integration Guide

Complete guide to integrate the AI Text Detector with your WhatsApp bot.

---

## Quick Start

### Your API Endpoint

After Modal deployment:
```
https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run
```

### Basic Request

```bash
curl -X POST https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{"text": "Text to analyze..."}'
```

### Response (Pre-formatted for WhatsApp)

```json
{
    "success": true,
    "message": "🤖 *AI Text Analysis*\n\n📊 *Result:* AI-Generated\n🟢 *Confidence:* Very High (92%)\n🗳️ *Agreement:* 3/3\n\n📋 *Analysis Details:*\n  🔵 DeTeCtive (Style): AI (95%)\n  🔵 Binoculars (Stats): AI (88%)\n  🔵 Fast-GPT (Curve): AI (91%)\n\n⛔ *Recommendation:* Likely AI content - verify source",
    "prediction": "AI",
    "confidence": 0.92,
    "is_ai": true,
    "suggested_action": "REJECT"
}
```

---

## Integration Code

### Node.js (Baileys - WhatsApp Web)

```javascript
const { default: makeWASocket, useMultiFileAuthState } = require('@whiskeysockets/baileys');
const axios = require('axios');

const DETECTOR_URL = 'https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run';

async function detectAI(text) {
    const response = await axios.post(DETECTOR_URL, { text }, { timeout: 30000 });
    return response.data;
}

async function startBot() {
    const { state, saveCreds } = await useMultiFileAuthState('./auth');
    const sock = makeWASocket({ auth: state });
    
    sock.ev.on('creds.update', saveCreds);
    
    sock.ev.on('messages.upsert', async ({ messages }) => {
        const msg = messages[0];
        if (!msg.message) return;
        
        const text = msg.message.conversation || 
                     msg.message.extendedTextMessage?.text || '';
        const jid = msg.key.remoteJid;
        
        // Check for /detect command
        if (text.toLowerCase().startsWith('/detect ')) {
            const textToCheck = text.substring(8).trim();
            
            if (textToCheck.length < 20) {
                await sock.sendMessage(jid, { 
                    text: '⚠️ Text too short. Need at least 20 characters.' 
                });
                return;
            }
            
            await sock.sendMessage(jid, { text: '🔍 Analyzing...' });
            
            try {
                const result = await detectAI(textToCheck);
                await sock.sendMessage(jid, { text: result.message });
            } catch (error) {
                await sock.sendMessage(jid, { 
                    text: '❌ Error analyzing text. Try again later.' 
                });
            }
        }
    });
}

startBot();
```

### Python (python-whatsapp)

```python
import requests
from flask import Flask, request

app = Flask(__name__)
DETECTOR_URL = 'https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run'

def detect_ai(text: str) -> dict:
    try:
        response = requests.post(DETECTOR_URL, json={'text': text}, timeout=30)
        return response.json()
    except:
        return {'success': False, 'message': '❌ Service unavailable'}

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = data.get('message', {}).get('text', '')
    
    if message.lower().startswith('/detect '):
        text_to_check = message[8:].strip()
        result = detect_ai(text_to_check)
        
        # Send result back via your WhatsApp API
        send_whatsapp_message(data['from'], result['message'])
    
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
```

### Twilio WhatsApp

```javascript
const express = require('express');
const twilio = require('twilio');
const axios = require('axios');

const app = express();
app.use(express.urlencoded({ extended: true }));

const DETECTOR_URL = 'https://blackpool25--ai-text-detector-whatsapp-endpoint.modal.run';

app.post('/webhook', async (req, res) => {
    const message = req.body.Body;
    const twiml = new twilio.twiml.MessagingResponse();
    
    if (message.toLowerCase().startsWith('/detect ')) {
        const text = message.substring(8).trim();
        
        try {
            const result = await axios.post(DETECTOR_URL, { text });
            twiml.message(result.data.message);
        } catch {
            twiml.message('❌ Error analyzing text.');
        }
    } else {
        twiml.message('Send: /detect <your text> to check for AI');
    }
    
    res.type('text/xml').send(twiml.toString());
});

app.listen(3000);
```

---

## Bot Commands

| Command | Example |
|---------|---------|
| `/detect <text>` | `/detect This is the text to analyze` |
| `/check <text>` | Alias for detect |
| `/ai <text>` | Short version |

---

## Response Meanings

| Emoji | Meaning |
|-------|---------|
| 🤖 | AI-generated text detected |
| 👤 | Human-written text |
| ❓ | Uncertain - needs review |
| 🟢 | High confidence (>85%) |
| 🟡 | Good confidence (70-85%) |
| 🟠 | Moderate confidence (55-70%) |
| 🔴 | Low confidence (<55%) |
| ⛔ | Recommend rejection |
| ⚠️ | Recommend review |
| ✅ | Appears authentic |

---

## Error Handling

```javascript
async function safeDetect(text, jid, sock) {
    // Validate
    if (!text || text.length < 20) {
        await sock.sendMessage(jid, { 
            text: '⚠️ Need at least 20 characters to analyze.' 
        });
        return;
    }
    
    // Truncate if too long
    if (text.length > 10000) {
        text = text.substring(0, 10000);
    }
    
    try {
        const result = await axios.post(DETECTOR_URL, { text }, {
            timeout: 30000
        });
        
        await sock.sendMessage(jid, { text: result.data.message });
        
    } catch (error) {
        let errorMsg = '❌ Error analyzing text.';
        
        if (error.code === 'ECONNABORTED') {
            errorMsg = '⏱️ Request timed out. Try shorter text.';
        } else if (error.response?.status === 503) {
            errorMsg = '🔄 Service starting up. Try again in 30 seconds.';
        }
        
        await sock.sendMessage(jid, { text: errorMsg });
    }
}
```

---

## Rate Limiting

```javascript
const cooldowns = new Map();
const COOLDOWN = 5000; // 5 seconds

function canRequest(userId) {
    const last = cooldowns.get(userId) || 0;
    if (Date.now() - last < COOLDOWN) {
        return false;
    }
    cooldowns.set(userId, Date.now());
    return true;
}

// In message handler:
if (!canRequest(msg.key.participant || msg.key.remoteJid)) {
    await sock.sendMessage(jid, { 
        text: '⏳ Please wait 5 seconds between requests.' 
    });
    return;
}
```

---

## Keep Warm (Avoid Cold Starts)

```javascript
// Ping every 4 minutes to keep the container warm
setInterval(async () => {
    try {
        await axios.get('https://blackpool25--ai-text-detector-health.modal.run');
    } catch {}
}, 240000);
```

---

## Test Samples

**AI-like (should detect as AI):**
```
Artificial intelligence has revolutionized various industries by providing innovative solutions to complex problems. The implementation of machine learning algorithms has enabled unprecedented levels of automation and efficiency across multiple sectors.
```

**Human-like (should detect as Human):**
```
honestly idk why everyone's making such a big deal about this lol... like yeah it's cool but maybe let's calm down? just saying 🤷
```
