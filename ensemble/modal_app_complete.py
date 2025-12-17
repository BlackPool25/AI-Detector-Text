"""
AI Text Detection Ensemble - Modal Serverless Deployment (COMPLETE)

Full 3-component ensemble with all calibrations:
- DeTeCtive (SimCSE RoBERTa + FAISS KNN)
- Binoculars (GPT-2 cross-perplexity, calibrated threshold=0.71)
- Fast-DetectGPT (probability curvature, calibrated params)

Deploy:
    modal deploy modal_app_complete.py

Test locally:
    modal serve modal_app_complete.py
"""

import modal
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("ai-text-detector")

# Create image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Web framework (required for Modal web endpoints)
        "fastapi>=0.104.0",
        # ML dependencies
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.20.0",  # Required for device_map in transformers
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
        "scipy>=1.10.0",
        "huggingface_hub>=0.19.0",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    })
    # Copy DeTeCtive model and database files to container
    .add_local_file("models/Deepfake_best.pth", "/root/ensemble/models/Deepfake_best.pth", copy=True)
    .add_local_dir("database/deepfake_sample", "/root/ensemble/database/deepfake_sample", copy=True)
    # Copy ensemble Python source code
    .add_local_file("detector.py", "/root/ensemble/detector.py", copy=True)
    .add_local_file("__init__.py", "/root/ensemble/__init__.py", copy=True)
    .add_local_dir("core", "/root/ensemble/core", copy=True)
    .run_commands(
        # Pre-download ALL models during image build
        "python -c \""
        "from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel; "
        "print('Downloading GPT-2...'); "
        "AutoTokenizer.from_pretrained('gpt2'); "
        "GPT2LMHeadModel.from_pretrained('gpt2'); "
        "print('Downloading GPT-2 Medium...'); "
        "AutoTokenizer.from_pretrained('gpt2-medium'); "
        "GPT2LMHeadModel.from_pretrained('gpt2-medium'); "
        "print('Downloading SimCSE RoBERTa...'); "
        "AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-roberta-base'); "
        "AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base'); "
        "print('All models downloaded!')"
        "\""
    )
)

# Volume to store the DeTeCtive model and database
model_volume = modal.Volume.from_name("ai-detector-models", create_if_missing=True)


# =============================================================================
# Main Detector Class
# =============================================================================

@app.cls(
    image=image,
    cpu=4,
    memory=8192,  # 8GB RAM for all models
    timeout=300,
    container_idle_timeout=120,  # 2 min idle
    volumes={"/models": model_volume},
)
class AITextDetector:
    """Full 3-component AI Text Detector."""
    
    detector = None
    start_time = None
    request_count = 0
    
    @modal.enter()
    def setup(self):
        """Load the full ensemble when container starts."""
        import sys
        import shutil
        
        print("=" * 60)
        print("Initializing Full AI Text Detection Ensemble...")
        print("=" * 60)
        
        start = time.time()
        
        # Add ensemble to path
        sys.path.insert(0, "/root")
        
        # Copy model files to volume if not present
        model_src = "/root/ensemble/models/Deepfake_best.pth"
        model_dst = "/models/Deepfake_best.pth"
        db_src = "/root/ensemble/database/deepfake_sample"
        db_dst = "/models/deepfake_sample"
        
        if not os.path.exists(model_dst):
            print("Copying DeTeCtive model to persistent volume...")
            shutil.copy(model_src, model_dst)
        
        if not os.path.exists(db_dst):
            print("Copying DeTeCtive database to persistent volume...")
            shutil.copytree(db_src, db_dst)
        
        model_volume.commit()
        
        # Import and initialize the full ensemble
        from ensemble.detector import EnsembleDetector
        
        self.detector = EnsembleDetector(
            detective_path=model_dst,
            detective_db=db_dst,
            include_binoculars=True,
            include_fast_detect=True,
            device="cpu"  # CPU for cost efficiency
        )
        
        self.start_time = time.time()
        print(f"\n✅ Ensemble loaded in {time.time() - start:.1f}s")
        print(f"   Weights: {self.detector.weights}")
    
    @modal.method()
    def detect(self, text: str, format_type: str = "web", include_breakdown: bool = True) -> Dict[str, Any]:
        """Run full ensemble detection."""
        self.request_count += 1
        
        if not text or len(text.strip()) < 20:
            return {
                "success": False,
                "error": "Text too short",
                "detail": "Minimum 20 characters required"
            }
        
        # Truncate very long text
        truncated = False
        if len(text) > 10000:
            text = text[:10000]
            truncated = True
        
        try:
            result = self.detector.detect(text, return_breakdown=include_breakdown)
            
            if format_type == "whatsapp":
                return self._format_whatsapp(result, truncated)
            else:
                return self._format_web(result, include_breakdown, truncated)
                
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _format_whatsapp(self, result, truncated: bool = False) -> Dict[str, Any]:
        """Format for WhatsApp."""
        if result.prediction == "AI":
            verdict = "🤖 *AI-Generated Content Detected*"
            action = "⛔ *Recommendation:* This text appears to be AI-generated"
        elif result.prediction == "Human":
            verdict = "👤 *Human-Written Content*"
            action = "✅ *Recommendation:* This text appears authentic"
        else:
            verdict = "❓ *Uncertain Result*"
            action = "⚠️ *Recommendation:* Manual review suggested"
        
        conf_pct = result.confidence * 100
        if conf_pct >= 85:
            conf_emoji = "🟢"
        elif conf_pct >= 70:
            conf_emoji = "🟡"
        elif conf_pct >= 55:
            conf_emoji = "🟠"
        else:
            conf_emoji = "🔴"
        
        lines = [
            "📊 *AI Text Analysis Complete*",
            "",
            verdict,
            "",
            f"{conf_emoji} *Confidence:* {conf_pct:.0f}%",
            f"📈 *Model Agreement:* {result.agreement}",
            "",
            "─────────────────",
            "📋 *Detector Breakdown:*",
        ]
        
        detector_names = {
            "detective": "DeTeCtive (Style)",
            "binoculars": "Binoculars (Stats)",
            "fast_detect": "Fast-GPT (Curve)"
        }
        
        for name, det in result.breakdown.items():
            emoji = "🤖" if det.prediction == "AI" else "👤"
            display_name = detector_names.get(name, name)
            lines.append(f"  {emoji} {display_name}: {det.prediction} ({det.score*100:.0f}%)")
        
        lines.extend(["", action])
        
        if truncated:
            lines.append("\n⚠️ _Text was truncated to 10,000 characters_")
        
        return {
            "success": True,
            "message": "\n".join(lines),
            "prediction": result.prediction,
            "confidence": round(result.confidence, 4),
            "is_ai": result.prediction == "AI",
            "suggested_action": result.suggested_action
        }
    
    def _format_web(self, result, include_breakdown: bool, truncated: bool = False) -> Dict[str, Any]:
        """Format for web API."""
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": {
                "prediction": result.prediction,
                "is_ai": result.prediction == "AI",
                "is_human": result.prediction == "Human",
                "is_uncertain": result.prediction == "UNCERTAIN",
                "confidence": round(result.confidence, 4),
                "confidence_percent": f"{result.confidence * 100:.1f}%",
                "ensemble_score": round(result.ensemble_score, 4),
                "agreement": result.agreement,
                "suggested_action": result.suggested_action,
                "action_color": {
                    "REJECT": "#dc3545",
                    "REVIEW": "#ffc107",
                    "ACCEPT": "#28a745"
                }.get(result.suggested_action, "#6c757d")
            },
            "truncated": truncated
        }
        
        if include_breakdown:
            response["breakdown"] = {
                name: {
                    "prediction": det.prediction,
                    "score": round(det.score, 4),
                    "confidence": round(det.confidence, 4),
                    "method": det.details.get("method", "Unknown")
                }
                for name, det in result.breakdown.items()
            }
        
        return response
    
    @modal.method()
    def health(self) -> Dict[str, Any]:
        """Health check with detailed status."""
        uptime = time.time() - self.start_time if self.start_time else 0
        return {
            "status": "healthy",
            "version": "3.0.0-complete",
            "ensemble_loaded": self.detector is not None,
            "detectors": list(self.detector.weights.keys()) if self.detector else [],
            "weights": self.detector.weights if self.detector else {},
            "uptime_seconds": round(uptime),
            "requests_handled": self.request_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# =============================================================================
# Web Endpoints
# =============================================================================

@app.function(image=image, cpu=1, memory=512)
@modal.web_endpoint(method="GET")
def health() -> Dict[str, Any]:
    """Quick health check."""
    return {"status": "ok", "version": "3.0.0-complete"}


@app.function(
    image=image, 
    cpu=4, 
    memory=8192,
    container_idle_timeout=120,
    volumes={"/models": model_volume}
)
@modal.web_endpoint(method="POST")
def detect_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """Main detection endpoint - full ensemble."""
    text = request.get("text", "")
    format_type = request.get("format", "web")
    include_breakdown = request.get("include_breakdown", True)
    
    detector = AITextDetector()
    return detector.detect.remote(text, format_type, include_breakdown)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    container_idle_timeout=120,
    volumes={"/models": model_volume}
)
@modal.web_endpoint(method="POST")
def whatsapp_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """WhatsApp endpoint - formatted for messaging."""
    text = request.get("text", request.get("message", ""))
    detector = AITextDetector()
    return detector.detect.remote(text, "whatsapp", True)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    container_idle_timeout=120,
    volumes={"/models": model_volume}
)
@modal.web_endpoint(method="POST")
def batch_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """Batch detection for multiple texts."""
    texts = request.get("texts", [])[:20]  # Max 20 at once
    format_type = request.get("format", "raw")
    
    detector = AITextDetector()
    results = []
    
    for text in texts:
        result = detector.detect.remote(text, format_type, False)
        results.append(result)
    
    successful = [r for r in results if r.get("success")]
    ai_count = sum(1 for r in successful if r.get("result", {}).get("is_ai") or r.get("is_ai"))
    
    return {
        "success": True,
        "results": results,
        "summary": {
            "total": len(texts),
            "processed": len(successful),
            "ai_detected": ai_count,
            "human_detected": len(successful) - ai_count,
            "ai_percentage": f"{ai_count / len(successful) * 100:.1f}%" if successful else "0%"
        }
    }


# =============================================================================
# Local Test
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the full ensemble locally."""
    print("🧪 Testing Full AI Detector Ensemble...")
    
    detector = AITextDetector()
    
    ai_text = """Artificial intelligence has revolutionized various industries 
    by providing innovative solutions to complex problems. The implementation 
    of machine learning algorithms has enabled unprecedented levels of automation 
    and efficiency across multiple sectors. This technological advancement 
    represents a paradigm shift in how we approach data analysis."""
    
    human_text = """honestly idk why everyone's making such a big deal about 
    this AI thing lol... like yeah it's cool but my grandma still can't figure 
    out her iPhone so maybe let's calm down a bit? just saying 🤷"""
    
    print("\n" + "=" * 60)
    print("📝 Testing AI-like text...")
    print("=" * 60)
    result = detector.detect.remote(ai_text, "whatsapp")
    print(result.get("message", result))
    
    print("\n" + "=" * 60)
    print("📝 Testing Human-like text...")
    print("=" * 60)
    result = detector.detect.remote(human_text, "whatsapp")
    print(result.get("message", result))
    
    print("\n" + "=" * 60)
    print("💚 Health check...")
    print("=" * 60)
    health_result = detector.health.remote()
    for key, value in health_result.items():
        print(f"  {key}: {value}")
