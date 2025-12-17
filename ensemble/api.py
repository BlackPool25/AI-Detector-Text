"""
AI Text Detection API - Modal-Ready Endpoints

This module provides FastAPI endpoints for the AI Text Detection Ensemble.
Designed for deployment on Modal.com or similar serverless platforms.

Usage (local):
    uvicorn api:app --host 0.0.0.0 --port 8000

Usage (Modal):
    modal deploy api.py
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy loading for faster cold starts
detector = None


def get_detector():
    """Lazy load detector to speed up cold starts"""
    global detector
    if detector is None:
        from .detector import EnsembleDetector
        detector = EnsembleDetector()
    return detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Pre-load detector on startup for faster first request
    print("Pre-loading AI Text Detector...")
    get_detector()
    print("Detector ready!")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Text Detection API",
    description="Ensemble AI text detection using DeTeCtive, Binoculars, and Fast-DetectGPT",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DetectionRequest(BaseModel):
    """Request model for single text detection"""
    text: str = Field(..., min_length=10, max_length=50000, description="Text to analyze")
    include_breakdown: bool = Field(True, description="Include per-detector breakdown")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Artificial intelligence represents a transformative technology that continues to reshape industries worldwide.",
                "include_breakdown": True
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection"""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    include_breakdown: bool = Field(False, description="Include per-detector breakdown")


class DetectorBreakdown(BaseModel):
    """Individual detector result"""
    prediction: str
    score: float
    confidence: float
    method: str


class DetectionResponse(BaseModel):
    """Response model for detection result"""
    prediction: str = Field(..., description="AI, Human, or UNCERTAIN")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    agreement: str = Field(..., description="Detector agreement e.g. '3/3'")
    ensemble_score: float = Field(..., ge=0, le=1, description="Weighted ensemble score")
    suggested_action: str = Field(..., description="ACCEPT, REVIEW, or REJECT")
    breakdown: Optional[dict] = Field(None, description="Per-detector results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "AI",
                "confidence": 0.85,
                "agreement": "3/3",
                "ensemble_score": 0.78,
                "suggested_action": "REJECT",
                "breakdown": {
                    "detective": {"prediction": "AI", "score": 0.8, "confidence": 0.6},
                    "binoculars": {"prediction": "AI", "score": 0.75, "confidence": 0.5},
                    "fast_detect": {"prediction": "AI", "score": 0.72, "confidence": 0.44}
                }
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response model for batch detection"""
    results: List[DetectionResponse]
    total: int
    ai_count: int
    human_count: int
    uncertain_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    detector_loaded: bool


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        detector_loaded=detector is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        detector_loaded=detector is not None
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest):
    """
    Detect if a single text is AI-generated
    
    Returns prediction with confidence, agreement level, and optional breakdown.
    """
    try:
        det = get_detector()
        result = det.detect(request.text, return_breakdown=request.include_breakdown)
        
        breakdown = None
        if request.include_breakdown and result.breakdown:
            breakdown = {}
            for name, det_result in result.breakdown.items():
                breakdown[name] = {
                    "prediction": det_result.prediction,
                    "score": det_result.score,
                    "confidence": det_result.confidence,
                    "method": det_result.details.get("method", "unknown")
                }
        
        return DetectionResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            agreement=result.agreement,
            ensemble_score=result.ensemble_score,
            suggested_action=result.suggested_action,
            breakdown=breakdown
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(request: BatchDetectionRequest):
    """
    Detect if multiple texts are AI-generated
    
    Processes texts sequentially and returns aggregated results.
    Maximum 100 texts per request.
    """
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    try:
        det = get_detector()
        results = []
        ai_count = 0
        human_count = 0
        uncertain_count = 0
        
        for text in request.texts:
            try:
                result = det.detect(text, return_breakdown=request.include_breakdown)
                
                breakdown = None
                if request.include_breakdown and result.breakdown:
                    breakdown = {}
                    for name, det_result in result.breakdown.items():
                        breakdown[name] = {
                            "prediction": det_result.prediction,
                            "score": det_result.score,
                            "confidence": det_result.confidence,
                            "method": det_result.details.get("method", "unknown")
                        }
                
                response = DetectionResponse(
                    prediction=result.prediction,
                    confidence=result.confidence,
                    agreement=result.agreement,
                    ensemble_score=result.ensemble_score,
                    suggested_action=result.suggested_action,
                    breakdown=breakdown
                )
                results.append(response)
                
                if result.prediction == "AI":
                    ai_count += 1
                elif result.prediction == "Human":
                    human_count += 1
                else:
                    uncertain_count += 1
                    
            except ValueError:
                # Skip texts that are too short
                results.append(DetectionResponse(
                    prediction="ERROR",
                    confidence=0.0,
                    agreement="0/0",
                    ensemble_score=0.0,
                    suggested_action="REVIEW",
                    breakdown=None
                ))
                uncertain_count += 1
        
        return BatchDetectionResponse(
            results=results,
            total=len(results),
            ai_count=ai_count,
            human_count=human_count,
            uncertain_count=uncertain_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@app.get("/info")
async def get_info():
    """Get detector configuration info"""
    det = get_detector()
    return {
        "components": list(det.weights.keys()),
        "weights": det.weights,
        "device": det.device,
        "models": {
            "detective": "princeton-nlp/unsup-simcse-roberta-base",
            "binoculars_observer": "gpt2",
            "binoculars_performer": "gpt2-medium",
            "fast_detect": "gpt2"
        }
    }


# ============================================================================
# Modal Deployment Configuration
# ============================================================================

# This section is for Modal.com deployment
# Uncomment and modify for your Modal setup

"""
import modal

# Create Modal app
stub = modal.Stub("ai-text-detector")

# Create image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "faiss-cpu",
    "numpy",
    "fastapi",
    "uvicorn",
)

# Mount model files
model_mount = modal.Mount.from_local_dir(
    local_path="./models",
    remote_path="/models"
)
database_mount = modal.Mount.from_local_dir(
    local_path="./database",
    remote_path="/database"
)

@stub.function(
    image=image,
    mounts=[model_mount, database_mount],
    gpu="T4",  # or "A10G" for better performance
    memory=8192,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    return app
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
