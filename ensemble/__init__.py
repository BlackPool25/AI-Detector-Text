"""
AI Text Detection Ensemble

A self-contained 3-component ensemble for detecting AI-generated text:
- DeTeCtive: Style clustering via contrastive learning
- Binoculars: Cross-model probability divergence
- Fast-DetectGPT: Probability curvature smoothness

Usage:
    from ensemble import EnsembleDetector
    
    detector = EnsembleDetector()
    result = detector.detect("Your text here...")
    print(result.prediction, result.confidence)
"""

from .detector import (
    EnsembleDetector,
    DeTeCtiveDetector,
    BinocularsDetector,
    FastDetectGPTDetector,
    DetectionResult,
    EnsembleResult,
    format_result,
    get_device,
    get_ensemble_dir,
    get_models_dir,
    get_database_dir,
)

__version__ = "1.0.0"

__all__ = [
    'EnsembleDetector',
    'DeTeCtiveDetector',
    'BinocularsDetector',
    'FastDetectGPTDetector',
    'DetectionResult',
    'EnsembleResult',
    'format_result',
    'get_device',
    'get_ensemble_dir',
    'get_models_dir',
    'get_database_dir',
]
