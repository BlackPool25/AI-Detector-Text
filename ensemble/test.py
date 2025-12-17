#!/usr/bin/env python3
"""
AI Text Detection Ensemble - Test Suite

This script tests the self-contained ensemble detector.
All components should work from within the ensemble folder only.
"""

import os
import sys
import time

# Add ensemble to path
ENSEMBLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ENSEMBLE_DIR))

# Test results tracker
RESULTS = {}


def test_paths():
    """Test that all required paths exist within ensemble folder"""
    print("=" * 60)
    print("TEST 1: Path Validation (Self-Contained)")
    print("=" * 60)
    
    from ensemble.detector import get_ensemble_dir, get_models_dir, get_database_dir, get_cache_dir
    
    paths = {
        "ensemble_dir": get_ensemble_dir(),
        "models_dir": get_models_dir(),
        "database_dir": get_database_dir(),
        "cache_dir": get_cache_dir(),
    }
    
    all_ok = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Check for model files
    model_dir = get_models_dir()
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        print(f"\n  Model files found: {len(model_files)}")
        for f in model_files:
            size = os.path.getsize(os.path.join(model_dir, f)) / (1024*1024)
            print(f"    ✓ {f}: {size:.1f} MB")
    
    # Check database
    db_dir = os.path.join(get_database_dir(), "deepfake_sample")
    if os.path.exists(db_dir):
        required = ['index.faiss', 'index_meta.faiss', 'label_dict.pkl']
        print(f"\n  Database files:")
        for f in required:
            exists = os.path.exists(os.path.join(db_dir, f))
            status = "✓" if exists else "✗"
            print(f"    {status} {f}")
            if not exists:
                all_ok = False
    
    return all_ok


def test_imports():
    """Test that all required imports work"""
    print("\n" + "=" * 60)
    print("TEST 2: Imports")
    print("=" * 60)
    
    try:
        import torch
        print(f"  ✓ torch: {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import faiss
        print(f"  ✓ faiss: available")
    except ImportError as e:
        print(f"  ✗ faiss: {e}")
        return False
    
    try:
        from ensemble import EnsembleDetector, get_device
        print(f"  ✓ ensemble module: importable")
    except ImportError as e:
        print(f"  ✗ ensemble module: {e}")
        return False
    
    return True


def test_device():
    """Test device availability"""
    print("\n" + "=" * 60)
    print("TEST 3: Device")
    print("=" * 60)
    
    import torch
    from ensemble import get_device
    
    device = get_device()
    print(f"  Device: {device}")
    
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    
    return True


def test_detective():
    """Test DeTeCtive detector standalone"""
    print("\n" + "=" * 60)
    print("TEST 4: DeTeCtive Detector")
    print("=" * 60)
    
    try:
        from ensemble import DeTeCtiveDetector
        
        detector = DeTeCtiveDetector()  # Uses default paths
        
        test_text = "Artificial intelligence has become a crucial technology in modern society."
        result = detector.detect(test_text)
        
        print(f"  Text: {test_text[:40]}...")
        print(f"  Prediction: {result.prediction}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Votes: {result.details['ai_votes']} AI, {result.details['human_votes']} Human")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_binoculars():
    """Test Binoculars detector standalone"""
    print("\n" + "=" * 60)
    print("TEST 5: Binoculars Detector")
    print("=" * 60)
    
    try:
        from ensemble import BinocularsDetector
        
        # Note: use_bfloat16=False for AMD GPU compatibility
        detector = BinocularsDetector(use_bfloat16=False)
        
        test_texts = [
            ("AI-like", "Artificial intelligence represents a transformative technology."),
            ("Human-like", "omg i cant believe what happened lol so crazy"),
        ]
        
        for label, text in test_texts:
            result = detector.detect(text)
            print(f"\n  [{label}]")
            print(f"    Text: {text[:40]}...")
            print(f"    Prediction: {result.prediction}")
            print(f"    Score: {result.score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fast_detect():
    """Test Fast-DetectGPT detector standalone"""
    print("\n" + "=" * 60)
    print("TEST 6: Fast-DetectGPT Detector")
    print("=" * 60)
    
    try:
        from ensemble import FastDetectGPTDetector
        
        detector = FastDetectGPTDetector()
        
        test_texts = [
            ("AI-like", "Machine learning algorithms have revolutionized data processing."),
            ("Human-like", "honestly idk why but my cat keeps staring at the wall lmao"),
        ]
        
        for label, text in test_texts:
            result = detector.detect(text)
            print(f"\n  [{label}]")
            print(f"    Text: {text[:40]}...")
            print(f"    Prediction: {result.prediction}")
            print(f"    Score: {result.score:.4f}")
            if 'raw_curvature' in result.details:
                print(f"    Curvature: {result.details['raw_curvature']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble():
    """Test full 3-component ensemble"""
    print("\n" + "=" * 60)
    print("TEST 7: Full Ensemble")
    print("=" * 60)
    
    try:
        from ensemble import EnsembleDetector, format_result
        
        ensemble = EnsembleDetector()
        
        test_texts = [
            ("AI-generated", "Climate change represents one of the most pressing challenges "
             "facing humanity today. The scientific consensus is clear that human activities "
             "have significantly contributed to global warming."),
            ("Human-like", "omg i cant believe what happened yesterday lol my cat knocked "
             "everything off the desk and then just sat there looking at me like nothing happened")
        ]
        
        for label, text in test_texts:
            print(f"\n  [{label}]")
            result = ensemble.detect(text)
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Agreement: {result.agreement}")
            print(f"  Suggested Action: {result.suggested_action}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """Test that API module loads correctly"""
    print("\n" + "=" * 60)
    print("TEST 8: API Module")
    print("=" * 60)
    
    try:
        from ensemble.api import app, DetectionRequest, DetectionResponse
        print(f"  ✓ FastAPI app loaded")
        print(f"  ✓ Routes: {[route.path for route in app.routes if hasattr(route, 'path')]}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AI TEXT DETECTOR - SELF-CONTAINED ENSEMBLE TESTS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    RESULTS['paths'] = test_paths()
    RESULTS['imports'] = test_imports()
    RESULTS['device'] = test_device()
    RESULTS['detective'] = test_detective()
    RESULTS['binoculars'] = test_binoculars()
    RESULTS['fast_detect'] = test_fast_detect()
    RESULTS['ensemble'] = test_ensemble()
    RESULTS['api'] = test_api()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in RESULTS.items():
        if result is True:
            status = "✓ PASSED"
            passed += 1
        else:
            status = "✗ FAILED"
            failed += 1
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    print(f"  Time: {elapsed:.1f}s")
    
    if failed == 0:
        print("\n  ✓ All tests passed! The ensemble is ready.")
    else:
        print(f"\n  ✗ {failed} test(s) failed. Check errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
