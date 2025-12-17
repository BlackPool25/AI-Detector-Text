#!/usr/bin/env python3
"""
Evaluate the ensemble detector on HC3 dataset to tune weights.

HC3 contains human and ChatGPT answers to questions.
This helps validate and optimize detector performance.
"""

import json
import os
import sys
import time
import random
from collections import defaultdict

# Add ensemble to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


def load_hc3_samples(file_path: str, n_samples: int = 50):
    """Load samples from HC3 dataset"""
    samples = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Get human answers
            for answer in data.get('human_answers', []):
                if len(answer) > 100:  # Skip very short answers
                    samples.append({
                        'text': answer,
                        'label': 'Human',
                        'source': data.get('source', 'unknown')
                    })
            
            # Get ChatGPT answers
            for answer in data.get('chatgpt_answers', []):
                if len(answer) > 100:
                    samples.append({
                        'text': answer,
                        'label': 'AI',
                        'source': data.get('source', 'unknown')
                    })
    
    # Randomly sample
    random.seed(42)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    
    return samples


def evaluate_detector(detector, samples, name: str = "Detector"):
    """Evaluate a single detector on samples"""
    results = {
        'correct': 0,
        'incorrect': 0,
        'ai_as_ai': 0,
        'human_as_human': 0,
        'ai_as_human': 0,
        'human_as_ai': 0,
        'predictions': []
    }
    
    print(f"\n{name} Evaluation:")
    print("-" * 40)
    
    for sample in samples:
        try:
            result = detector.detect(sample['text'])
            pred = result.prediction
            true = sample['label']
            
            # Get score (different attribute for EnsembleResult)
            if hasattr(result, 'ensemble_score'):
                score = result.ensemble_score
            else:
                score = result.score
            
            # Skip INSUFFICIENT_DATA
            if pred == "INSUFFICIENT_DATA" or pred == "UNCERTAIN":
                continue
            
            results['predictions'].append({
                'true': true,
                'pred': pred,
                'score': score,
                'confidence': result.confidence
            })
            
            if pred == true:
                results['correct'] += 1
                if true == 'AI':
                    results['ai_as_ai'] += 1
                else:
                    results['human_as_human'] += 1
            else:
                results['incorrect'] += 1
                if true == 'AI':
                    results['ai_as_human'] += 1
                else:
                    results['human_as_ai'] += 1
                    
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    total = results['correct'] + results['incorrect']
    if total > 0:
        accuracy = results['correct'] / total
        ai_recall = results['ai_as_ai'] / max(1, results['ai_as_ai'] + results['ai_as_human'])
        human_recall = results['human_as_human'] / max(1, results['human_as_human'] + results['human_as_ai'])
        
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  AI Recall: {ai_recall:.2%} (correctly detecting AI)")
        print(f"  Human Recall: {human_recall:.2%} (correctly detecting Human)")
        print(f"  False Positive Rate: {results['human_as_ai'] / max(1, results['human_as_human'] + results['human_as_ai']):.2%}")
        print(f"  Samples: {total}")
        
        results['accuracy'] = accuracy
        results['ai_recall'] = ai_recall
        results['human_recall'] = human_recall
    
    return results


def main():
    print("=" * 60)
    print("HC3 Dataset Evaluation")
    print("=" * 60)
    
    # Load samples
    hc3_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'HC3', 'all.jsonl')
    
    if not os.path.exists(hc3_path):
        print(f"Error: HC3 dataset not found at {hc3_path}")
        return
    
    print(f"Loading samples from {hc3_path}...")
    samples = load_hc3_samples(hc3_path, n_samples=100)
    
    ai_samples = [s for s in samples if s['label'] == 'AI']
    human_samples = [s for s in samples if s['label'] == 'Human']
    
    print(f"Loaded {len(samples)} samples:")
    print(f"  - AI (ChatGPT): {len(ai_samples)}")
    print(f"  - Human: {len(human_samples)}")
    
    # Test individual detectors first
    print("\n" + "=" * 60)
    print("Individual Detector Performance")
    print("=" * 60)
    
    from ensemble import DeTeCtiveDetector, BinocularsDetector, FastDetectGPTDetector
    
    # Test DeTeCtive
    detective = DeTeCtiveDetector()
    detective_results = evaluate_detector(detective, samples, "DeTeCtive")
    
    # Test Binoculars
    binoculars = BinocularsDetector(use_bfloat16=False)
    binoculars_results = evaluate_detector(binoculars, samples, "Binoculars")
    
    # Test Fast-DetectGPT
    fast_detect = FastDetectGPTDetector()
    fast_detect_results = evaluate_detector(fast_detect, samples, "Fast-DetectGPT")
    
    # Test Ensemble
    print("\n" + "=" * 60)
    print("Ensemble Performance")
    print("=" * 60)
    
    from ensemble import EnsembleDetector
    ensemble = EnsembleDetector()
    ensemble_results = evaluate_detector(ensemble, samples, "Ensemble (3-component)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"\n{'Detector':<20} {'Accuracy':<12} {'AI Recall':<12} {'Human Recall':<12}")
    print("-" * 56)
    
    for name, results in [
        ("DeTeCtive", detective_results),
        ("Binoculars", binoculars_results),
        ("Fast-DetectGPT", fast_detect_results),
        ("Ensemble", ensemble_results)
    ]:
        if 'accuracy' in results:
            print(f"{name:<20} {results['accuracy']:<12.2%} {results['ai_recall']:<12.2%} {results['human_recall']:<12.2%}")
    
    print("\nRecommendations:")
    # Find best individual detector
    individual_detectors = [
        ("DeTeCtive", detective_results),
        ("Binoculars", binoculars_results),
        ("Fast-DetectGPT", fast_detect_results)
    ]
    best = max(individual_detectors, key=lambda x: x[1].get('accuracy', 0))
    print(f"  - Best individual detector: {best[0]} ({best[1].get('accuracy', 0):.2%})")
    
    if 'accuracy' in ensemble_results:
        if ensemble_results['accuracy'] >= best[1].get('accuracy', 0):
            print("  - Ensemble improves upon individual detectors ✓")
        else:
            print("  - Consider adjusting ensemble weights")


if __name__ == "__main__":
    main()
