"""
Binoculars Threshold Calibration Script

The Binoculars paper used Falcon-7B/Falcon-7B-instruct models.
The current ensemble uses GPT-2/GPT-2-medium which have different score distributions.

This script analyzes raw scores to find appropriate thresholds.
"""

import os
import sys
import json
import random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_samples(hc3_path: str, n_per_class: int = 50):
    """Load balanced samples from HC3"""
    samples = {"AI": [], "Human": []}
    
    if not os.path.exists(hc3_path):
        print(f"HC3 not found at {hc3_path}")
        return samples
    
    jsonl_files = [f for f in os.listdir(hc3_path) if f.endswith('.jsonl')]
    
    for filename in jsonl_files:
        filepath = os.path.join(hc3_path, filename)
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    for ans in data.get('human_answers', []):
                        if len(ans) >= 100 and len(samples["Human"]) < n_per_class:
                            samples["Human"].append(ans[:2000])
                    
                    for ans in data.get('chatgpt_answers', []):
                        if len(ans) >= 100 and len(samples["AI"]) < n_per_class:
                            samples["AI"].append(ans[:2000])
                except:
                    continue
    
    return samples


def analyze_binoculars_scores():
    """Analyze raw Binoculars scores to find optimal thresholds"""
    
    from ensemble.detector import BinocularsDetector
    
    # Load detector
    print("Loading Binoculars detector...")
    detector = BinocularsDetector(mode="low-fpr")
    
    # Load samples
    samples = load_samples("/home/lightdesk/Projects/AI-Text/HC3", n_per_class=100)
    
    if not samples["AI"] or not samples["Human"]:
        print("No samples loaded!")
        return
    
    # Collect raw scores
    ai_scores = []
    human_scores = []
    
    print("\nScoring AI samples...")
    for text in tqdm(samples["AI"]):
        score = detector.compute_score(text)
        ai_scores.append(score)
    
    print("\nScoring Human samples...")
    for text in tqdm(samples["Human"]):
        score = detector.compute_score(text)
        human_scores.append(score)
    
    # Analyze distributions
    print("\n" + "="*60)
    print("BINOCULARS SCORE ANALYSIS")
    print("="*60)
    
    print(f"\nAI Scores (n={len(ai_scores)}):")
    print(f"  Mean:   {np.mean(ai_scores):.4f}")
    print(f"  Std:    {np.std(ai_scores):.4f}")
    print(f"  Min:    {np.min(ai_scores):.4f}")
    print(f"  Max:    {np.max(ai_scores):.4f}")
    print(f"  Median: {np.median(ai_scores):.4f}")
    
    print(f"\nHuman Scores (n={len(human_scores)}):")
    print(f"  Mean:   {np.mean(human_scores):.4f}")
    print(f"  Std:    {np.std(human_scores):.4f}")
    print(f"  Min:    {np.min(human_scores):.4f}")
    print(f"  Max:    {np.max(human_scores):.4f}")
    print(f"  Median: {np.median(human_scores):.4f}")
    
    # Current thresholds from Falcon-7B
    print(f"\nCurrent threshold (from Falcon-7B): {detector.threshold:.4f}")
    
    # Test different thresholds
    print("\n" + "-"*60)
    print("THRESHOLD ANALYSIS")
    print("(Score < threshold → AI, Score >= threshold → Human)")
    print("-"*60)
    
    thresholds = np.linspace(0.5, 1.5, 21)
    
    best_f1 = 0
    best_threshold = 0.9
    best_low_fpr_threshold = 0.9
    lowest_fpr = 1.0
    
    print(f"\n{'Threshold':>10} {'AI_Acc':>10} {'Hum_Acc':>10} {'Accuracy':>10} {'FPR':>10} {'F1':>10}")
    
    for thresh in thresholds:
        ai_correct = sum(1 for s in ai_scores if s < thresh)
        human_correct = sum(1 for s in human_scores if s >= thresh)
        
        ai_acc = ai_correct / len(ai_scores) if ai_scores else 0
        human_acc = human_correct / len(human_scores) if human_scores else 0
        
        accuracy = (ai_correct + human_correct) / (len(ai_scores) + len(human_scores))
        fpr = 1 - human_acc  # False Positive Rate = human classified as AI
        
        # Calculate F1
        tp = ai_correct
        fp = len(human_scores) - human_correct
        fn = len(ai_scores) - ai_correct
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{thresh:>10.4f} {ai_acc:>10.2%} {human_acc:>10.2%} {accuracy:>10.2%} {fpr:>10.2%} {f1:>10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
        
        if fpr < lowest_fpr and ai_acc > 0.5:  # Require at least 50% AI detection
            lowest_fpr = fpr
            best_low_fpr_threshold = thresh
    
    print(f"\n" + "="*60)
    print("RECOMMENDED THRESHOLDS FOR GPT-2 MODELS:")
    print("="*60)
    print(f"Best F1 threshold:      {best_threshold:.4f} (F1={best_f1:.4f})")
    print(f"Best Low-FPR threshold: {best_low_fpr_threshold:.4f} (FPR={lowest_fpr:.2%})")
    
    # Midpoint between AI and Human distributions
    midpoint = (np.mean(ai_scores) + np.mean(human_scores)) / 2
    print(f"Distribution midpoint:  {midpoint:.4f}")
    
    return {
        "best_f1_threshold": best_threshold,
        "best_low_fpr_threshold": best_low_fpr_threshold,
        "midpoint": midpoint,
        "ai_mean": np.mean(ai_scores),
        "human_mean": np.mean(human_scores)
    }


if __name__ == "__main__":
    results = analyze_binoculars_scores()
    
    if results:
        print("\n" + "="*60)
        print("SUGGESTED CODE CHANGES:")
        print("="*60)
        print(f"""
# For GPT-2/GPT-2-medium model pair:
BINOCULARS_GPT2_ACCURACY_THRESHOLD = {results['best_f1_threshold']:.4f}
BINOCULARS_GPT2_FPR_THRESHOLD = {results['best_low_fpr_threshold']:.4f}

# The original Falcon-7B thresholds don't work for GPT-2:
# - Falcon threshold: 0.8536 (low-fpr)
# - GPT-2 needs: ~{results['midpoint']:.4f} (based on distribution midpoint)
""")
