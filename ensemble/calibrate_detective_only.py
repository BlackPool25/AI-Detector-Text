#!/usr/bin/env python3
"""
Quick calibration script for DeTeCtive only.
Uses pre-computed Binoculars and Fast-DetectGPT results from the massive calibration run.
"""

import os
import sys
import json
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Sample:
    text: str
    label: str  # "AI" or "Human"
    source: str  # dataset name
    model: str = "unknown"  # AI model name if AI-generated

def load_deepfake_samples(max_samples: int = 5000) -> List[Sample]:
    """Load samples from local Deepfake dataset"""
    samples = []
    base_path = "/home/lightdesk/Projects/AI-Text/LLMtext_detect_dataset/Deepfake"
    
    # Load AI samples
    ai_path = os.path.join(base_path, "chatgpt_polished")
    if os.path.exists(ai_path):
        for fname in os.listdir(ai_path)[:max_samples//2]:
            fpath = os.path.join(ai_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if len(text) > 50:
                        samples.append(Sample(text=text, label="AI", source="Deepfake", model="ChatGPT"))
            except:
                continue
    
    # Load human samples
    human_path = os.path.join(base_path, "human")
    if os.path.exists(human_path):
        for fname in os.listdir(human_path)[:max_samples//2]:
            fpath = os.path.join(human_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if len(text) > 50:
                        samples.append(Sample(text=text, label="Human", source="Deepfake", model="human"))
            except:
                continue
    
    return samples

def load_hc3_samples(max_samples: int = 5000) -> List[Sample]:
    """Load samples from local HC3 dataset"""
    samples = []
    hc3_path = "/home/lightdesk/Projects/AI-Text/HC3"
    
    for jsonl_file in ['open_qa.jsonl', 'finance.jsonl', 'medicine.jsonl', 'wiki_csai.jsonl', 'reddit_eli5.jsonl']:
        fpath = os.path.join(hc3_path, jsonl_file)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(samples) >= max_samples:
                            break
                        data = json.loads(line)
                        
                        # Add human answers
                        for ans in data.get('human_answers', [])[:1]:
                            if len(ans) > 50:
                                samples.append(Sample(text=ans, label="Human", source="HC3", model="human"))
                        
                        # Add ChatGPT answers
                        for ans in data.get('chatgpt_answers', [])[:1]:
                            if len(ans) > 50:
                                samples.append(Sample(text=ans, label="AI", source="HC3", model="ChatGPT"))
            except:
                continue
    
    return samples[:max_samples]

def calibrate_detective(samples: List[Sample]):
    """Evaluate DeTeCtive detector"""
    from ensemble.detector import DeTeCtiveDetector
    
    print("\n" + "="*70)
    print("EVALUATING DETECTIVE")
    print("="*70)
    
    model_path = "/home/lightdesk/Projects/AI-Text/ensemble/models/Deepfake_best.pth"
    db_path = "/home/lightdesk/Projects/AI-Text/ensemble/database/deepfake_sample"
    
    detector = DeTeCtiveDetector(
        model_path=model_path,
        database_path=db_path,
        device=DEVICE,
        k=10
    )
    
    ai_scores = []
    human_scores = []
    model_scores = defaultdict(list)
    
    print("Computing scores...")
    for sample in tqdm(samples, desc="DeTeCtive"):
        try:
            # DeTeCtive uses detect() method
            result = detector.detect(sample.text)
            score = result.score  # AI probability from KNN voting
            if sample.label == "AI":
                ai_scores.append(score)
                model_scores[sample.model].append(score)
            else:
                human_scores.append(score)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Check if we got any scores
    if len(ai_scores) == 0 or len(human_scores) == 0:
        print(f"\n⚠️ Warning: Not enough scores collected (AI: {len(ai_scores)}, Human: {len(human_scores)})")
        return None
    
    print(f"\nAI Scores: n={len(ai_scores)}, mean={np.mean(ai_scores):.4f}, std={np.std(ai_scores):.4f}")
    print(f"Human Scores: n={len(human_scores)}, mean={np.mean(human_scores):.4f}, std={np.std(human_scores):.4f}")
    
    # DeTeCtive uses > 0.5 as AI
    ai_correct = sum(1 for s in ai_scores if s > 0.5)
    human_correct = sum(1 for s in human_scores if s <= 0.5)
    
    accuracy = (ai_correct + human_correct) / (len(ai_scores) + len(human_scores))
    fpr = 1 - (human_correct / len(human_scores)) if human_scores else 1.0
    fnr = 1 - (ai_correct / len(ai_scores)) if ai_scores else 1.0
    
    print(f"\n📊 Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   FPR: {fpr:.2%}")
    print(f"   FNR: {fnr:.2%}")
    
    # Per-model breakdown
    print("\n📊 Per-Model Performance:")
    for model, scores in sorted(model_scores.items(), key=lambda x: -len(x[1]))[:10]:
        detected = sum(1 for s in scores if s > 0.5)
        rate = detected / len(scores) if scores else 0
        print(f"   {model}: {rate:.1%} detected ({len(scores)} samples)")
    
    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "ai_mean": float(np.mean(ai_scores)),
        "human_mean": float(np.mean(human_scores)),
    }


def main():
    print("="*70)
    print("DETECTIVE-ONLY CALIBRATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Load samples
    print("\n📂 Loading datasets...")
    
    deepfake_samples = load_deepfake_samples(2000)
    print(f"   Deepfake: {sum(1 for s in deepfake_samples if s.label=='AI')} AI, {sum(1 for s in deepfake_samples if s.label=='Human')} Human")
    
    hc3_samples = load_hc3_samples(2000)
    print(f"   HC3: {sum(1 for s in hc3_samples if s.label=='AI')} AI, {sum(1 for s in hc3_samples if s.label=='Human')} Human")
    
    all_samples = deepfake_samples + hc3_samples
    random.shuffle(all_samples)
    
    print(f"\nTotal: {len(all_samples)} samples")
    
    # Calibrate DeTeCtive
    result = calibrate_detective(all_samples)
    
    if result:
        print("\n" + "="*70)
        print("FINAL CALIBRATION SUMMARY")
        print("="*70)
        
        # Pre-computed values from previous run
        print("\n📊 BINOCULARS (from previous run):")
        print("   Optimal Accuracy Threshold: 0.7100")
        print("   Accuracy: 72.84%")
        print("   Low-FPR Threshold: 0.6100 (FPR: 0.83%)")
        
        print("\n📊 FAST-DETECTGPT (from previous run):")
        print("   mu0=0.6717, sigma0=1.3280 (Human)")
        print("   mu1=2.6677, sigma1=3.0683 (AI)")
        print("   Accuracy: 74.84%")
        
        print("\n📊 DETECTIVE:")
        print(f"   Accuracy: {result['accuracy']:.2%}")
        print(f"   FPR: {result['fpr']:.2%}")
        print(f"   FNR: {result['fnr']:.2%}")
        print(f"   AI mean score: {result['ai_mean']:.4f}")
        print(f"   Human mean score: {result['human_mean']:.4f}")
        
        print("\n" + "="*70)
        print("RECOMMENDED UPDATES TO detector.py")
        print("="*70)
        print("""
# Binoculars thresholds (CONFIRMED - close to current):
BINOCULARS_ACCURACY_THRESHOLD = 0.71  # Was 0.70
BINOCULARS_LOWFPR_THRESHOLD = 0.61    # Was 0.71 - SIGNIFICANT CHANGE

# Fast-DetectGPT parameters (UPDATE RECOMMENDED):
FASTDETECT_MU0 = 0.6717      # Was 0.6557
FASTDETECT_SIGMA0 = 1.3280   # Was 1.1536 - INCREASE
FASTDETECT_MU1 = 2.6677      # Was 4.7479 - DECREASE SIGNIFICANTLY  
FASTDETECT_SIGMA1 = 3.0683   # Was 1.5231 - INCREASE SIGNIFICANTLY

# DeTeCtive weights (current seems ok):
# Keep k=10 and threshold=0.5
        """)


if __name__ == "__main__":
    main()
