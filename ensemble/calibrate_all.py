"""
Comprehensive Calibration Suite for AI Text Detection Ensemble

This script:
1. Downloads multiple datasets (RAID, HC3, MAGE)
2. Runs extensive calibration for each detector
3. Finds optimal thresholds and parameters
4. Updates detector.py with calibrated values
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Sample:
    text: str
    label: str  # "AI" or "Human"
    source: str
    model: str = "unknown"


def load_hc3_samples(path: str, max_per_class: int = 500) -> List[Sample]:
    """Load HC3 dataset samples"""
    samples = []
    ai_count = human_count = 0
    
    if not os.path.exists(path):
        print(f"HC3 not found at {path}")
        return samples
    
    for filename in os.listdir(path):
        if not filename.endswith('.jsonl'):
            continue
        
        with open(os.path.join(path, filename), 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    for ans in data.get('human_answers', []):
                        if len(ans) >= 100 and human_count < max_per_class:
                            samples.append(Sample(
                                text=ans[:2000],
                                label="Human",
                                source="HC3",
                                model="human"
                            ))
                            human_count += 1
                    
                    for ans in data.get('chatgpt_answers', []):
                        if len(ans) >= 100 and ai_count < max_per_class:
                            samples.append(Sample(
                                text=ans[:2000],
                                label="AI",
                                source="HC3",
                                model="ChatGPT"
                            ))
                            ai_count += 1
                except:
                    continue
    
    print(f"HC3: {ai_count} AI, {human_count} Human")
    return samples


def load_raid_samples(max_per_class: int = 500) -> List[Sample]:
    """Load RAID dataset samples"""
    try:
        from datasets import load_dataset
        print("Loading RAID dataset...")
        
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
        
        samples = []
        ai_count = human_count = 0
        
        for item in tqdm(dataset, desc="Loading RAID", total=max_per_class * 3):
            if ai_count >= max_per_class and human_count >= max_per_class:
                break
            
            text = item.get('generation', item.get('text', ''))
            model = item.get('model', 'unknown')
            
            if len(text) < 100:
                continue
            
            is_human = model.lower() == 'human'
            
            if is_human and human_count < max_per_class:
                samples.append(Sample(
                    text=text[:2000],
                    label="Human",
                    source="RAID",
                    model="human"
                ))
                human_count += 1
            elif not is_human and ai_count < max_per_class:
                samples.append(Sample(
                    text=text[:2000],
                    label="AI",
                    source="RAID",
                    model=model
                ))
                ai_count += 1
        
        print(f"RAID: {ai_count} AI, {human_count} Human")
        return samples
        
    except Exception as e:
        print(f"Error loading RAID: {e}")
        return []


def calibrate_binoculars(samples: List[Sample], device: str = "cuda") -> Dict:
    """Calibrate Binoculars detector thresholds"""
    from ensemble.detector import BinocularsDetector
    
    print("\n" + "="*60)
    print("CALIBRATING BINOCULARS")
    print("="*60)
    
    # Load with default threshold (we'll find optimal)
    detector = BinocularsDetector(device=device, mode="accuracy")
    
    ai_scores = []
    human_scores = []
    
    print("Computing Binoculars scores...")
    for sample in tqdm(samples):
        try:
            score = detector.compute_score(sample.text)
            if sample.label == "AI":
                ai_scores.append(score)
            else:
                human_scores.append(score)
        except Exception as e:
            continue
    
    print(f"\nAI Scores: mean={np.mean(ai_scores):.4f}, std={np.std(ai_scores):.4f}")
    print(f"Human Scores: mean={np.mean(human_scores):.4f}, std={np.std(human_scores):.4f}")
    
    # Find optimal thresholds
    best_accuracy_thresh = 0.7
    best_accuracy = 0
    best_lowfpr_thresh = 0.6
    best_lowfpr_recall = 0
    
    for thresh in np.linspace(0.4, 1.2, 81):
        ai_correct = sum(1 for s in ai_scores if s < thresh)
        human_correct = sum(1 for s in human_scores if s >= thresh)
        
        accuracy = (ai_correct + human_correct) / (len(ai_scores) + len(human_scores))
        recall = ai_correct / len(ai_scores) if ai_scores else 0
        fpr = 1 - (human_correct / len(human_scores)) if human_scores else 1
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_thresh = thresh
        
        # For low-FPR: maximize recall while keeping FPR < 5%
        if fpr <= 0.05 and recall > best_lowfpr_recall:
            best_lowfpr_recall = recall
            best_lowfpr_thresh = thresh
    
    # If no threshold gives < 5% FPR, find the one with lowest FPR
    if best_lowfpr_recall == 0:
        best_fpr = 1.0
        for thresh in np.linspace(0.4, 1.2, 81):
            human_correct = sum(1 for s in human_scores if s >= thresh)
            fpr = 1 - (human_correct / len(human_scores)) if human_scores else 1
            ai_correct = sum(1 for s in ai_scores if s < thresh)
            recall = ai_correct / len(ai_scores) if ai_scores else 0
            
            if fpr < best_fpr and recall > 0.3:  # Require at least 30% recall
                best_fpr = fpr
                best_lowfpr_thresh = thresh
                best_lowfpr_recall = recall
    
    print(f"\nOptimal Accuracy Threshold: {best_accuracy_thresh:.4f} (acc={best_accuracy:.2%})")
    print(f"Optimal Low-FPR Threshold: {best_lowfpr_thresh:.4f} (recall={best_lowfpr_recall:.2%})")
    
    return {
        "accuracy_threshold": best_accuracy_thresh,
        "lowfpr_threshold": best_lowfpr_thresh,
        "ai_mean": np.mean(ai_scores),
        "ai_std": np.std(ai_scores),
        "human_mean": np.mean(human_scores),
        "human_std": np.std(human_scores),
        "best_accuracy": best_accuracy
    }


def calibrate_fastdetect(samples: List[Sample], device: str = "cuda") -> Dict:
    """Calibrate Fast-DetectGPT parameters using proper two-model approach
    
    Research shows best results with:
    - Sampling model (reference): Larger model for probability distribution
    - Scoring model: Smaller model for actual log probabilities
    
    We use GPT-2-medium (sampling) + GPT-2 (scoring) following research guidelines.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "="*60)
    print("CALIBRATING FAST-DETECTGPT (Two-Model Approach)")
    print("="*60)
    
    cache_dir = "/home/lightdesk/Projects/AI-Text/ensemble/cache"
    
    # Load scoring model (smaller)
    print("Loading scoring model (gpt2)...")
    scoring_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    scoring_model.eval()
    
    # Load sampling/reference model (larger)
    print("Loading sampling model (gpt2-medium)...")
    sampling_model = AutoModelForCausalLM.from_pretrained("gpt2-medium", cache_dir=cache_dir).to(device)
    sampling_model.eval()
    
    print("Using TWO-MODEL setup: gpt2-medium (sampling) + gpt2 (scoring)")
    
    def get_curvature(text: str) -> float:
        """Calculate curvature using two different models (research-validated approach)"""
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(device)
        
        if tokenized.input_ids.shape[1] < 10:
            return None
        
        labels = tokenized.input_ids[:, 1:]
        
        with torch.no_grad():
            # Get logits from SCORING model (gpt2)
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            # Get logits from SAMPLING/REFERENCE model (gpt2-medium)
            logits_ref = sampling_model(**tokenized).logits[:, :-1]
        
        # Handle vocabulary size mismatch if any
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]
        
        labels_exp = labels.unsqueeze(-1)
        
        # Use scoring model's log probs but reference model's probability distribution
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        
        log_likelihood = lprobs_score.gather(dim=-1, index=labels_exp).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        
        discrepancy = (
            (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) 
            / var_ref.sum(dim=-1).sqrt()
        )
        return discrepancy.mean().item()
    
    ai_curvatures = []
    human_curvatures = []
    
    print("Computing curvatures...")
    for sample in tqdm(samples):
        try:
            c = get_curvature(sample.text)
            if c is not None:
                if sample.label == "AI":
                    ai_curvatures.append(c)
                else:
                    human_curvatures.append(c)
        except:
            continue
    
    mu0 = np.mean(human_curvatures)
    sigma0 = np.std(human_curvatures)
    mu1 = np.mean(ai_curvatures)
    sigma1 = np.std(ai_curvatures)
    
    print(f"\nHuman: mu0={mu0:.4f}, sigma0={sigma0:.4f}")
    print(f"AI: mu1={mu1:.4f}, sigma1={sigma1:.4f}")
    
    # Test accuracy
    from scipy.stats import norm
    
    def compute_prob(x):
        pdf0 = norm.pdf(x, loc=mu0, scale=sigma0)
        pdf1 = norm.pdf(x, loc=mu1, scale=sigma1)
        return pdf1 / (pdf0 + pdf1 + 1e-10)
    
    ai_correct = sum(1 for c in ai_curvatures if compute_prob(c) > 0.5)
    human_correct = sum(1 for c in human_curvatures if compute_prob(c) <= 0.5)
    
    accuracy = (ai_correct + human_correct) / (len(ai_curvatures) + len(human_curvatures))
    fpr = 1 - (human_correct / len(human_curvatures))
    
    print(f"Calibrated Accuracy: {accuracy:.2%}, FPR: {fpr:.2%}")
    
    return {
        "mu0": mu0,
        "sigma0": sigma0,
        "mu1": mu1,
        "sigma1": sigma1,
        "accuracy": accuracy,
        "fpr": fpr
    }


def calibrate_ensemble_weights(samples: List[Sample], device: str = "cuda") -> Dict:
    """Find optimal ensemble weights"""
    from ensemble.detector import EnsembleDetector
    
    print("\n" + "="*60)
    print("CALIBRATING ENSEMBLE WEIGHTS")
    print("="*60)
    
    # Load ensemble
    ensemble = EnsembleDetector(device=device)
    
    # Collect individual predictions
    results = []
    
    print("Collecting predictions...")
    for sample in tqdm(samples[:500]):  # Limit for speed
        try:
            result = ensemble.detect(sample.text)
            results.append({
                "label": sample.label,
                "detective": result.breakdown["detective"].score,
                "binoculars": result.breakdown["binoculars"].score,
                "fast_detect": result.breakdown["fast_detect"].score
            })
        except:
            continue
    
    # Grid search for optimal weights
    best_weights = {"detective": 0.55, "binoculars": 0.30, "fast_detect": 0.15}
    best_score = 0
    
    print("Searching for optimal weights...")
    for d_weight in np.arange(0.3, 0.8, 0.05):
        for b_weight in np.arange(0.1, 0.5, 0.05):
            f_weight = 1.0 - d_weight - b_weight
            if f_weight < 0.05 or f_weight > 0.4:
                continue
            
            correct = 0
            fp = 0
            total_human = 0
            
            for r in results:
                score = (d_weight * r["detective"] + 
                        b_weight * r["binoculars"] + 
                        f_weight * r["fast_detect"])
                pred = "AI" if score > 0.5 else "Human"
                
                if pred == r["label"]:
                    correct += 1
                
                if r["label"] == "Human":
                    total_human += 1
                    if pred == "AI":
                        fp += 1
            
            accuracy = correct / len(results)
            fpr = fp / total_human if total_human > 0 else 0
            
            # Score: accuracy - 2*FPR (penalize false positives heavily)
            combined_score = accuracy - 2 * fpr
            
            if combined_score > best_score:
                best_score = combined_score
                best_weights = {
                    "detective": round(d_weight, 2),
                    "binoculars": round(b_weight, 2),
                    "fast_detect": round(f_weight, 2)
                }
    
    print(f"\nOptimal Weights: {best_weights}")
    print(f"Combined Score: {best_score:.4f}")
    
    return best_weights


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load all samples
    samples = []
    
    # HC3
    hc3_samples = load_hc3_samples("/home/lightdesk/Projects/AI-Text/HC3", max_per_class=300)
    samples.extend(hc3_samples)
    
    # RAID
    raid_samples = load_raid_samples(max_per_class=300)
    samples.extend(raid_samples)
    
    random.shuffle(samples)
    
    print(f"\nTotal samples: {len(samples)}")
    print(f"AI: {sum(1 for s in samples if s.label == 'AI')}")
    print(f"Human: {sum(1 for s in samples if s.label == 'Human')}")
    
    # Calibrate each component
    binoculars_params = calibrate_binoculars(samples, device)
    fastdetect_params = calibrate_fastdetect(samples, device)
    
    # Print final calibration results
    print("\n" + "="*60)
    print("FINAL CALIBRATION RESULTS")
    print("="*60)
    
    print("\n# Binoculars GPT-2/GPT-2-medium Thresholds:")
    print(f"# Observer: gpt2, Performer: gpt2-medium")
    print(f"BINOCULARS_GPT2_ACCURACY_THRESHOLD = {binoculars_params['accuracy_threshold']:.4f}")
    print(f"BINOCULARS_GPT2_FPR_THRESHOLD = {binoculars_params['lowfpr_threshold']:.4f}")
    
    print("\n# Fast-DetectGPT GPT-2-medium/GPT-2 Parameters (Two-Model):")
    print(f"# Sampling: gpt2-medium, Scoring: gpt2")
    print(f"'gpt2-medium_gpt2': {{")
    print(f"    'mu0': {fastdetect_params['mu0']:.4f},")
    print(f"    'sigma0': {fastdetect_params['sigma0']:.4f},")
    print(f"    'mu1': {fastdetect_params['mu1']:.4f},")
    print(f"    'sigma1': {fastdetect_params['sigma1']:.4f}")
    print(f"}}")
    
    return {
        "binoculars": binoculars_params,
        "fastdetect": fastdetect_params
    }


if __name__ == "__main__":
    results = main()
