#!/usr/bin/env python3
"""
Comprehensive evaluation of the AI Text Detection Ensemble.

Tests on multiple datasets:
- HC3 (ChatGPT vs Human)
- Deepfake (various LLMs from the Deepfake dataset)

This helps ensure the detector is robust across different AI models and domains.

Metrics computed (matching DeTeCtive research paper):
- Accuracy, Precision, Recall, F1
- Human Recall, Machine Recall, Average Recall
- Confusion Matrix
- Per-model breakdown
"""

import json
import os
import sys
import csv
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


def load_hc3_samples(n_per_class: int = 25) -> List[Dict]:
    """Load balanced samples from HC3 dataset (ChatGPT vs Human)"""
    hc3_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'HC3', 'all.jsonl')
    
    if not os.path.exists(hc3_path):
        print(f"HC3 not found at {hc3_path}")
        return []
    
    human_samples = []
    ai_samples = []
    
    with open(hc3_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            for answer in data.get('human_answers', []):
                if len(answer) > 100:
                    human_samples.append({
                        'text': answer,
                        'label': 'Human',
                        'source': 'HC3',
                        'model': 'human'
                    })
            
            for answer in data.get('chatgpt_answers', []):
                if len(answer) > 100:
                    ai_samples.append({
                        'text': answer,
                        'label': 'AI',
                        'source': 'HC3',
                        'model': 'chatgpt'
                    })
    
    random.seed(42)
    human_samples = random.sample(human_samples, min(n_per_class, len(human_samples)))
    ai_samples = random.sample(ai_samples, min(n_per_class, len(ai_samples)))
    
    return human_samples + ai_samples


def load_deepfake_samples(n_per_model: int = 10) -> List[Dict]:
    """Load samples from Deepfake dataset (various LLMs)"""
    base_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'LLMtext_detect_dataset', 'Deepfake', 'processed')
    
    if not os.path.exists(base_path):
        print(f"Deepfake dataset not found at {base_path}")
        return []
    
    samples = []
    
    def get_text(row):
        """Extract text handling BOM-prefixed column names"""
        for key in row.keys():
            if 'text' in key.lower():
                return row[key]
        return row.get('text', row.get('content', ''))
    
    # Actual domains in the dataset
    domains = ['cmv', 'eli5', 'wp', 'xsum', 'tldr', 'roct']
    
    # Load human samples from multiple domains
    for domain in domains:
        human_file = os.path.join(base_path, f'{domain}_human.csv')
        if os.path.exists(human_file):
            try:
                with open(human_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
                    reader = csv.DictReader(f)
                    count = 0
                    for row in reader:
                        if count >= n_per_model // len(domains) + 1:
                            break
                        text = get_text(row)
                        if len(text) > 100:
                            samples.append({
                                'text': text[:2000],  # Truncate very long texts
                                'label': 'Human',
                                'source': f'Deepfake/{domain}',
                                'model': 'human'
                            })
                            count += 1
            except Exception as e:
                print(f"Error loading {human_file}: {e}")
    
    # Load AI samples from different models
    ai_models = [
        'gpt-3.5-trubo',
        'text-davinci-003',
        'text-davinci-002', 
        'bloom_7b',
        'opt_13b',
        'flan_t5_xxl',
        'gpt_j',
        'gpt_neox'
    ]
    
    for model in ai_models:
        for domain in ['cmv', 'eli5', 'wp']:
            for gen_type in ['continuation', 'topical', 'specified']:
                ai_file = os.path.join(base_path, f'{domain}_machine_{gen_type}_{model}.csv')
                if os.path.exists(ai_file):
                    try:
                        with open(ai_file, 'r', encoding='utf-8-sig') as f:
                            reader = csv.DictReader(f)
                            count = 0
                            for row in reader:
                                if count >= max(1, n_per_model // (len(ai_models) * 2)):
                                    break
                                text = get_text(row)
                                if len(text) > 100:
                                    samples.append({
                                        'text': text[:2000],
                                        'label': 'AI',
                                        'source': f'Deepfake/{domain}/{gen_type}',
                                        'model': model
                                    })
                                    count += 1
                    except Exception as e:
                        pass  # Skip files with issues
    
    random.seed(42)
    random.shuffle(samples)
    return samples


def evaluate_on_samples(detector, samples: List[Dict], name: str = "Detector") -> Dict:
    """Evaluate detector on samples and return comprehensive metrics"""
    # Collect predictions
    y_true = []  # 1 for AI, 0 for Human
    y_pred = []  # 1 for AI, 0 for Human
    confidences = []
    by_model = defaultdict(lambda: {'correct': 0, 'total': 0, 'y_true': [], 'y_pred': []})
    skipped = 0
    
    print(f"\n{name}:")
    print("-" * 50)
    
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(samples)}...", end="\r")
        
        try:
            result = detector.detect(sample['text'])
            pred = result.prediction
            true_label = sample['label']
            model = sample['model']
            
            if pred in ("INSUFFICIENT_DATA", "UNCERTAIN"):
                skipped += 1
                continue
            
            # Convert to binary: AI=1, Human=0
            true_bin = 1 if true_label == 'AI' else 0
            pred_bin = 1 if pred == 'AI' else 0
            
            y_true.append(true_bin)
            y_pred.append(pred_bin)
            confidences.append(result.confidence)
            
            by_model[model]['y_true'].append(true_bin)
            by_model[model]['y_pred'].append(pred_bin)
            by_model[model]['total'] += 1
            if true_bin == pred_bin:
                by_model[model]['correct'] += 1
                    
        except Exception as e:
            skipped += 1
            print(f"\n  Error on sample {i}: {e}")
    
    print(f"  Completed processing {len(samples)} samples.      ")
    
    # Compute comprehensive metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    results = compute_all_metrics(y_true, y_pred, name)
    results['skipped'] = skipped
    results['by_model'] = dict(by_model)
    results['confidences'] = confidences
    
    return results


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> Dict:
    """Compute all metrics matching DeTeCtive research paper methodology"""
    
    # Confusion matrix: TN, FP, FN, TP
    # True labels: AI=1 (positive), Human=0 (negative)
    tp = np.sum((y_true == 1) & (y_pred == 1))  # AI correctly detected as AI
    tn = np.sum((y_true == 0) & (y_pred == 0))  # Human correctly detected as Human
    fp = np.sum((y_true == 0) & (y_pred == 1))  # Human incorrectly detected as AI (false positive)
    fn = np.sum((y_true == 1) & (y_pred == 0))  # AI incorrectly detected as Human (miss)
    
    total = len(y_true)
    
    # Core metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Of predicted AI, how many are actually AI
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Of actual AI, how many detected (Machine Recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # DeTeCtive-style recalls
    machine_rec = tp / (tp + fn) if (tp + fn) > 0 else 0  # AI detected correctly
    human_rec = tn / (tn + fp) if (tn + fp) > 0 else 0    # Human detected correctly  
    avg_rec = (human_rec + machine_rec) / 2
    
    # False positive rate (Human classified as AI)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'human_recall': human_rec,
        'machine_recall': machine_rec,
        'avg_recall': avg_rec,
        'fpr': fpr,
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn), 
            'FP': int(fp), 'FN': int(fn)
        },
        'total_evaluated': total
    }
    
    # Print metrics
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Human    AI")
    print(f"  Actual Human      {tn:5d}   {fp:5d}")
    print(f"  Actual AI         {fn:5d}   {tp:5d}")
    print()
    print(f"  Accuracy:         {accuracy:.4f} ({tp+tn}/{total})")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall (AI):      {recall:.4f}")
    print(f"  F1 Score:         {f1:.4f}")
    print(f"  Human Recall:     {human_rec:.4f}")
    print(f"  Machine Recall:   {machine_rec:.4f}")
    print(f"  Avg Recall:       {avg_rec:.4f}")
    print(f"  False Pos Rate:   {fpr:.4f}")
    
    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE AI TEXT DETECTOR EVALUATION")
    print("=" * 70)
    
    # Load samples from multiple sources
    print("\nLoading datasets...")
    
    hc3_samples = load_hc3_samples(n_per_class=50)  # 100 total
    print(f"  HC3: {len(hc3_samples)} samples")
    
    deepfake_samples = load_deepfake_samples(n_per_model=50)  
    print(f"  Deepfake: {len(deepfake_samples)} samples")
    
    all_samples = hc3_samples + deepfake_samples
    random.seed(42)
    random.shuffle(all_samples)
    
    ai_count = sum(1 for s in all_samples if s['label'] == 'AI')
    human_count = len(all_samples) - ai_count
    print(f"\nTotal: {len(all_samples)} samples ({ai_count} AI, {human_count} Human)")
    
    # Show sample distribution by model
    model_counts = defaultdict(int)
    for s in all_samples:
        model_counts[s['model']] += 1
    print("\nDistribution by source model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
    
    # Import detectors
    from ensemble import DeTeCtiveDetector, BinocularsDetector, FastDetectGPTDetector, EnsembleDetector
    
    # Test each detector
    print("\n" + "=" * 70)
    print("INDIVIDUAL DETECTOR EVALUATION")
    print("=" * 70)
    
    print("\n[1] DeTeCtive (RoBERTa + KNN):")
    detective = DeTeCtiveDetector()
    detective_results = evaluate_on_samples(detective, all_samples, "DeTeCtive")
    
    print("\n[2] Binoculars (GPT-2 cross-model):")
    binoculars = BinocularsDetector(use_bfloat16=False)
    binoculars_results = evaluate_on_samples(binoculars, all_samples, "Binoculars")
    
    print("\n[3] Fast-DetectGPT (probability curvature):")
    fast_detect = FastDetectGPTDetector()
    fast_detect_results = evaluate_on_samples(fast_detect, all_samples, "Fast-DetectGPT")
    
    # Test ensemble
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)
    
    print("\n[4] Full Ensemble (3 components):")
    ensemble = EnsembleDetector()
    ensemble_results = evaluate_on_samples(ensemble, all_samples, "Ensemble")
    
    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    all_results = [
        ("DeTeCtive", detective_results),
        ("Binoculars", binoculars_results),
        ("Fast-DetectGPT", fast_detect_results),
        ("Ensemble", ensemble_results)
    ]
    
    print(f"\n{'Detector':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'H-Rec':<8} {'M-Rec':<8} {'FPR':<8}")
    print("-" * 74)
    
    for name, r in all_results:
        if 'accuracy' in r:
            print(f"{name:<18} {r['accuracy']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} "
                  f"{r['f1']:<8.4f} {r['human_recall']:<8.4f} {r['machine_recall']:<8.4f} {r['fpr']:<8.4f}")
    
    # Find best by different metrics
    print("\n" + "-" * 50)
    print("Best by metric:")
    
    best_acc = max(all_results, key=lambda x: x[1].get('accuracy', 0))
    best_f1 = max(all_results, key=lambda x: x[1].get('f1', 0))
    best_human = max(all_results, key=lambda x: x[1].get('human_recall', 0))
    best_machine = max(all_results, key=lambda x: x[1].get('machine_recall', 0))
    
    print(f"  Best Accuracy:      {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
    print(f"  Best F1:            {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    print(f"  Best Human Recall:  {best_human[0]} ({best_human[1]['human_recall']:.4f})")
    print(f"  Best Machine Recall:{best_machine[0]} ({best_machine[1]['machine_recall']:.4f})")
    
    # Performance by model for ensemble
    print("\n" + "-" * 50)
    print("Ensemble performance by AI model:")
    for model, stats in sorted(ensemble_results.get('by_model', {}).items()):
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"  {model:<25}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # Confusion matrices summary
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES SUMMARY")
    print("=" * 70)
    for name, r in all_results:
        if 'confusion_matrix' in r:
            cm = r['confusion_matrix']
            print(f"\n{name}:")
            print(f"  TP={cm['TP']:4d}  FN={cm['FN']:4d}  (AI samples)")
            print(f"  FP={cm['FP']:4d}  TN={cm['TN']:4d}  (Human samples)")


if __name__ == "__main__":
    main()
