"""
Comprehensive Ensemble Test Suite

Tests the 3-component ensemble on multiple datasets:
1. RAID dataset (includes GPT-4)
2. HC3 dataset (existing)
3. Manual test cases

Evaluates:
- Individual detector accuracy
- Ensemble performance
- False positive rate
- Model-specific performance
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Add ensemble to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestSample:
    text: str
    label: str  # "AI" or "Human"
    model: str = "unknown"  # Which model generated it
    domain: str = "unknown"  # Domain/source


def load_raid_samples(n_samples: int = 500) -> List[TestSample]:
    """Load samples from RAID dataset"""
    try:
        from datasets import load_dataset
        print("Loading RAID dataset...")
        
        # Load a subset for testing
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
        
        samples = []
        ai_count = 0
        human_count = 0
        
        for item in tqdm(dataset, desc="Loading RAID samples", total=n_samples * 2):
            if len(samples) >= n_samples * 2:
                break
                
            # RAID has 'generation' field with text and 'model' field
            text = item.get('generation', item.get('text', ''))
            model = item.get('model', 'unknown')
            domain = item.get('domain', 'unknown')
            
            # Skip short texts
            if len(text) < 100:
                continue
            
            # Determine label (in RAID, human is 'human' model)
            is_human = model.lower() == 'human'
            label = "Human" if is_human else "AI"
            
            # Balance the dataset
            if label == "AI" and ai_count >= n_samples:
                continue
            if label == "Human" and human_count >= n_samples:
                continue
            
            samples.append(TestSample(
                text=text[:2000],  # Truncate for speed
                label=label,
                model=model,
                domain=domain
            ))
            
            if label == "AI":
                ai_count += 1
            else:
                human_count += 1
        
        print(f"Loaded {len(samples)} RAID samples ({ai_count} AI, {human_count} Human)")
        return samples
        
    except Exception as e:
        print(f"Error loading RAID: {e}")
        return []


def load_hc3_samples(hc3_path: str, n_samples: int = 200) -> List[TestSample]:
    """Load samples from HC3 dataset"""
    samples = []
    
    if not os.path.exists(hc3_path):
        print(f"HC3 path not found: {hc3_path}")
        return samples
    
    print("Loading HC3 dataset...")
    
    # Get all .jsonl files
    jsonl_files = [f for f in os.listdir(hc3_path) if f.endswith('.jsonl')]
    
    for filename in jsonl_files:
        filepath = os.path.join(hc3_path, filename)
        domain = filename.replace('.jsonl', '')
        
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # HC3 format: human_answers and chatgpt_answers
                    human_answers = data.get('human_answers', [])
                    chatgpt_answers = data.get('chatgpt_answers', [])
                    
                    for ans in human_answers:
                        if len(ans) >= 100 and len(samples) < n_samples:
                            samples.append(TestSample(
                                text=ans[:2000],
                                label="Human",
                                model="human",
                                domain=domain
                            ))
                    
                    for ans in chatgpt_answers:
                        if len(ans) >= 100 and len(samples) < n_samples * 2:
                            samples.append(TestSample(
                                text=ans[:2000],
                                label="AI",
                                model="ChatGPT",
                                domain=domain
                            ))
                            
                except json.JSONDecodeError:
                    continue
    
    random.shuffle(samples)
    samples = samples[:n_samples * 2]
    
    ai_count = sum(1 for s in samples if s.label == "AI")
    human_count = len(samples) - ai_count
    print(f"Loaded {len(samples)} HC3 samples ({ai_count} AI, {human_count} Human)")
    
    return samples


def get_manual_test_cases() -> List[TestSample]:
    """Get curated manual test cases"""
    return [
        # Clear AI samples
        TestSample(
            text="Artificial intelligence has revolutionized various industries by providing innovative solutions to complex problems. The implementation of machine learning algorithms has enabled unprecedented levels of automation and efficiency across multiple sectors, from healthcare to finance. These technological advancements continue to reshape our understanding of what is possible in the digital age.",
            label="AI",
            model="typical_llm",
            domain="manual"
        ),
        TestSample(
            text="The integration of neural networks into modern computing infrastructure represents a paradigm shift in how we approach data processing and analysis. Deep learning models have demonstrated remarkable capabilities in pattern recognition, natural language understanding, and predictive analytics. This transformation is fundamentally altering the landscape of technological innovation.",
            label="AI",
            model="typical_llm",
            domain="manual"
        ),
        # Clear human samples
        TestSample(
            text="honestly idk why everyone's making such a big deal about this whole AI thing lol... like yeah it's cool but my grandma still can't figure out how to use her iPhone so maybe let's calm down a bit? just saying 🤷",
            label="Human",
            model="human",
            domain="manual"
        ),
        TestSample(
            text="Was walking home yesterday when I saw the weirdest thing - someone had put a tiny hat on every single parking meter on Main St. Took me a solid 5 mins to stop laughing. Never change, Portland. Never change.",
            label="Human",
            model="human",
            domain="manual"
        ),
        # Mixed/borderline samples
        TestSample(
            text="I think the new iPhone is pretty good. The camera is definitely better than last year's model, and battery life seems improved. Not sure if it's worth the upgrade though if you have last year's phone. The new colors are nice but that's about it for noticeable differences day to day.",
            label="Human",
            model="human",
            domain="manual"
        ),
        TestSample(
            text="Machine learning models require careful consideration of several key factors during development. First, one must ensure adequate data quality and quantity. Second, appropriate model architecture selection is crucial. Third, hyperparameter tuning can significantly impact performance. Finally, proper evaluation metrics should be established before training begins.",
            label="AI",
            model="typical_llm",
            domain="manual"
        ),
    ]


def calculate_metrics(predictions: List[str], labels: List[str]) -> Dict:
    """Calculate accuracy, precision, recall, F1, and FPR"""
    tp = sum(1 for p, l in zip(predictions, labels) if p == "AI" and l == "AI")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "Human" and l == "Human")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "AI" and l == "Human")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "Human" and l == "AI")
    
    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total
    }


def test_ensemble(samples: List[TestSample], ensemble, verbose: bool = False) -> Dict:
    """Test ensemble on samples and return metrics"""
    
    # Collect predictions
    detective_preds = []
    binoculars_preds = []
    fastdetect_preds = []
    ensemble_preds = []
    labels = []
    
    # Track by model
    by_model = defaultdict(lambda: {"preds": [], "labels": []})
    
    print(f"\nTesting on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            result = ensemble.detect(sample.text)
            
            labels.append(sample.label)
            ensemble_preds.append(result.prediction if result.prediction in ["AI", "Human"] else "Human")
            
            if "detective" in result.breakdown:
                detective_preds.append(result.breakdown["detective"].prediction)
            if "binoculars" in result.breakdown:
                binoculars_preds.append(result.breakdown["binoculars"].prediction)
            if "fast_detect" in result.breakdown:
                fastdetect_preds.append(result.breakdown["fast_detect"].prediction)
            
            # Track by source model
            by_model[sample.model]["preds"].append(result.prediction)
            by_model[sample.model]["labels"].append(sample.label)
            
            if verbose:
                match = "✓" if result.prediction == sample.label else "✗"
                print(f"{match} Label:{sample.label} Pred:{result.prediction} ({sample.model})")
                
        except Exception as e:
            print(f"Error on sample: {e}")
            labels.append(sample.label)
            ensemble_preds.append("Human")
            detective_preds.append("Human")
            binoculars_preds.append("Human")
            fastdetect_preds.append("Human")
    
    # Calculate metrics
    results = {
        "ensemble": calculate_metrics(ensemble_preds, labels),
        "detective": calculate_metrics(detective_preds, labels) if detective_preds else {},
        "binoculars": calculate_metrics(binoculars_preds, labels) if binoculars_preds else {},
        "fast_detect": calculate_metrics(fastdetect_preds, labels) if fastdetect_preds else {},
    }
    
    # Metrics by model
    results["by_model"] = {}
    for model, data in by_model.items():
        if data["preds"]:
            results["by_model"][model] = calculate_metrics(data["preds"], data["labels"])
    
    return results


def print_results(results: Dict, dataset_name: str):
    """Pretty print results"""
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name}")
    print('='*60)
    
    for component in ["ensemble", "detective", "binoculars", "fast_detect"]:
        if component in results and results[component]:
            m = results[component]
            print(f"\n{component.upper()}:")
            print(f"  Accuracy:  {m['accuracy']:.2%}")
            print(f"  Precision: {m['precision']:.2%}")
            print(f"  Recall:    {m['recall']:.2%} (AI detection rate)")
            print(f"  F1:        {m['f1']:.2%}")
            print(f"  FPR:       {m['fpr']:.2%} (Human misclassified as AI)")
            print(f"  TP/TN/FP/FN: {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")
    
    if "by_model" in results and results["by_model"]:
        print(f"\n{'-'*40}")
        print("BY SOURCE MODEL:")
        for model, m in results["by_model"].items():
            print(f"  {model}: Acc={m['accuracy']:.2%}, FPR={m['fpr']:.2%} (n={m['total']})")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Ensemble Testing")
    parser.add_argument("--raid-samples", type=int, default=100, help="Number of RAID samples per class")
    parser.add_argument("--hc3-path", type=str, default="/home/lightdesk/Projects/AI-Text/HC3", help="Path to HC3 dataset")
    parser.add_argument("--hc3-samples", type=int, default=100, help="Number of HC3 samples per class")
    parser.add_argument("--skip-raid", action="store_true", help="Skip RAID dataset")
    parser.add_argument("--skip-hc3", action="store_true", help="Skip HC3 dataset")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample results")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()
    
    # Load ensemble
    print("Loading ensemble detector...")
    from ensemble.detector import EnsembleDetector
    
    ensemble = EnsembleDetector(device=args.device)
    
    all_results = {}
    
    # Test on manual samples
    print("\n" + "="*60)
    print("TESTING MANUAL SAMPLES")
    print("="*60)
    manual_samples = get_manual_test_cases()
    manual_results = test_ensemble(manual_samples, ensemble, verbose=True)
    print_results(manual_results, "Manual Test Cases")
    all_results["manual"] = manual_results
    
    # Test on HC3
    if not args.skip_hc3:
        print("\n" + "="*60)
        print("TESTING HC3 DATASET")
        print("="*60)
        hc3_samples = load_hc3_samples(args.hc3_path, args.hc3_samples)
        if hc3_samples:
            hc3_results = test_ensemble(hc3_samples, ensemble, verbose=args.verbose)
            print_results(hc3_results, "HC3 Dataset")
            all_results["hc3"] = hc3_results
    
    # Test on RAID
    if not args.skip_raid:
        print("\n" + "="*60)
        print("TESTING RAID DATASET")
        print("="*60)
        raid_samples = load_raid_samples(args.raid_samples)
        if raid_samples:
            raid_results = test_ensemble(raid_samples, ensemble, verbose=args.verbose)
            print_results(raid_results, "RAID Dataset")
            all_results["raid"] = raid_results
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    for dataset, results in all_results.items():
        if "ensemble" in results:
            m = results["ensemble"]
            print(f"\n{dataset.upper()}:")
            print(f"  Ensemble Accuracy: {m['accuracy']:.2%}, FPR: {m['fpr']:.2%}")
    
    # Recommendations
    print("\n" + "-"*60)
    print("WEIGHT ADJUSTMENT RECOMMENDATIONS:")
    print("-"*60)
    
    avg_fpr = {}
    avg_recall = {}
    for component in ["detective", "binoculars", "fast_detect"]:
        fprs = [r[component]["fpr"] for r in all_results.values() if component in r and r[component]]
        recalls = [r[component]["recall"] for r in all_results.values() if component in r and r[component]]
        if fprs:
            avg_fpr[component] = np.mean(fprs)
            avg_recall[component] = np.mean(recalls)
            print(f"{component}: Avg FPR={avg_fpr[component]:.2%}, Avg Recall={avg_recall[component]:.2%}")
    
    print("\nSuggested weights (lower FPR = higher weight):")
    if avg_fpr:
        # Weight inversely proportional to FPR
        total_inv_fpr = sum(1/(fpr + 0.01) for fpr in avg_fpr.values())
        for comp, fpr in avg_fpr.items():
            suggested_weight = (1/(fpr + 0.01)) / total_inv_fpr
            print(f"  {comp}: {suggested_weight:.3f}")
    
    return all_results


if __name__ == "__main__":
    main()
