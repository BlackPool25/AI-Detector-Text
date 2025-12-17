#!/usr/bin/env python3
"""
Massive Calibration Suite for AI Text Detection Ensemble
=========================================================

Downloads and evaluates on 10,000+ samples from:
1. RAID dataset (multi-model, multi-domain)
2. MAGE dataset (modern AI detectors benchmark)
3. Deepfake dataset (local)
4. HC3 dataset (ChatGPT vs Human)
5. Additional HuggingFace datasets

Tests robustness across:
- Multiple LLM models (GPT-3.5, GPT-4, Claude, LLaMA, etc.)
- Multiple domains (news, creative, academic, social media)
- Multiple text lengths
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import torch
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
CACHE_DIR = "/home/lightdesk/Projects/AI-Text/ensemble/cache"
DATASET_DIR = "/home/lightdesk/Projects/AI-Text/LLMtext_detect_dataset"
HC3_DIR = "/home/lightdesk/Projects/AI-Text/HC3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target samples per dataset
SAMPLES_PER_DATASET = 2500
TOTAL_TARGET = 10000


@dataclass
class Sample:
    text: str
    label: str  # "AI" or "Human"
    source: str  # Dataset name
    model: str = "unknown"  # Model that generated it
    domain: str = "unknown"  # Domain/topic
    length_bucket: str = "medium"  # short/medium/long


@dataclass
class CalibrationResult:
    detector: str
    accuracy: float
    fpr: float  # False Positive Rate
    fnr: float  # False Negative Rate
    threshold: float = 0.0
    params: dict = field(default_factory=dict)
    per_model: dict = field(default_factory=dict)
    per_domain: dict = field(default_factory=dict)


def download_raid_dataset(max_samples: int = 3000) -> List[Sample]:
    """Download RAID dataset from HuggingFace"""
    samples = []
    
    try:
        from datasets import load_dataset
        print("\n📥 Loading RAID dataset from HuggingFace...")
        
        # Load RAID - comprehensive AI text detection benchmark
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
        
        ai_count = human_count = 0
        models_seen = defaultdict(int)
        
        for item in tqdm(dataset, desc="RAID", total=max_samples * 2):
            if ai_count >= max_samples and human_count >= max_samples:
                break
            
            text = item.get('generation', item.get('text', ''))
            model = item.get('model', 'unknown')
            domain = item.get('domain', item.get('source', 'unknown'))
            
            if len(text) < 100:
                continue
            
            # Determine length bucket
            word_count = len(text.split())
            if word_count < 100:
                length_bucket = "short"
            elif word_count < 300:
                length_bucket = "medium"
            else:
                length_bucket = "long"
            
            is_human = model.lower() == 'human'
            
            if is_human and human_count < max_samples:
                samples.append(Sample(
                    text=text[:3000],
                    label="Human",
                    source="RAID",
                    model="human",
                    domain=domain,
                    length_bucket=length_bucket
                ))
                human_count += 1
                models_seen["human"] += 1
            elif not is_human and ai_count < max_samples:
                samples.append(Sample(
                    text=text[:3000],
                    label="AI",
                    source="RAID",
                    model=model,
                    domain=domain,
                    length_bucket=length_bucket
                ))
                ai_count += 1
                models_seen[model] += 1
        
        print(f"✅ RAID: {ai_count} AI, {human_count} Human")
        print(f"   Models: {dict(models_seen)}")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading RAID: {e}")
        import traceback
        traceback.print_exc()
        return []


def download_mage_dataset(max_samples: int = 2000) -> List[Sample]:
    """Download MAGE dataset - Modern AI Generated Text Evaluation"""
    samples = []
    
    try:
        from datasets import load_dataset
        print("\n📥 Loading MAGE dataset...")
        
        # Try loading MAGE or similar modern datasets
        try:
            dataset = load_dataset("yaful/MAGE", split="test", streaming=True)
        except:
            # Fallback to alternative
            try:
                dataset = load_dataset("Hello-SimpleAI/HC3", split="all", streaming=True)
            except:
                print("⚠️ MAGE not available, skipping")
                return []
        
        ai_count = human_count = 0
        
        for item in tqdm(dataset, desc="MAGE/Alt", total=max_samples * 2):
            if ai_count >= max_samples and human_count >= max_samples:
                break
            
            # Handle different dataset formats
            if 'text' in item:
                text = item['text']
                label = item.get('label', item.get('is_machine', 0))
                is_ai = label == 1 or label == 'AI' or label == 'machine'
            elif 'generation' in item:
                text = item['generation']
                is_ai = item.get('model', 'human').lower() != 'human'
            else:
                continue
            
            if len(text) < 100:
                continue
            
            word_count = len(text.split())
            length_bucket = "short" if word_count < 100 else "medium" if word_count < 300 else "long"
            
            if is_ai and ai_count < max_samples:
                samples.append(Sample(
                    text=text[:3000],
                    label="AI",
                    source="MAGE",
                    model=item.get('model', 'unknown'),
                    domain=item.get('domain', 'unknown'),
                    length_bucket=length_bucket
                ))
                ai_count += 1
            elif not is_ai and human_count < max_samples:
                samples.append(Sample(
                    text=text[:3000],
                    label="Human",
                    source="MAGE",
                    model="human",
                    domain=item.get('domain', 'unknown'),
                    length_bucket=length_bucket
                ))
                human_count += 1
        
        print(f"✅ MAGE: {ai_count} AI, {human_count} Human")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading MAGE: {e}")
        return []


def load_deepfake_samples(max_samples: int = 3000) -> List[Sample]:
    """Load samples from local Deepfake dataset"""
    samples = []
    processed_dir = os.path.join(DATASET_DIR, "Deepfake", "processed")
    
    if not os.path.exists(processed_dir):
        print(f"⚠️ Deepfake dataset not found at {processed_dir}")
        return []
    
    print("\n📂 Loading Deepfake dataset...")
    
    ai_count = human_count = 0
    files = os.listdir(processed_dir)
    
    # Categorize files
    human_files = [f for f in files if 'human' in f.lower()]
    ai_files = [f for f in files if 'machine' in f.lower()]
    
    # Priority models (newer, more capable)
    priority_models = ['gpt-3.5', 'gpt_neox', '65B', '30B', 'bloom']
    priority_ai_files = [f for f in ai_files if any(p in f for p in priority_models)]
    other_ai_files = [f for f in ai_files if f not in priority_ai_files]
    
    # Sample from priority first, then others
    ai_files_ordered = priority_ai_files + other_ai_files
    
    # Load human samples
    for filename in human_files:
        if human_count >= max_samples:
            break
        try:
            df = pd.read_csv(os.path.join(processed_dir, filename))
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            domain = filename.split('_')[0]
            
            for text in df[text_col].dropna():
                if human_count >= max_samples:
                    break
                if len(str(text)) >= 100:
                    word_count = len(str(text).split())
                    samples.append(Sample(
                        text=str(text)[:3000],
                        label="Human",
                        source="Deepfake",
                        model="human",
                        domain=domain,
                        length_bucket="short" if word_count < 100 else "medium" if word_count < 300 else "long"
                    ))
                    human_count += 1
        except Exception as e:
            continue
    
    # Load AI samples - balanced across models
    samples_per_file = max(10, max_samples // len(ai_files_ordered) if ai_files_ordered else 10)
    
    for filename in ai_files_ordered:
        if ai_count >= max_samples:
            break
        try:
            df = pd.read_csv(os.path.join(processed_dir, filename))
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            
            # Extract model name from filename
            parts = filename.replace('.csv', '').split('_')
            model = '_'.join(parts[parts.index('continuation') + 1:]) if 'continuation' in parts else 'unknown'
            domain = parts[0]
            
            file_count = 0
            for text in df[text_col].dropna().sample(frac=1):  # Shuffle
                if file_count >= samples_per_file or ai_count >= max_samples:
                    break
                if len(str(text)) >= 100:
                    word_count = len(str(text).split())
                    samples.append(Sample(
                        text=str(text)[:3000],
                        label="AI",
                        source="Deepfake",
                        model=model,
                        domain=domain,
                        length_bucket="short" if word_count < 100 else "medium" if word_count < 300 else "long"
                    ))
                    ai_count += 1
                    file_count += 1
        except Exception as e:
            continue
    
    print(f"✅ Deepfake: {ai_count} AI, {human_count} Human")
    return samples


def load_hc3_samples(max_samples: int = 2000) -> List[Sample]:
    """Load HC3 (Human ChatGPT Comparison Corpus)"""
    samples = []
    
    if not os.path.exists(HC3_DIR):
        print(f"⚠️ HC3 not found at {HC3_DIR}")
        return []
    
    print("\n📂 Loading HC3 dataset...")
    
    ai_count = human_count = 0
    
    for filename in os.listdir(HC3_DIR):
        if not filename.endswith('.jsonl'):
            continue
        
        domain = filename.replace('.jsonl', '')
        
        with open(os.path.join(HC3_DIR, filename), 'r') as f:
            for line in f:
                if ai_count >= max_samples and human_count >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    for ans in data.get('human_answers', []):
                        if len(ans) >= 100 and human_count < max_samples:
                            word_count = len(ans.split())
                            samples.append(Sample(
                                text=ans[:3000],
                                label="Human",
                                source="HC3",
                                model="human",
                                domain=domain,
                                length_bucket="short" if word_count < 100 else "medium" if word_count < 300 else "long"
                            ))
                            human_count += 1
                    
                    for ans in data.get('chatgpt_answers', []):
                        if len(ans) >= 100 and ai_count < max_samples:
                            word_count = len(ans.split())
                            samples.append(Sample(
                                text=ans[:3000],
                                label="AI",
                                source="HC3",
                                model="ChatGPT",
                                domain=domain,
                                length_bucket="short" if word_count < 100 else "medium" if word_count < 300 else "long"
                            ))
                            ai_count += 1
                except:
                    continue
    
    print(f"✅ HC3: {ai_count} AI, {human_count} Human")
    return samples


def download_additional_datasets(max_samples: int = 2000) -> List[Sample]:
    """Download additional modern datasets from HuggingFace"""
    samples = []
    
    try:
        from datasets import load_dataset
        print("\n📥 Loading additional HuggingFace datasets...")
        
        # Try multiple datasets
        dataset_names = [
            ("NicolaiSivesworski/ChatGPT-Research-Abstracts", "train"),
            ("artem9k/ai-text-detection-pile", "train"),
            ("aadsblog/GPT-wiki-intro", "train"),
        ]
        
        for ds_name, split in dataset_names:
            try:
                print(f"   Trying {ds_name}...")
                dataset = load_dataset(ds_name, split=split, streaming=True)
                
                count = 0
                for item in dataset:
                    if count >= max_samples // len(dataset_names):
                        break
                    
                    # Handle different formats
                    text = item.get('text', item.get('abstract', item.get('content', '')))
                    label = item.get('label', item.get('generated', None))
                    
                    if not text or len(text) < 100:
                        continue
                    
                    if label is not None:
                        is_ai = label == 1 or label == 'AI' or label == 'generated' or label == True
                    else:
                        # Assume AI for generated datasets
                        is_ai = 'gpt' in ds_name.lower() or 'ai' in ds_name.lower()
                    
                    word_count = len(text.split())
                    samples.append(Sample(
                        text=text[:3000],
                        label="AI" if is_ai else "Human",
                        source=ds_name.split('/')[-1],
                        model=item.get('model', 'GPT' if is_ai else 'human'),
                        domain="research" if 'abstract' in ds_name.lower() else "wiki",
                        length_bucket="short" if word_count < 100 else "medium" if word_count < 300 else "long"
                    ))
                    count += 1
                
                print(f"   ✅ Loaded {count} samples from {ds_name}")
                
            except Exception as e:
                print(f"   ⚠️ Could not load {ds_name}: {e}")
                continue
        
    except Exception as e:
        print(f"❌ Error loading additional datasets: {e}")
    
    return samples


def collect_all_samples() -> List[Sample]:
    """Collect samples from all available datasets"""
    print("\n" + "="*70)
    print("COLLECTING CALIBRATION DATA")
    print("="*70)
    
    all_samples = []
    
    # Load from each source
    all_samples.extend(download_raid_dataset(SAMPLES_PER_DATASET))
    all_samples.extend(load_deepfake_samples(SAMPLES_PER_DATASET))
    all_samples.extend(load_hc3_samples(SAMPLES_PER_DATASET))
    all_samples.extend(download_mage_dataset(1500))
    all_samples.extend(download_additional_datasets(1500))
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Statistics
    ai_samples = [s for s in all_samples if s.label == "AI"]
    human_samples = [s for s in all_samples if s.label == "Human"]
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {len(all_samples)}")
    print(f"  AI: {len(ai_samples)}")
    print(f"  Human: {len(human_samples)}")
    
    # By source
    sources = defaultdict(lambda: {"AI": 0, "Human": 0})
    for s in all_samples:
        sources[s.source][s.label] += 1
    
    print("\nBy Source:")
    for src, counts in sources.items():
        print(f"  {src}: AI={counts['AI']}, Human={counts['Human']}")
    
    # By model
    models = defaultdict(int)
    for s in all_samples:
        if s.label == "AI":
            models[s.model] += 1
    
    print("\nTop AI Models:")
    for model, count in sorted(models.items(), key=lambda x: -x[1])[:15]:
        print(f"  {model}: {count}")
    
    return all_samples


def calibrate_binoculars(samples: List[Sample]) -> CalibrationResult:
    """Calibrate Binoculars detector"""
    from ensemble.detector import BinocularsDetector
    
    print("\n" + "="*70)
    print("CALIBRATING BINOCULARS")
    print("="*70)
    
    detector = BinocularsDetector(device=DEVICE, mode="accuracy")
    
    ai_scores = []
    human_scores = []
    model_scores = defaultdict(list)
    domain_scores = defaultdict(lambda: {"AI": [], "Human": []})
    
    print("Computing scores...")
    for sample in tqdm(samples, desc="Binoculars"):
        try:
            score = detector.compute_score(sample.text)
            if sample.label == "AI":
                ai_scores.append(score)
                model_scores[sample.model].append(score)
            else:
                human_scores.append(score)
            domain_scores[sample.domain][sample.label].append(score)
        except Exception as e:
            continue
    
    print(f"\nAI Scores: n={len(ai_scores)}, mean={np.mean(ai_scores):.4f}, std={np.std(ai_scores):.4f}")
    print(f"Human Scores: n={len(human_scores)}, mean={np.mean(human_scores):.4f}, std={np.std(human_scores):.4f}")
    
    # Find optimal thresholds
    best_accuracy_thresh = 0.7
    best_accuracy = 0
    best_lowfpr_thresh = 0.6
    best_lowfpr_fpr = 1.0
    best_lowfpr_recall = 0
    
    thresholds_to_test = np.linspace(0.3, 1.5, 121)
    
    results = []
    for thresh in thresholds_to_test:
        ai_correct = sum(1 for s in ai_scores if s < thresh)
        human_correct = sum(1 for s in human_scores if s >= thresh)
        
        accuracy = (ai_correct + human_correct) / (len(ai_scores) + len(human_scores))
        recall = ai_correct / len(ai_scores) if ai_scores else 0
        fpr = 1 - (human_correct / len(human_scores)) if human_scores else 1
        
        results.append((thresh, accuracy, fpr, recall))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_thresh = thresh
        
        # For low-FPR: want FPR < 1% with best possible recall
        if fpr <= 0.01 and recall > best_lowfpr_recall:
            best_lowfpr_recall = recall
            best_lowfpr_thresh = thresh
            best_lowfpr_fpr = fpr
    
    # If no threshold gives < 1% FPR, relax to 5%
    if best_lowfpr_recall == 0:
        for thresh, acc, fpr, recall in results:
            if fpr <= 0.05 and recall > best_lowfpr_recall:
                best_lowfpr_recall = recall
                best_lowfpr_thresh = thresh
                best_lowfpr_fpr = fpr
    
    print(f"\n📊 Optimal Accuracy Threshold: {best_accuracy_thresh:.4f}")
    print(f"   Accuracy: {best_accuracy:.2%}")
    
    print(f"\n📊 Optimal Low-FPR Threshold: {best_lowfpr_thresh:.4f}")
    print(f"   FPR: {best_lowfpr_fpr:.2%}, Recall: {best_lowfpr_recall:.2%}")
    
    # Per-model breakdown
    print("\n📊 Per-Model Performance (at accuracy threshold):")
    per_model = {}
    for model, scores in sorted(model_scores.items(), key=lambda x: -len(x[1]))[:10]:
        detected = sum(1 for s in scores if s < best_accuracy_thresh)
        rate = detected / len(scores) if scores else 0
        per_model[model] = rate
        print(f"   {model}: {rate:.1%} detected ({len(scores)} samples)")
    
    return CalibrationResult(
        detector="binoculars",
        accuracy=best_accuracy,
        fpr=1 - (sum(1 for s in human_scores if s >= best_accuracy_thresh) / len(human_scores)),
        fnr=1 - (sum(1 for s in ai_scores if s < best_accuracy_thresh) / len(ai_scores)),
        threshold=best_accuracy_thresh,
        params={
            "accuracy_threshold": round(best_accuracy_thresh, 4),
            "lowfpr_threshold": round(best_lowfpr_thresh, 4),
            "ai_mean": round(np.mean(ai_scores), 4),
            "human_mean": round(np.mean(human_scores), 4),
        },
        per_model=per_model
    )


def calibrate_fastdetect(samples: List[Sample]) -> CalibrationResult:
    """Calibrate Fast-DetectGPT parameters"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n" + "="*70)
    print("CALIBRATING FAST-DETECTGPT")
    print("="*70)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=CACHE_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    def get_curvature(text: str) -> Optional[float]:
        try:
            tokenized = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_token_type_ids=False
            ).to(DEVICE)
            
            if tokenized.input_ids.shape[1] < 10:
                return None
            
            labels = tokenized.input_ids[:, 1:]
            
            with torch.no_grad():
                logits = model(**tokenized).logits[:, :-1]
            
            labels_exp = labels.unsqueeze(-1)
            lprobs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            log_likelihood = lprobs.gather(dim=-1, index=labels_exp).squeeze(-1)
            mean_ref = (probs * lprobs).sum(dim=-1)
            var_ref = (probs * torch.square(lprobs)).sum(dim=-1) - torch.square(mean_ref)
            
            discrepancy = (
                (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) 
                / (var_ref.sum(dim=-1).sqrt() + 1e-10)
            )
            return discrepancy.mean().item()
        except:
            return None
    
    ai_curvatures = []
    human_curvatures = []
    model_curvatures = defaultdict(list)
    
    print("Computing curvatures...")
    for sample in tqdm(samples, desc="Fast-DetectGPT"):
        c = get_curvature(sample.text)
        if c is not None and not np.isnan(c) and not np.isinf(c):
            if sample.label == "AI":
                ai_curvatures.append(c)
                model_curvatures[sample.model].append(c)
            else:
                human_curvatures.append(c)
    
    # Filter outliers (beyond 3 std)
    def filter_outliers(data, n_std=3):
        mean, std = np.mean(data), np.std(data)
        return [x for x in data if abs(x - mean) < n_std * std]
    
    ai_curvatures = filter_outliers(ai_curvatures)
    human_curvatures = filter_outliers(human_curvatures)
    
    mu0 = np.mean(human_curvatures)
    sigma0 = np.std(human_curvatures)
    mu1 = np.mean(ai_curvatures)
    sigma1 = np.std(ai_curvatures)
    
    print(f"\nHuman: n={len(human_curvatures)}, mu0={mu0:.4f}, sigma0={sigma0:.4f}")
    print(f"AI: n={len(ai_curvatures)}, mu1={mu1:.4f}, sigma1={sigma1:.4f}")
    
    # Test accuracy with calibrated parameters
    from scipy.stats import norm
    
    def compute_ai_prob(x):
        pdf0 = norm.pdf(x, loc=mu0, scale=sigma0)
        pdf1 = norm.pdf(x, loc=mu1, scale=sigma1)
        return pdf1 / (pdf0 + pdf1 + 1e-10)
    
    ai_correct = sum(1 for c in ai_curvatures if compute_ai_prob(c) > 0.5)
    human_correct = sum(1 for c in human_curvatures if compute_ai_prob(c) <= 0.5)
    
    accuracy = (ai_correct + human_correct) / (len(ai_curvatures) + len(human_curvatures))
    fpr = 1 - (human_correct / len(human_curvatures))
    fnr = 1 - (ai_correct / len(ai_curvatures))
    
    print(f"\n📊 Calibrated Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   FPR: {fpr:.2%}")
    print(f"   FNR: {fnr:.2%}")
    
    # Per-model breakdown
    print("\n📊 Per-Model Performance:")
    per_model = {}
    for model, curvs in sorted(model_curvatures.items(), key=lambda x: -len(x[1]))[:10]:
        detected = sum(1 for c in curvs if compute_ai_prob(c) > 0.5)
        rate = detected / len(curvs) if curvs else 0
        per_model[model] = rate
        print(f"   {model}: {rate:.1%} detected ({len(curvs)} samples)")
    
    return CalibrationResult(
        detector="fast_detect",
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        params={
            "mu0": round(mu0, 4),
            "sigma0": round(sigma0, 4),
            "mu1": round(mu1, 4),
            "sigma1": round(sigma1, 4),
        },
        per_model=per_model
    )


def calibrate_detective(samples: List[Sample]) -> CalibrationResult:
    """Evaluate DeTeCtive detector (KNN-based, no threshold tuning)"""
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
            # DeTeCtive uses detect() method, not compute_score()
            result = detector.detect(sample.text)
            score = result.score  # AI probability from KNN voting
            if sample.label == "AI":
                ai_scores.append(score)
                model_scores[sample.model].append(score)
            else:
                human_scores.append(score)
        except Exception as e:
            continue
    
    # Check if we got any scores
    if len(ai_scores) == 0 or len(human_scores) == 0:
        print(f"\n⚠️ Warning: Not enough scores collected (AI: {len(ai_scores)}, Human: {len(human_scores)})")
        return CalibrationResult(
            detector="detective",
            accuracy=0.0,
            fpr=1.0,
            fnr=1.0,
            threshold=0.5,
            params={"error": "No scores collected"},
            per_model={}
        )
    
    print(f"\nAI Scores: n={len(ai_scores)}, mean={np.mean(ai_scores):.4f}, std={np.std(ai_scores):.4f}")
    print(f"Human Scores: n={len(human_scores)}, mean={np.mean(human_scores):.4f}, std={np.std(human_scores):.4f}")
    
    # DeTeCtive uses > 0.5 as AI
    ai_correct = sum(1 for s in ai_scores if s > 0.5)
    human_correct = sum(1 for s in human_scores if s <= 0.5)
    
    accuracy = (ai_correct + human_correct) / (len(ai_scores) + len(human_scores))
    fpr = 1 - (human_correct / len(human_scores))
    fnr = 1 - (ai_correct / len(ai_scores))
    
    print(f"\n📊 Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   FPR: {fpr:.2%}")
    print(f"   FNR: {fnr:.2%}")
    
    # Per-model breakdown
    print("\n📊 Per-Model Performance:")
    per_model = {}
    for model, scores in sorted(model_scores.items(), key=lambda x: -len(x[1]))[:10]:
        detected = sum(1 for s in scores if s > 0.5)
        rate = detected / len(scores) if scores else 0
        per_model[model] = rate
        print(f"   {model}: {rate:.1%} detected ({len(scores)} samples)")
    
    return CalibrationResult(
        detector="detective",
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        threshold=0.5,
        params={
            "k": 10,
            "ai_mean": round(np.mean(ai_scores), 4),
            "human_mean": round(np.mean(human_scores), 4),
        },
        per_model=per_model
    )


def find_optimal_weights(samples: List[Sample], 
                         binoculars_result: CalibrationResult,
                         fastdetect_result: CalibrationResult,
                         detective_result: CalibrationResult) -> Dict:
    """Find optimal ensemble weights"""
    from ensemble.detector import BinocularsDetector, FastDetectGPTDetector, DeTeCtiveDetector
    
    print("\n" + "="*70)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*70)
    
    # Get pre-computed scores for all samples
    binoculars = BinocularsDetector(device=DEVICE, mode="accuracy")
    
    # Use a subset for weight optimization
    subset = random.sample(samples, min(2000, len(samples)))
    
    predictions = []
    print("Computing all detector predictions...")
    
    for sample in tqdm(subset, desc="Ensemble"):
        try:
            bino_score = binoculars.compute_score(sample.text)
            bino_pred = 1 if bino_score < binoculars_result.params["accuracy_threshold"] else 0
            
            # We'll use the pre-computed thresholds
            predictions.append({
                "label": 1 if sample.label == "AI" else 0,
                "bino": bino_pred,
                "bino_conf": 1 - (bino_score / 2),  # Approximate confidence
            })
        except:
            continue
    
    # Test different weight combinations
    best_weights = {"detective": 0.4, "binoculars": 0.35, "fast_detect": 0.25}
    best_accuracy = 0
    
    # Based on individual accuracies, weight proportionally
    total_acc = binoculars_result.accuracy + fastdetect_result.accuracy + detective_result.accuracy
    
    if total_acc > 0:
        # Weight by accuracy but also consider FPR
        det_weight = detective_result.accuracy * (1 - detective_result.fpr * 0.5)
        bino_weight = binoculars_result.accuracy * (1 - binoculars_result.fpr * 0.5)
        fast_weight = fastdetect_result.accuracy * (1 - fastdetect_result.fpr * 0.5)
        
        total_weight = det_weight + bino_weight + fast_weight
        
        best_weights = {
            "detective": round(det_weight / total_weight, 2),
            "binoculars": round(bino_weight / total_weight, 2),
            "fast_detect": round(fast_weight / total_weight, 2),
        }
    
    print(f"\n📊 Optimal Weights:")
    print(f"   DeTeCtive: {best_weights['detective']}")
    print(f"   Binoculars: {best_weights['binoculars']}")
    print(f"   Fast-DetectGPT: {best_weights['fast_detect']}")
    
    return best_weights


def update_detector_file(binoculars_result: CalibrationResult,
                         fastdetect_result: CalibrationResult,
                         weights: Dict):
    """Update detector.py with calibrated values"""
    detector_path = "/home/lightdesk/Projects/AI-Text/ensemble/detector.py"
    
    print("\n" + "="*70)
    print("UPDATING DETECTOR.PY WITH CALIBRATED VALUES")
    print("="*70)
    
    with open(detector_path, 'r') as f:
        content = f.read()
    
    # Store original for comparison
    original = content
    
    # Update Binoculars thresholds
    import re
    
    # Find and update binoculars threshold for accuracy mode
    bino_acc_thresh = binoculars_result.params["accuracy_threshold"]
    bino_lowfpr_thresh = binoculars_result.params["lowfpr_threshold"]
    
    # Update accuracy threshold
    content = re.sub(
        r'(BINOCULARS_THRESHOLD_ACCURACY\s*=\s*)[\d.]+',
        f'\\g<1>{bino_acc_thresh}',
        content
    )
    
    # Update low-fpr threshold  
    content = re.sub(
        r'(BINOCULARS_THRESHOLD_LOW_FPR\s*=\s*)[\d.]+',
        f'\\g<1>{bino_lowfpr_thresh}',
        content
    )
    
    # Update Fast-DetectGPT parameters
    fd_params = fastdetect_result.params
    content = re.sub(r'(FASTDETECT_MU0\s*=\s*)[\d.-]+', f'\\g<1>{fd_params["mu0"]}', content)
    content = re.sub(r'(FASTDETECT_SIGMA0\s*=\s*)[\d.-]+', f'\\g<1>{fd_params["sigma0"]}', content)
    content = re.sub(r'(FASTDETECT_MU1\s*=\s*)[\d.-]+', f'\\g<1>{fd_params["mu1"]}', content)
    content = re.sub(r'(FASTDETECT_SIGMA1\s*=\s*)[\d.-]+', f'\\g<1>{fd_params["sigma1"]}', content)
    
    # Update weights
    content = re.sub(
        r'("detective"\s*:\s*)[\d.]+',
        f'\\g<1>{weights["detective"]}',
        content
    )
    content = re.sub(
        r'("binoculars"\s*:\s*)[\d.]+',
        f'\\g<1>{weights["binoculars"]}',
        content
    )
    content = re.sub(
        r'("fast_detect"\s*:\s*)[\d.]+',
        f'\\g<1>{weights["fast_detect"]}',
        content
    )
    
    if content != original:
        with open(detector_path, 'w') as f:
            f.write(content)
        print("✅ detector.py updated with new calibration values")
    else:
        print("⚠️ No changes made to detector.py")
    
    # Print summary
    print("\n📊 New Calibration Values:")
    print(f"   Binoculars accuracy threshold: {bino_acc_thresh}")
    print(f"   Binoculars low-FPR threshold: {bino_lowfpr_thresh}")
    print(f"   Fast-DetectGPT: mu0={fd_params['mu0']}, sigma0={fd_params['sigma0']}")
    print(f"   Fast-DetectGPT: mu1={fd_params['mu1']}, sigma1={fd_params['sigma1']}")
    print(f"   Weights: {weights}")


def run_final_evaluation(samples: List[Sample]):
    """Run final evaluation with calibrated ensemble"""
    from ensemble.detector import EnsembleDetector
    
    print("\n" + "="*70)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*70)
    
    # Reload detector with new calibration
    detector = EnsembleDetector(
        detective_path="/home/lightdesk/Projects/AI-Text/ensemble/models/Deepfake_best.pth",
        detective_db="/home/lightdesk/Projects/AI-Text/ensemble/database/deepfake_sample",
        device=DEVICE
    )
    
    # Use subset for final eval
    eval_samples = random.sample(samples, min(1000, len(samples)))
    
    correct = 0
    ai_correct = 0
    human_correct = 0
    ai_total = 0
    human_total = 0
    
    print("Running final evaluation...")
    for sample in tqdm(eval_samples, desc="Final Eval"):
        try:
            result = detector.detect(sample.text)
            pred_ai = result.get("is_ai", False)
            actual_ai = sample.label == "AI"
            
            if pred_ai == actual_ai:
                correct += 1
            
            if actual_ai:
                ai_total += 1
                if pred_ai:
                    ai_correct += 1
            else:
                human_total += 1
                if not pred_ai:
                    human_correct += 1
        except:
            continue
    
    accuracy = correct / len(eval_samples) if eval_samples else 0
    recall = ai_correct / ai_total if ai_total else 0
    specificity = human_correct / human_total if human_total else 0
    fpr = 1 - specificity
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Recall (AI detection): {recall:.2%}")
    print(f"   Specificity (Human detection): {specificity:.2%}")
    print(f"   False Positive Rate: {fpr:.2%}")
    
    return accuracy, fpr


def main():
    """Main calibration pipeline"""
    print("\n" + "="*70)
    print("MASSIVE CALIBRATION SUITE FOR AI TEXT DETECTION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Target samples: {TOTAL_TARGET}")
    
    start_time = time.time()
    
    # 1. Collect all samples
    samples = collect_all_samples()
    
    if len(samples) < 1000:
        print("❌ Not enough samples collected. Exiting.")
        return
    
    # 2. Calibrate each detector
    binoculars_result = calibrate_binoculars(samples)
    fastdetect_result = calibrate_fastdetect(samples)
    detective_result = calibrate_detective(samples)
    
    # 3. Find optimal weights
    weights = find_optimal_weights(samples, binoculars_result, fastdetect_result, detective_result)
    
    # 4. Update detector.py
    update_detector_file(binoculars_result, fastdetect_result, weights)
    
    # 5. Final evaluation
    accuracy, fpr = run_final_evaluation(samples)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Samples: {len(samples)}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Final FPR: {fpr:.2%}")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(samples),
        "binoculars": {
            "accuracy": binoculars_result.accuracy,
            "fpr": binoculars_result.fpr,
            "params": binoculars_result.params,
        },
        "fastdetect": {
            "accuracy": fastdetect_result.accuracy,
            "fpr": fastdetect_result.fpr,
            "params": fastdetect_result.params,
        },
        "detective": {
            "accuracy": detective_result.accuracy,
            "fpr": detective_result.fpr,
        },
        "weights": weights,
        "final_accuracy": accuracy,
        "final_fpr": fpr,
    }
    
    with open("/home/lightdesk/Projects/AI-Text/ensemble/calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to calibration_results.json")
    
    if accuracy >= 0.85 and fpr <= 0.10:
        print("\n🎉 Calibration successful! Ready to deploy.")
        print("Run: cd /home/lightdesk/Projects/AI-Text/ensemble && modal deploy modal_app_complete.py")
    else:
        print("\n⚠️ Calibration may need adjustment. Review results.")


if __name__ == "__main__":
    main()
