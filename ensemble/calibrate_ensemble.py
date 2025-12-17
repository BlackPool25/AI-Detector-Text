#!/usr/bin/env python3
"""
AI Detection Calibration Framework
Tests and calibrates the ensemble detector on diverse AI/Human samples.
"""

import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
import sys

# Add ensemble to path
sys.path.insert(0, str(Path(__file__).parent))

# Sample AI-generated texts from latest models (manually collected examples)
# These represent the style of latest LLMs with "undetectable" prompts
LATEST_AI_SAMPLES = [
    # Claude-style "undetectable" essay
    {
        "text": """In the grand tapestry of human existence, lifelong learning is a pillar of enlightenment and a pathway to a meaningful and good life. Through the pursuit of knowledge, we embark on a transformative journey, expanding our horizons and nourishing our souls. Lifelong learning is not confined to the classroom or formal education; it is a mindset, a relentless thirst for understanding that permeates every aspect of our lives.

By embracing lifelong learning, we open ourselves to many opportunities for growth and self-improvement. It instills a sense of curiosity, enabling us to explore new ideas, perspectives, and cultures. It empowers us to challenge the status quo, to question deeply held beliefs, and to evolve as individuals. Lifelong learning fuels our creativity and imagination while preparing us for an ever-changing world.""",
        "label": "AI",
        "source": "Claude-style (user sample)"
    },
    {
        "text": """*The Dawn of Intelligent Machines*

Artificial intelligence has quietly woven itself into the fabric of our daily lives, transforming how we work, communicate, and solve problems. What once seemed like science fiction now powers our smartphones, recommends our entertainment, and even drives our cars.

The practical applications of AI span nearly every industry imaginable. In healthcare, algorithms analyze medical images with remarkable accuracy, helping doctors detect diseases earlier than ever before. Businesses use AI to predict market trends and personalize customer experiences, while educators employ intelligent tutoring systems that adapt to each student's learning pace. Even creative fields have embraced these tools, with AI assisting in music composition, writing, and visual design.

Yet this technological revolution brings both promise and uncertainty. AI excels at processing vast amounts of data and recognizing patterns humans might miss, potentially solving challenges from climate change to drug discovery. However, questions about job displacement, privacy, and the ethical use of such powerful technology demand our attention.

As we stand at this crossroads, the key lies not in fearing AI but in thoughtfully directing its development. The goal should be creating tools that enhance human capability rather than replace human judgment, ensuring this technology serves humanity's broader interests while preserving what makes us fundamentally human.""",
        "label": "AI",  
        "source": "GPT-4 style (user sample)"
    },
    {
        "text": """# The Transformative Impact of Artificial Intelligence

Artificial intelligence has emerged as one of the most consequential technologies of our era, fundamentally reshaping how we live and work. From voice assistants that answer our questions to algorithms that detect diseases in medical scans, AI systems are becoming increasingly integrated into daily life.

The technology's rapid advancement stems from breakthroughs in machine learning, where systems learn patterns from vast amounts of data rather than following explicitly programmed rules. This has enabled remarkable achievements in areas like natural language processing, computer vision, and predictive analytics.

However, AI's rise brings significant challenges alongside its benefits. Concerns about job displacement, privacy, and algorithmic bias have prompted governments and institutions worldwide to develop frameworks for responsible AI development and deployment.

Moving forward, the key lies in harnessing AI's potential while mitigating its risks. This requires ongoing dialogue between technologists, policymakers, and the public to ensure these powerful tools serve humanity's collective interests.""",
        "label": "AI",
        "source": "GPT-4 formal essay"
    },
    # Typical AI patterns - very fluent, well-structured
    {
        "text": """The intersection of technology and education represents one of the most significant transformations in modern society. Digital tools have revolutionized how knowledge is acquired, shared, and applied across diverse contexts. From online learning platforms to interactive simulations, technology has democratized access to educational resources on an unprecedented scale.

Students today can access world-class instruction from leading institutions regardless of their geographic location or socioeconomic background. This shift has profound implications for global education equity and workforce development. However, the digital divide remains a persistent challenge, with significant disparities in access to technology and digital literacy skills across different communities.""",
        "label": "AI",
        "source": "Generated formal education essay"
    },
]

# Human-written samples (diverse styles)
HUMAN_SAMPLES = [
    {
        "text": """I remember the first time I saw the ocean. I was maybe six or seven, can't remember exactly. Mom had packed sandwiches that got all soggy from the humidity and Dad kept complaining about the traffic. But when we finally got there, and I saw that endless stretch of blue... nothing else mattered. Kids don't understand infinity, but I think I got close that day.""",
        "label": "Human",
        "source": "Personal narrative"
    },
    {
        "text": """The experiment wasn't working and I didn't know why. I'd checked the protocol three times, recalibrated the equipment twice, and still - the readings made no sense. Dr. Martinez suggested it might be contamination, but that seemed too simple. Turns out she was right. A tiny speck of dust had settled on the lens. Three weeks of work, derailed by dust. That's science for you.""",
        "label": "Human",
        "source": "Academic anecdote"
    },
    {
        "text": """gonna be honest this whole situation is kinda ridiculous. like we've been waiting for what feels like forever and nobody seems to know what's going on?? my friend texted me earlier saying the same thing happened to her last week. at this point im just gonna go home and deal with it tomorrow i guess""",
        "label": "Human",
        "source": "Casual text/social media"
    },
    {
        "text": """Historical analysis reveals that the economic conditions preceding the 1929 crash were far more complex than popular narratives suggest. While speculative excess certainly played a role, structural imbalances in agricultural exports, unsustainable consumer credit expansion, and Federal Reserve policy missteps created a perfect storm. The crash itself was merely the trigger, not the cause.""",
        "label": "Human",
        "source": "Academic writing"
    },
    {
        "text": """The patient presented with acute onset chest pain radiating to the left arm, accompanied by diaphoresis and shortness of breath. ECG showed ST elevation in leads II, III, and aVF, consistent with inferior STEMI. Troponin levels were elevated at 2.4 ng/mL. Patient was taken immediately to cath lab for primary PCI.""",
        "label": "Human", 
        "source": "Medical note"
    },
]


def load_hc3_samples(hc3_path: str, n_samples: int = 50) -> List[Dict]:
    """Load human samples from HC3 dataset."""
    samples = []
    
    # Try different HC3 files
    for filename in ['open_qa.jsonl', 'wiki_csai.jsonl', 'finance.jsonl']:
        filepath = Path(hc3_path) / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= n_samples // 3:  # Distribute across files
                        break
                    try:
                        data = json.loads(line)
                        # HC3 has human_answers field
                        if 'human_answers' in data:
                            for answer in data['human_answers'][:1]:  # Take first human answer
                                if len(answer) > 100:
                                    samples.append({
                                        "text": answer,
                                        "label": "Human",
                                        "source": f"HC3/{filename}"
                                    })
                    except:
                        continue
    
    return samples[:n_samples]


def load_gpt_essays(essay_path: str, n_samples: int = 50) -> List[Dict]:
    """Load GPT essays from one-class-essay-detection dataset."""
    samples = []
    filepath = Path(essay_path) / 'short-essays-en-balanced-merged-cv.csv'
    
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            # Get AI samples (class=1 typically means AI)
            ai_samples = df[df['class'] == 1].sample(min(n_samples, len(df[df['class'] == 1])))
            for _, row in ai_samples.iterrows():
                if len(str(row['essay'])) > 100:
                    samples.append({
                        "text": str(row['essay']),
                        "label": "AI",
                        "source": "GPT-essays-dataset"
                    })
        except Exception as e:
            print(f"Error loading essays: {e}")
    
    return samples[:n_samples]


def load_claude_samples(claude_path: str, n_samples: int = 50) -> List[Dict]:
    """Load and clean Claude 4.5 samples (remove <think> sections)."""
    samples = []
    
    jsonl_file = Path(claude_path) / 'claude-4.5-high-reasoning-250x.jsonl'
    if not jsonl_file.exists():
        return []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        # Remove <think> sections
                        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                        clean_content = clean_content.strip()
                        
                        if len(clean_content) > 200:
                            samples.append({
                                "text": clean_content[:3000],  # Limit length
                                "label": "AI",
                                "source": "Claude-4.5-reasoning"
                            })
                            break
            except:
                continue
    
    return samples[:n_samples]


def build_test_dataset(project_root: str) -> List[Dict]:
    """Build comprehensive test dataset."""
    dataset = []
    
    # Add hardcoded latest AI samples
    dataset.extend(LATEST_AI_SAMPLES)
    print(f"Added {len(LATEST_AI_SAMPLES)} latest AI samples")
    
    # Add hardcoded human samples
    dataset.extend(HUMAN_SAMPLES)
    print(f"Added {len(HUMAN_SAMPLES)} human samples")
    
    # Load HC3 human samples
    hc3_path = Path(project_root) / 'HC3'
    if hc3_path.exists():
        hc3_samples = load_hc3_samples(str(hc3_path), n_samples=30)
        dataset.extend(hc3_samples)
        print(f"Added {len(hc3_samples)} HC3 human samples")
    
    # Load GPT essay samples
    essay_path = Path(project_root) / 'one-class-essay-detection' / 'english'
    if essay_path.exists():
        essay_samples = load_gpt_essays(str(essay_path), n_samples=30)
        dataset.extend(essay_samples)
        print(f"Added {len(essay_samples)} GPT essay samples")
    
    # Load Claude samples
    claude_path = Path(project_root) / 'claude-sonnet-4.5-high-reasoning-250x'
    if claude_path.exists():
        claude_samples = load_claude_samples(str(claude_path), n_samples=30)
        dataset.extend(claude_samples)
        print(f"Added {len(claude_samples)} Claude 4.5 samples")
    
    # Shuffle
    random.shuffle(dataset)
    
    return dataset


def run_calibration_test(dataset: List[Dict], verbose: bool = True) -> Dict[str, Any]:
    """Run detector on test dataset and compute metrics."""
    # Set up imports - detector.py uses relative imports so we need ensemble as package
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Add project root so we can import ensemble as a package
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from ensemble.detector import EnsembleDetector
    except ImportError as e:
        print(f"Error: Cannot import EnsembleDetector: {e}")
        print("Make sure you have the required dependencies installed.")
        return {}
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = EnsembleDetector()
    
    results = {
        "total": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "uncertain": 0,
        "false_positives": [],  # Human classified as AI
        "false_negatives": [],  # AI classified as Human
        "by_source": {}
    }
    
    print(f"\nTesting on {len(dataset)} samples...\n")
    
    for i, sample in enumerate(dataset):
        text = sample['text']
        true_label = sample['label']
        source = sample['source']
        
        # Run detection
        result = detector.detect(text)
        predicted = result.prediction
        
        # Track results
        if source not in results['by_source']:
            results['by_source'][source] = {"total": 0, "correct": 0}
        results['by_source'][source]['total'] += 1
        
        # Check if correct
        if predicted == true_label:
            results['correct'] += 1
            results['by_source'][source]['correct'] += 1
        elif predicted == "UNCERTAIN":
            results['uncertain'] += 1
        else:
            results['incorrect'] += 1
            
            if true_label == "Human" and predicted == "AI":
                results['false_positives'].append({
                    "text_preview": text[:150] + "...",
                    "source": source,
                    "confidence": result.confidence,
                    "breakdown": {k: v.prediction for k, v in result.breakdown.items()}
                })
            elif true_label == "AI" and predicted == "Human":
                results['false_negatives'].append({
                    "text_preview": text[:150] + "...",
                    "source": source,
                    "confidence": result.confidence,
                    "breakdown": {k: v.prediction for k, v in result.breakdown.items()}
                })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} samples...")
    
    # Calculate metrics
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    results['fp_rate'] = len(results['false_positives']) / results['total']
    results['fn_rate'] = len(results['false_negatives']) / results['total']
    
    return results


def print_results(results: Dict[str, Any]):
    """Print test results summary."""
    print("\n" + "="*60)
    print("CALIBRATION TEST RESULTS")
    print("="*60)
    
    print(f"\nTotal Samples: {results['total']}")
    print(f"Correct: {results['correct']} ({results['accuracy']*100:.1f}%)")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Uncertain: {results['uncertain']}")
    
    print(f"\nFalse Positive Rate: {results['fp_rate']*100:.1f}%")
    print(f"False Negative Rate: {results['fn_rate']*100:.1f}%")
    
    print("\n--- Results by Source ---")
    for source, data in results['by_source'].items():
        acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"  {source}: {data['correct']}/{data['total']} ({acc:.1f}%)")
    
    if results['false_positives']:
        print("\n--- False Positives (Human → AI) ---")
        for i, fp in enumerate(results['false_positives'][:5]):
            print(f"\n[{i+1}] Source: {fp['source']}")
            print(f"    Confidence: {fp['confidence']*100:.1f}%")
            print(f"    Breakdown: {fp['breakdown']}")
            print(f"    Text: {fp['text_preview'][:100]}...")
    
    if results['false_negatives']:
        print("\n--- False Negatives (AI → Human) ---")
        for i, fn in enumerate(results['false_negatives'][:5]):
            print(f"\n[{i+1}] Source: {fn['source']}")
            print(f"    Confidence: {fn['confidence']*100:.1f}%")
            print(f"    Breakdown: {fn['breakdown']}")
            print(f"    Text: {fn['text_preview'][:100]}...")


def main():
    """Main entry point."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("="*60)
    print("AI Detection Calibration Framework")
    print("="*60)
    
    # Build test dataset
    print("\n1. Building test dataset...")
    dataset = build_test_dataset(str(project_root))
    print(f"\nTotal test samples: {len(dataset)}")
    
    # Count by label
    ai_count = sum(1 for s in dataset if s['label'] == 'AI')
    human_count = sum(1 for s in dataset if s['label'] == 'Human')
    print(f"  AI samples: {ai_count}")
    print(f"  Human samples: {human_count}")
    
    # Save dataset for reference
    dataset_path = script_dir / 'calibration_dataset.json'
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nDataset saved to: {dataset_path}")
    
    # Run calibration test
    print("\n2. Running calibration test...")
    results = run_calibration_test(dataset)
    
    if results:
        # Print results
        print_results(results)
        
        # Save results
        results_path = script_dir / 'calibration_results.json'
        # Remove non-serializable items
        save_results = {k: v for k, v in results.items()}
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
