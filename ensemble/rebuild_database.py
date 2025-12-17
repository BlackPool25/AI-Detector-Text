#!/usr/bin/env python3
"""
Rebuild DeTeCtive Database with More Diverse AI Samples

This script builds a larger, more diverse FAISS database for the DeTeCtive detector
using the official Deepfake cross_domains_cross_models training data.

The Deepfake_best.pth model was trained on this data, so we should use the same
data source for the database to ensure proper embedding matching.

Usage:
    python rebuild_database.py --output database/deepfake_full --samples 20000
"""

import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.text_embedding import TextEmbeddingModel
from core.faiss_index import Indexer


def load_deepfake_training_data(deepfake_path: str, max_samples: int = 20000) -> List[Tuple[str, int, str]]:
    """
    Load samples from Deepfake cross_domains_cross_models training set.
    This is the same data used to train the Deepfake_best.pth model.
    
    Returns:
        List of (text, label, source) tuples
        label: 0 = AI, 1 = Human
    """
    train_file = os.path.join(deepfake_path, "cross_domains_cross_models", "train.csv")
    
    if not os.path.exists(train_file):
        print(f"ERROR: Training file not found at {train_file}")
        return []
    
    print(f"Loading Deepfake training data from {train_file}...")
    
    df = pd.read_csv(train_file)
    print(f"Total samples in training set: {len(df)}")
    
    # Filter by minimum length
    df = df[df['text'].str.len() >= 100]
    print(f"After length filter (>=100 chars): {len(df)}")
    
    # Balance the dataset
    human_df = df[df['label'] == 1]
    ai_df = df[df['label'] == 0]
    
    print(f"Human samples: {len(human_df)}, AI samples: {len(ai_df)}")
    
    n_each = min(len(human_df), len(ai_df), max_samples // 2)
    print(f"Sampling {n_each} from each class...")
    
    human_sample = human_df.sample(n=n_each, random_state=42)
    ai_sample = ai_df.sample(n=n_each, random_state=42)
    
    samples = []
    for _, row in human_sample.iterrows():
        samples.append((row['text'], 1, row['src']))
    for _, row in ai_sample.iterrows():
        samples.append((row['text'], 0, row['src']))
    
    random.shuffle(samples)
    
    # Print source distribution for AI samples
    ai_sources = ai_sample['src'].value_counts()
    print(f"\nAI source distribution (top 10):")
    for src, count in ai_sources.head(10).items():
        print(f"  {src}: {count}")
    
    return samples


def load_hc3_samples(hc3_path: str, max_samples: int = 5000) -> List[Tuple[str, int, str]]:
    """
    Load samples from HC3 dataset (optional, for additional ChatGPT coverage)
    
    Returns:
        List of (text, label, source) tuples
        label: 0 = AI, 1 = Human
    """
    samples = []
    all_jsonl = os.path.join(hc3_path, "all.jsonl")
    
    if not os.path.exists(all_jsonl):
        print(f"HC3 all.jsonl not found at {all_jsonl}, skipping...")
        return samples
    
    print(f"Loading HC3 samples from {all_jsonl}...")
    
    with open(all_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                question = data.get('question', '')
                
                # Human answers
                human_answers = data.get('human_answers', [])
                for answer in human_answers:
                    if len(answer) >= 100:  # Minimum length
                        text = f"Question: {question}\nAnswer: {answer}"
                        samples.append((text, 1, 'hc3_human'))
                
                # ChatGPT answers  
                chatgpt_answers = data.get('chatgpt_answers', [])
                for answer in chatgpt_answers:
                    if len(answer) >= 100:
                        text = f"Question: {question}\nAnswer: {answer}"
                        samples.append((text, 0, 'hc3_chatgpt'))
                        
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(samples)} samples from HC3")
    
    # Balance and limit
    human_samples = [s for s in samples if s[1] == 1]
    ai_samples = [s for s in samples if s[1] == 0]
    
    n_each = min(len(human_samples), len(ai_samples), max_samples // 2)
    
    random.shuffle(human_samples)
    random.shuffle(ai_samples)
    
    return human_samples[:n_each] + ai_samples[:n_each]


def build_database(
    samples: List[Tuple[str, int, str]],
    model_path: str,
    output_path: str,
    model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
    batch_size: int = 32,
    embedding_dim: int = 768
):
    """Build FAISS database from samples"""
    
    print(f"\nBuilding database with {len(samples)} samples...")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = TextEmbeddingModel(model_name)
    if device == "cuda":
        model = model.cuda()
    
    # Load trained weights
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model weights not found at {model_path}, using base model")
    
    model.eval()
    tokenizer = model.tokenizer
    
    # Generate embeddings
    all_embeddings = []
    all_ids = []
    all_labels = []
    
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    print("Generating embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            encoded = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            if device == "cuda":
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            embeddings = model(encoded)
            embeddings = F.normalize(embeddings, dim=-1)
            
            for j, emb in enumerate(embeddings):
                # Generate unique ID
                text_hash = hash(batch_texts[j]) & ((1 << 63) - 1)
                all_ids.append(text_hash)
                all_embeddings.append(emb.cpu().numpy())
                all_labels.append(batch_labels[j])
    
    # Stack embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Build FAISS index
    print("Building FAISS index...")
    indexer = Indexer(embedding_dim, device='cpu')
    indexer.index_data(all_ids, all_embeddings)
    indexer.serialize(output_path)
    
    # Save label dictionary
    label_dict = {all_ids[i]: all_labels[i] for i in range(len(all_ids))}
    with open(os.path.join(output_path, 'label_dict.pkl'), 'wb') as f:
        pickle.dump(label_dict, f)
    
    # Save metadata
    metadata = {
        'total_samples': len(samples),
        'human_samples': sum(1 for l in all_labels if l == 1),
        'ai_samples': sum(1 for l in all_labels if l == 0),
        'embedding_dim': embedding_dim,
        'model_name': model_name,
        'sources': list(set(s[2] for s in samples))
    }
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDatabase built successfully!")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Human samples: {metadata['human_samples']}")
    print(f"  AI samples: {metadata['ai_samples']}")
    print(f"  Sources: {metadata['sources']}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Rebuild DeTeCtive database")
    parser.add_argument("--output", type=str, default="database/deepfake_full",
                        help="Output directory for the database")
    parser.add_argument("--samples", type=int, default=20000,
                        help="Target number of samples (will be balanced)")
    parser.add_argument("--model-path", type=str, default="models/Deepfake_best.pth",
                        help="Path to trained model weights")
    parser.add_argument("--hc3-path", type=str, default="../HC3",
                        help="Path to HC3 dataset")
    parser.add_argument("--deepfake-path", type=str, default="../LLMtext_detect_dataset/Deepfake",
                        help="Path to Deepfake dataset")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--include-hc3", action="store_true",
                        help="Also include HC3 samples for ChatGPT coverage")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths
    output_path = os.path.join(script_dir, args.output)
    model_path = os.path.join(script_dir, args.model_path)
    hc3_path = os.path.join(script_dir, args.hc3_path)
    deepfake_path = os.path.join(script_dir, args.deepfake_path)
    
    # Load samples - PRIMARY source is Deepfake training data
    all_samples = []
    
    # Deepfake training data (primary - this is what the model was trained on)
    deepfake_samples = load_deepfake_training_data(deepfake_path, max_samples=args.samples)
    all_samples.extend(deepfake_samples)
    
    # Optionally add HC3 for additional ChatGPT coverage
    if args.include_hc3:
        hc3_samples = load_hc3_samples(hc3_path, max_samples=args.samples // 4)
        all_samples.extend(hc3_samples)
    
    if not all_samples:
        print("ERROR: No samples loaded! Check dataset paths.")
        sys.exit(1)
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Limit to target
    if len(all_samples) > args.samples:
        # Balance before limiting
        human = [s for s in all_samples if s[1] == 1]
        ai = [s for s in all_samples if s[1] == 0]
        n_each = min(len(human), len(ai), args.samples // 2)
        all_samples = human[:n_each] + ai[:n_each]
        random.shuffle(all_samples)
    
    print(f"\nTotal samples to process: {len(all_samples)}")
    print(f"  Human: {sum(1 for s in all_samples if s[1] == 1)}")
    print(f"  AI: {sum(1 for s in all_samples if s[1] == 0)}")
    
    # Build database
    build_database(
        samples=all_samples,
        model_path=model_path,
        output_path=output_path,
        batch_size=args.batch_size
    )
    
    print(f"\nDone! New database at: {output_path}")
    print("Update detector.py to use: database_path='database/diverse_v2'")


if __name__ == "__main__":
    main()
