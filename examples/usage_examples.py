#!/usr/bin/env python3
"""
Example usage script for the AI Text Detection Dataset Preprocessor.

This script demonstrates various ways to use preprocess.py and load the results.
"""

import pandas as pd
from pathlib import Path
import json

def example_1_basic_usage():
    """Example 1: Basic usage with minimal options."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    print("\nCommand:")
    print("python preprocess.py --input data.csv --output processed_data")
    print("\nThis uses default settings:")
    print("  - Chunk size: 100,000 rows")
    print("  - Text length: 10-100,000 characters")
    print("  - Compression: snappy")
    print("  - Remove duplicates: enabled")


def example_2_high_quality():
    """Example 2: High-quality processing with stricter filters."""
    print("\n" + "="*70)
    print("EXAMPLE 2: High-Quality Processing")
    print("="*70)
    print("\nCommand:")
    print("""python preprocess.py \\
    --input dataset.csv \\
    --output high_quality_data \\
    --min-text-length 50 \\
    --max-text-length 10000 \\
    --compression gzip \\
    --remove-duplicates""")
    print("\nThis creates a high-quality dataset by:")
    print("  - Filtering out very short texts (<50 chars)")
    print("  - Filtering out very long texts (>10,000 chars)")
    print("  - Using gzip compression (better ratio)")
    print("  - Removing all duplicates")


def example_3_load_with_pandas():
    """Example 3: Loading processed data with pandas."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Loading Data with Pandas")
    print("="*70)
    
    if not Path('test_output').exists():
        print("\nNo test_output directory found. Run the preprocessor first.")
        return
    
    print("\nLoading human data:")
    human_files = list(Path('test_output/human').glob('*.parquet'))
    if human_files:
        df_human = pd.concat([pd.read_parquet(f) for f in human_files])
        print(f"  - Loaded {len(df_human)} human samples")
        print(f"  - Columns: {df_human.columns.tolist()}")
        print(f"\nFirst human sample:")
        print(f"  Text: {df_human.iloc[0]['text'][:100]}...")
        print(f"  Label: {df_human.iloc[0]['label_name']}")
        print(f"  Length: {df_human.iloc[0]['text_length']}")
    
    print("\nLoading AI data:")
    for model_dir in Path('test_output/ai').iterdir():
        if model_dir.is_dir():
            model_files = list(model_dir.glob('*.parquet'))
            if model_files:
                df_model = pd.concat([pd.read_parquet(f) for f in model_files])
                print(f"  - {model_dir.name}: {len(df_model)} samples")


def example_4_statistics():
    """Example 4: Reading and analyzing statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Analyzing Statistics")
    print("="*70)
    
    stats_file = Path('test_output/statistics.json')
    if not stats_file.exists():
        print("\nNo statistics.json found. Run the preprocessor first.")
        return
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    print("\nDataset Overview:")
    print(f"  Total Samples: {stats['total_samples']:,}")
    print(f"  Human: {stats['human_samples']:,} ({stats['human_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  AI: {stats['ai_samples']:,} ({stats['ai_samples']/stats['total_samples']*100:.1f}%)")
    
    if 'ai_model_breakdown' in stats:
        print("\n  AI Models:")
        for model, count in sorted(stats['ai_model_breakdown'].items(), key=lambda x: x[1], reverse=True):
            pct = count / stats['ai_samples'] * 100 if stats['ai_samples'] > 0 else 0
            print(f"    - {model}: {count:,} ({pct:.1f}%)")
    
    if 'text_length_stats' in stats:
        tls = stats['text_length_stats']
        print(f"\n  Text Lengths:")
        print(f"    - Min: {tls['min']}")
        print(f"    - Max: {tls['max']}")
        print(f"    - Mean: {tls['mean']:.0f}")
        print(f"    - Median: {tls['median']:.0f}")
    
    print(f"\nProcessing Info:")
    print(f"  Time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"  Duplicates Removed: {stats['duplicates_removed']:,}")
    print(f"  Invalid Rows: {stats['invalid_rows']:,}")


def example_5_huggingface():
    """Example 5: Loading with HuggingFace datasets (code example only)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Loading with HuggingFace Datasets")
    print("="*70)
    print("\nInstall datasets library:")
    print("  pip install datasets")
    print("\nLoad all data:")
    print("""
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset('parquet', data_dir='test_output')
print(dataset)

# Load specific subset
dataset_human = load_dataset('parquet', data_dir='test_output/human')
dataset_ai_gpt4 = load_dataset('parquet', data_dir='test_output/ai/gpt4')

# Create train/val/test splits
splits = dataset['train'].train_test_split(test_size=0.2, seed=42)
test_valid = splits['test'].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    'train': splits['train'],        # 80%
    'validation': test_valid['train'], # 10%
    'test': test_valid['test']         # 10%
}

print(f"Train: {len(final_dataset['train'])} samples")
print(f"Validation: {len(final_dataset['validation'])} samples")
print(f"Test: {len(final_dataset['test'])} samples")
""")


def example_6_balanced_sampling():
    """Example 6: Creating balanced datasets."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Balanced Sampling")
    print("="*70)
    
    print("\nIf your dataset is imbalanced, create a balanced subset:")
    print("""
import pandas as pd
from pathlib import Path

# Load all data
human_files = list(Path('test_output/human').glob('*.parquet'))
ai_files = list(Path('test_output/ai').rglob('*.parquet'))

df_human = pd.concat([pd.read_parquet(f) for f in human_files])
df_ai = pd.concat([pd.read_parquet(f) for f in ai_files])

# Balance by sampling
min_samples = min(len(df_human), len(df_ai))
df_human_balanced = df_human.sample(n=min_samples, random_state=42)
df_ai_balanced = df_ai.sample(n=min_samples, random_state=42)

# Combine
df_balanced = pd.concat([df_human_balanced, df_ai_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

print(f"Balanced dataset: {len(df_balanced)} samples")
print(f"Human: {(df_balanced['label']==0).sum()} ({(df_balanced['label']==0).sum()/len(df_balanced)*100:.1f}%)")
print(f"AI: {(df_balanced['label']==1).sum()} ({(df_balanced['label']==1).sum()/len(df_balanced)*100:.1f}%)")
""")


def example_7_model_specific():
    """Example 7: Training on specific AI models."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Model-Specific Analysis")
    print("="*70)
    
    print("\nAnalyze or train on specific AI models:")
    print("""
import pandas as pd
from pathlib import Path

# Load human data
human_files = list(Path('test_output/human').glob('*.parquet'))
df_human = pd.concat([pd.read_parquet(f) for f in human_files])

# Load only GPT-4 data
gpt4_files = list(Path('test_output/ai/gpt4').glob('*.parquet'))
df_gpt4 = pd.concat([pd.read_parquet(f) for f in gpt4_files])

# Create binary classifier: human vs GPT-4
df_binary = pd.concat([df_human, df_gpt4])

print(f"Human vs GPT-4 dataset: {len(df_binary)} samples")
print(f"Human: {len(df_human)}")
print(f"GPT-4: {len(df_gpt4)}")

# Multi-class: human vs multiple AI models
claude_files = list(Path('test_output/ai/claude').glob('*.parquet'))
gpt35_files = list(Path('test_output/ai/gpt3_5').glob('*.parquet'))

df_claude = pd.concat([pd.read_parquet(f) for f in claude_files]) if claude_files else pd.DataFrame()
df_gpt35 = pd.concat([pd.read_parquet(f) for f in gpt35_files]) if gpt35_files else pd.DataFrame()

# Create model labels
df_human['model_label'] = 'human'
df_gpt4['model_label'] = 'gpt4'
if len(df_claude) > 0:
    df_claude['model_label'] = 'claude'
if len(df_gpt35) > 0:
    df_gpt35['model_label'] = 'gpt35'

# Combine for multi-class
df_multiclass = pd.concat([df_human, df_gpt4, df_claude, df_gpt35])
print(f"\\nMulti-class dataset: {len(df_multiclass)} samples")
print(df_multiclass['model_label'].value_counts())
""")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("AI TEXT DETECTION DATASET PREPROCESSOR - USAGE EXAMPLES")
    print("="*70)
    
    example_1_basic_usage()
    example_2_high_quality()
    example_3_load_with_pandas()
    example_4_statistics()
    example_5_huggingface()
    example_6_balanced_sampling()
    example_7_model_specific()
    
    print("\n" + "="*70)
    print("For more information, see README.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
