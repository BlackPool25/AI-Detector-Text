#!/usr/bin/env python3
"""
Validate processed dataset for training readiness.
"""

import pandas as pd
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default processed data directory
PROCESSED_DIR = project_root / "data" / "processed"

if not PROCESSED_DIR.exists():
    print(f"Error: {PROCESSED_DIR} not found!")
    print(f"Please process your data first using:")
    print(f"  python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed")
    sys.exit(1)

print("="*70)
print("PROCESSED DATA VALIDATION")
print("="*70)

# Load statistics
stats_file = PROCESSED_DIR / "statistics.json"
if not stats_file.exists():
    print(f"Error: {stats_file} not found!")
    sys.exit(1)

with open(stats_file, 'r') as f:
    stats = json.load(f)

print("\n1. PROCESSING SUMMARY:")
print(f"   Total samples: {stats['total_samples']:,}")
print(f"   Human samples: {stats['human_samples']:,} ({stats['human_samples']/stats['total_samples']*100:.2f}%)")
print(f"   AI samples: {stats['ai_samples']:,} ({stats['ai_samples']/stats['total_samples']*100:.2f}%)")
print(f"   Processing time: {stats['processing_time_seconds']:.1f} seconds")
print(f"   Speed: {stats['total_samples']/stats['processing_time_seconds']:.0f} rows/sec")
print(f"   Duplicates removed: {stats['duplicates_removed']:,}")
print(f"   Invalid rows: {stats['invalid_rows']:,}")

print("\n2. AI MODEL DISTRIBUTION:")
total_ai = sum(stats['ai_model_breakdown'].values())
for model, count in sorted(stats['ai_model_breakdown'].items(), key=lambda x: x[1], reverse=True):
    pct = count / total_ai * 100
    print(f"   {model:15} {count:8,} ({pct:5.2f}%)")

print("\n3. DOMAIN DISTRIBUTION:")
total_domain = sum(stats['domain_breakdown'].values())
for domain, count in sorted(stats['domain_breakdown'].items(), key=lambda x: x[1], reverse=True):
    pct = count / total_domain * 100
    print(f"   {domain:15} {count:8,} ({pct:5.2f}%)")

print("\n4. DATA QUALITY CHECKS:")

# Check human data
print("\n   A. HUMAN DATA:")
human_dir = PROCESSED_DIR / "real" / "RAID-Dataset" / "human"
if not human_dir.exists():
    # Try alternative structure
    human_dir = PROCESSED_DIR / "human"

if human_dir.exists():
    human_files = list(human_dir.glob('*.parquet'))
    print(f"      Files: {len(human_files)}")

    if human_files:
        # Sample first file
        df_human = pd.read_parquet(human_files[0])
        print(f"      Columns: {df_human.columns.tolist()}")
        print(f"      Sample size (first file): {len(df_human)}")
        print(f"      Labels (should all be 0): {df_human['label'].unique()}")
        print(f"      Label names: {df_human['label_name'].unique()}")
        print(f"      Models: {df_human['model'].unique()}")
        print(f"\n      Sample text (first row):")
        print(f"      '{df_human['text'].iloc[0][:100]}...'")
        print(f"      Text length: {df_human['text_length'].iloc[0]} chars")

        # Verify all human data
        all_human = []
        for f in human_files[:5]:  # Check first 5 files
            df = pd.read_parquet(f)
            all_human.append(df)

        df_human_all = pd.concat(all_human, ignore_index=True)
        print(f"\n      Total human samples checked (first 5 files): {len(df_human_all)}")
        print(f"      All labels = 0? {(df_human_all['label'] == 0).all()}")
        print(f"      All models = 'human'? {(df_human_all['model'] == 'human').all()}")
    else:
        print("      No human parquet files found!")
else:
    print("      Human directory not found!")

# Check AI data
print("\n   B. AI DATA:")
ai_base = PROCESSED_DIR / "ai" / "RAID-Dataset"
if not ai_base.exists():
    # Try alternative structure
    ai_base = PROCESSED_DIR / "ai"

if ai_base.exists():
    # Check GPT-4 (or first available model)
    model_dirs = [d for d in ai_base.iterdir() if d.is_dir()]
    if model_dirs:
        first_model = model_dirs[0]
        model_files = list(first_model.glob('*.parquet'))
        print(f"\n      {first_model.name}:")
        print(f"      Files: {len(model_files)}")
        
        if model_files:
            df_ai = pd.read_parquet(model_files[0])
            print(f"      Sample size (first file): {len(df_ai)}")
            print(f"      Labels (should all be 1): {df_ai['label'].unique()}")
            print(f"      Label names: {df_ai['label_name'].unique()}")
            print(f"      Models: {df_ai['model'].unique()}")
            print(f"      Sample text: '{df_ai['text'].iloc[0][:100]}...'")

        # Check multiple models
        for model_dir in model_dirs[:3]:
            model_files = list(model_dir.glob('*.parquet'))
            if model_files:
                df = pd.read_parquet(model_files[0])
                print(f"\n      {model_dir.name}: {len(df)} samples, all label=1? {(df['label'] == 1).all()}")
    else:
        print("      No AI model directories found!")
else:
    print("      AI directory not found!")

print("\n5. SCHEMA VALIDATION:")
if 'df_human' in locals():
    print(f"   Required columns present: {set(df_human.columns) >= {'text', 'label', 'label_name', 'model', 'domain', 'attack_type', 'text_length'}}")
else:
    print("   Skipped - no data loaded")

print("\n6. DATA CORRUPTION CHECK:")
# Check for null values in critical columns
if 'df_human_all' in locals():
    print(f"   Human data nulls in 'text': {df_human_all['text'].isna().sum()}")
    print(f"   Human data nulls in 'label': {df_human_all['label'].isna().sum()}")
else:
    print("   Human data: Not checked")

if 'df_ai' in locals():
    print(f"   AI data nulls in 'text': {df_ai['text'].isna().sum()}")
    print(f"   AI data nulls in 'label': {df_ai['label'].isna().sum()}")
else:
    print("   AI data: Not checked")

print("\n7. TEXT LENGTH VALIDATION:")
if 'text_length_stats' in stats:
    print(f"   Min text length: {stats['text_length_stats']['min']}")
    print(f"   Max text length: {stats['text_length_stats']['max']}")
    print(f"   Mean text length: {stats['text_length_stats']['mean']:.0f}")
    print(f"   Median text length: {stats['text_length_stats']['median']}")
else:
    print("   Text length stats not available in statistics.json")

print("\n8. TRAINING READINESS:")
checks_passed = []
checks_passed.append(("Folder structure correct", PROCESSED_DIR.exists()))

if 'df_human_all' in locals():
    checks_passed.append(("Human labels all = 0", (df_human_all['label'] == 0).all()))
if 'df_ai' in locals():
    checks_passed.append(("AI labels all = 1", (df_ai['label'] == 1).all()))
if 'df_human_all' in locals() and 'df_ai' in locals():
    checks_passed.append(("No null text values", df_human_all['text'].isna().sum() == 0 and df_ai['text'].isna().sum() == 0))
    checks_passed.append(("No null labels", df_human_all['label'].isna().sum() == 0 and df_ai['label'].isna().sum() == 0))
if 'df_human' in locals():
    checks_passed.append(("All required columns present", set(df_human.columns) >= {'text', 'label', 'label_name', 'model', 'domain', 'attack_type', 'text_length'}))
if 'human_files' in locals() and 'model_files' in locals():
    checks_passed.append(("Parquet files readable", len(human_files) > 0 and len(model_files) > 0))

print("\n   VALIDATION RESULTS:")
all_passed = True
for check_name, passed in checks_passed:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"   [{status}] {check_name}")
    if not passed:
        all_passed = False

print("\n" + "="*70)
if all_passed:
    print("✓✓✓ DATASET IS VALID AND READY FOR TRAINING! ✓✓✓")
    print("="*70)
    print("\nSUGGESTED NEXT STEPS:")
    print("1. Load dataset:")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('parquet', data_dir='processed_data')")
    print("\n2. Or load specific splits:")
    print("   # Load all human data")
    print("   human_data = pd.read_parquet('processed_data/real/RAID-Dataset/human/')")
    print("   # Load specific AI model")
    print("   gpt4_data = pd.read_parquet('processed_data/ai/RAID-Dataset/gpt4/')")
    print("\n3. Create balanced train/val/test splits")
    print("4. Train your AI text detector!")
else:
    print("✗✗✗ DATASET HAS ISSUES - REVIEW FAILURES ABOVE ✗✗✗")
print("="*70)
