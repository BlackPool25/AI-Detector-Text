#!/usr/bin/env python3
"""Analyze train.csv to understand its structure before processing."""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default data file
DATA_FILE = project_root / "data" / "raw" / "train.csv"

if not DATA_FILE.exists():
    print(f"Error: {DATA_FILE} not found!")
    print(f"Please place your train.csv file in {DATA_FILE.parent}")
    sys.exit(1)

# Read sample data
print(f"Reading {DATA_FILE.name} (first 100 rows)...")
df = pd.read_csv(DATA_FILE, nrows=100)

print("\n" + "="*70)
print("DETAILED DATA ANALYSIS")
print("="*70)

print("\n1. UNIQUE VALUES PER COLUMN:")
print("-"*70)
for col in df.columns:
    unique_count = df[col].nunique()
    null_count = df[col].isna().sum()
    print(f"\n{col}:")
    print(f"  - Unique values: {unique_count}")
    print(f"  - Null values: {null_count}")
    print(f"  - Sample values: {df[col].dropna().head(3).tolist()}")

print("\n" + "="*70)
print("2. MODEL COLUMN ANALYSIS:")
print("-"*70)
print("\nAll unique models in sample:")
print(df['model'].value_counts())

print("\n" + "="*70)
print("3. SOURCE_ID ANALYSIS (checking for human-written text):")
print("-"*70)
print("\nAll unique source_id values:")
print(df['source_id'].value_counts())

print("\n" + "="*70)
print("4. ATTACK COLUMN ANALYSIS:")
print("-"*70)
print("\nAll unique attack values:")
print(df['attack'].value_counts())

print("\n" + "="*70)
print("5. DOMAIN COLUMN ANALYSIS:")
print("-"*70)
print("\nAll unique domains:")
print(df['domain'].value_counts())

print("\n" + "="*70)
print("6. GENERATION COLUMN (text samples):")
print("-"*70)
print("\nFirst 3 generation texts (truncated to 200 chars):")
for i, text in enumerate(df['generation'].head(3), 1):
    print(f"\n[{i}] {str(text)[:200]}...")

print("\n" + "="*70)
print("7. CHECKING IF source_id INDICATES HUMAN TEXT:")
print("-"*70)
# Check if source_id being non-null means human-written
print(f"\nRows with non-null source_id: {df['source_id'].notna().sum()}")
print(f"Rows with null source_id: {df['source_id'].isna().sum()}")
print(f"Rows where source_id != adv_source_id: {(df['source_id'] != df['adv_source_id']).sum()}")

print("\n" + "="*70)
print("8. KEY INSIGHT - LABEL INFERENCE:")
print("-"*70)
print("\nPossible labeling strategies:")
print("1. If source_id != model → Human-written (label=0)")
print("2. If source_id == model → AI-generated (label=1)")
print("3. Or: If model is not null/NaN → AI-generated (label=1)")
print("\nLet me check the actual values...")

# Read more rows for better analysis
df_large = pd.read_csv(DATA_FILE, nrows=1000)
print(f"\nAnalyzing first 1000 rows...")
print(f"Total rows with model value: {df_large['model'].notna().sum()}")
print(f"Unique models: {df_large['model'].nunique()}")
print("\nModel distribution:")
print(df_large['model'].value_counts())
