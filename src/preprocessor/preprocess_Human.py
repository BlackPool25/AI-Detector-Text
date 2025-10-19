#!/usr/bin/env python3
"""
AI Text Detection Dataset Preprocessor

A production-grade Python script for processing large CSV datasets with automatic
schema detection, memory-efficient chunked processing, intelligent data segregation,
and Parquet output optimization.

Author: AI Assistant
Date: 2025-10-18
"""

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class SchemaInfo:
    """Detected schema information from CSV file."""
    text_column: str
    label_column: str
    model_column: Optional[str] = None
    domain_column: Optional[str] = None
    attack_column: Optional[str] = None
    detected_columns: Dict[str, str] = None
    label_format: str = "unknown"
    unique_labels: List[str] = None
    unique_models: List[str] = None
    
    def __post_init__(self):
        if self.detected_columns is None:
            self.detected_columns = {}
        if self.unique_labels is None:
            self.unique_labels = []
        if self.unique_models is None:
            self.unique_models = []


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline."""
    input_path: str
    output_dir: str
    chunk_size: int = 100000
    min_text_length: int = 10
    max_text_length: int = 100000
    compression: str = 'snappy'
    remove_duplicates: bool = True
    resume: bool = False
    validate_only: bool = False
    threads: int = 4
    checkpoint_interval: int = 10
    dataset_name: str = 'RAID-Dataset'
    force_human: bool = False


class StatisticsTracker:
    """Track and display processing statistics."""
    
    def __init__(self):
        self.total_rows = 0
        self.human_count = 0
        self.ai_count = 0
        self.model_counts = defaultdict(int)
        self.domain_counts = defaultdict(int)
        self.duplicates_removed = 0
        self.invalid_rows = 0
        self.text_lengths = []
        self.start_time = time.time()
        self.chunks_processed = 0
        
    def update(self, chunk_stats: Dict[str, Any]):
        """Update statistics with chunk data."""
        self.total_rows += chunk_stats.get('rows_processed', 0)
        self.human_count += chunk_stats.get('human_count', 0)
        self.ai_count += chunk_stats.get('ai_count', 0)
        self.duplicates_removed += chunk_stats.get('duplicates_removed', 0)
        self.invalid_rows += chunk_stats.get('invalid_rows', 0)
        self.chunks_processed += 1
        
        for model, count in chunk_stats.get('model_counts', {}).items():
            self.model_counts[model] += count
            
        for domain, count in chunk_stats.get('domain_counts', {}).items():
            self.domain_counts[domain] += count
            
        if 'text_lengths' in chunk_stats:
            self.text_lengths.extend(chunk_stats['text_lengths'])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        elapsed = time.time() - self.start_time
        rows_per_sec = self.total_rows / elapsed if elapsed > 0 else 0
        
        summary = {
            'total_rows': self.total_rows,
            'human_count': self.human_count,
            'ai_count': self.ai_count,
            'human_percentage': (self.human_count / self.total_rows * 100) if self.total_rows > 0 else 0,
            'ai_percentage': (self.ai_count / self.total_rows * 100) if self.total_rows > 0 else 0,
            'model_counts': dict(self.model_counts),
            'domain_counts': dict(self.domain_counts),
            'duplicates_removed': self.duplicates_removed,
            'invalid_rows': self.invalid_rows,
            'processing_time': elapsed,
            'rows_per_second': rows_per_sec,
            'chunks_processed': self.chunks_processed
        }
        
        if self.text_lengths:
            summary['text_length_stats'] = {
                'min': min(self.text_lengths),
                'max': max(self.text_lengths),
                'mean': sum(self.text_lengths) / len(self.text_lengths),
                'median': sorted(self.text_lengths)[len(self.text_lengths) // 2]
            }
        
        return summary
    
    def display(self, current_label: str = ""):
        """Display formatted statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print(f"Current Processing: {current_label}")
        print(f"Speed: {summary['rows_per_second']:.1f} rows/sec")
        print(f"Memory: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        print(f"\n--- Cumulative Statistics ---")
        print(f"Total Processed: {summary['total_rows']:,} rows")
        print(f"  Human: {summary['human_count']:,} ({summary['human_percentage']:.1f}%)")
        print(f"  AI: {summary['ai_count']:,} ({summary['ai_percentage']:.1f}%)")
        
        if self.model_counts:
            print(f"\nTop AI Models:")
            sorted_models = sorted(self.model_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for model, count in sorted_models:
                pct = (count / self.ai_count * 100) if self.ai_count > 0 else 0
                print(f"  - {model}: {count:,} ({pct:.1f}%)")
        
        if 'text_length_stats' in summary:
            stats = summary['text_length_stats']
            print(f"\nText Length: min={stats['min']}, max={stats['max']}, "
                  f"avg={stats['mean']:.0f}, median={stats['median']:.0f}")
        
        print(f"\nDuplicates Removed: {summary['duplicates_removed']:,}")
        print(f"Invalid Rows: {summary['invalid_rows']:,}")
        print("="*70 + "\n")


# ============================================================================
# SCHEMA DETECTION
# ============================================================================

def detect_schema(csv_path: str, sample_rows: int = 1000) -> SchemaInfo:
    """
    Auto-detect schema from CSV or Parquet file using fuzzy column matching.
    
    Args:
        csv_path: Path to input CSV or Parquet file
        sample_rows: Number of rows to sample for detection
        
    Returns:
        SchemaInfo object with detected column mappings
    """
    logging.info(f"Detecting schema from first {sample_rows} rows...")
    
    try:
        # Check file type and read accordingly
        file_ext = os.path.splitext(csv_path)[1].lower()
        df_sample = None
        
        if file_ext == '.parquet':
            # Read Parquet file
            try:
                df_sample = pd.read_parquet(csv_path)
                if len(df_sample) > sample_rows:
                    df_sample = df_sample.head(sample_rows)
                logging.info(f"Successfully read Parquet file")
            except Exception as e:
                raise ValueError(f"Could not read Parquet file: {e}")
        else:
            # Read sample with various encoding attempts for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df_sample = pd.read_csv(csv_path, nrows=sample_rows, encoding=encoding)
                    logging.info(f"Successfully read CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
        
        if df_sample is None:
            raise ValueError("Could not read file with any supported encoding")
        
        columns = df_sample.columns.tolist()
        logging.info(f"Found columns: {columns}")
        
        # Fuzzy pattern matching (ordered by priority - more specific first)
        text_patterns = r'^(generation|text|content|document|passage|output)$'
        label_patterns = r'^(label|class|is_ai|category|type|source)$'
        model_patterns = r'^(model|generator|llm|model_name)$'
        domain_patterns = r'^(domain|genre|task|topic|source)$'
        attack_patterns = r'^(attack|perturbation|adversarial)$'
        
        detected = {
            'text_column': None,
            'label_column': None,
            'model_column': None,
            'domain_column': None,
            'attack_column': None
        }
        
        for col in columns:
            col_lower = col.lower()
            if detected['text_column'] is None and re.search(text_patterns, col_lower):
                detected['text_column'] = col
            elif detected['label_column'] is None and re.search(label_patterns, col_lower):
                detected['label_column'] = col
            elif detected['model_column'] is None and re.search(model_patterns, col_lower):
                detected['model_column'] = col
            elif detected['domain_column'] is None and re.search(domain_patterns, col_lower):
                detected['domain_column'] = col
            elif detected['attack_column'] is None and re.search(attack_patterns, col_lower):
                detected['attack_column'] = col
        
        # Validate required columns
        if detected['text_column'] is None:
            raise ValueError(
                f"Could not detect text column. Available columns: {columns}\n"
                f"Please ensure your CSV has a column matching: {text_patterns}"
            )
        
        # SPECIAL CASE: Check if model column contains "human" - it serves as both label and model
        if detected['label_column'] is None and detected['model_column'] is not None:
            model_values = df_sample[detected['model_column']].dropna().unique()
            # Check if "human" appears in model values (indicating it's also a label column)
            if any(str(val).lower() == 'human' for val in model_values):
                logging.info(f"Model column '{detected['model_column']}' contains 'human' - using as label column too")
                detected['label_column'] = detected['model_column']
                detected['model_column'] = detected['model_column']  # Same column serves both purposes
        
        # For human-only datasets (no label column), use text column as placeholder
        if detected['label_column'] is None:
            logging.warning("No label column detected. If this is a human-only dataset, use --force-human flag.")
            # Use text column as placeholder - will be overridden by force_human flag
            detected['label_column'] = detected['text_column']
        
        # Analyze label format
        label_col = detected['label_column']
        unique_labels = df_sample[label_col].dropna().unique().tolist()
        label_format = "unknown"
        
        if set(unique_labels).issubset({0, 1, '0', '1'}):
            label_format = "binary_numeric"
        elif set(str(x).lower() for x in unique_labels).issubset({'human', 'ai', 'machine'}):
            label_format = "binary_string"
        elif set(str(x).lower() for x in unique_labels).issubset({'true', 'false'}):
            label_format = "binary_boolean"
        else:
            label_format = "multi_class"
        
        # Get unique models if column exists
        unique_models = []
        if detected['model_column']:
            unique_models = df_sample[detected['model_column']].dropna().unique().tolist()[:20]
        
        schema = SchemaInfo(
            text_column=detected['text_column'],
            label_column=detected['label_column'],
            model_column=detected['model_column'],
            domain_column=detected['domain_column'],
            attack_column=detected['attack_column'],
            detected_columns=detected,
            label_format=label_format,
            unique_labels=unique_labels,
            unique_models=unique_models
        )
        
        # Print schema summary
        print("\n" + "="*70)
        print("DETECTED SCHEMA")
        print("="*70)
        print(f"Text Column: {schema.text_column}")
        print(f"Label Column: {schema.label_column} (format: {schema.label_format})")
        print(f"  Unique Labels: {unique_labels[:10]}")
        if schema.model_column:
            print(f"Model Column: {schema.model_column}")
            print(f"  Unique Models: {unique_models[:10]}")
        if schema.domain_column:
            print(f"Domain Column: {schema.domain_column}")
        if schema.attack_column:
            print(f"Attack Column: {schema.attack_column}")
        print("="*70 + "\n")
        
        return schema
        
    except Exception as e:
        logging.error(f"Schema detection failed: {e}")
        raise


def normalize_labels(df: pd.DataFrame, label_col: str, schema_info: SchemaInfo, force_human: bool = False) -> pd.DataFrame:
    """
    Normalize labels to binary format (0=human, 1=ai).
    
    Args:
        df: DataFrame to normalize
        label_col: Name of label column
        schema_info: Detected schema information
        force_human: If True, force all labels to be 'human' (0)
        
    Returns:
        DataFrame with normalized 'label' and 'label_name' columns
    """
    df = df.copy()
    
    # If force_human is True, set all to human
    if force_human:
        df['label'] = 0
        df['label_name'] = 'human'
        return df
    
    # SPECIAL HANDLING: If label_col is same as model_col, preserve original model info
    preserve_model = (schema_info.model_column == label_col)
    if preserve_model and 'model' not in df.columns:
        # Save original model values before normalization
        df['model'] = df[label_col].copy()
    
    if schema_info.label_format == "binary_numeric":
        df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(-1).astype(int)
        df['label_name'] = df['label'].map({0: 'human', 1: 'ai'}).fillna('unknown')
        
    elif schema_info.label_format == "binary_string":
        label_map = {}
        for val in df[label_col].dropna().unique():
            val_lower = str(val).lower()
            if val_lower in ['human', 'person', 'real']:
                label_map[val] = 0
            elif val_lower in ['ai', 'machine', 'generated', 'synthetic']:
                label_map[val] = 1
            else:
                label_map[val] = -1
        
        df['label'] = df[label_col].map(label_map).fillna(-1).astype(int)
        df['label_name'] = df['label'].map({0: 'human', 1: 'ai'}).fillna('unknown')
        
    elif schema_info.label_format == "binary_boolean":
        bool_map = {}
        for val in df[label_col].dropna().unique():
            val_lower = str(val).lower()
            if val_lower in ['true', 't', '1']:
                bool_map[val] = 1
            elif val_lower in ['false', 'f', '0']:
                bool_map[val] = 0
            else:
                bool_map[val] = -1
        
        df['label'] = df[label_col].map(bool_map).fillna(-1).astype(int)
        df['label_name'] = df['label'].map({0: 'human', 1: 'ai'}).fillna('unknown')
        
    elif schema_info.label_format == "multi_class":
        # Convert multi-class to binary: human=0, all others=1
        human_values = ['human', 'person', 'real', 'original']
        label_map = {}
        
        for val in df[label_col].dropna().unique():
            val_lower = str(val).lower()
            if any(hv in val_lower for hv in human_values):
                label_map[val] = 0
            else:
                label_map[val] = 1
        
        df['label'] = df[label_col].map(label_map).fillna(-1).astype(int)
        df['label_name'] = df['label'].map({0: 'human', 1: 'ai'}).fillna('unknown')
    
    return df


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_chunk(df: pd.DataFrame, schema_info: SchemaInfo, config: ProcessingConfig, force_human: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean and validate a chunk of data.
    
    Args:
        df: DataFrame chunk to clean
        schema_info: Schema information
        config: Processing configuration
        force_human: If True, force all labels to be 'human'
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dict)
    """
    initial_rows = len(df)
    stats = {
        'rows_processed': 0,
        'duplicates_removed': 0,
        'invalid_rows': 0,
        'human_count': 0,
        'ai_count': 0,
        'model_counts': defaultdict(int),
        'domain_counts': defaultdict(int),
        'text_lengths': []
    }
    
    text_col = schema_info.text_column
    
    # Ensure text column exists and is string type
    if text_col not in df.columns:
        logging.error(f"Text column '{text_col}' not found in chunk")
        return pd.DataFrame(), stats
    
    # Convert to string and handle NaN
    df[text_col] = df[text_col].astype(str)
    
    # Strip whitespace
    df[text_col] = df[text_col].str.strip()
    
    # Remove null bytes
    df[text_col] = df[text_col].str.replace('\x00', '', regex=False)
    
    # UTF-8 normalization (handle errors)
    try:
        df[text_col] = df[text_col].str.normalize('NFKC')
    except Exception as e:
        logging.warning(f"UTF-8 normalization failed: {e}")
    
    # Drop rows with missing or invalid text
    df = df[df[text_col].notna()]
    df = df[df[text_col] != 'nan']
    df = df[df[text_col] != '']
    
    # Filter by length
    df['text_length'] = df[text_col].str.len()
    df = df[
        (df['text_length'] >= config.min_text_length) & 
        (df['text_length'] <= config.max_text_length)
    ]
    
    # Normalize labels
    df = normalize_labels(df, schema_info.label_column, schema_info, force_human=force_human)
    
    # Fill missing model column
    if schema_info.model_column and schema_info.model_column in df.columns:
        df['model'] = df[schema_info.model_column].fillna('unspecified').astype(str)
    elif 'model' not in df.columns:
        df['model'] = 'unspecified'
    
    # Fill missing domain column
    if schema_info.domain_column and schema_info.domain_column in df.columns:
        df['domain'] = df[schema_info.domain_column].fillna('unspecified').astype(str)
    else:
        df['domain'] = 'unspecified'
    
    # Fill missing attack column
    if schema_info.attack_column and schema_info.attack_column in df.columns:
        df['attack_type'] = df[schema_info.attack_column].fillna('none').astype(str)
    else:
        df['attack_type'] = 'none'
    
    # Remove duplicates
    if config.remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[text_col], keep='first')
        stats['duplicates_removed'] = before_dedup - len(df)
    
    # Calculate statistics
    stats['rows_processed'] = len(df)
    stats['invalid_rows'] = initial_rows - len(df) - stats['duplicates_removed']
    stats['human_count'] = int((df['label'] == 0).sum())
    stats['ai_count'] = int((df['label'] == 1).sum())
    
    # Model counts
    for model in df[df['label'] == 1]['model'].value_counts().items():
        stats['model_counts'][model[0]] = int(model[1])
    
    # Domain counts
    if 'domain' in df.columns:
        for domain in df['domain'].value_counts().items():
            stats['domain_counts'][domain[0]] = int(domain[1])
    
    # Text length samples (limit to avoid memory issues)
    stats['text_lengths'] = df['text_length'].sample(min(1000, len(df))).tolist()
    
    return df, stats


# ============================================================================
# DATA SEGREGATION AND SAVING
# ============================================================================

def segregate_and_save(
    df: pd.DataFrame,
    schema_info: SchemaInfo,
    output_dir: Path,
    chunk_num: int,
    compression: str = 'snappy',
    dataset_name: str = 'RAID-Dataset'
) -> Dict[str, int]:
    """
    Segregate data by label and model, then save to Parquet files.
    
    Args:
        df: DataFrame to segregate and save
        schema_info: Schema information
        output_dir: Output directory path
        chunk_num: Current chunk number
        compression: Compression method
        dataset_name: Name of the dataset folder (default: 'RAID-Dataset')
        
    Returns:
        Dictionary with saved file counts
    """
    saved_files = {'human': 0, 'ai': 0, 'unknown': 0}
    
    if len(df) == 0:
        return saved_files
    
    # Select columns to save
    columns_to_save = [
        schema_info.text_column,
        'label',
        'label_name',
        'model',
        'domain',
        'attack_type',
        'text_length'
    ]
    
    # Add row_id if it exists
    if 'row_id' not in df.columns:
        df['row_id'] = range(len(df))
    columns_to_save.append('row_id')
    
    # Add chunk_id
    df['chunk_id'] = chunk_num
    columns_to_save.append('chunk_id')
    
    # Rename text column to 'text' for consistency
    if schema_info.text_column != 'text':
        df = df.rename(columns={schema_info.text_column: 'text'})
        columns_to_save[0] = 'text'
    
    df_save = df[columns_to_save].copy()
    
    # Split by label
    df_human = df_save[df_save['label'] == 0]
    df_ai = df_save[df_save['label'] == 1]
    df_unknown = df_save[df_save['label'] == -1]
    
    # Save human data: real/{dataset_name}/human/
    if len(df_human) > 0:
        human_dir = output_dir / 'real' / dataset_name / 'human'
        human_dir.mkdir(parents=True, exist_ok=True)
        human_file = human_dir / f'chunk_{chunk_num:04d}.parquet'
        df_human.to_parquet(human_file, compression=compression, index=False, engine='pyarrow')
        saved_files['human'] = len(df_human)
    
    # Save AI data by model: ai/{dataset_name}/{model_name}/
    if len(df_ai) > 0:
        for model_name, model_df in df_ai.groupby('model'):
            # Sanitize model name for filesystem
            safe_model_name = re.sub(r'[^\w\-_]', '_', str(model_name).lower())
            if not safe_model_name:
                safe_model_name = 'unknown_model'
            
            model_dir = output_dir / 'ai' / dataset_name / safe_model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / f'chunk_{chunk_num:04d}.parquet'
            model_df.to_parquet(model_file, compression=compression, index=False, engine='pyarrow')
            saved_files['ai'] += len(model_df)
    
    # Save unknown labels: unknown/{dataset_name}/
    if len(df_unknown) > 0:
        unknown_dir = output_dir / 'unknown' / dataset_name
        unknown_dir.mkdir(parents=True, exist_ok=True)
        unknown_file = unknown_dir / f'chunk_{chunk_num:04d}.parquet'
        df_unknown.to_parquet(unknown_file, compression=compression, index=False, engine='pyarrow')
        saved_files['unknown'] = len(df_unknown)
    
    return saved_files


# ============================================================================
# CHUNK PROCESSOR
# ============================================================================

class ChunkProcessor:
    """Memory-efficient chunk processor with checkpointing."""
    
    def __init__(self, config: ProcessingConfig, schema_info: SchemaInfo):
        self.config = config
        self.schema_info = schema_info
        self.stats_tracker = StatisticsTracker()
        self.checkpoint_file = Path(config.output_dir) / 'checkpoint.json'
        self.last_processed_row = 0
        self.chunks_completed = 0
        
        # Setup logging
        log_file = Path(config.output_dir) / 'processing_log.txt'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Load checkpoint if resuming
        if config.resume and self.checkpoint_file.exists():
            self.load_checkpoint()
    
    def save_checkpoint(self):
        """Save processing checkpoint."""
        checkpoint = {
            'last_processed_row': self.last_processed_row,
            'chunks_completed': self.chunks_completed,
            'statistics': self.stats_tracker.get_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self):
        """Load processing checkpoint."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            self.last_processed_row = checkpoint.get('last_processed_row', 0)
            self.chunks_completed = checkpoint.get('chunks_completed', 0)
            
            logging.info(f"Resumed from checkpoint: {self.chunks_completed} chunks, "
                        f"{self.last_processed_row} rows")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
    
    def process_csv_in_chunks(self) -> Dict[str, Any]:
        """
        Process CSV or Parquet file in chunks with progress tracking.
        
        Returns:
            Final statistics dictionary
        """
        input_path = self.config.input_path
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check file type
        file_ext = os.path.splitext(input_path)[1].lower()
        is_parquet = (file_ext == '.parquet')
        
        if is_parquet:
            # Handle Parquet file
            logging.info("Processing Parquet file...")
            total_rows = None
            
            try:
                # Read entire parquet file (they're typically already compressed and efficient)
                df = pd.read_parquet(input_path)
                total_rows = len(df)
                logging.info(f"Total rows in Parquet file: {total_rows:,}")
                
                # Process in chunks for memory efficiency
                chunk_size = self.config.chunk_size
                num_chunks = (total_rows + chunk_size - 1) // chunk_size
                
                pbar = tqdm(
                    total=total_rows,
                    initial=0,
                    desc="Processing",
                    unit="rows",
                    dynamic_ncols=True
                )
                
                chunk_num = self.chunks_completed
                
                for i in range(0, total_rows, chunk_size):
                    chunk_num += 1
                    chunk_df = df.iloc[i:i+chunk_size].copy()
                    
                    # Clean chunk
                    cleaned_df, chunk_stats = clean_chunk(
                        chunk_df,
                        self.schema_info,
                        self.config,
                        force_human=self.config.force_human
                    )
                    
                    # Segregate and save
                    if len(cleaned_df) > 0:
                        segregate_and_save(
                            cleaned_df,
                            self.schema_info,
                            output_dir,
                            chunk_num,
                            self.config.compression,
                            dataset_name=self.config.dataset_name
                        )
                    
                    # Update statistics
                    self.stats_tracker.update(chunk_stats)
                    self.last_processed_row += len(chunk_df)
                    self.chunks_completed = chunk_num
                    
                    # Update progress bar
                    pbar.update(len(chunk_df))
                    pbar.set_postfix({
                        'Human': f"{self.stats_tracker.human_count:,}",
                        'AI': f"{self.stats_tracker.ai_count:,}",
                        'Speed': f"{self.stats_tracker.get_summary()['rows_per_second']:.0f} r/s"
                    })
                    
                    # Save checkpoint
                    if chunk_num % self.config.checkpoint_interval == 0:
                        self.save_checkpoint()
                    
                    # Memory management
                    del chunk_df, cleaned_df
                    gc.collect()
                
                pbar.close()
                
            except Exception as e:
                logging.error(f"Error processing Parquet file: {e}")
                raise
                
        else:
            # Handle CSV file (original code)
            # Determine encoding
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            encoding_to_use = 'utf-8'
            
            for encoding in encodings:
                try:
                    pd.read_csv(input_path, nrows=10, encoding=encoding)
                    encoding_to_use = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            logging.info(f"Using encoding: {encoding_to_use}")
            
            # Get total rows for progress bar using pandas (faster and more accurate)
            logging.info("Counting total rows (this may take a moment)...")
            total_rows = None
            try:
                # Fast row count using pandas in chunks
                row_count = 0
                for chunk in pd.read_csv(input_path, chunksize=100000, encoding=encoding_to_use, on_bad_lines='skip'):
                    row_count += len(chunk)
                total_rows = row_count
                logging.info(f"Total rows detected: {total_rows:,}")
            except Exception as e:
                logging.warning(f"Could not count total rows: {e}. Progress bar will use incremental count.")
                total_rows = None
            
            # Skip rows if resuming
            skiprows = None
            if self.config.resume and self.last_processed_row > 0:
                skiprows = range(1, self.last_processed_row + 1)
            
            # Process chunks
            chunk_iterator = pd.read_csv(
                input_path,
                chunksize=self.config.chunk_size,
                encoding=encoding_to_use,
                skiprows=skiprows,
                on_bad_lines='skip'
            )
            
            # Progress bar - if total_rows is None, show incremental counter
            if total_rows is not None:
                pbar = tqdm(
                    total=total_rows,
                    initial=self.last_processed_row if self.config.resume else 0,
                    desc="Processing",
                    unit="rows",
                    dynamic_ncols=True
                )
            else:
                # No total - just count up
                pbar = tqdm(
                    desc="Processing",
                    unit="rows",
                    dynamic_ncols=True
                )
            
            chunk_num = self.chunks_completed
            
            try:
                for chunk_df in chunk_iterator:
                    chunk_num += 1
                    
                    # Clean chunk
                    cleaned_df, chunk_stats = clean_chunk(
                        chunk_df,
                        self.schema_info,
                        self.config,
                        force_human=self.config.force_human
                    )
                    
                    # Segregate and save
                    if len(cleaned_df) > 0:
                        segregate_and_save(
                            cleaned_df,
                            self.schema_info,
                            output_dir,
                            chunk_num,
                            self.config.compression,
                            dataset_name=self.config.dataset_name
                        )
                    
                    # Update statistics
                    self.stats_tracker.update(chunk_stats)
                    self.last_processed_row += len(chunk_df)
                    self.chunks_completed = chunk_num
                    
                    # Update progress bar
                    pbar.update(len(chunk_df))
                    pbar.set_postfix({
                        'Human': f"{self.stats_tracker.human_count:,}",
                        'AI': f"{self.stats_tracker.ai_count:,}",
                        'Speed': f"{self.stats_tracker.get_summary()['rows_per_second']:.0f} r/s"
                    })
                    
                    # Save checkpoint
                    if chunk_num % self.config.checkpoint_interval == 0:
                        self.save_checkpoint()
                    
                    # Memory management
                    del chunk_df, cleaned_df
                    gc.collect()
                    
                    # Memory warning
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 80:
                        logging.warning(f"High memory usage: {memory_percent:.1f}%")
            
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_num}: {e}")
                self.save_checkpoint()
                raise
            
            finally:
                pbar.close()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Display final statistics
        self.stats_tracker.display("Processing Complete")
        
        return self.stats_tracker.get_summary()


# ============================================================================
# METADATA GENERATION
# ============================================================================

def generate_metadata(output_dir: Path, statistics: Dict[str, Any], config: ProcessingConfig):
    """
    Generate metadata files for the processed dataset.
    
    Args:
        output_dir: Output directory path
        statistics: Processing statistics
        config: Processing configuration
    """
    logging.info("Generating metadata files...")
    
    # Root statistics file
    root_stats = {
        'total_samples': statistics['total_rows'],
        'human_samples': statistics['human_count'],
        'ai_samples': statistics['ai_count'],
        'ai_model_breakdown': statistics['model_counts'],
        'domain_breakdown': statistics.get('domain_counts', {}),
        'processing_time_seconds': statistics['processing_time'],
        'duplicates_removed': statistics['duplicates_removed'],
        'invalid_rows': statistics['invalid_rows'],
        'chunks_processed': statistics['chunks_processed'],
        'date_processed': datetime.now().isoformat(),
        'source_csv': config.input_path,
        'config': {
            'chunk_size': config.chunk_size,
            'min_text_length': config.min_text_length,
            'max_text_length': config.max_text_length,
            'compression': config.compression,
            'remove_duplicates': config.remove_duplicates
        }
    }
    
    # Calculate file sizes
    try:
        original_size = os.path.getsize(config.input_path) / (1024**3)
        output_size = sum(
            f.stat().st_size for f in output_dir.rglob('*.parquet')
        ) / (1024**3)
        
        root_stats['original_csv_size_gb'] = round(original_size, 2)
        root_stats['output_size_gb'] = round(output_size, 2)
        root_stats['compression_ratio'] = round(output_size / original_size, 3) if original_size > 0 else 0
    except Exception as e:
        logging.warning(f"Could not calculate file sizes: {e}")
    
    # Add text length stats
    if 'text_length_stats' in statistics:
        root_stats['text_length_stats'] = statistics['text_length_stats']
    
    # Save root statistics
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(root_stats, f, indent=2)
    
    # Generate per-subfolder metadata
    for subfolder in output_dir.rglob('*/'):
        if subfolder == output_dir:
            continue
        
        parquet_files = list(subfolder.glob('*.parquet'))
        if not parquet_files:
            continue
        
        # Determine label and model
        rel_path = subfolder.relative_to(output_dir)
        parts = rel_path.parts
        
        if parts[0] == 'human':
            label = 'human'
            model_name = 'human'
        elif parts[0] == 'ai' and len(parts) > 1:
            label = 'ai'
            model_name = parts[1]
        else:
            label = 'unknown'
            model_name = 'unknown'
        
        # Count samples
        total_samples = 0
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                total_samples += len(df)
            except:
                pass
        
        metadata = {
            'label': label,
            'model_name': model_name,
            'total_samples': total_samples,
            'num_chunks': len(parquet_files),
            'date_processed': datetime.now().isoformat(),
            'source_csv': config.input_path,
            'chunk_files': [f.name for f in sorted(parquet_files)]
        }
        
        with open(subfolder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logging.info("Metadata generation complete")


# ============================================================================
# VALIDATION MODE
# ============================================================================

def validate_dataset(csv_path: str, sample_rows: int = 10000) -> Dict[str, Any]:
    """
    Validate dataset without processing (dry-run mode).
    
    Args:
        csv_path: Path to CSV or Parquet file
        sample_rows: Number of rows to sample
        
    Returns:
        Validation report dictionary
    """
    print("\n" + "="*70)
    print("DATASET VALIDATION (DRY RUN)")
    print("="*70 + "\n")
    
    # Detect schema
    schema_info = detect_schema(csv_path, sample_rows=min(sample_rows, 1000))
    
    # Check file type and read sample
    file_ext = os.path.splitext(csv_path)[1].lower()
    
    if file_ext == '.parquet':
        df_sample = pd.read_parquet(csv_path)
        if len(df_sample) > sample_rows:
            df_sample = df_sample.head(sample_rows)
    else:
        df_sample = pd.read_csv(csv_path, nrows=sample_rows)
    
    print(f"Sample Size: {len(df_sample):,} rows\n")
    
    # Show sample data
    print("Sample Data (first 5 rows):")
    print(df_sample[[schema_info.text_column, schema_info.label_column]].head())
    print()
    
    # Analyze labels
    print("Label Distribution:")
    print(df_sample[schema_info.label_column].value_counts())
    print()
    
    # Analyze text lengths
    text_lengths = df_sample[schema_info.text_column].astype(str).str.len()
    print("Text Length Statistics:")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Mean: {text_lengths.mean():.0f}")
    print(f"  Median: {text_lengths.median():.0f}")
    print()
    
    # Check for issues
    issues = []
    
    # Missing values
    missing = df_sample[schema_info.text_column].isna().sum()
    if missing > 0:
        issues.append(f"Found {missing} missing text values")
    
    # Empty strings
    empty = (df_sample[schema_info.text_column].astype(str).str.strip() == '').sum()
    if empty > 0:
        issues.append(f"Found {empty} empty text strings")
    
    # Very short texts
    very_short = (text_lengths < 10).sum()
    if very_short > 0:
        issues.append(f"Found {very_short} texts shorter than 10 characters")
    
    # Duplicates
    duplicates = df_sample[schema_info.text_column].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate texts in sample")
    
    if issues:
        print("⚠ Potential Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("✓ No obvious issues detected\n")
    
    # Estimate processing time
    file_size = os.path.getsize(csv_path) / (1024**3)  # GB
    
    # For Parquet files, we can get exact row count
    if file_ext == '.parquet':
        df_full = pd.read_parquet(csv_path)
        estimated_rows = len(df_full)
    else:
        estimated_rows = int(file_size * (len(df_sample) / (sys.getsizeof(df_sample) / (1024**3))))
    
    estimated_time = estimated_rows / 10000  # Assume 10k rows/sec
    
    print(f"File Size: {file_size:.2f} GB")
    print(f"{'Actual' if file_ext == '.parquet' else 'Estimated'} Total Rows: {estimated_rows:,}")
    print(f"Estimated Processing Time: {estimated_time/60:.1f} minutes")
    print(f"Estimated Output Size: {file_size * 0.2:.2f} GB (assuming 20% compression)")
    print()
    
    print("="*70)
    print("Validation complete. Run without --validate-only to process the dataset.")
    print("="*70 + "\n")
    
    return {
        'schema': asdict(schema_info),
        'sample_size': len(df_sample),
        'estimated_rows': estimated_rows,
        'issues': issues
    }


# ============================================================================
# POST-PROCESSING VERIFICATION
# ============================================================================

def verify_output(output_dir: Path, expected_total_rows: int) -> bool:
    """
    Verify output files after processing.
    
    Args:
        output_dir: Output directory path
        expected_total_rows: Expected total number of rows
        
    Returns:
        True if verification passed
    """
    logging.info("Verifying output files...")
    
    total_rows = 0
    corrupted_files = []
    
    for parquet_file in output_dir.rglob('*.parquet'):
        try:
            df = pd.read_parquet(parquet_file)
            total_rows += len(df)
        except Exception as e:
            logging.error(f"Corrupted file: {parquet_file} - {e}")
            corrupted_files.append(str(parquet_file))
    
    print("\n" + "="*70)
    print("OUTPUT VERIFICATION")
    print("="*70)
    print(f"Expected Rows: {expected_total_rows:,}")
    print(f"Actual Rows: {total_rows:,}")
    print(f"Difference: {expected_total_rows - total_rows:,}")
    
    if corrupted_files:
        print(f"\n⚠ Found {len(corrupted_files)} corrupted files:")
        for cf in corrupted_files:
            print(f"  - {cf}")
    else:
        print("\n✓ All files readable")
    
    print("="*70 + "\n")
    
    return len(corrupted_files) == 0


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_final_summary(statistics: Dict[str, Any], output_dir: Path):
    """Print final processing summary."""
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    
    elapsed = statistics['processing_time']
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"Total Time: {hours}h {minutes}m {seconds}s")
    print(f"Input Rows: {statistics['total_rows']:,}")
    print(f"  Human: {statistics['human_count']:,} ({statistics['human_percentage']:.1f}%)")
    print(f"  AI: {statistics['ai_count']:,} ({statistics['ai_percentage']:.1f}%)")
    
    if 'original_csv_size_gb' in statistics:
        print(f"\nOriginal Size: {statistics['original_csv_size_gb']:.2f} GB")
        print(f"Output Size: {statistics['output_size_gb']:.2f} GB")
        print(f"Compression: {(1 - statistics['compression_ratio']) * 100:.1f}% reduction")
    
    print(f"\nDuplicates Removed: {statistics['duplicates_removed']:,}")
    print(f"Invalid Rows: {statistics['invalid_rows']:,}")
    print(f"Processing Speed: {statistics['rows_per_second']:.0f} rows/sec")
    
    print("\n" + "-"*70)
    print("Suggested Next Steps:")
    print("-"*70)
    print("1. Load dataset:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('parquet', data_dir='{output_dir}')")
    print("\n2. Create train/val/test splits (80/10/10)")
    print("\n3. Train classifier with balanced sampling")
    print("="*70 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Text Detection Dataset Preprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=100000, 
                       help='Number of rows per chunk (default: 100000)')
    parser.add_argument('--min-text-length', type=int, default=10,
                       help='Minimum text length (default: 10)')
    parser.add_argument('--max-text-length', type=int, default=100000,
                       help='Maximum text length (default: 100000)')
    parser.add_argument('--compression', choices=['snappy', 'gzip'], default='snappy',
                       help='Compression method (default: snappy)')
    parser.add_argument('--remove-duplicates', action='store_true', default=True,
                       help='Remove duplicate texts (default: True)')
    parser.add_argument('--no-remove-duplicates', dest='remove_duplicates', action='store_false',
                       help='Keep duplicate texts')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                       help='Analyze schema without processing')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of I/O threads (default: 4)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N chunks (default: 10)')
    parser.add_argument('--dataset-name', type=str, default='RAID-Dataset',
                       help='Dataset folder name (default: RAID-Dataset)')
    parser.add_argument('--force-human', action='store_true',
                       help='Force all labels to be human (0) - use for human-only datasets')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Check disk space
    stat = shutil.disk_usage(os.path.dirname(args.output) or '.')
    available_gb = stat.free / (1024**3)
    input_size_gb = os.path.getsize(args.input) / (1024**3)
    required_gb = input_size_gb * 0.25
    
    print(f"\nDisk Space Check:")
    print(f"  Available: {available_gb:.2f} GB")
    print(f"  Required (estimated): {required_gb:.2f} GB")
    
    if available_gb < required_gb:
        print(f"\n⚠ Warning: May not have enough disk space!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    print()
    
    # Create config
    config = ProcessingConfig(
        input_path=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        compression=args.compression,
        remove_duplicates=args.remove_duplicates,
        resume=args.resume,
        validate_only=args.validate_only,
        threads=args.threads,
        checkpoint_interval=args.checkpoint_interval,
        dataset_name=args.dataset_name,
        force_human=args.force_human
    )
    
    # Detect schema
    try:
        schema_info = detect_schema(config.input_path)
    except Exception as e:
        print(f"\nError detecting schema: {e}")
        sys.exit(1)
    
    # Validation mode
    if config.validate_only:
        validate_dataset(config.input_path)
        sys.exit(0)
    
    # Process dataset
    try:
        processor = ChunkProcessor(config, schema_info)
        statistics = processor.process_csv_in_chunks()
        
        # Generate metadata
        output_path = Path(config.output_dir)
        generate_metadata(output_path, statistics, config)
        
        # Verify output
        verify_output(output_path, statistics['total_rows'])
        
        # Print summary
        print_final_summary(statistics, output_path)
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user. Progress saved to checkpoint.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
