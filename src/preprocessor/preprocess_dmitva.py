#!/usr/bin/env python3
"""
Dmitva Dataset Preprocessor for AI Text Detection

Specialized preprocessor for the dmitva/human_ai_generated_text dataset format.
This dataset has a unique paired structure with separate human_text and ai_text columns.

Dataset: https://huggingface.co/datasets/dmitva/human_ai_generated_text
Format: id, human_text, ai_text, instructions

Processing Strategy:
- Each row contains BOTH human-written and AI-generated text
- Unpivot to create 2 samples per row:
  * human_text → label=0, model='human'
  * ai_text → label=1, model='ai_generated'
- Output in RAID-Dataset format matching existing processed data

Author: AI Assistant
Date: 2025-10-18
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import psutil
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def unpivot_chunk(chunk_df: pd.DataFrame, chunk_num: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert paired format chunk to standard format.
    Each input row → 2 output rows (1 human + 1 AI).
    
    Args:
        chunk_df: Input chunk with human_text and ai_text columns
        chunk_num: Current chunk number
        
    Returns:
        Tuple of (unpivoted DataFrame, statistics dict)
    """
    rows_human = []
    rows_ai = []
    
    for idx, row in chunk_df.iterrows():
        row_id = row.get('id', f'row_{chunk_num}_{idx}')
        instructions = str(row.get('instructions', 'unspecified'))
        
        # Human text sample
        human_text = str(row.get('human_text', '')).strip()
        if human_text and human_text != 'nan' and len(human_text) >= 10:
            rows_human.append({
                'text': human_text,
                'label': 0,
                'label_name': 'human',
                'model': 'human',
                'domain': instructions,
                'attack_type': 'none',
                'text_length': len(human_text),
                'row_id': f"{row_id}_human",
                'chunk_id': chunk_num
            })
        
        # AI text sample
        ai_text = str(row.get('ai_text', '')).strip()
        if ai_text and ai_text != 'nan' and len(ai_text) >= 10:
            rows_ai.append({
                'text': ai_text,
                'label': 1,
                'label_name': 'ai',
                'model': 'ai_generated',
                'domain': instructions,
                'attack_type': 'none',
                'text_length': len(ai_text),
                'row_id': f"{row_id}_ai",
                'chunk_id': chunk_num
            })
    
    # Combine
    all_rows = rows_human + rows_ai
    df_unpivoted = pd.DataFrame(all_rows)
    
    # Statistics
    stats = {
        'input_rows': len(chunk_df),
        'output_rows': len(df_unpivoted),
        'human_count': len(rows_human),
        'ai_count': len(rows_ai),
        'text_lengths': df_unpivoted['text_length'].tolist()[:1000]  # Sample
    }
    
    return df_unpivoted, stats


def save_segregated_data(
    df: pd.DataFrame,
    output_dir: Path,
    chunk_num: int,
    compression: str = 'snappy'
) -> Dict[str, int]:
    """
    Save data segregated by label (human/ai) in dmitva-dataset format.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        chunk_num: Chunk number
        compression: Compression method
        
    Returns:
        Dictionary with saved counts
    """
    saved = {'human': 0, 'ai': 0}
    
    if len(df) == 0:
        return saved
    
    # Split by label
    df_human = df[df['label'] == 0].copy()
    df_ai = df[df['label'] == 1].copy()
    
    # Save human data: real/dmitva-dataset/human/
    if len(df_human) > 0:
        human_dir = output_dir / 'real' / 'dmitva-dataset' / 'human'
        human_dir.mkdir(parents=True, exist_ok=True)
        human_file = human_dir / f'chunk_{chunk_num:04d}.parquet'
        df_human.to_parquet(human_file, compression=compression, index=False, engine='pyarrow')
        saved['human'] = len(df_human)
        logger.info(f"Saved {len(df_human)} human samples to {human_file.name}")
    
    # Save AI data: ai/dmitva-dataset/ai_generated/
    if len(df_ai) > 0:
        ai_dir = output_dir / 'ai' / 'dmitva-dataset' / 'ai_generated'
        ai_dir.mkdir(parents=True, exist_ok=True)
        ai_file = ai_dir / f'chunk_{chunk_num:04d}.parquet'
        df_ai.to_parquet(ai_file, compression=compression, index=False, engine='pyarrow')
        saved['ai'] = len(df_ai)
        logger.info(f"Saved {len(df_ai)} AI samples to {ai_file.name}")
    
    return saved


def process_dataset(
    input_path: str,
    output_dir: str,
    chunk_size: int = 50000,
    remove_duplicates: bool = True,
    compression: str = 'snappy'
) -> Dict[str, Any]:
    """
    Process the dmitva dataset.
    
    Args:
        input_path: Path to input CSV
        output_dir: Output directory
        chunk_size: Rows per chunk
        remove_duplicates: Whether to remove duplicates
        compression: Compression method
        
    Returns:
        Processing statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_path / 'processing_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Processing dmitva dataset: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Chunk size: {chunk_size}")
    
    # Statistics
    stats = {
        'total_input_rows': 0,
        'total_output_rows': 0,
        'human_count': 0,
        'ai_count': 0,
        'duplicates_removed': 0,
        'chunks_processed': 0,
        'text_lengths': [],
        'start_time': time.time()
    }
    
    # Track seen texts for deduplication
    seen_texts = set() if remove_duplicates else None
    
    # Count total rows for progress bar
    logger.info("Counting total rows...")
    total_rows = 0
    try:
        for chunk in pd.read_csv(input_path, chunksize=100000, on_bad_lines='skip'):
            total_rows += len(chunk)
        logger.info(f"Total rows: {total_rows:,}")
    except Exception as e:
        logger.warning(f"Could not count rows: {e}")
        total_rows = None
    
    # Process chunks
    chunk_iterator = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        on_bad_lines='skip',
        encoding='utf-8'
    )
    
    pbar = tqdm(
        total=total_rows,
        desc="Processing",
        unit="rows",
        dynamic_ncols=True
    )
    
    chunk_num = 0
    
    try:
        for chunk_df in chunk_iterator:
            chunk_num += 1
            
            # Unpivot
            df_unpivoted, chunk_stats = unpivot_chunk(chunk_df, chunk_num)
            
            # Remove duplicates if requested
            if remove_duplicates and len(df_unpivoted) > 0:
                before = len(df_unpivoted)
                
                # Filter out seen texts
                if seen_texts is not None:
                    df_unpivoted = df_unpivoted[~df_unpivoted['text'].isin(seen_texts)]
                    # Add new texts to seen set
                    seen_texts.update(df_unpivoted['text'].tolist())
                
                # Remove duplicates within chunk
                df_unpivoted = df_unpivoted.drop_duplicates(subset=['text'], keep='first')
                
                after = len(df_unpivoted)
                stats['duplicates_removed'] += (before - after)
            
            # Save
            saved_counts = save_segregated_data(
                df_unpivoted,
                output_path,
                chunk_num,
                compression
            )
            
            # Update statistics
            stats['total_input_rows'] += len(chunk_df)
            stats['total_output_rows'] += len(df_unpivoted)
            stats['human_count'] += saved_counts['human']
            stats['ai_count'] += saved_counts['ai']
            stats['chunks_processed'] = chunk_num
            stats['text_lengths'].extend(chunk_stats['text_lengths'])
            
            # Update progress
            pbar.update(len(chunk_df))
            pbar.set_postfix({
                'Human': f"{stats['human_count']:,}",
                'AI': f"{stats['ai_count']:,}",
                'Total': f"{stats['total_output_rows']:,}"
            })
            
            # Memory cleanup
            del chunk_df, df_unpivoted
            gc.collect()
            
            # Memory warning
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
    
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_num}: {e}", exc_info=True)
        raise
    
    finally:
        pbar.close()
    
    # Calculate final statistics
    stats['processing_time'] = time.time() - stats['start_time']
    stats['rows_per_second'] = stats['total_output_rows'] / stats['processing_time']
    
    if stats['text_lengths']:
        stats['text_length_stats'] = {
            'min': min(stats['text_lengths']),
            'max': max(stats['text_lengths']),
            'mean': sum(stats['text_lengths']) / len(stats['text_lengths']),
            'median': sorted(stats['text_lengths'])[len(stats['text_lengths']) // 2]
        }
        del stats['text_lengths']  # Remove large list from saved stats
    
    # Save statistics
    stats_file = output_path / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_samples': stats['total_output_rows'],
            'human_samples': stats['human_count'],
            'ai_samples': stats['ai_count'],
            'ai_model_breakdown': {'ai_generated': stats['ai_count']},
            'processing_time_seconds': stats['processing_time'],
            'duplicates_removed': stats['duplicates_removed'],
            'chunks_processed': stats['chunks_processed'],
            'date_processed': datetime.now().isoformat(),
            'source_csv': input_path,
            'text_length_stats': stats.get('text_length_stats', {}),
            'config': {
                'chunk_size': chunk_size,
                'remove_duplicates': remove_duplicates,
                'compression': compression
            }
        }, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_file}")
    
    # Generate metadata files
    generate_metadata(output_path, stats, input_path)
    
    return stats


def generate_metadata(output_dir: Path, statistics: Dict[str, Any], source_csv: str):
    """
    Generate metadata files for each category.
    
    Args:
        output_dir: Output directory
        statistics: Processing statistics
        source_csv: Source CSV path
    """
    logger.info("Generating metadata files...")
    
    # Human metadata
    human_dir = output_dir / 'real' / 'dmitva-dataset' / 'human'
    if human_dir.exists():
        parquet_files = sorted(list(human_dir.glob('*.parquet')))
        if parquet_files:
            metadata = {
                'label': 'human',
                'model_name': 'human',
                'total_samples': statistics['human_count'],
                'num_chunks': len(parquet_files),
                'date_processed': datetime.now().isoformat(),
                'source_csv': source_csv,
                'chunk_files': [f.name for f in parquet_files]
            }
            with open(human_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Human metadata: {len(parquet_files)} chunks, {statistics['human_count']:,} samples")
    
    # AI metadata
    ai_dir = output_dir / 'ai' / 'dmitva-dataset' / 'ai_generated'
    if ai_dir.exists():
        parquet_files = sorted(list(ai_dir.glob('*.parquet')))
        if parquet_files:
            metadata = {
                'label': 'ai',
                'model_name': 'ai_generated',
                'total_samples': statistics['ai_count'],
                'num_chunks': len(parquet_files),
                'date_processed': datetime.now().isoformat(),
                'source_csv': source_csv,
                'chunk_files': [f.name for f in parquet_files]
            }
            with open(ai_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"AI metadata: {len(parquet_files)} chunks, {statistics['ai_count']:,} samples")


def print_summary(statistics: Dict[str, Any], output_dir: Path):
    """Print final processing summary."""
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - DMITVA DATASET")
    print("="*70)
    
    elapsed = statistics['processing_time']
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"Time: {hours}h {minutes}m {seconds}s")
    print(f"Speed: {statistics['rows_per_second']:.0f} output rows/sec")
    print()
    print(f"Input Rows: {statistics['total_input_rows']:,}")
    print(f"Output Samples: {statistics['total_output_rows']:,}")
    print(f"  Human: {statistics['human_count']:,} ({statistics['human_count']/statistics['total_output_rows']*100:.1f}%)")
    print(f"  AI: {statistics['ai_count']:,} ({statistics['ai_count']/statistics['total_output_rows']*100:.1f}%)")
    print()
    print(f"Duplicates Removed: {statistics['duplicates_removed']:,}")
    
    if 'text_length_stats' in statistics:
        stats = statistics['text_length_stats']
        print(f"\nText Length: min={stats['min']}, max={stats['max']}, "
              f"avg={stats['mean']:.0f}, median={stats['median']:.0f}")
    
    print("\n" + "-"*70)
    print("Output Structure:")
    print("-"*70)
    print(f"{output_dir}/")
    print("  ├── ai/dmitva-dataset/ai_generated/")
    print(f"  │   └── chunk_*.parquet ({statistics['ai_count']:,} samples)")
    print("  └── real/dmitva-dataset/human/")
    print(f"      └── chunk_*.parquet ({statistics['human_count']:,} samples)")
    print()
    print("Next steps:")
    print("  1. Verify data with: python scripts/validate_processed_data.py")
    print("  2. Train model with: python src/training/train.py")
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dmitva Dataset Preprocessor for AI Text Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=50000,
                       help='Number of rows per chunk (default: 50000)')
    parser.add_argument('--compression', choices=['snappy', 'gzip'], default='snappy',
                       help='Compression method (default: snappy)')
    parser.add_argument('--remove-duplicates', action='store_true', default=True,
                       help='Remove duplicate texts (default: True)')
    parser.add_argument('--no-remove-duplicates', dest='remove_duplicates', action='store_false',
                       help='Keep duplicate texts')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("DMITVA DATASET PREPROCESSOR")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Remove duplicates: {args.remove_duplicates}")
    print(f"Compression: {args.compression}")
    print("="*70 + "\n")
    
    # Process
    try:
        stats = process_dataset(
            input_path=args.input,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            remove_duplicates=args.remove_duplicates,
            compression=args.compression
        )
        
        print_summary(stats, Path(args.output))
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error("Processing failed", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
