# AI Text Detection Dataset Preprocessor

## Overview

Build a production-grade Python script (`preprocess.py`) that processes large CSV datasets (millions of rows) with automatic schema detection, memory-efficient chunked processing, intelligent data segregation, and Parquet output optimization.

## Implementation Plan

### 1. Core Script Structure & Dependencies

**File: `preprocess.py`**

Dependencies to install:

- `pandas` - CSV reading and data manipulation
- `pyarrow` - Parquet file I/O with compression
- `tqdm` - Progress bars and ETA tracking
- `psutil` - Memory monitoring
- `argparse` - CLI argument parsing

### 2. Schema Auto-Detection Module

**Function: `detect_schema(csv_path, sample_rows=1000)`**

- Read first 1000 rows using `pd.read_csv(nrows=1000)`
- Implement fuzzy column matching using pattern matching:
    - Text columns: search for `text|content|document|generation` (case-insensitive)
    - Label columns: `label|class|is_ai|source` 
    - Model columns: `model|generator|llm|source_model`
    - Domain columns: `domain|genre|category|task`
    - Attack columns: `attack|perturbation`
- Analyze data types with `df.dtypes`
- For categorical columns, get unique values and counts
- Return a `SchemaInfo` dataclass with detected column mappings
- Print formatted schema summary to console
- Validate that text and label columns exist, exit gracefully if missing

**Function: `normalize_labels(df, label_col)`**

- Handle multiple label formats:
    - Binary: `{0, 1}` → keep as is
    - String binary: `{human, ai}`, `{Human, AI}`, `{true, false}` → map to `{0, 1}`
    - Multi-class: model names like `gpt4`, `claude`, `human` → create binary (human=0, others=1) + keep original in `model` column
- Return normalized DataFrame with standardized `label` column (0=human, 1=ai)

### 3. Memory-Efficient Chunk Processor

**Class: `ChunkProcessor`**

Properties:

- `chunk_size`: rows per chunk (default 100K)
- `checkpoint_interval`: save progress every N chunks
- `processed_rows`: counter
- `statistics`: running stats dictionary

**Method: `process_csv_in_chunks()`**

- Use `pd.read_csv(chunksize=chunk_size)` iterator
- For each chunk:

    1. Clean and validate data
    2. Segregate by label/model
    3. Write to appropriate Parquet files
    4. Update statistics
    5. Call `gc.collect()` explicitly
    6. Log memory usage with `psutil.Process().memory_info().rss`
    7. Update progress bar with `tqdm`
    8. Save checkpoint every N chunks

**Checkpoint Format:**

```python
{
  "last_processed_row": 500000,
  "chunks_completed": 5,
  "statistics": {...}
}
```

### 4. Data Cleaning Pipeline

**Function: `clean_chunk(df, schema_info, config)`**

Operations:

- Strip whitespace: `df[text_col] = df[text_col].str.strip()`
- Remove null bytes: `df[text_col] = df[text_col].str.replace('\x00', '')`
- UTF-8 normalization: `df[text_col].str.normalize('NFKC')`
- Drop rows with missing text: `df.dropna(subset=[text_col])`
- Fill missing labels with "unknown"
- Fill missing models with "unspecified"
- Filter by length: drop rows where `text_col.str.len() < min_length` or `> max_length`
- Remove exact duplicates: `df.drop_duplicates(subset=[text_col], keep='first')`
- Add computed columns:
    - `text_length`: character count
    - `chunk_id`: current chunk number
    - `row_id`: original row index

Return cleaned DataFrame + stats dict (duplicates removed, invalid rows, etc.)

### 5. Smart Segregation System

**Function: `segregate_and_save(df, schema_info, output_dir, chunk_num)`**

Logic:

- Split DataFrame by label: `df_human = df[df['label'] == 0]`, `df_ai = df[df['label'] == 1]`
- For human data: save to `output_dir/human/chunk_{chunk_num:04d}.parquet`
- For AI data: group by model column, then save each to `output_dir/ai/{model_name}/chunk_{chunk_num:04d}.parquet`
- Handle edge cases:
    - Unknown models → `output_dir/ai/unknown_model/`
    - Unknown labels → `output_dir/unknown/`

**Parquet Schema:**

```python
{
  'text': pa.string(),
  'label': pa.int32(),  # 0=human, 1=ai
  'label_name': pa.string(),  # 'human' or 'ai'
  'model': pa.string(),
  'domain': pa.string(),  # if exists
  'attack_type': pa.string(),  # if exists
  'text_length': pa.int32(),
  'chunk_id': pa.int32(),
  'row_id': pa.int64()
}
```

**Parquet Write Settings:**

- Compression: `compression='snappy'` (default) or `'gzip'` (CLI option)
- Use `df.to_parquet()` with `engine='pyarrow'`

### 6. Real-Time Analytics Display

**Class: `StatisticsTracker`**

Track cumulative statistics:

- Total rows processed
- Human vs AI distribution (counts + percentages)
- Model breakdown (dict with counts)
- Domain distribution (if column exists)
- Text length stats (min/max/mean/median)
- Duplicates removed
- Invalid rows skipped
- Processing speed (rows/sec)
- Disk usage (original CSV size vs output size)

**Display Format (using tqdm and custom print):**

```
Processing Chunk 47/98 [████████████░░░░] 48% | 2.3M/4.9M rows
Current: AI-GPT4 | Speed: 12.3K rows/sec | ETA: 6m 32s
Memory: 4.2GB/32GB | Disk Written: +1.8GB

--- Cumulative Statistics ---
Total Processed: 2,345,678 rows
  Human: 145,234 (6.2%)
  AI: 2,200,444 (93.8%)

Top AI Models:
  - GPT-4: 523,419 (23.8%)
  - GPT-3.5: 612,358 (27.8%)
  - Claude: 445,123 (20.2%)

Text Length: min=12, max=98,432, avg=487
Duplicates Removed: 12,457
Invalid Rows: 234
```

Update after each chunk using `tqdm.write()` for clean output.

### 7. Metadata Generation

**Function: `generate_metadata(output_dir, statistics)`**

Create three metadata files:

**Per-subfolder `metadata.json`** (in each model folder):

```json
{
  "label": "ai",
  "model_name": "gpt4",
  "total_samples": 523419,
  "num_chunks": 6,
  "avg_text_length": 487,
  "domains": {"arxiv": 89231, "reddit": 145002},
  "date_processed": "2025-10-18T10:30:00Z",
  "source_csv": "input.csv",
  "chunk_files": ["chunk_0001.parquet", "chunk_0002.parquet"]
}
```

**Root `statistics.json`**:

```json
{
  "total_samples": 4900000,
  "human_samples": 149710,
  "ai_samples": 4750290,
  "ai_model_breakdown": {"gpt4": 523419, "gpt3.5": 612358},
  "processing_time_seconds": 3847,
  "original_csv_size_gb": 12.4,
  "output_size_gb": 2.1,
  "compression_ratio": 0.169,
  "duplicates_removed": 12457,
  "invalid_rows": 234,
  "config": {"chunk_size": 100000, "min_text_length": 10}
}
```

**Root `processing_log.txt`**: Timestamped log entries for debugging

### 8. CLI Interface & Configuration

**Arguments:**

```python
parser.add_argument('--input', required=True, help='Input CSV file path')
parser.add_argument('--output', required=True, help='Output directory')
parser.add_argument('--chunk-size', type=int, default=100000)
parser.add_argument('--min-text-length', type=int, default=10)
parser.add_argument('--max-text-length', type=int, default=100000)
parser.add_argument('--compression', choices=['snappy', 'gzip'], default='snappy')
parser.add_argument('--remove-duplicates', action='store_true', default=True)
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--validate-only', action='store_true', help='Analyze schema without processing')
parser.add_argument('--threads', type=int, default=4, help='Number of I/O threads')
```

### 9. Validation Mode

**Function: `validate_dataset(csv_path)`**

Dry-run functionality:

- Read first 10K rows
- Run schema detection
- Print detected schema with sample values (5 examples per category)
- Calculate statistics preview
- Estimate processing time based on file size and sample processing speed
- Estimate output size (assume 20% of CSV size with Parquet compression)
- Check for potential issues:
    - Encoding problems
    - Missing critical columns
    - Unusual data patterns
- Print warnings and exit without processing

### 10. Multi-Threaded I/O (Optional Optimization)

**Implementation with `ThreadPoolExecutor`:**

- Use 2 threads: one for reading CSV chunks, one for writing Parquet files
- Create queue-based system:
    - Reader thread: reads chunk → adds to processing queue
    - Main thread: processes chunk (clean/validate)
    - Writer thread: writes processed chunks to Parquet files
- Implement using `concurrent.futures.ThreadPoolExecutor` with max_workers=4
- Use `queue.Queue` for inter-thread communication
- Add proper exception handling and thread synchronization

**Dynamic chunk sizing:**

- Monitor memory with `psutil`
- If RAM > 20GB: reduce next chunk to 50K rows
- If RAM < 10GB: increase next chunk to 150K rows
- Update chunk size dynamically between iterations

### 11. Error Handling & Recovery

**Error Tracking:**

- Wrap chunk processing in try-except blocks
- Log errors to `processing_log.txt` with timestamps
- Save problematic rows to `errors.csv` with error descriptions
- Continue processing even if individual rows fail

**Pre-flight Checks:**

- Verify input CSV exists and is readable
- Check available disk space: `shutil.disk_usage()`
- Estimate required space (25% of CSV size)
- Exit gracefully if insufficient space

**Memory Warnings:**

- Monitor RAM usage each chunk
- Warn if > 80% of available RAM used
- Suggest reducing chunk size

### 12. Post-Processing Verification

**Function: `verify_output(output_dir, expected_total_rows)`**

After completion:

- Iterate through all Parquet files
- Verify each can be read: `pd.read_parquet(file)`
- Count total rows across all files
- Compare with input row count (accounting for removed duplicates/invalid rows)
- Print discrepancy report if mismatch

**Final Summary Report:**

```
=== Processing Complete ===
Total Time: 1h 4m 7s
Input: 4,900,000 rows (12.4 GB)
Output: 4,887,309 rows (2.1 GB)
Compression: 83.1% reduction
Duplicates: 12,457 removed
Invalid: 234 skipped

Suggested Next Steps:
1. Load dataset: datasets.load_dataset('parquet', data_files='processed_data/**/chunk_*.parquet')
2. Create train/val/test splits (80/10/10)
3. Train DeBERTa-v3-Small with balanced sampling
```

**Optional: HuggingFace Dataset Conversion:**

Provide helper function to convert to HuggingFace format:

```python
from datasets import load_dataset
dataset = load_dataset('parquet', data_files='processed_data/**/chunk_*.parquet')
dataset.save_to_disk('processed_data/hf_dataset')
```

## File Structure

```
/home/lightdesk/Projects/Text/
├── preprocess.py          # Main script (all-in-one)
├── requirements.txt       # Dependencies
└── README.md             # Usage instructions
```

## Implementation Notes

- Keep code modular with clear function separation
- Add comprehensive docstrings for all functions
- Use type hints for better IDE support
- Include logging throughout for debugging
- Follow PEP 8 style guidelines
- Add input validation for all user-provided parameters
- Make code robust to handle edge cases (empty chunks, single-row datasets, etc.)