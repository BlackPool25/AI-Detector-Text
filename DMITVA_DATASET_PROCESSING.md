# Dmitva Dataset Processing Guide

## Overview

Successfully created a specialized preprocessor for the **dmitva/human_ai_generated_text** dataset from Hugging Face.

**Dataset Source**: https://huggingface.co/datasets/dmitva/human_ai_generated_text  
**Original Size**: 1,000,000 rows (3.66 GB)  
**Output**: ~1.69M samples after unpivoting and deduplication

---

## Dataset Format

### Input Format (Unique Paired Structure)
The dmitva dataset has a unique structure where each row contains BOTH human-written and AI-generated text:

```csv
id,human_text,ai_text,instructions
uuid,"Human written text...","AI generated text...","Task description..."
```

### Key Characteristics
- **Paired Format**: Each row = 1 human text + 1 AI text
- **Instructions**: Task/prompt given to both human and AI
- **Total Rows**: 1,000,000 input rows → 2,000,000 potential samples
- **After Deduplication**: ~1.69M samples (310K duplicates removed)

---

## Processing Solution

### Created Specialized Script
`src/preprocessor/preprocess_dmitva.py` - A dedicated processor for this unique format

### Processing Strategy
1. **Unpivot**: Each input row becomes 2 output rows
   - Row 1: `human_text` → `label=0`, `model='human'`
   - Row 2: `ai_text` → `label=1`, `model='ai_generated'`

2. **Standardize**: Convert to training-ready format with columns:
   - `text`: The actual text content
   - `label`: 0 (human) or 1 (AI)
   - `label_name`: 'human' or 'ai'
   - `model`: 'human' or 'ai_generated'
   - `domain`: Preserved from 'instructions' column
   - `attack_type`: 'none'
   - `text_length`: Character count
   - `row_id`: Unique identifier
   - `chunk_id`: Processing chunk number

3. **Segregate**: Save in organized directory structure
   ```
   processed_data/
   ├── ai/dmitva-dataset/ai_generated/
   │   └── chunk_*.parquet
   └── real/dmitva-dataset/human/
       └── chunk_*.parquet
   ```

---

## Usage

### Basic Usage
```bash
python src/preprocessor/preprocess_dmitva.py \
    --input data/raw/model_training_dataset.csv \
    --output processed_data \
    --chunk-size 50000 \
    --remove-duplicates
```

### Options
- `--input`: Path to the dmitva CSV file
- `--output`: Output directory (will create subdirectories)
- `--chunk-size`: Rows per processing chunk (default: 50000)
- `--compression`: 'snappy' or 'gzip' (default: snappy)
- `--remove-duplicates`: Remove duplicate texts (default: True)
- `--no-remove-duplicates`: Keep all texts including duplicates

### Processing Stats (1M Row Dataset)
- **Processing Time**: ~1 minute 20 seconds
- **Speed**: ~21,000 output rows/sec
- **Memory**: Efficient chunked processing
- **Output Size**: ~310 MB (with snappy compression)

---

## Output Structure

### Directory Layout
```
processed_data/
├── statistics.json                      # Overall dataset stats
├── processing_log.txt                   # Detailed processing log
├── ai/
│   └── dmitva-dataset/
│       └── ai_generated/
│           ├── chunk_0001.parquet
│           ├── chunk_0002.parquet
│           ├── ...
│           └── metadata.json
└── real/
    └── dmitva-dataset/
        └── human/
            ├── chunk_0001.parquet
            ├── chunk_0002.parquet
            ├── ...
            └── metadata.json
```

### Sample Statistics (1M input rows)
```
Input Rows: 1,000,000
Output Samples: 1,689,937
  Human: 817,018 (48.3%)
  AI: 872,919 (51.7%)

Duplicates Removed: 310,063
Text Length: min=239, max=6044, avg=2332, median=2162
```

---

## Verification

### Test Script Functionality
```bash
# Create small test sample
head -1001 data/raw/model_training_dataset.csv > /tmp/test_sample.csv

# Process test sample
python src/preprocessor/preprocess_dmitva.py \
    --input /tmp/test_sample.csv \
    --output /tmp/test_output \
    --chunk-size 500
```

### Verify Parquet Files
```python
import pandas as pd

# Check human data
df_human = pd.read_parquet('processed_data/real/dmitva-dataset/human/chunk_0001.parquet')
print(df_human.columns)
print(df_human['label'].value_counts())  # Should be all 0s

# Check AI data
df_ai = pd.read_parquet('processed_data/ai/dmitva-dataset/ai_generated/chunk_0001.parquet')
print(df_ai.columns)
print(df_ai['label'].value_counts())  # Should be all 1s
```

---

## Compatibility with Training Pipeline

### Data Format
The processed data is **fully compatible** with the existing training pipeline (`src/training/train.py`):

✅ **Columns Match**: All required columns present (text, label, model, domain, etc.)  
✅ **Parquet Format**: Efficient storage and fast loading  
✅ **Directory Structure**: Follows ai/real segregation pattern  
✅ **Metadata**: Includes statistics.json and metadata.json files

### Loading with data_loader.py
```python
from src.training.data_loader import ProcessedDataLoader

loader = ProcessedDataLoader(data_dir="./processed_data")
parquet_files = loader.find_parquet_files()

# Will find:
# - 'real_human': processed_data/real/dmitva-dataset/human/*.parquet
# - 'ai_ai_generated': processed_data/ai/dmitva-dataset/ai_generated/*.parquet
```

---

## Dataset Characteristics

### Text Quality
- **Human Text**: Student essays, varied quality, 239-6044 characters
- **AI Text**: Generated responses, more polished, 335-2640 characters
- **Instructions**: Task descriptions preserved as domain metadata

### Label Distribution
- **Balanced**: ~48% human, ~52% AI (after deduplication)
- **Clean**: Binary classification ready
- **Deduplicated**: 310K duplicates removed for better generalization

### Domain Information
The `instructions` column provides context about the writing task:
- Persuasive essays
- Descriptive writing
- Argumentative pieces
- Opinion pieces

---

## Next Steps

1. **Verify Processed Data**
   ```bash
   python scripts/validate_processed_data.py
   ```

2. **Train Model**
   ```bash
   python src/training/train.py \
       --data-dir processed_data \
       --output-dir models/dmitva_model
   ```

3. **Combine with Other Datasets** (Optional)
   The dmitva-dataset folder structure allows it to coexist with other processed datasets like RAID-Dataset.

---

## Key Features

✅ **Automatic Format Detection**: Detects paired human_text/ai_text structure  
✅ **Memory Efficient**: Chunked processing handles large files  
✅ **Deduplication**: Removes duplicate texts across the entire dataset  
✅ **Progress Tracking**: Real-time progress bar and statistics  
✅ **Metadata Generation**: Automatic statistics and metadata files  
✅ **Error Handling**: Robust error handling and logging  
✅ **Resume Capability**: Can be extended with checkpoint support

---

## Citation

Dataset Citation:
```
@article{abiodunfinbarrsoketunji-agtd2023,
  title={Evaluating the Efficacy of Hybrid Deep Learning Models in Distinguishing AI-Generated Text},
  author={Abiodun Finbarrs Oketunji},
  journal={arXiv:2311.15565v2},
  year={2023}
}
```

---

## Troubleshooting

### Issue: Memory Usage High
**Solution**: Reduce `--chunk-size` (e.g., from 50000 to 25000)

### Issue: Processing Too Slow
**Solution**: Increase `--chunk-size` if you have sufficient RAM

### Issue: Want to Keep Duplicates
**Solution**: Use `--no-remove-duplicates` flag

### Issue: Need Different Compression
**Solution**: Use `--compression gzip` for better compression (slower) or `--compression snappy` (default, faster)

---

**Created**: October 18, 2025  
**Status**: ✅ Tested and Verified  
**Compatible with**: train.py, data_loader.py, validate_processed_data.py
