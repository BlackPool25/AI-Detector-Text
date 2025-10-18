# Quick Start Guide

## Installation (30 seconds)

```bash
# 1. Navigate to project directory
cd /home/lightdesk/Projects/Text

# 2. Create virtual environment (if not exists)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage (3 steps)

### Step 1: Validate Your Data (Optional but Recommended)

```bash
python preprocess.py --input your_data.csv --output processed_data --validate-only
```

This will show you:
- ✓ Detected schema
- ✓ Sample data
- ✓ Potential issues
- ✓ Processing estimates

### Step 2: Process Your Dataset

```bash
python preprocess.py --input your_data.csv --output processed_data
```

**For large datasets (>1M rows), customize:**

```bash
python preprocess.py \
    --input your_data.csv \
    --output processed_data \
    --chunk-size 50000 \
    --min-text-length 20 \
    --compression gzip
```

### Step 3: Load and Use Processed Data

```python
import pandas as pd

# Load human data
df_human = pd.concat([
    pd.read_parquet(f) 
    for f in Path('processed_data/human').glob('*.parquet')
])

# Load AI data (all models)
df_ai = pd.concat([
    pd.read_parquet(f) 
    for f in Path('processed_data/ai').rglob('*.parquet')
])

# Or load specific model
df_gpt4 = pd.concat([
    pd.read_parquet(f) 
    for f in Path('processed_data/ai/gpt4').glob('*.parquet')
])
```

## CSV Format Requirements

Your CSV needs at minimum:

1. **Text column** - names like: `text`, `content`, `document`, `passage`
2. **Label column** - names like: `label`, `class`, `is_ai`, `source`

**Supported label formats:**
- `0, 1` (0=human, 1=ai)
- `human, ai` (case insensitive)
- `true, false` (true=ai, false=human)
- Model names: `gpt4, claude, human`

**Example CSV:**
```csv
text,label,model
"Human written text here",human,human
"AI generated text here",ai,gpt4
```

## Common Use Cases

### Use Case 1: Basic Processing

```bash
python preprocess.py --input data.csv --output out
```

### Use Case 2: High Quality Dataset

```bash
python preprocess.py \
    --input data.csv \
    --output out \
    --min-text-length 50 \
    --max-text-length 10000 \
    --compression gzip \
    --remove-duplicates
```

### Use Case 3: Memory Constrained

```bash
python preprocess.py \
    --input data.csv \
    --output out \
    --chunk-size 25000
```

### Use Case 4: Resume After Interruption

```bash
python preprocess.py --input data.csv --output out --resume
```

## What You Get

After processing, you'll have:

```
processed_data/
├── human/
│   ├── chunk_0001.parquet
│   └── metadata.json
├── ai/
│   ├── gpt4/
│   │   ├── chunk_0001.parquet
│   │   └── metadata.json
│   └── claude/
│       ├── chunk_0001.parquet
│       └── metadata.json
├── statistics.json          # Overall stats
└── processing_log.txt       # Detailed log
```

## Loading with HuggingFace

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset('parquet', data_dir='processed_data')

# Load specific subset
dataset_gpt4 = load_dataset('parquet', data_dir='processed_data/ai/gpt4')

# Create splits
splits = dataset['train'].train_test_split(test_size=0.2, seed=42)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not detect text column" | Rename column to `text`, `content`, or `document` |
| "Could not detect label column" | Rename column to `label`, `class`, or `is_ai` |
| "High memory usage" | Use `--chunk-size 25000` or lower |
| "Processing too slow" | Use `--chunk-size 200000` (if RAM allows) |
| Process interrupted | Use `--resume` to continue from checkpoint |

## Performance Tips

**Speed up processing:**
- Increase chunk size: `--chunk-size 200000`
- Use Snappy compression: `--compression snappy` (default)
- Disable deduplication: `--no-remove-duplicates`

**Reduce memory:**
- Decrease chunk size: `--chunk-size 25000`
- Close other applications

**Disk space:**
- Plan for ~20-25% of original CSV size (with compression)

## Next Steps

1. ✅ Process your dataset
2. ✅ Check `statistics.json` for overview
3. ✅ Load data with pandas or HuggingFace
4. ✅ Create train/val/test splits
5. ✅ Train your AI detection model!

## Need Help?

Run the examples:
```bash
python examples.py
```

Read the full documentation:
```bash
cat README.md
```

Check processing logs:
```bash
cat processed_data/processing_log.txt
```

---

**Ready to process millions of rows? Let's go! 🚀**
