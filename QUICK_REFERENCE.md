# Quick Reference Card

## Common Commands

### Processing Data
```bash
# Basic processing
python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed

# With validation first
python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed --validate-only

# Resume interrupted processing
python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed --resume

# Custom settings
python src/preprocessor/preprocess.py \
    --input data/raw/train.csv \
    --output data/processed \
    --chunk-size 50000 \
    --min-text-length 20 \
    --compression gzip
```

### Using Makefile
```bash
make help         # Show all commands
make analyze      # Analyze raw data
make process      # Process train.csv
make validate     # Validate output
make test         # Run tests
make clean        # Remove processed data
```

### Analysis & Validation
```bash
# Analyze before processing
python scripts/analyze_data.py

# Validate after processing
python scripts/validate_processed_data.py
```

## Directory Quick Guide

| Directory | Purpose | Contains |
|-----------|---------|----------|
| `src/preprocessor/` | Core code | Main preprocessing script |
| `scripts/` | Utilities | Analysis and validation tools |
| `tests/` | Testing | Unit tests |
| `examples/` | Examples | Usage examples |
| `docs/` | Documentation | All docs and guides |
| `data/raw/` | Input data | Original CSV files |
| `data/samples/` | Test data | Small sample files |
| `data/processed/` | Output data | Processed Parquet files |

## File Locations

| What | Where |
|------|-------|
| Main script | `src/preprocessor/preprocess.py` |
| Analyze data | `scripts/analyze_data.py` |
| Validate output | `scripts/validate_processed_data.py` |
| Run tests | `tests/test_preprocess.py` |
| Examples | `examples/usage_examples.py` |
| Main README | `README.md` |
| Usage guide | `docs/USAGE.md` |
| Quick start | `docs/QUICKSTART.md` |

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# OR install as package
pip install -e .
```

## Data Workflow

```
1. Place CSV files in:     data/raw/
2. Analyze them:            make analyze
3. Process them:            make process
4. Validate output:         make validate
5. Output appears in:       data/processed/
```

## Common Options

| Option | Default | Purpose |
|--------|---------|---------|
| `--chunk-size` | 100000 | Rows per chunk |
| `--min-text-length` | 10 | Min text length |
| `--max-text-length` | 100000 | Max text length |
| `--compression` | snappy | Compression type |
| `--remove-duplicates` | True | Remove duplicates |
| `--validate-only` | False | Dry run |
| `--resume` | False | Resume processing |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check paths: use `data/raw/` and `data/processed/` |
| Import error | Activate venv: `source .venv/bin/activate` |
| Memory error | Reduce `--chunk-size` |
| Slow processing | Increase `--chunk-size` (if RAM allows) |

## Getting Help

```bash
# Command help
python src/preprocessor/preprocess.py --help

# Makefile help
make help

# Documentation
cat docs/USAGE.md
cat docs/QUICKSTART.md
```

---
**For full documentation, see README.md and docs/ directory**
