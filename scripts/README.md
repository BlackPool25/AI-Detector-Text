# Scripts Directory

Utility scripts for analyzing and validating data.

## Available Scripts

### 1. analyze_data.py

Analyzes the structure of raw CSV files before processing.

**Usage:**
```bash
python scripts/analyze_data.py
```

**What it does:**
- Reads sample data from `data/raw/train.csv`
- Shows unique values per column
- Displays model distribution
- Analyzes label patterns
- Helps you understand data structure

**Output:**
- Column analysis
- Sample values
- Distribution statistics
- Data quality insights

### 2. validate_processed_data.py

Validates processed data for training readiness.

**Usage:**
```bash
python scripts/validate_processed_data.py
```

**Prerequisites:**
You must have processed data in `data/processed/` directory. Run the preprocessor first:
```bash
python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed
```

**What it checks:**
- Processing statistics
- Label distribution (human vs AI)
- Model distribution
- Data quality (nulls, corruption)
- Schema validation
- Training readiness

**Output:**
- Processing summary
- Distribution breakdown
- Quality check results
- Validation status
- Training readiness confirmation

## When to Use These Scripts

### Before Processing
1. Run `analyze_data.py` to understand your raw data
2. Check column names and formats
3. Identify any data issues

### After Processing
1. Run `validate_processed_data.py` to verify output
2. Confirm data quality
3. Check training readiness

## Customization

Both scripts can be modified to work with different data paths:

### Analyze Different File
Edit `scripts/analyze_data.py`:
```python
DATA_FILE = project_root / "data" / "raw" / "your_file.csv"
```

### Validate Different Output
Edit `scripts/validate_processed_data.py`:
```python
PROCESSED_DIR = project_root / "data" / "your_output_dir"
```

## Examples

### Full Workflow
```bash
# 1. Analyze raw data
python scripts/analyze_data.py

# 2. Process data
python src/preprocessor/preprocess.py \
    --input data/raw/train.csv \
    --output data/processed

# 3. Validate output
python scripts/validate_processed_data.py
```

## Troubleshooting

### "File not found" Error
Make sure your data files are in the correct locations:
- Raw data: `data/raw/train.csv`
- Processed data: `data/processed/`

### "No module named" Error
Make sure you're running from the project root and have dependencies installed:
```bash
cd /path/to/project
source .venv/bin/activate
pip install -r requirements.txt
```
