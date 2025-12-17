# Usage Guide

## Running the Preprocessor

After organizing the project structure, the main preprocessing script is now located at:
```
src/preprocessor/preprocess.py
```

### Basic Command

```bash
python src/preprocessor/preprocess.py --input data/raw/input.csv --output data/processed
```

## Utility Scripts

### Analyze Data

Analyze your raw CSV files before processing:

```bash
python scripts/analyze_data.py
```

This script will:
- Read sample data from your CSV
- Show unique values per column
- Display model distribution
- Help you understand the data structure

### Validate Processed Data

After processing, validate the output:

```bash
python scripts/validate_processed_data.py
```

This provides:
- Processing summary
- Sample counts and distribution
- Model breakdown
- Data quality metrics

## Working with Data Directories

### Raw Data (`data/raw/`)
Place your original CSV files here:
- `train.csv` - Training dataset
- `test.csv` - Test dataset
- Any other raw CSV files

### Sample Data (`data/samples/`)
Small sample files for testing:
- `test_dataset.csv` - Sample test data
- `extra.csv` - Additional samples

### Processed Data (`data/processed/`)
Output directory (automatically created):
- Organized by label and model
- Contains Parquet files
- Includes metadata and statistics

## Running Tests

Execute the test suite:

```bash
python tests/test_preprocess.py
```

## Examples

See example usage patterns:

```bash
python examples/usage_examples.py
```

## Documentation

All documentation is now in the `docs/` directory:
- `QUICKSTART.md` - Quick start guide
- `GETTING_STARTED.md` - Detailed getting started
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `CHANGELOG.md` - Version history
- `planning/` - Project planning documents

## Installation as Package

You can install this as a Python package:

```bash
pip install -e .
```

This will make the preprocessor available as a command-line tool:

```bash
ai-preprocess --input data/raw/input.csv --output data/processed
```

## Tips

1. **Always use relative paths from project root**
   ```bash
   python src/preprocessor/preprocess.py --input data/raw/train.csv --output data/processed
   ```

2. **Keep raw data separate from processed data**
   - Raw: `data/raw/`
   - Processed: `data/processed/`

3. **Version control**
   - `.gitignore` is configured to exclude large data files
   - Only sample data and code are tracked

4. **Virtual environment**
   - Always activate your virtual environment:
     ```bash
     source .venv/bin/activate  # Linux/Mac
     .venv\Scripts\activate     # Windows
     ```
