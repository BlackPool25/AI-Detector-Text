# AI Text Detection Ensemble

A **self-contained** 3-component ensemble for detecting AI-generated text.

## Features

- **DeTeCtive** (Weight: 50%) - Style clustering via contrastive learning (RoBERTa-based)
- **Binoculars** (Weight: 30%) - Cross-model probability divergence (GPT-2 based)
- **Fast-DetectGPT** (Weight: 20%) - Probability curvature smoothness (GPT-2 based)

Each detector exploits a DIFFERENT AI weakness with zero overlap.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Python Usage

```python
from ensemble import EnsembleDetector

# Initialize (loads all 3 detectors)
detector = EnsembleDetector()

# Detect single text
result = detector.detect("Your text here...")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Agreement: {result.agreement}")  # e.g., "3/3"

# Batch detection
results = detector.batch_detect(["Text 1", "Text 2", "Text 3"])
```

### API Usage

```bash
# Start the API server
uvicorn ensemble.api:app --host 0.0.0.0 --port 8000
```

Then access:
- `GET /` - Health check
- `POST /detect` - Detect single text
- `POST /detect/batch` - Batch detection
- `GET /info` - Detector info
- `GET /docs` - Interactive API docs

## Project Structure

```
ensemble/
├── __init__.py              # Package exports
├── detector.py              # All detector implementations (MAIN FILE)
├── api.py                   # FastAPI endpoints for Modal deployment
├── test.py                  # Test suite
├── core/                    # Core modules
│   ├── text_embedding.py    # RoBERTa text embeddings
│   ├── faiss_index.py       # FAISS KNN indexer
│   └── binoculars_metrics.py # Perplexity/entropy metrics
├── models/                  # Pre-trained model weights
├── database/                # FAISS databases
└── cache/                   # HuggingFace model cache
```

## Setup

### 1. Install Dependencies

```bash
cd /home/lightdesk/Projects/AI-Text
pip install -r ensemble/requirements.txt
```

**Note:** Use `faiss-cpu` for AMD GPUs or CPU-only systems.

### 2. Verify Model Paths

Ensure the following exist:
- DeTeCtive models: `Models/DeTeCtive/*.pth`
- DeTeCtive database: `DeTeCtive/database/deepfake_sample/`

### 3. Run Integration Tests

```bash
cd /home/lightdesk/Projects/AI-Text/ensemble
python test_full_ensemble.py
```

## Usage

### Full 3-Component Ensemble

```python
from ensemble.full_ensemble import EnsembleDetector, format_result

# Initialize ensemble
ensemble = EnsembleDetector(
    detective_path="/path/to/model.pth",
    detective_db="/path/to/database",
    include_fast_detect=True,  # Enable 3rd component
    share_gpt2=True            # Share GPT-2 to save memory
)

# Detect single text
text = "Your text to analyze here..."
result = ensemble.detect(text)

# Print formatted result
print(format_result(result))

# Access structured data
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Agreement: {result.agreement}")
```

### Batch Processing

```python
texts = [
    "First text to analyze...",
    "Second text to analyze...",
    "Third text to analyze..."
]

results = ensemble.batch_detect(texts)

for i, result in enumerate(results, 1):
    print(f"\nText {i}: {result.prediction} ({result.confidence:.2%})")
```

### Running Tests

```bash
# Run all tests with default paths
python test_ensemble.py

# Run with custom paths
python test_ensemble.py /path/to/model.pth /path/to/database
```

## Output Format

The detector returns an `EnsembleResult` object with:

- `prediction`: "AI", "Human", or "UNCERTAIN"
- `confidence`: 0.0-1.0 (higher = more confident)
- `agreement`: "2/2", "1/2" (detector agreement)
- `ensemble_score`: 0.0-1.0 (weighted average, higher = more likely AI)
- `suggested_action`: "ACCEPT", "REVIEW", or "REJECT"
- `breakdown`: Individual detector results (optional)

### Example Output

```
==============================================================
PREDICTION: AI
Confidence: 87.50%
Agreement: 2/2
Ensemble Score: 0.8750
Suggested Action: REJECT

--------------------------------------------------------------
DETECTOR BREAKDOWN:

DETECTIVE:
  Prediction: AI
  Score: 0.9000
  Confidence: 0.8000
  Votes: 9 AI, 1 Human

BINOCULARS:
  Prediction: AI
  Score: 0.8250
  Confidence: 0.6500
  Votes: N/A (statistical method)
==============================================================
```

## Performance

Based on plan specifications:

- **Accuracy**: 94-98% on GPT-4/Claude/Gemini text
- **False Positive Rate**: <2% on human text
- **Speed**: ~40-60 texts/second on AMD 7900GRE (batch processing)
- **Memory**: ~1.6-1.8GB GPU memory

## Decision Rules

### Agreement Logic

- **2/2 Agreement**: High confidence, use ensemble score
- **1/2 Agreement**: 
  - If ensemble > 0.55: Predict AI (reduced confidence)
  - If ensemble < 0.45: Predict Human (reduced confidence)
  - Otherwise: UNCERTAIN

### Confidence Thresholds

- **High confidence** (≥80%): Strong prediction, suggested action ACCEPT/REJECT
- **Medium confidence** (60-80%): Reliable prediction, suggested action ACCEPT
- **Low confidence** (<60%): Uncertain, suggested action REVIEW

## Troubleshooting

### Import Errors

If you see import errors when running the detector:
- The script uses `sys.path.insert()` to dynamically load modules
- Ensure DeTeCtive and Binoculars folders exist in the parent directory

### CUDA Out of Memory

If you encounter OOM errors:
1. Use smaller models: `gpt2` instead of larger variants
2. Set `use_bfloat16=True` in Binoculars (default)
3. Process texts individually instead of batching
4. Use CPU fallback: set `CUDA_VISIBLE_DEVICES=""`

### Model Download Issues

If HuggingFace models fail to download:
- Check internet connection
- Set HF_TOKEN environment variable for gated models
- Use offline mode with pre-downloaded models

### Database Not Found

If DeTeCtive database is missing:
```bash
cd ../DeTeCtive
python gen_database.py --help  # See database generation options
```

## Error Handling

The detector validates:
- Text length (minimum 10 characters)
- Model and database paths
- GPU availability (falls back to CPU)

Edge cases:
- Empty/very short text: Raises `ValueError`
- Very long text: Automatically truncated to 512 tokens
- Processing errors: Returns UNCERTAIN result

## Integration Notes

### Compatibility

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, CPU supported)
- Tested on Ubuntu 22.04

### Model Loading Order

To avoid OOM:
1. GPT-2 loads first (shared observer)
2. GPT-2-medium loads second
3. RoBERTa loads last
4. Uses `torch.cuda.empty_cache()` between loads if needed

### Thread Safety

Not thread-safe due to CUDA state. For parallel processing:
- Use multiprocessing with separate instances
- Or implement queue-based batching

## Citation

If using this ensemble, please cite both underlying methods:

```bibtex
@article{detective2024,
  title={DeTeCtive: Detecting AI-Generated Text via Multi-Level Contrastive Learning},
  journal={arXiv preprint arXiv:2410.20964},
  year={2024}
}

@article{binoculars2024,
  title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
  journal={arXiv preprint arXiv:2401.12070},
  year={2024}
}
```

## License

This integration follows the licenses of the constituent projects:
- DeTeCtive: Check LICENSE in DeTeCtive folder
- Binoculars: Check LICENSE in Binoculars folder
