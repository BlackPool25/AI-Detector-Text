# Model Saving and Checkpointing Guide

## Overview

The training pipeline implements comprehensive model saving with two distinct strategies:

1. **Training Checkpoints** - For resumability, debugging, and training recovery
2. **Inference Models** - For deployment and production use

## Architecture

### Components

```
src/training/train.py
├── CheckpointManager      # Manages training checkpoints
├── ModelSaver            # Handles inference model exports
└── ModelTrainer          # Orchestrates training and saving
```

### Storage Structure

```
outputs/
├── checkpoints/                    # Training checkpoints
│   ├── checkpoint_latest.safetensors
│   ├── checkpoint_best_val_f1.safetensors
│   ├── checkpoint_epoch_5.safetensors
│   ├── checkpoint_step_1000.safetensors
│   └── *.meta.pt                  # Non-tensor metadata (if safetensors)
│
└── models/
    └── inference/                  # Deployment models
        ├── model_v1.0_epoch5/
        │   ├── config.json
        │   ├── full_model.pt
        │   ├── tokenizer_config.json
        │   ├── training_metadata.json
        │   ├── model_traced.pt    # Optional TorchScript
        │   └── README.md
        └── model_v1.0_final/
            └── ...
```

## Training Checkpoints

### Features

✅ **Complete Training State**
- Model weights
- Optimizer state (Adam, AdamW)
- Learning rate scheduler state
- EMA (Exponential Moving Average) weights
- Random number generator states (PyTorch, CUDA, NumPy, Python)
- Training metrics and history
- Current epoch and global step

✅ **Security**
- Uses `safetensors` format by default (prevents pickle attacks)
- Falls back to PyTorch format if safetensors unavailable

✅ **Automatic Management**
- Keeps top-K best checkpoints based on validation metric
- Periodic epoch checkpoints (every 5 epochs)
- Always maintains "latest" checkpoint
- Automatic cleanup of old checkpoints

### Usage

#### During Training

Checkpoints are automatically managed:

```python
from src.training.train import ModelTrainer, TrainingConfig

# Initialize trainer
config = TrainingConfig(
    save_total_limit=3,          # Keep top 3 checkpoints
    early_stopping_patience=3,
    metric_for_best_model="val_f1"
)
trainer = ModelTrainer(config, output_dir="./outputs")

# Train - checkpoints saved automatically
results = trainer.train(train_data, val_data)
```

#### Manual Checkpoint Saving

```python
# Access checkpoint manager
checkpoint_manager = trainer.checkpoint_manager

# Save a checkpoint
checkpoint_manager.save_checkpoint(
    model=trainer.model,
    optimizer=optimizer,
    scheduler=scheduler,
    ema=trainer.ema,
    epoch=5,
    step=1000,
    metrics={'val_f1': 0.98, 'val_loss': 0.05},
    config=trainer.config,
    checkpoint_type='best'  # 'latest', 'best', 'epoch', 'regular'
)
```

#### Loading Checkpoints

```python
# Load for inference only (fastest)
metadata = trainer.load_checkpoint(
    checkpoint_path="./outputs/checkpoints/checkpoint_best_val_f1.safetensors",
    load_training_state=False
)

# Load for resuming training (includes optimizer, etc.)
metadata = trainer.checkpoint_manager.load_checkpoint(
    filepath="./outputs/checkpoints/checkpoint_latest.safetensors",
    model=trainer.model,
    optimizer=optimizer,
    scheduler=scheduler,
    ema=trainer.ema,
    load_training_state=True
)

print(f"Resumed from epoch {metadata['epoch']}, step {metadata['global_step']}")
```

#### Resume Training Example

```python
# Create trainer
trainer = ModelTrainer(config, output_dir="./outputs")

# Resume from checkpoint
metadata = trainer.resume_from_checkpoint(
    checkpoint_path="./outputs/checkpoints/checkpoint_latest.safetensors"
)

# Continue training
results = trainer.train(train_data, val_data)
```

## Inference Models

### Features

✅ **Production-Ready Format**
- HuggingFace compatible structure
- Full model weights in single file
- Tokenizer configuration
- Comprehensive metadata
- Optional TorchScript compilation

✅ **Version Management**
- Semantic versioning (v1.0, v1.1, etc.)
- Automatic README generation
- Training date and metrics tracking

✅ **Deployment Options**
- Standard PyTorch format
- TorchScript (compiled, optimized)
- ONNX export (future support)

### Usage

#### Automatic Saving

Inference models are saved automatically:
- When best validation metric is achieved
- At end of training (final model)

```python
# Train - inference models saved automatically
trainer = ModelTrainer(config, output_dir="./outputs")
results = trainer.train(train_data, val_data)

# Best model: ./outputs/models/inference/model_v1.0_epoch5/
# Final model: ./outputs/models/inference/model_v1.0_final/
```

#### Manual Saving

```python
# Access model saver
model_saver = trainer.model_saver

# Save for inference
model_saver.save_for_inference(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    metrics={'val_f1': 0.98, 'val_accuracy': 0.97},
    config=trainer.config,
    version='v1.0',
    save_traced=True  # Also save TorchScript version
)
```

#### Loading Inference Models

```python
import torch
from transformers import AutoTokenizer
from src.training.train import DeBERTaAIDetector, TrainingConfig

# Load model
model_dir = "./outputs/models/inference/model_v1.0_final"

# Load configuration
with open(f"{model_dir}/training_metadata.json", 'r') as f:
    metadata = json.load(f)

# Reconstruct config
config = TrainingConfig(**metadata['hyperparameters'])

# Initialize model
model = DeBERTaAIDetector(config)

# Load weights
checkpoint = torch.load(f"{model_dir}/full_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Use for inference
text = "This is a sample text to classify."
inputs = tokenizer(
    text,
    return_tensors='pt',
    max_length=512,
    padding='max_length',
    truncation=True
)

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    prediction = outputs['logits'].argmax(dim=-1).item()
    print(f"Prediction: {'AI' if prediction == 1 else 'Human'}")
```

## Safetensors Format

### Why Safetensors?

- **Security**: Prevents arbitrary code execution (no pickle)
- **Speed**: Faster loading than pickle
- **Compatibility**: Cross-framework support
- **Safety**: Type validation and size checks

### Installation

```bash
pip install safetensors>=0.4.0
```

### Format Details

Safetensors files (`.safetensors`) contain only tensor data. Non-tensor data (optimizer state, config) is saved separately in `.meta.pt` files.

```
checkpoint_best_val_f1.safetensors   # Tensor weights
checkpoint_best_val_f1.safetensors.meta.pt  # Metadata
```

## Checkpoint Manager Configuration

### Parameters

```python
CheckpointManager(
    save_dir="./checkpoints",      # Directory for checkpoints
    keep_top_k=3,                  # Keep top 3 checkpoints
    metric='val_f1',               # Metric to track
    mode='max',                    # 'max' or 'min'
    use_safetensors=True           # Use safetensors if available
)
```

### Checkpoint Types

1. **Latest** (`checkpoint_latest.safetensors`)
   - Always overwritten with most recent state
   - Use for crash recovery

2. **Best** (`checkpoint_best_val_f1.safetensors`)
   - Best checkpoint based on tracked metric
   - Updated when metric improves

3. **Epoch** (`checkpoint_epoch_5.safetensors`)
   - Saved every 5 epochs
   - Never automatically deleted

4. **Regular** (`checkpoint_step_1000.safetensors`)
   - Saved based on top-K metric
   - Old ones deleted when limit exceeded

## Best Practices

### Training

1. **Always use CheckpointManager** - Handles cleanup and tracking
2. **Monitor disk space** - Checkpoints can be large (500MB+)
3. **Version your experiments** - Use meaningful output directories
4. **Save periodically** - Don't rely only on end-of-training saves

### Inference

1. **Test models before deployment** - Load and validate
2. **Document model versions** - Track what's in production
3. **Keep training metadata** - For reproducibility
4. **Consider model size** - Use quantization if needed

### Storage

1. **Use Git LFS** for version control of models
2. **Backup to cloud storage** (S3, GCS) for important models
3. **Clean old experiments** to save disk space
4. **Validate checksums** when loading from external sources

## Advanced Usage

### Custom Checkpoint Frequency

```python
# In training loop
for epoch in range(num_epochs):
    train_epoch()
    
    if epoch % 2 == 0:  # Every 2 epochs
        checkpoint_manager.save_epoch(
            model, optimizer, scheduler, ema,
            epoch, step, metrics, config
        )
```

### Export to ONNX (Future)

```python
import torch.onnx

model.eval()
example_input = torch.randint(0, 30000, (1, 512))
example_mask = torch.ones((1, 512), dtype=torch.long)

torch.onnx.export(
    model,
    (example_input, example_mask),
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)
```

### Model Quantization

```python
# Post-training quantization
import torch.quantization

model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized.pt')
```

## Troubleshooting

### Checkpoint Loading Errors

```python
# Check if file exists
from pathlib import Path
checkpoint_path = Path("checkpoint.safetensors")
if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")

# Verify format
if checkpoint_path.suffix == '.safetensors':
    from safetensors.torch import load_file
    checkpoint = load_file(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path)
```

### Disk Space Issues

```python
# List checkpoints by size
import os

checkpoint_dir = Path("./outputs/checkpoints")
for f in sorted(checkpoint_dir.glob("*.safetensors")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"{f.name}: {size_mb:.2f} MB")
```

### Version Compatibility

```python
# Save PyTorch and library versions
checkpoint['versions'] = {
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'python': sys.version,
}
```

## Performance Tips

1. **Use safetensors** - 2-3x faster loading than pickle
2. **Save to SSD** - Faster I/O than HDD
3. **Async saving** - Don't block training (future feature)
4. **Compress old checkpoints** - Use `gzip` for archival

## Summary

The model saving system provides:

✅ **Reliability** - Never lose training progress  
✅ **Flexibility** - Multiple save formats and strategies  
✅ **Security** - Safetensors prevents code injection  
✅ **Convenience** - Automatic management and cleanup  
✅ **Production-Ready** - Easy deployment workflows  

For questions or issues, refer to the main documentation or training script comments.
