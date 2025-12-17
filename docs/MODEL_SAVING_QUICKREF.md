# Model Saving Quick Reference

## Installation

```bash
pip install safetensors>=0.4.0
```

## Basic Training (Automatic Saving)

```python
from src.training.train import ModelTrainer, TrainingConfig
import pandas as pd

# Configure
config = TrainingConfig(
    num_train_epochs=5,
    save_total_limit=3,  # Keep top 3 checkpoints
    metric_for_best_model="val_f1"
)

# Train
trainer = ModelTrainer(config, output_dir="./outputs")
results = trainer.train(train_data, val_data)

# Checkpoints: ./outputs/checkpoints/
# Models: ./outputs/models/inference/
```

## Resume Training

```python
trainer = ModelTrainer(config, output_dir="./outputs")
metadata = trainer.resume_from_checkpoint(
    "./outputs/checkpoints/checkpoint_latest.safetensors"
)
results = trainer.train(train_data, val_data)
```

## Manual Checkpoint Save

```python
trainer.checkpoint_manager.save_checkpoint(
    model=trainer.model,
    optimizer=optimizer,
    scheduler=scheduler,
    ema=trainer.ema,
    epoch=5,
    step=1000,
    metrics={'val_f1': 0.98},
    config=config,
    checkpoint_type='best'  # 'latest', 'best', 'epoch', 'regular'
)
```

## Load Checkpoint

```python
# For inference only (fast)
metadata = trainer.load_checkpoint(
    checkpoint_path="./outputs/checkpoints/checkpoint_best_val_f1.safetensors",
    load_training_state=False
)

# For resuming training (includes optimizer, etc.)
metadata = trainer.checkpoint_manager.load_checkpoint(
    filepath="./outputs/checkpoints/checkpoint_latest.safetensors",
    model=trainer.model,
    optimizer=optimizer,
    scheduler=scheduler,
    ema=trainer.ema,
    load_training_state=True
)
```

## Save for Inference

```python
trainer.model_saver.save_for_inference(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    metrics={'val_f1': 0.98, 'val_accuracy': 0.97},
    config=config,
    version='v1.0',
    save_traced=False  # Set True for TorchScript
)
```

## Load for Inference

```python
import torch
import json
from transformers import AutoTokenizer
from src.training.train import DeBERTaAIDetector, TrainingConfig

# Load metadata
model_dir = "./outputs/models/inference/model_v1.0_final"
with open(f"{model_dir}/training_metadata.json") as f:
    metadata = json.load(f)

# Initialize model
config = TrainingConfig(**metadata['hyperparameters'])
model = DeBERTaAIDetector(config)

# Load weights
checkpoint = torch.load(f"{model_dir}/full_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Inference
text = "Your text here"
inputs = tokenizer(text, return_tensors='pt', max_length=512, 
                   padding='max_length', truncation=True)

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    prediction = outputs['logits'].argmax(dim=-1).item()
    # 0 = Human, 1 = AI
```

## Checkpoint Types

| Type | Filename | When Created | Auto-Delete |
|------|----------|--------------|-------------|
| Latest | `checkpoint_latest.safetensors` | Every epoch | No (overwritten) |
| Best | `checkpoint_best_val_f1.safetensors` | New best metric | No |
| Epoch | `checkpoint_epoch_5.safetensors` | Every 5 epochs | No |
| Regular | `checkpoint_step_1000.safetensors` | Regular saves | Yes (beyond top-K) |

## File Structure

```
outputs/
├── checkpoints/
│   ├── checkpoint_latest.safetensors      # Most recent
│   ├── checkpoint_best_val_f1.safetensors # Best metric
│   ├── checkpoint_epoch_5.safetensors     # Periodic
│   └── checkpoint_step_1000.safetensors   # Regular
│
└── models/inference/
    ├── model_v1.0_epoch5/                 # Best during training
    │   ├── full_model.pt
    │   ├── tokenizer_config.json
    │   ├── training_metadata.json
    │   └── README.md
    └── model_v1.0_final/                  # Final model
        └── ...
```

## Configuration Options

```python
TrainingConfig(
    save_total_limit=3,              # Keep top N checkpoints
    save_strategy="epoch",           # "epoch" or "steps"
    evaluation_strategy="epoch",     # When to evaluate
    metric_for_best_model="val_f1", # Metric to track
    early_stopping_patience=3,       # Stop after N epochs without improvement
)

CheckpointManager(
    save_dir="./checkpoints",        # Directory
    keep_top_k=3,                    # Number to keep
    metric='val_f1',                 # Metric name
    mode='max',                      # 'max' or 'min'
    use_safetensors=True             # Use safetensors format
)
```

## Common Tasks

### Check Saved Checkpoints
```python
from pathlib import Path

checkpoint_dir = Path("./outputs/checkpoints")
for f in sorted(checkpoint_dir.glob("*.safetensors")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"{f.name}: {size_mb:.2f} MB")
```

### List Top Checkpoints
```python
print(f"Top {trainer.checkpoint_manager.keep_top_k} checkpoints:")
for metric_value, filepath, metadata in trainer.checkpoint_manager.checkpoints:
    print(f"  F1={metric_value:.4f} - Epoch {metadata['epoch']} - {filepath.name}")
```

### Clean Old Checkpoints
```python
# Automatically handled by CheckpointManager
# But you can manually delete:
import shutil
shutil.rmtree("./outputs/checkpoints/old_experiment")
```

## Best Practices

✅ Always use `save_total_limit` to avoid disk bloat  
✅ Use safetensors format (default) for security  
✅ Save inference models separately from checkpoints  
✅ Include version numbers in model names  
✅ Test loading before deployment  
✅ Backup best models to cloud storage  
✅ Document training configuration in metadata  

## Troubleshooting

### ImportError: safetensors
```bash
pip install safetensors>=0.4.0
```

### Checkpoint not found
```python
from pathlib import Path
checkpoint_path = Path("checkpoint.safetensors")
if not checkpoint_path.exists():
    print(f"Not found: {checkpoint_path}")
    print(f"Available: {list(checkpoint_path.parent.glob('*.safetensors'))}")
```

### Out of disk space
```python
# Reduce save_total_limit
config = TrainingConfig(save_total_limit=2)

# Or delete old experiments
import shutil
shutil.rmtree("./outputs/old_experiment")
```

## Examples

Run comprehensive examples:
```bash
python examples/model_saving_examples.py
```

## Documentation

- **Full guide**: `docs/MODEL_SAVING_GUIDE.md`
- **Implementation**: `IMPLEMENTATION_MODEL_SAVING.md`
- **Examples**: `examples/model_saving_examples.py`
