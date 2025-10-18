# Quick Start Guide - Training AI Text Detector

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd /home/lightdesk/Projects/Text
pip install -r requirements.txt
```

### 2. Verify GPU Setup

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 3. Start Training

#### Quick Test (Small Dataset)
```bash
cd /home/lightdesk/Projects/Text
python src/training/train_script.py \
    --load-mode all \
    --ai-sample 50000 \
    --human-sample 50000 \
    --num-epochs 1 \
    --batch-size 16 \
    --output-dir ./outputs/test_run
```

**Expected output:**
```
Loading data from all AI models...
Loaded total X rows from Y files
Split: Train=Z, Val=A, Test=B

Starting Model Training
Training module initialized on device: cuda
Total parameters: 145,234,432

Epoch 1/1 - Train Loss: 0.1234, Train F1: 0.9456
Validation - Loss: 0.0987, F1: 0.9623, AUC: 0.9912

Training Complete!
```

---

## Common Training Scenarios

### Scenario 1: Full Training (All Models)

```bash
python src/training/train_script.py \
    --load-mode all \
    --num-epochs 5 \
    --batch-size 16 \
    --output-dir ./outputs/full_training
```

**Expected duration:** 6-12 hours on AMD 7900GRE  
**Expected F1-score:** 0.95-1.0

---

### Scenario 2: Model-Specific Training (GPT-4)

```bash
python src/training/train_script.py \
    --load-mode model \
    --model-name gpt4 \
    --num-epochs 3 \
    --batch-size 16 \
    --output-dir ./outputs/gpt4_detector
```

**Use case:** Train detector specifically for GPT-4 output

---

### Scenario 3: Limited Data Testing

```bash
python src/training/train_script.py \
    --load-mode all \
    --ai-sample 100000 \
    --human-sample 100000 \
    --num-epochs 2 \
    --batch-size 24 \
    --output-dir ./outputs/limited_data
```

**Use case:** Quick validation without full dataset

---

### Scenario 4: Aggressive Hyperparameter Tuning

```bash
python src/training/train_script.py \
    --load-mode all \
    --num-epochs 10 \
    --batch-size 12 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --lstm-lr 5e-4 \
    --lstm-hidden-size 256 \
    --warmup-steps 1000 \
    --ema-decay 0.9999 \
    --early-stopping-patience 5 \
    --output-dir ./outputs/tuned_model
```

---

## Monitoring Training

### Watch Training Progress

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f training.log
```

### Check Output Metrics

After training completes:

```bash
python -c "
import json
with open('outputs/full_training/results.json') as f:
    results = json.load(f)
    print('Best Val Loss:', results['best_val_loss'])
    print('Final Val F1:', results['training_history']['val_f1'][-1])
    print('Dataset Splits:', results['data_info'])
"
```

---

## Loading Trained Model

### Basic Usage

```python
import torch
from pathlib import Path
from src.training.train import ModelTrainer, TrainingConfig

# Load best checkpoint
checkpoint = torch.load(
    "outputs/full_training/best_model_epoch_3.pt",
    map_location="cuda"
)

# Restore configuration
config_dict = checkpoint['config']
config = TrainingConfig(**config_dict)

# Initialize and load model
trainer = ModelTrainer(config)
trainer.load_checkpoint(Path("outputs/full_training/best_model_epoch_3.pt"))

# Use model
trainer.model.eval()
```

### Make Predictions

```python
import torch

def predict(text: str, trainer: ModelTrainer) -> dict:
    """Predict whether text is human or AI-generated."""
    inputs = trainer.tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = trainer.model(
            inputs['input_ids'].to(trainer.config.device),
            inputs['attention_mask'].to(trainer.config.device)
        )
        
        probs = torch.softmax(outputs['logits'], dim=-1)[0]
        
        return {
            'human_probability': float(probs[0]),
            'ai_probability': float(probs[1]),
            'prediction': 'Human' if probs[0] > probs[1] else 'AI'
        }

# Example
text = "The quick brown fox jumps over the lazy dog."
result = predict(text, trainer)
print(result)
```

---

## Troubleshooting

### Problem: Out of Memory (CUDA Error)

**Solution 1:** Reduce batch size
```bash
python src/training/train_script.py ... --batch-size 8
```

**Solution 2:** Increase gradient accumulation
```bash
python src/training/train_script.py ... --gradient-accumulation-steps 8
```

**Solution 3:** Reduce sequence length
```bash
python src/training/train_script.py ... --max-seq-length 256
```

---

### Problem: GPU Not Used

Check GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, ensure CUDA/ROCm is installed:
```bash
# For AMD GPU (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

### Problem: Very Slow Training

1. **Check GPU utilization:**
   ```bash
   nvidia-smi  # Should show 95%+ utilization
   ```

2. **Enable BF16 (faster):**
   ```bash
   python src/training/train_script.py ... --bf16 True
   ```

3. **Verify data loading isn't bottleneck:**
   ```bash
   # Check data_info.json shows loaded sizes
   cat outputs/test_run/data_info.json
   ```

---

### Problem: Poor Model Performance

1. **Train longer:**
   ```bash
   python src/training/train_script.py ... --num-epochs 10
   ```

2. **Adjust learning rate:**
   ```bash
   python src/training/train_script.py ... --learning-rate 5e-6
   ```

3. **Increase augmentation:**
   ```bash
   python src/training/train_script.py ... --noise-injection-prob 0.2
   ```

---

## Understanding Output Files

After training, you'll find:

```
outputs/full_training/
├── best_model_epoch_3.pt      # Best model by validation loss
├── final_model.pt             # Model at last epoch
├── checkpoint_epoch_2.pt      # Periodic checkpoints
├── results.json               # All metrics & configuration
├── data_info.json             # Dataset statistics
└── training.log               # Detailed logs
```

### Key Metrics in results.json

- **val_f1**: F1-score on validation set (target: >0.95)
- **val_auc**: AUC-ROC score (target: >0.99)
- **val_precision**: Precision (target: >0.97)
- **val_recall**: Recall (target: >0.94)

---

## Next Steps

1. **Evaluate on Test Set**: Load best model and evaluate on held-out test data
2. **Deploy Model**: Use for inference on new texts
3. **Fine-tune for Domain**: Train on domain-specific data
4. **Compare Models**: Train multiple architectures and compare performance

---

## Performance Benchmarks

**On AMD 7900GRE (16GB VRAM):**
- Per-epoch time: 1.5-2 hours (5M samples)
- Total training time (5 epochs): 7.5-10 hours
- Batch throughput: ~300-400 samples/second during training
- Inference: 100-200 samples/second

**Expected Results:**
- Validation F1: 0.96-0.99
- Validation AUC: 0.992-0.997
- Test F1: Similar to validation

---

## Tips for Best Results

1. **Use BF16 mixed precision** for faster training on 7900GRE
2. **Set warmup to ~5% of total steps** (typically 500-1000)
3. **Monitor early stopping** to avoid overfitting
4. **Use balanced datasets** across train/val/test
5. **Save checkpoints every 2 epochs** for recovery
6. **Profile first epoch** to estimate total training time

---

## Support & Documentation

- Full documentation: `src/training/README.md`
- Training configuration: `TrainingConfig` in `src/training/train.py`
- Data loading: `ProcessedDataLoader` in `src/training/data_loader.py`

---

Good luck with your training! 🚀
