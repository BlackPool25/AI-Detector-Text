# AI Text Detection Model Training

Complete training pipeline for binary AI text detection using DeBERTa-v3-small with advanced optimization techniques.

## Architecture Overview

### Model Stack
- **Base Model**: DeBERTa-v3-small (optimal speed-accuracy tradeoff on 16GB GPU)
- **Frozen Embeddings**: Stabilizes training and reduces overfitting
- **Sequential Processing**: 2-layer Bi-LSTM (512 hidden units) captures long-range dependencies
- **Attention Pooling**: Learns to focus on discriminative features
- **Classification Head**: Binary cross-entropy with label smoothing

### Key Optimizations

#### 1. **Mixed Precision Training (BF16/FP16)**
- Reduces memory by ~50%
- Speeds up training by 2-3x
- Maintains model accuracy with proper gradient scaling

#### 2. **Gradient Management**
- **Accumulation**: Effective batch size = per_device × accumulation_steps (32-64)
- **Clipping**: Max norm 1.0 prevents exploding gradients
- **Component-wise optimization** for different layer groups

#### 3. **Advanced Optimizer Configuration**
- **AdamW** with differentiated learning rates:
  - Transformer backbone: 2e-5
  - LSTM + FC layers: 1e-3
- **Linear warmup** (500-1000 steps) + decay-to-zero schedule
- **Weight decay**: 0.01 for regularization

#### 4. **Data Augmentation**
- **Noise injection (10%)**: Random junk/garbled words for robustness
- **Back-translation (15%)**: Semantic variation
- **20-30%** of training samples augmented

#### 5. **Regularization & Stability**
- **Dropout**: 0.3-0.5 in LSTM and classification layers
- **Label smoothing**: 0.1 factor reduces overconfidence
- **Early stopping**: Patience=3 epochs

#### 6. **Model Averaging**
- **EMA (Exponential Moving Average)**: Decay=0.999 improves robustness
- Maintains shadow model without additional training cost
- Particularly effective for transformer fine-tuning

#### 7. **Class Imbalance Handling**
- **Stratified sampling** in train/val/test splits
- **Data processed**: ~5.5M samples with 2.87% human, 97.13% AI
- Balanced within train/val/test sets

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- AMD 7900GRE GPU (16GB VRAM) recommended or equivalent

### Setup

```bash
# Clone repository
cd /home/lightdesk/Projects/Text

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Format

Data is processed and stored in Parquet format organized by:
```
processed_data/
├── ai/RAID-Dataset/
│   ├── chatgpt/
│   ├── cohere/
│   ├── gpt2/
│   ├── gpt3/
│   ├── gpt4/
│   ├── llama-chat/
│   ├── mistral/
│   ├── mistral-chat/
│   ├── mpt/
│   ├── mpt-chat/
│   └── cohere-chat/
├── real/RAID-Dataset/
│   └── human/
├── statistics.json
└── checkpoint.json
```

**Dataset Statistics:**
- Total samples: 5,508,125
- Human samples: 158,078 (2.87%)
- AI samples: 5,350,047 (97.13%)
- Domains: 8 (abstracts, books, news, poetry, recipes, reddit, reviews, wiki)

## Training Configuration

### Default Configuration

```python
training_config = {
    # Model Architecture
    "base_model": "microsoft/deberta-v3-small",
    "freeze_embeddings": True,
    "lstm_layers": 2,
    "lstm_hidden_size": 512,
    "attention_pooling": True,
    "dropout": 0.4,
    
    # Training Hyperparameters
    "max_seq_length": 512,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,  # Effective batch: 64
    "num_train_epochs": 5,
    
    # Optimizer (Differentiated LR)
    "learning_rate": 2e-5,      # Transformer
    "lstm_lr": 1e-3,            # Task-specific layers
    "weight_decay": 0.01,
    
    # Mixed Precision
    "bf16": True,               # BF16 on AMD 7900GRE
    
    # Learning Rate Schedule
    "lr_scheduler_type": "linear",
    "warmup_steps": 500,
    
    # Regularization
    "label_smoothing": 0.1,
    "dropout": 0.4,
    
    # Augmentation
    "noise_injection_prob": 0.1,
    "back_translation_prob": 0.15,
    
    # Model Averaging
    "use_ema": True,
    "ema_decay": 0.999,
    
    # Early Stopping
    "early_stopping_patience": 3,
}
```

## Usage

### Quick Start: Train on All Models

```bash
python src/training/train_script.py \
    --load-mode all \
    --num-epochs 5 \
    --batch-size 16 \
    --output-dir ./outputs/deberta_all_models
```

### Train on Specific Model

```bash
python src/training/train_script.py \
    --load-mode model \
    --model-name gpt4 \
    --ai-sample 500000 \
    --human-sample 500000 \
    --num-epochs 5 \
    --output-dir ./outputs/deberta_gpt4
```

### Custom Configuration

```bash
python src/training/train_script.py \
    --load-mode all \
    --num-epochs 3 \
    --batch-size 24 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-5 \
    --lstm-lr 5e-4 \
    --lstm-hidden-size 256 \
    --warmup-steps 1000 \
    --bf16 True \
    --use-ema True \
    --ema-decay 0.9999 \
    --output-dir ./outputs/custom_config
```

### Available Arguments

**Data Loading:**
- `--data-dir` (default: `./processed_data`): Path to processed data
- `--load-mode` (choices: `all`, `model`): Load strategy
- `--model-name` (default: `gpt4`): Model name for mode='model'
- `--ai-sample`: Optional limit on AI samples
- `--human-sample`: Optional limit on human samples

**Model Architecture:**
- `--base-model`: HuggingFace model ID
- `--freeze-embeddings`: Freeze embedding layers
- `--lstm-layers`: Number of LSTM layers
- `--lstm-hidden-size`: LSTM hidden dimension
- `--attention-pooling`: Use attention pooling
- `--dropout`: Dropout rate

**Training:**
- `--num-epochs`: Training epochs (3-5 recommended)
- `--batch-size`: Batch size per device (8-24)
- `--gradient-accumulation-steps`: Accumulation steps (2-8)
- `--max-seq-length`: Max token sequence (512 optimal)

**Optimizer:**
- `--learning-rate`: Transformer LR (2e-5 recommended)
- `--lstm-lr`: LSTM/FC LR (1e-3 recommended)
- `--weight-decay`: L2 regularization (0.01 standard)
- `--lr-scheduler`: Scheduler type
- `--warmup-steps`: Warmup steps (500-1000)

**Mixed Precision:**
- `--bf16`: Use BF16 (True for 7900GRE)
- `--fp16`: Use FP16 (alternative)
- `--max-grad-norm`: Gradient clipping norm (1.0)

**Regularization:**
- `--label-smoothing`: Label smoothing (0.1)
- `--noise-injection-prob`: Augmentation probability (0.1)
- `--back-translation-prob`: Back-translation prob (0.15)

**Model Averaging:**
- `--use-ema`: Use EMA
- `--ema-decay`: EMA decay factor (0.999)

**Early Stopping:**
- `--early-stopping-patience`: Patience epochs (3)

## Output Files

Training generates in output directory:

```
outputs/deberta_all_models/
├── best_model_epoch_3.pt          # Best checkpoint
├── checkpoint_epoch_2.pt           # Periodic checkpoints
├── checkpoint_epoch_4.pt
├── final_model.pt                  # Final trained model
├── results.json                    # Training metrics & history
├── data_info.json                  # Dataset split information
└── training.log                    # Detailed training logs
```

### results.json Format

```json
{
  "config": { /* complete training configuration */ },
  "data_info": {
    "train_size": 4406500,
    "val_size": 550812,
    "test_size": 550813,
    "train_human": 126462,
    "train_ai": 4280038,
    "val_human": 15808,
    "val_ai": 535004,
    "test_human": 15808,
    "test_ai": 535005
  },
  "training_history": {
    "train_loss": [0.156, 0.124, ...],
    "train_f1": [0.951, 0.962, ...],
    "val_loss": [0.089, 0.087, ...],
    "val_f1": [0.967, 0.971, ...],
    "val_precision": [0.975, 0.978, ...],
    "val_recall": [0.959, 0.964, ...],
    "val_auc": [0.991, 0.993, ...]
  },
  "best_val_loss": 0.087
}
```

## Expected Performance

Based on the architecture and training setup:

- **Validation F1-score**: 0.95-1.0 (binary classification)
- **Validation AUC**: 0.99+
- **Training time**: 6-12 hours for 5M samples (3-5 epochs) on AMD 7900GRE
- **Memory usage**: 10-14GB VRAM with BF16 mixed precision
- **Inference speed**: 100-200 samples/second

## Loading and Using Trained Models

### Load Checkpoint

```python
from src.training.train import ModelTrainer, TrainingConfig
import torch
from pathlib import Path

# Load configuration
checkpoint = torch.load("outputs/deberta_all_models/best_model_epoch_3.pt")
config_dict = checkpoint['config']
config = TrainingConfig(**config_dict)

# Initialize trainer
trainer = ModelTrainer(config)

# Load model weights
trainer.load_checkpoint(Path("outputs/deberta_all_models/best_model_epoch_3.pt"))

# Use for inference
trainer.model.eval()
```

### Inference Example

```python
import torch
from transformers import AutoTokenizer

# Prepare input
text = "This is a sample text to classify."
inputs = trainer.tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Forward pass
with torch.no_grad():
    outputs = trainer.model(
        inputs['input_ids'].to(trainer.config.device),
        inputs['attention_mask'].to(trainer.config.device)
    )
    logits = outputs['logits']
    prediction = torch.softmax(logits, dim=-1)
    
    # prediction[:, 0] = human probability
    # prediction[:, 1] = AI probability
    print(f"Human: {prediction[0, 0].item():.4f}")
    print(f"AI: {prediction[0, 1].item():.4f}")
```

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**: `--batch-size 8` or `--batch-size 12`
2. **Increase gradient accumulation**: `--gradient-accumulation-steps 8`
3. **Reduce sequence length**: `--max-seq-length 256`
4. **Enable gradient checkpointing**: (add to code if needed)

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Set device explicitly
export CUDA_VISIBLE_DEVICES=0
```

### Slow Training

1. Verify GPU utilization: `nvidia-smi`
2. Check data loading bottleneck: Increase `--num-workers`
3. Verify BF16 is enabled: Check logs for "dtype=torch.bfloat16"

### Poor Performance

1. Increase training epochs: `--num-epochs 7-10`
2. Adjust learning rates for fine-tuning
3. Increase augmentation: `--noise-injection-prob 0.2`
4. Load more balanced data across models

## References

- DeBERTa-v3-small: https://huggingface.co/microsoft/deberta-v3-small
- PyTorch Mixed Precision: https://pytorch.org/docs/stable/amp.html
- Gradient Accumulation: https://pytorch.org/docs/stable/notes/amp_examples.html
- EMA techniques: https://arxiv.org/abs/1505.3212

## License

See parent repository license
