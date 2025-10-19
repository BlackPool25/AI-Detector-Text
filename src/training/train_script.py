#!/usr/bin/env python3
"""
Training Script for AI Text Detection Model

Complete training pipeline with DeBERTa-v3-small architecture,
data loading from processed_data, and comprehensive logging.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
import pandas as pd

# Handle both direct execution and module import
try:
    from .train import ModelTrainer, TrainingConfig
    from .data_loader import ProcessedDataLoader
except ImportError:
    # Direct execution - add parent directory to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.training.train import ModelTrainer, TrainingConfig
    from src.training.data_loader import ProcessedDataLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    return TrainingConfig(
        base_model=args.base_model,
        freeze_embeddings=args.freeze_embeddings,
        lstm_layers=args.lstm_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        attention_pooling=args.attention_pooling,
        dropout=args.dropout,
        
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        
        learning_rate=args.learning_rate,
        lstm_lr=args.lstm_lr,
        weight_decay=args.weight_decay,
        
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        
        label_smoothing=args.label_smoothing,
        
        noise_injection_prob=args.noise_injection_prob,
        back_translation_prob=args.back_translation_prob,
        
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        
        early_stopping_patience=args.early_stopping_patience,
        
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main(args):
    """Main training function."""
    
    logger.info("=" * 80)
    logger.info("Starting AI Text Detection Model Training")
    logger.info("=" * 80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, training will be on CPU")
    
    # Load training configuration
    config = get_training_config(args)
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("Loading Data")
    logger.info("=" * 80)
    
    data_loader = ProcessedDataLoader(args.data_dir)
    
    # Print data summary
    summary = data_loader.get_data_summary()
    logger.info("Processed Data Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Load datasets
    if args.load_mode == "all":
        logger.info("Loading data from all AI models...")
        train_data, val_data, test_data = data_loader.load_all_models(
            ai_sample_per_model=args.ai_sample,
            human_sample_per_model=args.human_sample,
        )
    else:
        logger.info(f"Loading data for model: {args.model_name}")
        train_data, val_data, test_data = data_loader.load_by_model(
            model_name=args.model_name,
            sample_size=args.ai_sample,
        )
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(train_data)}")
    logger.info(f"  Val:   {len(val_data)}")
    logger.info(f"  Test:  {len(test_data)}")
    
    # Save data info
    data_info = {
        'train_size': int(len(train_data)),
        'val_size': int(len(val_data)),
        'test_size': int(len(test_data)),
        'train_human': int((train_data['label'] == 0).sum()),
        'train_ai': int((train_data['label'] == 1).sum()),
        'val_human': int((val_data['label'] == 0).sum()),
        'val_ai': int((val_data['label'] == 1).sum()),
        'test_human': int((test_data['label'] == 0).sum()),
        'test_ai': int((test_data['label'] == 1).sum()),
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "data_info.json", 'w') as f:
        json.dump(data_info, f, indent=2)
    
    # Initialize trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Model Trainer")
    logger.info("=" * 80)
    
    trainer = ModelTrainer(config, output_dir=args.output_dir)
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    
    results = trainer.train(
        train_data=train_data,
        val_data=val_data,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    
    # Save training results
    logger.info("\n" + "=" * 80)
    logger.info("Saving Training Results")
    logger.info("=" * 80)
    
    # Convert history to serializable format
    history = results['history']
    history_serializable = {
        key: [float(v) for v in values] 
        for key, values in history.items()
    }
    
    results_summary = {
        'config': config.to_dict(),
        'data_info': data_info,
        'training_history': history_serializable,
        'best_val_loss': float(results['best_val_loss']),
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'results.json'}")
    
    # Save final model
    trainer.save_checkpoint(output_dir / "final_model.pt")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AI Text Detection Model with DeBERTa-v3-small"
    )
    
    # Data loading arguments
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--data-dir",
        type=str,
        default="./processed_data",
        help="Directory containing processed data"
    )
    data_group.add_argument(
        "--load-mode",
        type=str,
        choices=["all", "model"],
        default="all",
        help="Load all models or specific model"
    )
    data_group.add_argument(
        "--model-name",
        type=str,
        default="gpt4",
        help="Model name when load_mode='model'"
    )
    data_group.add_argument(
        "--ai-sample",
        type=int,
        default=None,
        help="Number of AI samples to load"
    )
    data_group.add_argument(
        "--human-sample",
        type=int,
        default=None,
        help="Number of human samples to load"
    )
    data_group.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in data"
    )
    data_group.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of label column in data"
    )
    
    # Model architecture arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--base-model",
        type=str,
        default="microsoft/deberta-v3-small",
        help="Base transformer model"
    )
    model_group.add_argument(
        "--freeze-embeddings",
        type=bool,
        default=True,
        help="Freeze embedding layers"
    )
    model_group.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )
    model_group.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=512,
        help="LSTM hidden size"
    )
    model_group.add_argument(
        "--attention-pooling",
        type=bool,
        default=True,
        help="Use attention pooling"
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate"
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device"
    )
    train_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Optimizer arguments
    opt_group = parser.add_argument_group("Optimizer")
    opt_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for transformer"
    )
    opt_group.add_argument(
        "--lstm-lr",
        type=float,
        default=1e-3,
        help="Learning rate for LSTM/FC layers"
    )
    opt_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    opt_group.add_argument(
        "--lr-scheduler",
        type=str,
        default="linear",
        help="Learning rate scheduler type"
    )
    opt_group.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Warmup steps"
    )
    
    # Mixed precision arguments
    precision_group = parser.add_argument_group("Mixed Precision")
    precision_group.add_argument(
        "--bf16",
        type=bool,
        default=True,
        help="Use bfloat16 precision"
    )
    precision_group.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="Use float16 precision"
    )
    precision_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # Regularization arguments
    reg_group = parser.add_argument_group("Regularization")
    reg_group.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor"
    )
    reg_group.add_argument(
        "--noise-injection-prob",
        type=float,
        default=0.1,
        help="Noise injection probability"
    )
    reg_group.add_argument(
        "--back-translation-prob",
        type=float,
        default=0.15,
        help="Back-translation probability"
    )
    
    # Model averaging arguments
    avg_group = parser.add_argument_group("Model Averaging")
    avg_group.add_argument(
        "--use-ema",
        type=bool,
        default=True,
        help="Use exponential moving average"
    )
    avg_group.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay factor"
    )
    
    # Early stopping arguments
    stop_group = parser.add_argument_group("Early Stopping")
    stop_group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/model_checkpoints",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Run training
    main(args)
