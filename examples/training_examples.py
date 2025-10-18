#!/usr/bin/env python3
"""
Example Usage and Integration Tests for Training Module

Demonstrates:
1. Loading processed data
2. Training model with various configurations  
3. Making predictions
4. Evaluating performance
"""

import logging
from pathlib import Path
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.training.train import ModelTrainer, TrainingConfig
from src.training.data_loader import ProcessedDataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_load_data():
    """Example 1: Load processed data."""
    logger.info("\n" + "="*80)
    logger.info("Example 1: Loading Processed Data")
    logger.info("="*80)
    
    loader = ProcessedDataLoader(data_dir=Path("./processed_data"))
    
    # Get summary
    summary = loader.get_data_summary()
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Human samples: {summary['human_samples']}")
    logger.info(f"AI samples: {summary['ai_samples']}")
    logger.info(f"Available models: {summary['ai_models']}")
    
    # Find available data
    files = loader.find_parquet_files()
    logger.info(f"\nFound {len(files)} data categories")
    for category in list(files.keys())[:3]:
        logger.info(f"  {category}: {len(files[category])} files")


def example_2_quick_training():
    """Example 2: Quick training with small dataset."""
    logger.info("\n" + "="*80)
    logger.info("Example 2: Quick Training (Small Dataset)")
    logger.info("="*80)
    
    # Load small dataset for demonstration
    loader = ProcessedDataLoader(data_dir=Path("./processed_data"))
    
    try:
        train_data, val_data, test_data = loader.load_all_models(
            ai_sample_per_model=5000,      # 5k AI samples per model
            human_sample_per_model=5000,   # 5k human samples
        )
        
        logger.info(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Configure for quick training
        config = TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            lstm_hidden_size=256,
            dropout=0.3,
            bf16=torch.cuda.is_available(),
        )
        
        logger.info(f"Using device: {config.device}")
        
        # Initialize trainer
        trainer = ModelTrainer(config, output_dir="./outputs/example_quick_train")
        
        # Train
        logger.info("Starting training...")
        results = trainer.train(
            train_data=train_data,
            val_data=val_data,
            text_column='text',
            label_column='label',
        )
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in quick training: {e}")
        logger.info("This is expected if processed_data is not available")


def example_3_inference():
    """Example 3: Make predictions with trained model."""
    logger.info("\n" + "="*80)
    logger.info("Example 3: Model Inference")
    logger.info("="*80)
    
    # Create a simple trained model for demonstration
    config = TrainingConfig(
        num_train_epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    trainer = ModelTrainer(config)
    trainer.model.eval()
    
    # Example texts
    texts = [
        "The quantum mechanics of subatomic particles has been extensively studied.",
        "In this paper we propose a novel approach to artificial intelligence.",
        "I absolutely loved this book, couldn't put it down!",
        "The recipe calls for two cups of flour and one cup of sugar.",
    ]
    
    logger.info("Making predictions on sample texts...\n")
    
    for text in texts:
        inputs = trainer.tokenizer(
            text[:512],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = trainer.model(
                inputs['input_ids'].to(config.device),
                inputs['attention_mask'].to(config.device)
            )
            
            logits = outputs['logits'][0]
            probs = torch.softmax(logits, dim=-1)
            
            prediction = "Human" if probs[0] > probs[1] else "AI"
            confidence = max(probs).item()
            
            logger.info(f"Text: '{text[:60]}...'")
            logger.info(f"  Prediction: {prediction}")
            logger.info(f"  Confidence: {confidence:.4f}")
            logger.info(f"  Human prob: {probs[0]:.4f}, AI prob: {probs[1]:.4f}\n")


def example_4_evaluation():
    """Example 4: Evaluate model on test set."""
    logger.info("\n" + "="*80)
    logger.info("Example 4: Model Evaluation")
    logger.info("="*80)
    
    try:
        # Load test data
        loader = ProcessedDataLoader(data_dir=Path("./processed_data"))
        _, _, test_data = loader.load_all_models(
            ai_sample_per_model=1000,
            human_sample_per_model=1000,
        )
        
        # Create model
        config = TrainingConfig(device="cuda" if torch.cuda.is_available() else "cpu")
        trainer = ModelTrainer(config)
        trainer.model.eval()
        
        # Create evaluation dataset
        from src.training.train import AITextDataset
        from torch.utils.data import DataLoader, SequentialSampler
        
        eval_dataset = AITextDataset(
            texts=test_data['text'].tolist(),
            labels=test_data['label'].tolist(),
            tokenizer=trainer.tokenizer,
            max_length=512,
            augment=False,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=32,
            sampler=SequentialSampler(eval_dataset),
        )
        
        # Evaluate
        metrics = trainer.validate(eval_loader)
        
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.info("This is expected if test data is not available")


def example_5_training_config_presets():
    """Example 5: Different training configuration presets."""
    logger.info("\n" + "="*80)
    logger.info("Example 5: Training Configuration Presets")
    logger.info("="*80)
    
    presets = {
        "Fast": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 32,
            "lstm_hidden_size": 256,
            "warmup_steps": 100,
        },
        "Balanced": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "lstm_hidden_size": 512,
            "warmup_steps": 500,
        },
        "Thorough": {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "lstm_hidden_size": 512,
            "warmup_steps": 1000,
            "use_ema": True,
            "ema_decay": 0.9999,
        },
    }
    
    for preset_name, params in presets.items():
        config = TrainingConfig(**params)
        
        # Calculate effective batch size
        eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
        
        logger.info(f"\n{preset_name} Preset:")
        logger.info(f"  Epochs: {config.num_train_epochs}")
        logger.info(f"  Batch size: {config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch: {eff_batch}")
        logger.info(f"  LSTM hidden: {config.lstm_hidden_size}")
        logger.info(f"  Warmup steps: {config.warmup_steps}")
        logger.info(f"  Use EMA: {config.use_ema}")


def example_6_config_saving():
    """Example 6: Save and load training configuration."""
    logger.info("\n" + "="*80)
    logger.info("Example 6: Configuration Management")
    logger.info("="*80)
    
    # Create config
    config = TrainingConfig(
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=500,
    )
    
    # Save to JSON
    config_dict = config.to_dict()
    config_path = Path("./outputs/example_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Load from JSON
    with open(config_path, 'r') as f:
        loaded_config_dict = json.load(f)
    
    loaded_config = TrainingConfig(**loaded_config_dict)
    logger.info("Configuration reloaded successfully")
    logger.info(f"Learning rate: {loaded_config.learning_rate}")
    logger.info(f"Num epochs: {loaded_config.num_train_epochs}")


def example_7_device_check():
    """Example 7: Check device and setup."""
    logger.info("\n" + "="*80)
    logger.info("Example 7: Device & Environment Check")
    logger.info("="*80)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1e9:.2f} GB")
        
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available - training will be on CPU (slow)")


if __name__ == "__main__":
    logger.info("AI Text Detection - Training Module Examples")
    
    # Run examples
    example_7_device_check()
    example_5_training_config_presets()
    example_6_config_saving()
    example_1_load_data()
    example_3_inference()
    
    # These require data/time:
    # example_2_quick_training()
    # example_4_evaluation()
    
    logger.info("\n" + "="*80)
    logger.info("Examples completed!")
    logger.info("See src/training/README.md for full documentation")
    logger.info("="*80)
