#!/usr/bin/env python3
"""
Model Saving Examples - Demonstrates checkpoint and inference model management

This script shows practical examples of:
1. Training with automatic checkpoint management
2. Resuming training from checkpoints
3. Saving inference models
4. Loading models for deployment
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from src.training.train import (
    ModelTrainer,
    TrainingConfig,
    DeBERTaAIDetector,
    CheckpointManager,
    ModelSaver
)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Training with Automatic Checkpoint Saving
# ============================================================================

def example_1_basic_training():
    """Train model with automatic checkpoint management."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Training with Automatic Checkpoints")
    logger.info("=" * 80)
    
    # Configure training
    config = TrainingConfig(
        base_model="microsoft/deberta-v3-small",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=3,  # Keep top 3 checkpoints
        early_stopping_patience=2,
        metric_for_best_model="val_f1",
        use_ema=True,
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config, output_dir="./outputs/example1")
    
    # Load sample data (replace with your data)
    train_data = pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2'] * 10,
        'label': [0, 1] * 10  # 0=human, 1=AI
    })
    
    val_data = pd.DataFrame({
        'text': ['Val text 1', 'Val text 2'] * 5,
        'label': [0, 1] * 5
    })
    
    # Train - checkpoints saved automatically
    logger.info("Starting training...")
    results = trainer.train(train_data, val_data)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")
    logger.info(f"Checkpoints saved in: {trainer.checkpoint_manager.save_dir}")
    logger.info(f"Inference models saved in: {trainer.model_saver.inference_dir}")
    
    return trainer


# ============================================================================
# EXAMPLE 2: Resuming Training from Checkpoint
# ============================================================================

def example_2_resume_training():
    """Resume training from a saved checkpoint."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Resuming Training from Checkpoint")
    logger.info("=" * 80)
    
    # Initialize trainer
    config = TrainingConfig(
        num_train_epochs=5,
        per_device_train_batch_size=8,
    )
    trainer = ModelTrainer(config, output_dir="./outputs/example2")
    
    # Resume from checkpoint
    checkpoint_path = "./outputs/example1/checkpoints/checkpoint_latest.safetensors"
    
    if Path(checkpoint_path).exists():
        logger.info(f"Resuming from: {checkpoint_path}")
        metadata = trainer.resume_from_checkpoint(checkpoint_path)
        
        logger.info(f"Resumed from epoch {metadata['epoch']}")
        logger.info(f"Resumed from step {metadata['global_step']}")
        logger.info(f"Previous metrics: {metadata.get('metrics', {})}")
        
        # Continue training
        train_data = pd.DataFrame({
            'text': ['Sample text'] * 20,
            'label': [0, 1] * 10
        })
        
        logger.info("Continuing training...")
        results = trainer.train(train_data)
        logger.info(f"Training completed! Total steps: {results['total_steps']}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Run example_1_basic_training() first!")


# ============================================================================
# EXAMPLE 3: Manual Checkpoint Management
# ============================================================================

def example_3_manual_checkpoints():
    """Manually save and load checkpoints."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Manual Checkpoint Management")
    logger.info("=" * 80)
    
    config = TrainingConfig()
    trainer = ModelTrainer(config, output_dir="./outputs/example3")
    
    # Create dummy optimizer and scheduler
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    # Manually save checkpoint
    logger.info("Saving checkpoint manually...")
    checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
        model=trainer.model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=trainer.ema,
        epoch=1,
        step=100,
        metrics={'val_f1': 0.85, 'val_loss': 0.25},
        config=config,
        checkpoint_type='best'
    )
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    # Load checkpoint
    logger.info("\nLoading checkpoint...")
    metadata = trainer.checkpoint_manager.load_checkpoint(
        filepath=checkpoint_path,
        model=trainer.model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=trainer.ema,
        load_training_state=True
    )
    
    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}, step {metadata['global_step']}")
    logger.info(f"Metrics: {metadata['metrics']}")


# ============================================================================
# EXAMPLE 4: Saving Inference Models
# ============================================================================

def example_4_inference_models():
    """Save models for inference and deployment."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Saving Inference Models")
    logger.info("=" * 80)
    
    config = TrainingConfig()
    trainer = ModelTrainer(config, output_dir="./outputs/example4")
    
    # Define metrics
    metrics = {
        'val_f1': 0.95,
        'val_accuracy': 0.94,
        'val_precision': 0.96,
        'val_recall': 0.94,
        'val_auc': 0.98,
    }
    
    # Save for inference
    logger.info("Saving inference model...")
    trainer.model_saver.save_for_inference(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        metrics=metrics,
        config=config,
        version='v1.0_demo',
        save_traced=False  # Set True for TorchScript
    )
    
    model_dir = trainer.model_saver.inference_dir / 'model_v1.0_demo'
    logger.info(f"\nInference model saved to: {model_dir}")
    logger.info("Files created:")
    for file in sorted(model_dir.glob("*")):
        logger.info(f"  - {file.name}")


# ============================================================================
# EXAMPLE 5: Loading Inference Models
# ============================================================================

def example_5_load_inference_model():
    """Load an inference model for deployment."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Loading Inference Model")
    logger.info("=" * 80)
    
    model_dir = "./outputs/example4/models/inference/model_v1.0_demo"
    
    if not Path(model_dir).exists():
        logger.warning(f"Model directory not found: {model_dir}")
        logger.info("Run example_4_inference_models() first!")
        return
    
    logger.info(f"Loading model from: {model_dir}")
    
    # Load metadata
    import json
    with open(f"{model_dir}/training_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model version: {metadata['version']}")
    logger.info(f"Training date: {metadata['training_date']}")
    logger.info(f"Metrics: {metadata['metrics']}")
    
    # Reconstruct configuration
    config = TrainingConfig(**metadata['hyperparameters'])
    
    # Initialize model
    model = DeBERTaAIDetector(config)
    
    # Load weights
    checkpoint = torch.load(f"{model_dir}/full_model.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    logger.info("\n✓ Model loaded successfully!")
    
    # Test inference
    logger.info("\nTesting inference...")
    test_text = "This is a sample text to classify as human or AI-generated."
    
    inputs = tokenizer(
        test_text,
        return_tensors='pt',
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs['logits']
        prediction = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
    
    label = "AI-generated" if prediction == 1 else "Human"
    logger.info(f"Text: {test_text[:50]}...")
    logger.info(f"Prediction: {label} (confidence: {confidence:.2%})")


# ============================================================================
# EXAMPLE 6: Checkpoint Manager Advanced Usage
# ============================================================================

def example_6_checkpoint_manager_advanced():
    """Advanced checkpoint manager usage."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: Advanced Checkpoint Manager")
    logger.info("=" * 80)
    
    # Create checkpoint manager with custom settings
    checkpoint_dir = Path("./outputs/example6/checkpoints")
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        keep_top_k=5,  # Keep top 5 checkpoints
        metric='val_f1',
        mode='max',
        use_safetensors=True
    )
    
    # Initialize model and optimizer
    config = TrainingConfig()
    model = DeBERTaAIDetector(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Simulate multiple checkpoints with different metrics
    logger.info("Simulating checkpoint saves with different metrics...")
    
    metrics_list = [
        {'val_f1': 0.85, 'val_loss': 0.25},
        {'val_f1': 0.88, 'val_loss': 0.22},
        {'val_f1': 0.90, 'val_loss': 0.20},  # Best
        {'val_f1': 0.87, 'val_loss': 0.23},
        {'val_f1': 0.89, 'val_loss': 0.21},
        {'val_f1': 0.86, 'val_loss': 0.24},
        {'val_f1': 0.91, 'val_loss': 0.19},  # New best
    ]
    
    for epoch, metrics in enumerate(metrics_list, 1):
        logger.info(f"\nEpoch {epoch}: val_f1={metrics['val_f1']:.4f}")
        checkpoint_manager.manage_checkpoints(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            ema=None,
            epoch=epoch,
            step=epoch * 100,
            metrics=metrics,
            config=config
        )
    
    # Show saved checkpoints
    logger.info("\n" + "=" * 80)
    logger.info("Saved checkpoints:")
    for checkpoint_file in sorted(checkpoint_dir.glob("*.safetensors")):
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {checkpoint_file.name} ({size_mb:.2f} MB)")
    
    logger.info(f"\nTop {checkpoint_manager.keep_top_k} checkpoints by val_f1:")
    for metric_value, filepath, metadata in checkpoint_manager.checkpoints:
        logger.info(f"  F1={metric_value:.4f} - Epoch {metadata['epoch']} - {filepath.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL SAVING EXAMPLES")
    logger.info("=" * 80)
    
    print("\nAvailable examples:")
    print("1. Basic training with automatic checkpoints")
    print("2. Resume training from checkpoint")
    print("3. Manual checkpoint management")
    print("4. Save inference models")
    print("5. Load inference models")
    print("6. Advanced checkpoint manager usage")
    print("0. Run all examples")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    examples = {
        '1': example_1_basic_training,
        '2': example_2_resume_training,
        '3': example_3_manual_checkpoints,
        '4': example_4_inference_models,
        '5': example_5_load_inference_model,
        '6': example_6_checkpoint_manager_advanced,
    }
    
    if choice == '0':
        # Run all examples
        for func in examples.values():
            try:
                func()
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
    elif choice in examples:
        examples[choice]()
    else:
        logger.warning("Invalid choice")
    
    logger.info("\n" + "=" * 80)
    logger.info("Examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
