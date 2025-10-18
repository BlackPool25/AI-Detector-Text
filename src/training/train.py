#!/usr/bin/env python3
"""
Model Training Module for AI Text Detection with DeBERTa-v3-small

Implements the optimized architecture for binary classification (human vs AI text)
with DeBERTa-v3-small backbone, Bi-LSTM layers, and advanced training techniques.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, asdict
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logging.warning("safetensors not available. Install with: pip install safetensors")


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration matching optimized DeBERTa-v3-small setup."""
    
    # Model Architecture
    base_model: str = "microsoft/deberta-v3-small"
    freeze_embeddings: bool = True
    lstm_layers: int = 2
    lstm_hidden_size: int = 512
    attention_pooling: bool = True
    dropout: float = 0.4
    
    # Training Hyperparameters
    max_seq_length: int = 512
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 5
    max_steps: Optional[int] = None
    
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 2e-5  # Transformer
    lstm_lr: float = 1e-3  # Task-specific layers
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning Rate Schedule
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 500
    
    # Mixed Precision & Gradient Management
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    
    # Regularization
    label_smoothing: float = 0.1
    
    # Data Augmentation
    noise_injection_prob: float = 0.1
    back_translation_prob: float = 0.15
    
    # Model Averaging
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Early Stopping
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_loss"
    
    # Checkpointing
    save_total_limit: int = 3
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 100
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class AITextDataset(Dataset):
    """Dataset for AI text detection with optional augmentation."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        augment: bool = False,
        noise_injection_prob: float = 0.1,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (0=human, 1=AI)
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            augment: Whether to apply augmentation
            noise_injection_prob: Probability of noise injection
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.noise_injection_prob = noise_injection_prob
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def _inject_noise(self, text: str) -> str:
        """Inject random noise/garbled words into text."""
        if np.random.random() > self.noise_injection_prob:
            return text
        
        words = text.split()
        noise_words = ['xyzabc', 'qwerty', 'zxcvbn', 'asdfgh', 'poiuyt']
        
        # Add random junk (3-8 chars)
        num_insertions = np.random.randint(1, 4)
        for _ in range(num_insertions):
            idx = np.random.randint(0, len(words))
            noise_word = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 
                                                   size=np.random.randint(3, 9)))
            words.insert(idx, noise_word)
        
        return ' '.join(words)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment:
            text = self._inject_noise(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class AttentionPooling(nn.Module):
    """Linear attention pooling layer."""
    
    def __init__(self, hidden_size: int):
        """Initialize attention pooling."""
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_size)
            mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Pooled representation (batch_size, hidden_size)
        """
        # Compute attention weights
        scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Mask attention scores
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)
        weights = weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        pooled = torch.sum(weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        return pooled


class DeBERTaAIDetector(nn.Module):
    """
    DeBERTa-v3-small based AI text detector.
    
    Architecture:
    - Frozen DeBERTa embeddings for feature extraction
    - Bi-LSTM layers for sequential processing
    - Attention pooling for discriminative feature selection
    - Classification head with dropout
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize model."""
        super().__init__()
        self.config = config
        
        # Load pre-trained DeBERTa model
        self.deberta = AutoModel.from_pretrained(config.base_model)
        
        # Freeze embeddings
        if config.freeze_embeddings:
            for param in self.deberta.embeddings.parameters():
                param.requires_grad = False
        
        hidden_size = self.deberta.config.hidden_size
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        
        lstm_output_size = config.lstm_hidden_size * 2  # bidirectional
        
        # Attention pooling
        if config.attention_pooling:
            self.attention_pool = AttentionPooling(lstm_output_size)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_output_size, 2)  # binary classification
        
        # Loss function with label smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Dictionary with logits and loss (if labels provided)
        """
        # DeBERTa forward pass
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use last hidden state
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_len, lstm_output_size)
        
        # Attention pooling
        if self.config.attention_pooling:
            pooled = self.attention_pool(lstm_output, attention_mask)
        else:
            # Use CLS token representation
            pooled = lstm_output[:, 0, :]
        
        # Classification head
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch_size, 2)
        
        output_dict = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output_dict["loss"] = loss
        
        return output_dict


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================================

class ExponentialMovingAverage:
    """Exponential Moving Average for weight averaging."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay factor
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register_shadow_params()
    
    def _register_shadow_params(self):
        """Register shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA shadow state dict."""
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA shadow state dict."""
        self.shadow = state_dict.copy()


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoints with top-k best checkpoints tracking.
    Implements best practices for model saving and loading.
    """
    
    def __init__(
        self, 
        save_dir: Path,
        keep_top_k: int = 3,
        metric: str = 'val_f1',
        mode: str = 'max',
        use_safetensors: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            keep_top_k: Number of best checkpoints to keep
            metric: Metric name to track for best checkpoints
            mode: 'max' or 'min' for metric tracking
            use_safetensors: Use safetensors format if available
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.use_safetensors = use_safetensors and SAFETENSORS_AVAILABLE
        self.checkpoints = []  # List of (metric_value, filepath, metadata)
        
        logger.info(f"CheckpointManager initialized at {self.save_dir}")
        logger.info(f"Using {'safetensors' if self.use_safetensors else 'torch'} format")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        ema: Optional[ExponentialMovingAverage],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: TrainingConfig,
        is_best: bool = False,
        checkpoint_type: str = 'regular'
    ) -> Path:
        """
        Save a complete training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            ema: EMA state if used
            epoch: Current epoch
            step: Global training step
            metrics: Training metrics dictionary
            config: Training configuration
            is_best: Whether this is the best checkpoint
            checkpoint_type: 'latest', 'best', 'epoch', or 'regular'
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': step,
            'metrics': metrics,
            'config': config.to_dict(),
        }
        
        # Add EMA if available
        if ema is not None:
            checkpoint['ema_state_dict'] = ema.state_dict()
        
        # Add scheduler if available
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save random states for reproducibility
        checkpoint['rng_state'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        
        # Determine filename
        if checkpoint_type == 'latest':
            filename = 'checkpoint_latest'
        elif checkpoint_type == 'best':
            filename = f'checkpoint_best_{self.metric}'
        elif checkpoint_type == 'epoch':
            filename = f'checkpoint_epoch_{epoch}'
        else:
            filename = f'checkpoint_step_{step}'
        
        # Add extension
        ext = '.safetensors' if self.use_safetensors else '.pt'
        filepath = self.save_dir / f'{filename}{ext}'
        
        # Save checkpoint
        if self.use_safetensors:
            self._save_safetensors(checkpoint, filepath, epoch, step, metrics)
        else:
            torch.save(checkpoint, filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def _save_safetensors(
        self, 
        checkpoint: Dict[str, Any], 
        filepath: Path,
        epoch: int,
        step: int,
        metrics: Dict[str, float]
    ):
        """Save checkpoint using safetensors format with metadata."""
        # Prepare metadata (must be strings)
        metadata = {
            'epoch': str(epoch),
            'global_step': str(step),
            'timestamp': datetime.now().isoformat(),
            'format': 'safetensors',
        }
        
        # Add metrics to metadata
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metadata[f'metric_{key}'] = str(value)
        
        # Extract tensors for safetensors
        tensors = {}
        non_tensors = {}
        
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, dict) and key == 'model_state_dict':
                # Flatten model state dict
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        tensors[f'model.{sub_key}'] = sub_value
            elif isinstance(value, dict) and key == 'optimizer_state_dict':
                # Save optimizer separately as it contains non-tensor data
                non_tensors['optimizer_state_dict'] = value
            elif isinstance(value, dict) and key == 'ema_state_dict':
                # Flatten EMA state dict
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        tensors[f'ema.{sub_key}'] = sub_value
            elif isinstance(value, dict) and key == 'scheduler_state_dict':
                non_tensors['scheduler_state_dict'] = value
            else:
                non_tensors[key] = value
        
        # Save tensors with safetensors
        save_file(tensors, filepath, metadata=metadata)
        
        # Save non-tensor data separately
        if non_tensors:
            non_tensor_path = filepath.with_suffix('.meta.pt')
            torch.save(non_tensors, non_tensor_path)
    
    def save_latest(self, model, optimizer, scheduler, ema, epoch, step, metrics, config):
        """Save the latest checkpoint (always kept)."""
        return self.save_checkpoint(
            model, optimizer, scheduler, ema, epoch, step, 
            metrics, config, checkpoint_type='latest'
        )
    
    def save_best(self, model, optimizer, scheduler, ema, epoch, step, metrics, config):
        """Save the best checkpoint based on metric."""
        return self.save_checkpoint(
            model, optimizer, scheduler, ema, epoch, step,
            metrics, config, is_best=True, checkpoint_type='best'
        )
    
    def save_epoch(self, model, optimizer, scheduler, ema, epoch, step, metrics, config):
        """Save periodic epoch checkpoint."""
        return self.save_checkpoint(
            model, optimizer, scheduler, ema, epoch, step,
            metrics, config, checkpoint_type='epoch'
        )
    
    def manage_checkpoints(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        ema: Optional[ExponentialMovingAverage],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: TrainingConfig
    ):
        """
        Manage checkpoints: save latest, track best, clean old checkpoints.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            ema: EMA state if used
            epoch: Current epoch
            step: Global training step
            metrics: Training metrics dictionary
            config: Training configuration
        """
        # Always save latest checkpoint
        self.save_latest(model, optimizer, scheduler, ema, epoch, step, metrics, config)
        
        # Save periodic epoch checkpoint
        if epoch % 5 == 0:
            self.save_epoch(model, optimizer, scheduler, ema, epoch, step, metrics, config)
        
        # Track and save best checkpoints
        if self.metric in metrics:
            metric_value = metrics[self.metric]
            checkpoint_path = self.save_checkpoint(
                model, optimizer, scheduler, ema, epoch, step,
                metrics, config, checkpoint_type='regular'
            )
            
            # Add to tracked checkpoints
            self.checkpoints.append((metric_value, checkpoint_path, {'epoch': epoch, 'step': step}))
            
            # Sort by metric
            reverse = (self.mode == 'max')
            self.checkpoints.sort(reverse=reverse, key=lambda x: x[0])
            
            # Check if this is the best checkpoint
            if self.checkpoints[0][1] == checkpoint_path:
                self.save_best(model, optimizer, scheduler, ema, epoch, step, metrics, config)
                logger.info(f"New best checkpoint! {self.metric}={metric_value:.4f}")
            
            # Remove old checkpoints beyond top-k
            if len(self.checkpoints) > self.keep_top_k:
                for _, old_path, _ in self.checkpoints[self.keep_top_k:]:
                    if old_path.exists() and 'latest' not in str(old_path) and 'best' not in str(old_path):
                        try:
                            old_path.unlink()
                            # Also remove .meta.pt if exists
                            meta_path = old_path.with_suffix('.meta.pt')
                            if meta_path.exists():
                                meta_path.unlink()
                            logger.info(f"Removed old checkpoint: {old_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove checkpoint {old_path}: {e}")
                
                self.checkpoints = self.checkpoints[:self.keep_top_k]
    
    def load_checkpoint(
        self,
        filepath: Path,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema: Optional[ExponentialMovingAverage] = None,
        load_training_state: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            ema: EMA to load state into
            load_training_state: Whether to load optimizer/scheduler state
            
        Returns:
            Dictionary with loaded metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        logger.info(f"Loading checkpoint from {filepath}")
        
        # Load checkpoint
        if self.use_safetensors and filepath.suffix == '.safetensors':
            checkpoint = self._load_safetensors(filepath)
        else:
            checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded")
        
        # Load optimizer state if requested
        if load_training_state and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded")
        
        # Load scheduler state
        if load_training_state and scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded")
        
        # Load EMA state
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            logger.info("EMA state loaded")
        
        # Restore random states if requested
        if load_training_state and 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['cuda'])
            np.random.set_state(rng_state['numpy'])
            random.setstate(rng_state['python'])
            logger.info("Random states restored")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
        }
    
    def _load_safetensors(self, filepath: Path) -> Dict[str, Any]:
        """Load checkpoint from safetensors format."""
        # Load tensors
        tensors = load_file(filepath)
        
        # Load non-tensor metadata
        meta_path = filepath.with_suffix('.meta.pt')
        if meta_path.exists():
            non_tensors = torch.load(meta_path, map_location='cpu')
        else:
            non_tensors = {}
        
        # Reconstruct checkpoint
        checkpoint = non_tensors.copy()
        
        # Reconstruct model state dict
        model_state = {}
        ema_state = {}
        for key, tensor in tensors.items():
            if key.startswith('model.'):
                model_state[key[6:]] = tensor
            elif key.startswith('ema.'):
                ema_state[key[4:]] = tensor
            else:
                checkpoint[key] = tensor
        
        if model_state:
            checkpoint['model_state_dict'] = model_state
        if ema_state:
            checkpoint['ema_state_dict'] = ema_state
        
        return checkpoint


# ============================================================================
# MODEL SAVER (For Inference/Deployment)
# ============================================================================

class ModelSaver:
    """
    Handles saving models for inference and deployment.
    Implements best practices for production-ready model saving.
    """
    
    def __init__(self, base_dir: str = './models'):
        """
        Initialize model saver.
        
        Args:
            base_dir: Base directory for saving models
        """
        self.base_dir = Path(base_dir)
        self.inference_dir = self.base_dir / 'inference'
        self.inference_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"ModelSaver initialized at {self.base_dir}")
    
    def save_for_inference(
        self,
        model: nn.Module,
        tokenizer: Any,
        metrics: Dict[str, float],
        config: TrainingConfig,
        version: str = 'v1.0',
        save_traced: bool = False
    ):
        """
        Save model for inference/deployment in HuggingFace format.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            metrics: Final metrics
            config: Training configuration
            version: Model version string
            save_traced: Whether to also save TorchScript traced version
        """
        # Create versioned directory
        version_dir = self.inference_dir / f'model_{version}'
        version_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Saving inference model to {version_dir}")
        
        # Save model in HuggingFace format
        try:
            model.deberta.save_pretrained(
                version_dir,
                safe_serialization=SAFETENSORS_AVAILABLE,
                max_shard_size="2GB"
            )
            logger.info("DeBERTa backbone saved in HuggingFace format")
        except Exception as e:
            logger.warning(f"Failed to save in HuggingFace format: {e}")
        
        # Save full model state dict
        full_model_path = version_dir / 'full_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'metrics': metrics,
        }, full_model_path)
        logger.info(f"Full model saved to {full_model_path}")
        
        # Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(version_dir)
            logger.info("Tokenizer saved")
        
        # Save training metadata
        metadata = {
            'model_type': config.base_model,
            'task': 'ai-text-detection',
            'architecture': f'{config.base_model} + BiLSTM',
            'training_date': datetime.now().isoformat(),
            'version': version,
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in metrics.items()},
            'hyperparameters': config.to_dict(),
            'model_config': {
                'lstm_layers': config.lstm_layers,
                'lstm_hidden_size': config.lstm_hidden_size,
                'attention_pooling': config.attention_pooling,
                'dropout': config.dropout,
                'max_seq_length': config.max_seq_length,
            }
        }
        
        metadata_path = version_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save TorchScript version if requested
        if save_traced:
            try:
                model.eval()
                example_input_ids = torch.randint(0, 30000, (1, config.max_seq_length))
                example_attention_mask = torch.ones((1, config.max_seq_length), dtype=torch.long)
                
                traced_model = torch.jit.trace(
                    model,
                    (example_input_ids, example_attention_mask)
                )
                
                # Optimize for inference
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                traced_path = version_dir / 'model_traced.pt'
                traced_model.save(str(traced_path))
                logger.info(f"TorchScript model saved to {traced_path}")
            except Exception as e:
                logger.warning(f"Failed to save TorchScript version: {e}")
        
        # Create README
        readme_content = f"""# AI Text Detection Model {version}

## Model Information
- **Architecture**: {config.base_model} + BiLSTM
- **Task**: Binary classification (Human vs AI-generated text)
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Performance Metrics
"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                readme_content += f"- **{key}**: {value:.4f}\n"
        
        readme_content += f"""
## Model Configuration
- LSTM Layers: {config.lstm_layers}
- LSTM Hidden Size: {config.lstm_hidden_size}
- Attention Pooling: {config.attention_pooling}
- Max Sequence Length: {config.max_seq_length}
- Dropout: {config.dropout}

## Usage

### Loading the Model
```python
import torch
from transformers import AutoTokenizer

# Load full model
checkpoint = torch.load('full_model.pt')
# Initialize your model architecture
model.load_state_dict(checkpoint['model_state_dict'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./')
```

### Inference
```python
model.eval()
text = "Your text here"
inputs = tokenizer(text, return_tensors='pt', max_length=512, 
                   padding='max_length', truncation=True)

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    prediction = outputs['logits'].argmax(dim=-1)
    # 0 = Human, 1 = AI-generated
```

## Files
- `full_model.pt`: Complete model with all layers
- `tokenizer_config.json`: Tokenizer configuration
- `training_metadata.json`: Detailed training information
- `model_traced.pt`: TorchScript version (if available)

---
Generated by AI Text Detection Training Pipeline
"""
        
        readme_path = version_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"README saved to {readme_path}")
        
        logger.info(f"✓ Inference model successfully saved to {version_dir}")


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA) - REMOVED DUPLICATE
# ============================================================================

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    """Handles model training, validation, and checkpoint management."""
    
    def __init__(
        self, 
        config: TrainingConfig,
        output_dir: str = "./outputs"
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = DeBERTaAIDetector(config).to(config.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Initialize EMA if enabled
        self.ema = ExponentialMovingAverage(self.model, config.ema_decay) if config.use_ema else None
        
        # Initialize checkpoint manager
        checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_dir,
            keep_top_k=config.save_total_limit,
            metric='val_f1',
            mode='max',
            use_safetensors=SAFETENSORS_AVAILABLE
        )
        
        # Initialize model saver for inference
        self.model_saver = ModelSaver(base_dir=str(self.output_dir / 'models'))
        
        # Training state
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_auc': [],
        }
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.current_epoch = 0
        
        logger.info(f"Model initialized on device: {config.device}")
        logger.info(f"Total parameters: {self._count_parameters():,}")
        logger.info(f"Trainable parameters: {self._count_trainable_parameters():,}")
    
    def _count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_optimizer_and_scheduler(
        self, 
        num_training_steps: int
    ) -> Tuple[optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Create optimizer with differentiated learning rates and scheduler.
        
        Args:
            num_training_steps: Total training steps
            
        Returns:
            Optimizer and scheduler
        """
        # Differentiated learning rates
        transformer_params = list(self.model.deberta.parameters())
        lstm_fc_params = list(self.model.lstm.parameters()) + list(self.model.classifier.parameters())
        
        optimizer_grouped_params = [
            {
                'params': transformer_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': lstm_fc_params,
                'lr': self.config.lstm_lr,
                'weight_decay': self.config.weight_decay,
            },
        ]
        
        # AdamW optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_params,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        # Linear schedule with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return optimizer, scheduler
    
    def train(
        self, 
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        text_column: str = 'text',
        label_column: str = 'label',
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting model training...")
        
        # Prepare datasets
        train_texts = train_data[text_column].tolist()
        train_labels = train_data[label_column].tolist()
        
        train_dataset = AITextDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
            augment=True,
            noise_injection_prob=self.config.noise_injection_prob,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=0,
        )
        
        # Prepare validation
        val_loader = None
        if val_data is not None:
            val_texts = val_data[text_column].tolist()
            val_labels = val_data[label_column].tolist()
            
            val_dataset = AITextDataset(
                texts=val_texts,
                labels=val_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                augment=False,
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.per_device_train_batch_size * 2,
                sampler=SequentialSampler(val_dataset),
                num_workers=0,
            )
        
        # Calculate number of training steps
        num_update_steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        max_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        
        if self.config.max_steps and self.config.max_steps > 0:
            max_steps = self.config.max_steps
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = self._get_optimizer_and_scheduler(max_steps)
        
        # Training loop
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        for epoch in range(self.config.num_train_epochs):
            # Training phase
            train_loss, train_f1 = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                progress_bar,
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_train_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                for key, value in val_metrics.items():
                    self.history[key].append(value)
                
                logger.info(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                           f"F1: {val_metrics['val_f1']:.4f}, "
                           f"AUC: {val_metrics['val_auc']:.4f}")
                
                # Prepare metrics for checkpoint manager
                checkpoint_metrics = {
                    'train_loss': train_loss,
                    'train_f1': train_f1,
                    **val_metrics
                }
                
                # Save checkpoints using checkpoint manager
                self.checkpoint_manager.manage_checkpoints(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=self.ema,
                    epoch=epoch + 1,
                    step=self.global_step,
                    metrics=checkpoint_metrics,
                    config=self.config
                )
                
                # Track best model for early stopping
                if val_metrics['val_f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['val_f1']
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    
                    # Save inference model for best checkpoint
                    logger.info("Saving best model for inference...")
                    self.model_saver.save_for_inference(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        metrics=checkpoint_metrics,
                        config=self.config,
                        version=f'v1.0_epoch{epoch+1}',
                        save_traced=False  # Set to True if you want TorchScript
                    )
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
                        break
            else:
                # No validation data - just save checkpoints based on training
                checkpoint_metrics = {
                    'train_loss': train_loss,
                    'train_f1': train_f1,
                }
                
                self.checkpoint_manager.manage_checkpoints(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=self.ema,
                    epoch=epoch + 1,
                    step=self.global_step,
                    metrics=checkpoint_metrics,
                    config=self.config
                )
        
        progress_bar.close()
        
        # Save final inference model
        logger.info("Training completed! Saving final inference model...")
        final_metrics = {
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0.0,
            'train_f1': self.history['train_f1'][-1] if self.history['train_f1'] else 0.0,
        }
        if val_loader is not None:
            final_metrics.update({
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0.0,
                'val_f1': self.history['val_f1'][-1] if self.history['val_f1'] else 0.0,
                'val_auc': self.history['val_auc'][-1] if self.history['val_auc'] else 0.0,
                'best_val_f1': self.best_val_f1,
                'best_val_loss': self.best_val_loss,
            })
        
        self.model_saver.save_for_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            metrics=final_metrics,
            config=self.config,
            version='v1.0_final',
            save_traced=False
        )
        
        logger.info("=" * 80)
        logger.info("Training Summary:")
        logger.info(f"  Total epochs: {epoch + 1}")
        logger.info(f"  Total steps: {self.global_step}")
        logger.info(f"  Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"  Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"  Checkpoints saved in: {self.checkpoint_manager.save_dir}")
        logger.info(f"  Inference models saved in: {self.model_saver.inference_dir}")
        logger.info("=" * 80)
        
        return {
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step,
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        progress_bar: tqdm,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            progress_bar: Progress bar
            
        Returns:
            Epoch loss and F1 score
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Enable mixed precision if configured
        use_amp = self.config.bf16 or self.config.fp16
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA
                if self.ema:
                    self.ema.update()
                
                self.global_step += 1
                progress_bar.update(1)
                
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")
            
            # Collect predictions
            with torch.no_grad():
                predictions = outputs['logits'].argmax(dim=-1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return avg_loss, f1
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                predictions = logits.argmax(dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        return {
            'val_loss': avg_loss,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall,
            'val_auc': auc,
        }
    
    def save_checkpoint(self, checkpoint_path: Path, epoch: int = 0, metrics: Dict = None) -> None:
        """
        Save model checkpoint (deprecated - use checkpoint_manager instead).
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            metrics: Training metrics
        """
        logger.warning("save_checkpoint is deprecated. Use checkpoint_manager.save_checkpoint instead.")
        
        if metrics is None:
            metrics = {
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0.0,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0.0,
                'val_f1': self.history['val_f1'][-1] if self.history['val_f1'] else 0.0,
            }
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'epoch': epoch,
            'global_step': self.global_step,
            'metrics': metrics,
        }
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: Path,
        load_training_state: bool = False,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_training_state: Whether to load optimizer/scheduler state
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Dictionary with checkpoint metadata
        """
        return self.checkpoint_manager.load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=optimizer if load_training_state else None,
            scheduler=scheduler if load_training_state else None,
            ema=self.ema if load_training_state else None,
            load_training_state=load_training_state
        )
    
    def resume_from_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with loaded state information
        """
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # We'll need to create optimizer and scheduler first
        # This is a simplified version - in practice, you'd call this after
        # setting up the training loop
        metadata = self.checkpoint_manager.load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=None,  # Pass optimizer if available
            scheduler=None,  # Pass scheduler if available
            ema=self.ema,
            load_training_state=True
        )
        
        # Restore training state
        self.current_epoch = metadata.get('epoch', 0)
        self.global_step = metadata.get('global_step', 0)
        
        # Restore history if available
        if 'metrics' in metadata:
            logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
            logger.info(f"Previous metrics: {metadata['metrics']}")
        
        return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training module initialized")

