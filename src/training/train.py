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
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        
        logger.info(f"Model initialized on device: {config.device}")
        logger.info(f"Total parameters: {self._count_parameters()}")
    
    def _count_parameters(self) -> int:
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
                
                # Early stopping
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(self.output_dir / f"best_model_epoch_{epoch+1}.pt")
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping after {epoch+1} epochs")
                        break
            
            # Save periodic checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(self.output_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        progress_bar.close()
        logger.info("Training completed!")
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
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
    
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
        }
        
        if self.ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        if self.ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training module initialized")
