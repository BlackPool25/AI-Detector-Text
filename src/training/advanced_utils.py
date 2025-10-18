#!/usr/bin/env python3
"""
Advanced Training Utilities for AI Text Detection

Implements optional advanced techniques:
- Stochastic Weight Averaging (SWA)
- Adversarial Weight Perturbation (AWP)
- Checkpoint averaging
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import copy

import torch
import torch.nn as nn
from pathlib import Path


logger = logging.getLogger(__name__)


class StochasticWeightAveraging:
    """Stochastic Weight Averaging for improved generalization."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize SWA.
        
        Args:
            model: Model to average
        """
        self.model = model
        self.averaged_model = None
        self.swa_n = 0
    
    def update(self):
        """Add current model weights to average."""
        if self.averaged_model is None:
            self.averaged_model = copy.deepcopy(self.model)
            self.swa_n = 1
        else:
            # Incremental averaging
            for param_avg, param_model in zip(
                self.averaged_model.parameters(),
                self.model.parameters()
            ):
                param_avg.data = (
                    (param_avg.data * self.swa_n + param_model.data) / 
                    (self.swa_n + 1)
                )
            self.swa_n += 1
    
    def get_model(self) -> nn.Module:
        """Get averaged model."""
        return self.averaged_model


class AdversarialWeightPerturbation:
    """
    Adversarial Weight Perturbation for robustness.
    
    Simulates adversarial scenarios by perturbing model weights
    during training to improve robustness to noise/adversarial examples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_lr: float = 1e-2,
        adv_eps: float = 1e-3,
    ):
        """
        Initialize AWP.
        
        Args:
            model: Model to apply AWP to
            optimizer: Optimizer
            adv_lr: Adversarial learning rate
            adv_eps: Perturbation magnitude
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
    
    def _save_weights(self):
        """Save current model weights."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def _restore_weights(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
    
    def perturb(self, loss: torch.Tensor):
        """
        Apply adversarial perturbation.
        
        Args:
            loss: Loss value
        """
        # Save original weights
        self._save_weights()
        
        # Compute gradient w.r.t. weights
        loss.backward(retain_graph=True)
        
        # Perturb weights in gradient direction
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Normalize gradient
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    param.data += self.adv_eps * param.grad / grad_norm
    
    def restore_and_update(self):
        """Restore weights and update optimizer."""
        self._restore_weights()
        self.optimizer.step()


class CheckpointAveraging:
    """Average multiple model checkpoints."""
    
    @staticmethod
    def average_checkpoints(
        checkpoint_paths: List[Path],
        output_path: Path,
        device: str = "cuda",
    ) -> None:
        """
        Average weights from multiple checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            output_path: Path to save averaged model
            device: Device to load on
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoints provided")
        
        logger.info(f"Averaging {len(checkpoint_paths)} checkpoints...")
        
        # Load first checkpoint as template
        first_checkpoint = torch.load(checkpoint_paths[0], map_location=device)
        averaged_state = {k: v.clone() for k, v in first_checkpoint['model_state_dict'].items()}
        
        # Average remaining checkpoints
        for checkpoint_path in checkpoint_paths[1:]:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            for key, value in state_dict.items():
                averaged_state[key] += value
        
        # Divide by number of checkpoints
        for key in averaged_state:
            averaged_state[key] /= len(checkpoint_paths)
        
        # Save averaged checkpoint
        first_checkpoint['model_state_dict'] = averaged_state
        torch.save(first_checkpoint, output_path)
        logger.info(f"Averaged checkpoint saved to {output_path}")
    
    @staticmethod
    def average_last_n(
        checkpoint_dir: Path,
        n: int = 3,
        output_path: Path = None,
        device: str = "cuda",
    ) -> Path:
        """
        Average last N checkpoints from a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            n: Number of last checkpoints to average
            output_path: Path to save averaged model
            device: Device to load on
            
        Returns:
            Path to averaged checkpoint
        """
        # Find all checkpoint files
        checkpoint_files = sorted(
            Path(checkpoint_dir).glob("checkpoint_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if len(checkpoint_files) < n:
            logger.warning(
                f"Found {len(checkpoint_files)} checkpoints, but requested {n}. "
                f"Using all available checkpoints."
            )
            checkpoints_to_average = checkpoint_files
        else:
            checkpoints_to_average = checkpoint_files[-n:]
        
        if output_path is None:
            output_path = Path(checkpoint_dir) / f"averaged_last_{n}.pt"
        
        CheckpointAveraging.average_checkpoints(
            checkpoints_to_average,
            output_path,
            device
        )
        
        return output_path


class LearningRateSchedules:
    """Custom learning rate schedules."""
    
    @staticmethod
    def linear_with_warmup(current_step: int, num_warmup_steps: int, num_training_steps: int) -> float:
        """Linear schedule with warmup."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    @staticmethod
    def cosine_with_warmup(current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5) -> float:
        """Cosine schedule with warmup."""
        import math
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    @staticmethod
    def polynomial_decay(current_step: int, num_training_steps: int, lr_init: float, lr_end: float = 0.0, power: float = 1.0) -> float:
        """Polynomial decay schedule."""
        if current_step > num_training_steps:
            return lr_end
        lr_range = lr_init - lr_end
        pct_remaining = 1 - (current_step / num_training_steps)
        return lr_end + lr_range * (pct_remaining ** power)


class MetricsTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.step_count = 0
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        self.step_count += 1
    
    def get_average(self, metric_name: str) -> float:
        """Get average of metric."""
        if metric_name not in self.metrics:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        return {
            key: self.get_average(key)
            for key in self.metrics.keys()
        }
    
    def reset(self):
        """Reset metrics."""
        self.metrics = {}
        self.step_count = 0
    
    def get_summary(self) -> str:
        """Get summary string."""
        averages = self.get_all_averages()
        return " | ".join(f"{k}: {v:.4f}" for k, v in averages.items())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Advanced training utilities module loaded")
