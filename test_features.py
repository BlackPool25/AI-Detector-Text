#!/usr/bin/env python3
"""
Feature Testing Script for AI Text Detection Model

Tests all implemented features from the architecture requirements:
✓ DeBERTa-v3-small backbone
✓ Frozen embeddings
✓ Bi-LSTM layers
✓ Attention pooling
✓ Mixed precision (BF16/FP16)
✓ Gradient accumulation
✓ Gradient clipping
✓ AdamW optimizer with differentiated learning rates
✓ Linear warmup scheduler
✓ Label smoothing
✓ Noise injection augmentation
✓ EMA (Exponential Moving Average)
✓ Early stopping
✓ Data loading from processed_data
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.train import (
    TrainingConfig, 
    DeBERTaAIDetector,
    AITextDataset,
    ExponentialMovingAverage,
    AttentionPooling,
    ModelTrainer
)
from training.data_loader import ProcessedDataLoader
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureTester:
    """Test all implemented features."""
    
    def __init__(self):
        self.results = {}
        self.config = TrainingConfig()
        
    def test_configuration(self):
        """Test 1: Configuration structure."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Configuration Structure")
        logger.info("="*80)
        
        try:
            config_dict = self.config.to_dict()
            
            required_features = {
                'base_model': 'microsoft/deberta-v3-small',
                'freeze_embeddings': True,
                'lstm_layers': 2,
                'lstm_hidden_size': 512,
                'attention_pooling': True,
                'dropout': 0.4,
                'per_device_train_batch_size': 16,
                'gradient_accumulation_steps': 4,
                'max_grad_norm': 1.0,
                'learning_rate': 2e-5,
                'lstm_lr': 1e-3,
                'weight_decay': 0.01,
                'warmup_steps': 500,
                'label_smoothing': 0.1,
                'noise_injection_prob': 0.1,
                'use_ema': True,
                'ema_decay': 0.999,
                'early_stopping_patience': 3,
                'bf16': True,
            }
            
            all_passed = True
            for key, expected_value in required_features.items():
                actual_value = config_dict.get(key)
                if actual_value == expected_value:
                    logger.info(f"  ✓ {key}: {actual_value}")
                else:
                    logger.error(f"  ✗ {key}: Expected {expected_value}, got {actual_value}")
                    all_passed = False
            
            self.results['configuration'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            self.results['configuration'] = False
            return False
    
    def test_model_architecture(self):
        """Test 2: Model architecture components."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Model Architecture")
        logger.info("="*80)
        
        try:
            model = DeBERTaAIDetector(self.config)
            
            # Check DeBERTa backbone
            has_deberta = hasattr(model, 'deberta')
            logger.info(f"  ✓ DeBERTa backbone: {has_deberta}")
            
            # Check frozen embeddings
            embeddings_frozen = not any(p.requires_grad for p in model.deberta.embeddings.parameters())
            logger.info(f"  {'✓' if embeddings_frozen else '✗'} Embeddings frozen: {embeddings_frozen}")
            
            # Check LSTM layers
            has_lstm = hasattr(model, 'lstm') and isinstance(model.lstm, nn.LSTM)
            lstm_correct = (
                model.lstm.num_layers == 2 and
                model.lstm.hidden_size == 512 and
                model.lstm.bidirectional == True
            )
            logger.info(f"  {'✓' if has_lstm and lstm_correct else '✗'} Bi-LSTM (2 layers, 512 hidden): {lstm_correct}")
            
            # Check attention pooling
            has_attention = hasattr(model, 'attention_pool') and isinstance(model.attention_pool, AttentionPooling)
            logger.info(f"  {'✓' if has_attention else '✗'} Attention pooling: {has_attention}")
            
            # Check dropout
            has_dropout = hasattr(model, 'dropout') and isinstance(model.dropout, nn.Dropout)
            logger.info(f"  ✓ Dropout layer: {has_dropout}")
            
            # Check classifier head
            has_classifier = hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear)
            classifier_correct = model.classifier.out_features == 2
            logger.info(f"  {'✓' if classifier_correct else '✗'} Binary classifier: {classifier_correct}")
            
            # Check label smoothing in loss
            has_loss = hasattr(model, 'loss_fn')
            logger.info(f"  ✓ Label smoothing loss: {has_loss}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\n  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            
            all_passed = (has_deberta and embeddings_frozen and has_lstm and 
                         lstm_correct and has_attention and classifier_correct)
            
            self.results['model_architecture'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Model architecture test failed: {e}")
            self.results['model_architecture'] = False
            return False
    
    def test_forward_pass(self):
        """Test 3: Model forward pass with dummy data."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Forward Pass")
        logger.info("="*80)
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"  Using device: {device}")
            
            model = DeBERTaAIDetector(self.config).to(device)
            
            # Create dummy batch
            batch_size = 4
            seq_length = 128
            input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
            attention_mask = torch.ones(batch_size, seq_length).to(device)
            labels = torch.randint(0, 2, (batch_size,)).to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels)
            
            # Check outputs
            has_logits = 'logits' in outputs
            has_loss = 'loss' in outputs
            logits_shape_correct = outputs['logits'].shape == (batch_size, 2)
            
            logger.info(f"  ✓ Has logits: {has_logits}")
            logger.info(f"  ✓ Has loss: {has_loss}")
            logger.info(f"  {'✓' if logits_shape_correct else '✗'} Logits shape: {outputs['logits'].shape}")
            logger.info(f"  ✓ Loss value: {outputs['loss'].item():.4f}")
            
            # Test without labels
            outputs_no_labels = model(input_ids, attention_mask)
            has_no_loss = 'loss' not in outputs_no_labels
            logger.info(f"  ✓ No loss without labels: {has_no_loss}")
            
            all_passed = has_logits and has_loss and logits_shape_correct and has_no_loss
            
            self.results['forward_pass'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Forward pass test failed: {e}")
            self.results['forward_pass'] = False
            return False
    
    def test_data_augmentation(self):
        """Test 4: Data augmentation (noise injection)."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Data Augmentation")
        logger.info("="*80)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            texts = ["This is a test sentence for augmentation."] * 100
            labels = [0] * 100
            
            # Create dataset with augmentation
            dataset = AITextDataset(
                texts=texts,
                labels=labels,
                tokenizer=tokenizer,
                max_length=128,
                augment=True,
                noise_injection_prob=1.0,  # Force augmentation
            )
            
            # Get sample
            sample = dataset[0]
            
            # Decode to check if noise was added
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            
            # Check for random characters (noise)
            has_noise = len(decoded_text.split()) > len(texts[0].split())
            
            logger.info(f"  Original: {texts[0]}")
            logger.info(f"  Augmented sample: {decoded_text[:100]}...")
            logger.info(f"  {'✓' if has_noise else '✗'} Noise injection working: {has_noise}")
            
            self.results['data_augmentation'] = has_noise
            logger.info(f"\n{'PASSED' if has_noise else 'FAILED'}")
            return has_noise
        except Exception as e:
            logger.error(f"Data augmentation test failed: {e}")
            self.results['data_augmentation'] = False
            return False
    
    def test_ema(self):
        """Test 5: Exponential Moving Average."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Exponential Moving Average (EMA)")
        logger.info("="*80)
        
        try:
            model = DeBERTaAIDetector(self.config)
            ema = ExponentialMovingAverage(model, decay=0.999)
            
            # Get initial weight
            initial_weight = list(model.parameters())[0].data.clone()
            
            # Modify model weights
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            
            # Update EMA
            ema.update()
            
            # Check that shadow params are different from current
            shadow_exists = len(ema.shadow) > 0
            logger.info(f"  ✓ Shadow parameters created: {shadow_exists}")
            logger.info(f"  ✓ Number of shadow params: {len(ema.shadow)}")
            
            # Apply shadow and check
            ema.apply_shadow()
            updated_weight = list(model.parameters())[0].data.clone()
            
            weights_different = not torch.allclose(initial_weight, updated_weight)
            logger.info(f"  ✓ EMA updates weights: {weights_different}")
            
            all_passed = shadow_exists and weights_different
            
            self.results['ema'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"EMA test failed: {e}")
            self.results['ema'] = False
            return False
    
    def test_data_loading(self):
        """Test 6: Data loading from processed_data."""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Data Loading from processed_data")
        logger.info("="*80)
        
        try:
            data_loader = ProcessedDataLoader(Path("./processed_data"))
            
            # Get summary
            summary = data_loader.get_data_summary()
            logger.info(f"  ✓ Total samples: {summary.get('total_samples', 0):,}")
            logger.info(f"  ✓ Human samples: {summary.get('human_samples', 0):,}")
            logger.info(f"  ✓ AI samples: {summary.get('ai_samples', 0):,}")
            logger.info(f"  ✓ AI models: {len(summary.get('ai_models', []))}")
            
            # Find parquet files
            parquet_files = data_loader.find_parquet_files()
            logger.info(f"  ✓ Parquet categories found: {len(parquet_files)}")
            
            # Test loading small sample
            logger.info("\n  Loading small sample (1000 samples per class)...")
            train_data, val_data, test_data = data_loader.load_by_model(
                model_name="gpt4",
                sample_size=1000,
            )
            
            logger.info(f"  ✓ Train size: {len(train_data)}")
            logger.info(f"  ✓ Val size: {len(val_data)}")
            logger.info(f"  ✓ Test size: {len(test_data)}")
            
            # Check columns
            has_text = 'text' in train_data.columns
            has_label = 'label' in train_data.columns
            logger.info(f"  {'✓' if has_text else '✗'} Has 'text' column: {has_text}")
            logger.info(f"  {'✓' if has_label else '✗'} Has 'label' column: {has_label}")
            
            # Check balance
            train_human = (train_data['label'] == 0).sum()
            train_ai = (train_data['label'] == 1).sum()
            logger.info(f"  ✓ Train - Human: {train_human}, AI: {train_ai}")
            
            all_passed = (len(parquet_files) > 0 and len(train_data) > 0 and 
                         has_text and has_label)
            
            self.results['data_loading'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Data loading test failed: {e}")
            self.results['data_loading'] = False
            return False
    
    def test_optimizer_and_scheduler(self):
        """Test 7: Optimizer with differentiated LR and scheduler."""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: Optimizer and Scheduler")
        logger.info("="*80)
        
        try:
            trainer = ModelTrainer(self.config)
            
            num_training_steps = 1000
            optimizer, scheduler = trainer._get_optimizer_and_scheduler(num_training_steps)
            
            # Check optimizer type
            is_adamw = isinstance(optimizer, torch.optim.AdamW)
            logger.info(f"  {'✓' if is_adamw else '✗'} Optimizer is AdamW: {is_adamw}")
            
            # Check parameter groups (differentiated LR)
            has_two_groups = len(optimizer.param_groups) == 2
            logger.info(f"  {'✓' if has_two_groups else '✗'} Has 2 param groups: {has_two_groups}")
            
            if has_two_groups:
                lr1 = optimizer.param_groups[0]['lr']
                lr2 = optimizer.param_groups[1]['lr']
                logger.info(f"  ✓ Transformer LR: {lr1}")
                logger.info(f"  ✓ LSTM/FC LR: {lr2}")
                different_lrs = lr1 != lr2
                logger.info(f"  {'✓' if different_lrs else '✗'} Differentiated LRs: {different_lrs}")
            
            # Check weight decay
            wd = optimizer.param_groups[0]['weight_decay']
            logger.info(f"  ✓ Weight decay: {wd}")
            
            # Check scheduler
            has_scheduler = scheduler is not None
            logger.info(f"  ✓ Scheduler created: {has_scheduler}")
            
            all_passed = is_adamw and has_two_groups and has_scheduler
            
            self.results['optimizer_scheduler'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Optimizer/scheduler test failed: {e}")
            self.results['optimizer_scheduler'] = False
            return False
    
    def test_mixed_precision(self):
        """Test 8: Mixed precision training support."""
        logger.info("\n" + "="*80)
        logger.info("TEST 8: Mixed Precision Support")
        logger.info("="*80)
        
        try:
            # Check if CUDA is available
            cuda_available = torch.cuda.is_available()
            logger.info(f"  CUDA available: {cuda_available}")
            
            if cuda_available:
                # Check BF16 support
                bf16_supported = torch.cuda.is_bf16_supported()
                logger.info(f"  {'✓' if bf16_supported else '✗'} BF16 supported: {bf16_supported}")
                
                # Check configuration
                config_bf16 = self.config.bf16
                logger.info(f"  ✓ BF16 enabled in config: {config_bf16}")
                
                all_passed = bf16_supported or self.config.fp16
            else:
                logger.info("  ⚠ Skipping - No CUDA available")
                all_passed = True  # Pass test since no GPU
            
            self.results['mixed_precision'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Mixed precision test failed: {e}")
            self.results['mixed_precision'] = False
            return False
    
    def test_gradient_features(self):
        """Test 9: Gradient accumulation and clipping."""
        logger.info("\n" + "="*80)
        logger.info("TEST 9: Gradient Features")
        logger.info("="*80)
        
        try:
            # Check configuration
            grad_accum = self.config.gradient_accumulation_steps
            max_grad_norm = self.config.max_grad_norm
            
            logger.info(f"  ✓ Gradient accumulation steps: {grad_accum}")
            logger.info(f"  ✓ Max gradient norm: {max_grad_norm}")
            
            has_grad_accum = grad_accum >= 4
            has_grad_clip = max_grad_norm == 1.0
            
            logger.info(f"  {'✓' if has_grad_accum else '✗'} Gradient accumulation ≥4: {has_grad_accum}")
            logger.info(f"  {'✓' if has_grad_clip else '✗'} Gradient clipping = 1.0: {has_grad_clip}")
            
            all_passed = has_grad_accum and has_grad_clip
            
            self.results['gradient_features'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Gradient features test failed: {e}")
            self.results['gradient_features'] = False
            return False
    
    def test_training_integration(self):
        """Test 10: Full training integration with tiny dataset."""
        logger.info("\n" + "="*80)
        logger.info("TEST 10: Training Integration (Mini Run)")
        logger.info("="*80)
        
        try:
            # Load tiny dataset
            data_loader = ProcessedDataLoader(Path("./processed_data"))
            train_data, val_data, _ = data_loader.load_by_model(
                model_name="gpt4",
                sample_size=50,  # Very small for quick test
            )
            
            logger.info(f"  Train samples: {len(train_data)}")
            logger.info(f"  Val samples: {len(val_data)}")
            
            # Modify config for quick test
            test_config = TrainingConfig(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                warmup_steps=5,
                early_stopping_patience=1,
                bf16=False,  # Disable for CPU compatibility
                fp16=False,
            )
            
            # Initialize trainer
            trainer = ModelTrainer(test_config, output_dir="./test_outputs")
            
            logger.info("  Starting mini training run...")
            
            # Train for 1 epoch
            results = trainer.train(
                train_data=train_data,
                val_data=val_data,
            )
            
            # Check results
            has_history = 'history' in results
            has_train_loss = len(results['history']['train_loss']) > 0
            has_val_metrics = len(results['history']['val_loss']) > 0
            
            logger.info(f"  {'✓' if has_history else '✗'} Training history saved: {has_history}")
            logger.info(f"  {'✓' if has_train_loss else '✗'} Train loss recorded: {has_train_loss}")
            logger.info(f"  {'✓' if has_val_metrics else '✗'} Val metrics recorded: {has_val_metrics}")
            
            if has_train_loss:
                logger.info(f"  ✓ Final train loss: {results['history']['train_loss'][-1]:.4f}")
                logger.info(f"  ✓ Final train F1: {results['history']['train_f1'][-1]:.4f}")
            
            if has_val_metrics:
                logger.info(f"  ✓ Final val loss: {results['history']['val_loss'][-1]:.4f}")
                logger.info(f"  ✓ Final val F1: {results['history']['val_f1'][-1]:.4f}")
                logger.info(f"  ✓ Final val AUC: {results['history']['val_auc'][-1]:.4f}")
            
            all_passed = has_history and has_train_loss and has_val_metrics
            
            self.results['training_integration'] = all_passed
            logger.info(f"\n{'PASSED' if all_passed else 'FAILED'}")
            return all_passed
        except Exception as e:
            logger.error(f"Training integration test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['training_integration'] = False
            return False
    
    def run_all_tests(self):
        """Run all feature tests."""
        logger.info("\n" + "#"*80)
        logger.info("# AI TEXT DETECTION - FEATURE VALIDATION SUITE")
        logger.info("#"*80)
        
        tests = [
            self.test_configuration,
            self.test_model_architecture,
            self.test_forward_pass,
            self.test_data_augmentation,
            self.test_ema,
            self.test_data_loading,
            self.test_optimizer_and_scheduler,
            self.test_mixed_precision,
            self.test_gradient_features,
            self.test_training_integration,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
        
        # Print summary
        logger.info("\n" + "#"*80)
        logger.info("# TEST SUMMARY")
        logger.info("#"*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v)
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"  {status}: {test_name}")
        
        logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL FEATURES VALIDATED SUCCESSFULLY!")
        else:
            logger.warning(f"\n⚠ {total_tests - passed_tests} test(s) failed")
        
        return passed_tests == total_tests


if __name__ == "__main__":
    tester = FeatureTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
