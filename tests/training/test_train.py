#!/usr/bin/env python3
"""
Unit tests for training module.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import unittest
from training.train import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"model": "test_model"}
        self.trainer = ModelTrainer(self.config)
    
    def test_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.config, self.config)
        self.assertIsNone(self.trainer.model)
        self.assertIsNone(self.trainer.history)


if __name__ == "__main__":
    unittest.main()
