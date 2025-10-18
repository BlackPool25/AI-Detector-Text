#!/usr/bin/env python3
"""
Unit tests for testing module.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import unittest
from testing.test import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {"metrics": ["accuracy", "f1"]}
        self.evaluator = ModelEvaluator(metrics_config=self.config)
    
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNone(self.evaluator.model)
        self.assertEqual(self.evaluator.config, self.config)
        self.assertIsNone(self.evaluator.results)


if __name__ == "__main__":
    unittest.main()
