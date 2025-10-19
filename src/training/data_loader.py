#!/usr/bin/env python3
"""
Data Loading Module for Processed AI Text Detection Data

Handles loading and processing Parquet files from the processed_data folder
with efficient memory management and stratified sampling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class ProcessedDataLoader:
    """Load data from processed_data folder (Parquet format)."""
    
    def __init__(self, data_dir: Path = Path("./processed_data")):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load statistics and metadata."""
        metadata_file = self.data_dir / "statistics.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file not found at {metadata_file}")
            return {}
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of processed data."""
        return {
            'total_samples': self.metadata.get('total_samples', 0),
            'human_samples': self.metadata.get('human_samples', 0),
            'ai_samples': self.metadata.get('ai_samples', 0),
            'ai_models': list(self.metadata.get('ai_model_breakdown', {}).keys()),
            'domains': list(self.metadata.get('domain_breakdown', {}).keys()),
            'text_length_stats': self.metadata.get('text_length_stats', {}),
        }
    
    def find_parquet_files(self) -> Dict[str, List[Path]]:
        """
        Find all parquet files organized by source (ai/real) and category.
        
        Returns:
            Dictionary mapping category to list of parquet files
        """
        parquet_files = {}
        
        # Check if data is in train subdirectory (processed_data/train/ai/) or root (processed_data/ai/)
        train_subdir = self.data_dir / "train"
        base_dir = train_subdir if train_subdir.exists() else self.data_dir
        
        # Find AI-generated text files - check multiple dataset structures
        ai_dirs = [
            base_dir / "ai" / "RAID-Dataset",
            base_dir / "ai" / "dmitva-dataset",
            base_dir / "ai" / "AI-Vs-Real-Dataset",
        ]
        
        for ai_dir in ai_dirs:
            if ai_dir.exists():
                for model_dir in ai_dir.iterdir():
                    if model_dir.is_dir():
                        files = list(model_dir.glob("*.parquet"))
                        if files:
                            # Use just the model name as key (e.g., "gpt4" instead of "ai_gpt4")
                            parquet_files[model_dir.name] = sorted(files)
        
        # Find real/human text files
        real_dirs = [
            base_dir / "real" / "RAID-Dataset",
            base_dir / "real" / "dmitva-dataset",
            base_dir / "real" / "AI-Vs-Real-Dataset",
            base_dir / "real" / "Wikipedia_C4-Web",
        ]
        
        for real_dir in real_dirs:
            if real_dir.exists():
                for category_dir in real_dir.iterdir():
                    if category_dir.is_dir():
                        files = list(category_dir.glob("*.parquet"))
                        if files:
                            parquet_files[f"real_{category_dir.name}"] = sorted(files)
        
        logger.info(f"Found parquet files in {len(parquet_files)} categories")
        for category, files in parquet_files.items():
            logger.info(f"  {category}: {len(files)} files")
        
        return parquet_files
    
    def load_parquet_files(
        self, 
        file_paths: List[Path],
        columns: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load multiple parquet files into a single DataFrame.
        
        Args:
            file_paths: List of parquet file paths
            columns: Columns to load (None = all columns)
            sample_size: Optional downsampling size
            
        Returns:
            Combined DataFrame
        """
        dfs = []
        
        for file_path in file_paths:
            try:
                df = pd.read_parquet(file_path, columns=columns)
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No parquet files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded total {len(combined_df)} rows from {len(file_paths)} files")
        
        # Optional downsampling
        if sample_size and len(combined_df) > sample_size:
            combined_df = combined_df.sample(n=sample_size, random_state=42)
            logger.info(f"Downsampled to {len(combined_df)} rows")
        
        return combined_df
    
    def load_balanced_dataset(
        self,
        ai_files: List[Path],
        human_files: List[Path],
        ai_samples: Optional[int] = None,
        human_samples: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load balanced AI and human datasets with train/val/test split.
        
        Args:
            ai_files: Parquet files for AI-generated text
            human_files: Parquet files for human text
            ai_samples: Number of AI samples to load
            human_samples: Number of human samples to load
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load datasets
        ai_data = self.load_parquet_files(ai_files, sample_size=ai_samples)
        human_data = self.load_parquet_files(human_files, sample_size=human_samples)
        
        # Add labels
        ai_data['label'] = 1
        human_data['label'] = 0
        
        # Combine and shuffle
        combined = pd.concat([ai_data, human_data], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Combined dataset: {len(combined)} samples "
                   f"({len(human_data)} human, {len(ai_data)} AI)")
        
        # Stratified split
        train_val, test = train_test_split(
            combined,
            test_size=test_size,
            stratify=combined['label'],
            random_state=42
        )
        
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val['label'],
            random_state=42
        )
        
        logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        logger.info(f"Train - Human: {(train['label']==0).sum()}, AI: {(train['label']==1).sum()}")
        logger.info(f"Val   - Human: {(val['label']==0).sum()}, AI: {(val['label']==1).sum()}")
        logger.info(f"Test  - Human: {(test['label']==0).sum()}, AI: {(test['label']==1).sum()}")
        
        return train, val, test
    
    def load_by_model(
        self,
        model_name: str,
        sample_size: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data for a specific AI model vs human text.
        
        Args:
            model_name: Name of AI model (e.g., 'gpt4', 'chatgpt')
            sample_size: Number of AI samples to load
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        parquet_files = self.find_parquet_files()
        
        # Find model files - check both with and without 'ai_' prefix
        ai_key = model_name if model_name in parquet_files else f"ai_{model_name}"
        human_key = "real_human" if "real_human" in parquet_files else "human"
        
        if ai_key not in parquet_files:
            available_models = [k for k in parquet_files.keys() if not k.startswith('real_')]
            raise ValueError(
                f"Model {model_name} not found in processed data. "
                f"Available models: {', '.join(available_models)}"
            )
        
        if human_key not in parquet_files:
            raise ValueError(f"Human text data not found in processed data")
        
        ai_files = parquet_files[ai_key]
        human_files = parquet_files[human_key]
        
        logger.info(f"Loading {len(ai_files)} AI files for model '{model_name}'")
        logger.info(f"Loading {len(human_files)} human files")
        
        return self.load_balanced_dataset(
            ai_files=ai_files,
            human_files=human_files,
            ai_samples=sample_size,
            human_samples=sample_size,
            test_size=test_size,
            val_size=val_size,
        )
    
    def load_all_models(
        self,
        ai_sample_per_model: Optional[int] = None,
        human_sample_per_model: Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data combining all AI models vs human text.
        
        Args:
            ai_sample_per_model: Samples per model
            human_sample_per_model: Human samples per model
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        parquet_files = self.find_parquet_files()
        
        ai_files = []
        human_files = []
        
        for key, files in parquet_files.items():
            if key.startswith("ai_"):
                ai_files.extend(files)
            elif key.startswith("real_human"):
                human_files.extend(files)
        
        if not ai_files:
            raise ValueError("No AI data found in processed_data")
        if not human_files:
            raise ValueError("No human data found in processed_data")
        
        return self.load_balanced_dataset(
            ai_files=ai_files,
            human_files=human_files,
            ai_samples=ai_sample_per_model,
            human_samples=human_sample_per_model,
            test_size=test_size,
            val_size=val_size,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    loader = ProcessedDataLoader()
    summary = loader.get_data_summary()
    print("Data Summary:", summary)
    
    # Find available data
    files = loader.find_parquet_files()
    print(f"\nFound {len(files)} data categories")
