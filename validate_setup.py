#!/usr/bin/env python3
"""
Validation Script - Verify Training Module Setup

Checks:
1. Dependencies installation
2. CUDA/GPU availability  
3. Data accessibility
4. Model architecture
5. Training pipeline
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version."""
    logger.info("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"✗ Python 3.8+ required, got {version.major}.{version.minor}")
        return False


def check_dependencies():
    """Check required dependencies."""
    logger.info("\nChecking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'pyarrow': 'PyArrow',
        'tqdm': 'tqdm',
    }
    
    all_installed = True
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {display_name}: {version}")
        except ImportError:
            logger.error(f"✗ {display_name} not installed")
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA/GPU availability."""
    logger.info("\nChecking CUDA/GPU...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✓ CUDA available")
            logger.info(f"  Device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                logger.info(f"  Device {i}: {props.name} ({memory_gb:.2f} GB)")
            
            return True
        else:
            logger.warning("⚠ CUDA not available - training will be on CPU (very slow)")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking CUDA: {e}")
        return False


def check_data():
    """Check processed data availability."""
    logger.info("\nChecking processed data...")
    
    data_dir = Path("./processed_data")
    
    if not data_dir.exists():
        logger.error(f"✗ Data directory not found: {data_dir}")
        return False
    
    logger.info(f"✓ Data directory exists: {data_dir}")
    
    # Check metadata
    metadata_file = data_dir / "statistics.json"
    if metadata_file.exists():
        logger.info(f"✓ Metadata file exists")
        
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)
            logger.info(f"  Total samples: {metadata.get('total_samples', 'unknown')}")
            logger.info(f"  Human samples: {metadata.get('human_samples', 'unknown')}")
            logger.info(f"  AI samples: {metadata.get('ai_samples', 'unknown')}")
    else:
        logger.warning("⚠ Metadata file not found")
    
    # Check parquet files
    ai_dir = data_dir / "ai" / "RAID-Dataset"
    if ai_dir.exists():
        ai_files = list(ai_dir.glob("*/*.parquet"))
        logger.info(f"✓ Found {len(ai_files)} AI parquet files")
    else:
        logger.warning("⚠ AI data directory not found")
    
    real_dir = data_dir / "real" / "RAID-Dataset"
    if real_dir.exists():
        real_files = list(real_dir.glob("*/*.parquet"))
        logger.info(f"✓ Found {len(real_files)} real/human parquet files")
    else:
        logger.warning("⚠ Real/human data directory not found")
    
    return ai_dir.exists() and real_dir.exists()


def check_training_module():
    """Check training module files."""
    logger.info("\nChecking training module files...")
    
    files_required = {
        'src/training/train.py': 'Main training module',
        'src/training/data_loader.py': 'Data loading module',
        'src/training/train_script.py': 'CLI training script',
        'src/training/advanced_utils.py': 'Advanced utilities',
        'src/training/README.md': 'Training documentation',
    }
    
    all_exist = True
    for filepath, description in files_required.items():
        if Path(filepath).exists():
            logger.info(f"✓ {description}: {filepath}")
        else:
            logger.error(f"✗ {description} not found: {filepath}")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if training modules can be imported."""
    logger.info("\nChecking module imports...")
    
    try:
        from src.training.train import ModelTrainer, TrainingConfig, DeBERTaAIDetector
        logger.info("✓ Successfully imported training classes")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import training modules: {e}")
        return False


def check_model_instantiation():
    """Check if model can be instantiated."""
    logger.info("\nChecking model instantiation...")
    
    try:
        import torch
        from src.training.train import ModelTrainer, TrainingConfig
        
        config = TrainingConfig(
            num_train_epochs=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        trainer = ModelTrainer(config)
        logger.info(f"✓ Model instantiated successfully")
        logger.info(f"  Total parameters: {trainer._count_parameters():,}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to instantiate model: {e}")
        return False


def check_config_serialization():
    """Check configuration serialization."""
    logger.info("\nChecking configuration serialization...")
    
    try:
        from src.training.train import TrainingConfig
        import json
        
        config = TrainingConfig(num_train_epochs=5, learning_rate=2e-5)
        config_dict = config.to_dict()
        
        # Try to serialize
        json_str = json.dumps(config_dict)
        logger.info(f"✓ Configuration serializable")
        logger.info(f"  Config params: {len(config_dict)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration serialization failed: {e}")
        return False


def check_data_loader():
    """Check data loader functionality."""
    logger.info("\nChecking data loader...")
    
    try:
        from src.training.data_loader import ProcessedDataLoader
        from pathlib import Path
        
        loader = ProcessedDataLoader(data_dir=Path("./processed_data"))
        summary = loader.get_data_summary()
        
        logger.info(f"✓ Data loader initialized")
        logger.info(f"  Total samples: {summary.get('total_samples', 0)}")
        logger.info(f"  Models available: {len(summary.get('ai_models', []))}")
        
        # Check file discovery
        files = loader.find_parquet_files()
        logger.info(f"  Data categories found: {len(files)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loader check failed: {e}")
        logger.error(f"  This is expected if processed_data folder structure differs")
        return False


def check_output_directory():
    """Check output directory setup."""
    logger.info("\nChecking output directory...")
    
    output_dir = Path("./outputs")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Output directory ready: {output_dir.absolute()}")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot create output directory: {e}")
        return False


def run_all_checks():
    """Run all validation checks."""
    logger.info("="*80)
    logger.info("AI Text Detection - Training Module Validation")
    logger.info("="*80)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA/GPU", check_cuda),
        ("Training Module Files", check_training_module),
        ("Module Imports", check_imports),
        ("Model Instantiation", check_model_instantiation),
        ("Config Serialization", check_config_serialization),
        ("Data Loader", check_data_loader),
        ("Processed Data", check_data),
        ("Output Directory", check_output_directory),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            logger.error(f"Exception in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    logger.info(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n🎉 All checks passed! Ready for training.")
        return True
    elif passed >= total - 2:
        logger.info("\n⚠ Most checks passed. Training should work.")
        return True
    else:
        logger.error("\n❌ Several checks failed. Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
