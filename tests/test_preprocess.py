#!/usr/bin/env python3
"""
Test script to verify all features of preprocess.py are working correctly.
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path
import json
import shutil

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ PASSED")
            return True
        else:
            print("❌ FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FAILED with exception: {e}")
        return False

def test_validation_mode():
    """Test validation mode."""
    cmd = ".venv/bin/python preprocess.py --input test_dataset.csv --output test_validation --validate-only"
    return run_command(cmd, "Validation Mode")

def test_basic_processing():
    """Test basic processing."""
    # Clean up if exists
    if Path('test_basic').exists():
        shutil.rmtree('test_basic')
    
    cmd = ".venv/bin/python preprocess.py --input test_dataset.csv --output test_basic --chunk-size 10"
    success = run_command(cmd, "Basic Processing")
    
    if success:
        # Verify output
        print("\nVerifying output structure...")
        
        # Check directories exist
        if not Path('test_basic/human').exists():
            print("❌ Missing human directory")
            return False
        if not Path('test_basic/ai').exists():
            print("❌ Missing ai directory")
            return False
        
        # Check parquet files exist
        human_files = list(Path('test_basic/human').glob('*.parquet'))
        if not human_files:
            print("❌ No human parquet files found")
            return False
        
        # Check statistics.json exists
        if not Path('test_basic/statistics.json').exists():
            print("❌ Missing statistics.json")
            return False
        
        # Verify data can be loaded
        try:
            df = pd.read_parquet(human_files[0])
            print(f"✅ Successfully loaded parquet file with {len(df)} rows")
            
            # Check required columns
            required_cols = ['text', 'label', 'label_name', 'model', 'domain', 'text_length']
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ Missing required column: {col}")
                    return False
            print("✅ All required columns present")
            
        except Exception as e:
            print(f"❌ Failed to load parquet: {e}")
            return False
        
        print("✅ Output verification complete")
    
    return success

def test_compression_options():
    """Test different compression options."""
    if Path('test_gzip').exists():
        shutil.rmtree('test_gzip')
    
    cmd = ".venv/bin/python preprocess.py --input test_dataset.csv --output test_gzip --chunk-size 10 --compression gzip"
    return run_command(cmd, "Gzip Compression")

def test_text_length_filters():
    """Test text length filtering."""
    if Path('test_filter').exists():
        shutil.rmtree('test_filter')
    
    cmd = ".venv/bin/python preprocess.py --input test_dataset.csv --output test_filter --chunk-size 10 --min-text-length 60 --max-text-length 1000"
    success = run_command(cmd, "Text Length Filtering")
    
    if success:
        # Verify filtering worked
        stats_file = Path('test_filter/statistics.json')
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            # Should have fewer samples due to filtering
            if stats['total_samples'] < 20:
                print(f"✅ Filtering worked: {stats['total_samples']} samples (< 20 original)")
            else:
                print(f"⚠ Warning: Expected fewer samples after filtering")
        
        # Check text lengths in output
        try:
            files = list(Path('test_filter').rglob('*.parquet'))
            if files:
                df = pd.read_parquet(files[0])
                if len(df) > 0:
                    min_len = df['text_length'].min()
                    max_len = df['text_length'].max()
                    print(f"✅ Text lengths in range: {min_len} to {max_len}")
                    if min_len < 60:
                        print(f"❌ Found text shorter than minimum: {min_len}")
                        return False
        except Exception as e:
            print(f"⚠ Could not verify text lengths: {e}")
    
    return success

def test_statistics_accuracy():
    """Test that statistics are accurate."""
    print(f"\n{'='*70}")
    print("TEST: Statistics Accuracy")
    print(f"{'='*70}")
    
    if not Path('test_basic/statistics.json').exists():
        print("❌ statistics.json not found")
        return False
    
    with open('test_basic/statistics.json') as f:
        stats = json.load(f)
    
    print(f"Total samples: {stats['total_samples']}")
    print(f"Human: {stats['human_samples']}")
    print(f"AI: {stats['ai_samples']}")
    
    # Count actual files
    actual_count = 0
    for pfile in Path('test_basic').rglob('*.parquet'):
        try:
            df = pd.read_parquet(pfile)
            actual_count += len(df)
        except:
            pass
    
    print(f"Actual rows in parquet files: {actual_count}")
    
    if actual_count == stats['total_samples']:
        print("✅ Statistics match actual data")
        return True
    else:
        print(f"❌ Mismatch: {actual_count} actual vs {stats['total_samples']} reported")
        return False

def test_metadata_generation():
    """Test metadata file generation."""
    print(f"\n{'='*70}")
    print("TEST: Metadata Generation")
    print(f"{'='*70}")
    
    # Check root metadata
    if not Path('test_basic/statistics.json').exists():
        print("❌ Missing statistics.json")
        return False
    print("✅ Found statistics.json")
    
    # Check per-folder metadata
    metadata_files = list(Path('test_basic').rglob('metadata.json'))
    metadata_files = [f for f in metadata_files if f.parent != Path('test_basic')]
    
    if not metadata_files:
        print("❌ No per-folder metadata.json files found")
        return False
    
    print(f"✅ Found {len(metadata_files)} metadata.json files")
    
    # Verify metadata content
    try:
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        required_fields = ['label', 'model_name', 'total_samples', 'num_chunks']
        for field in required_fields:
            if field not in metadata:
                print(f"❌ Missing field in metadata: {field}")
                return False
        
        print("✅ Metadata contains all required fields")
        return True
    except Exception as e:
        print(f"❌ Failed to read metadata: {e}")
        return False

def test_checkpoint_creation():
    """Test that checkpoint file is created."""
    print(f"\n{'='*70}")
    print("TEST: Checkpoint Creation")
    print(f"{'='*70}")
    
    if not Path('test_basic/checkpoint.json').exists():
        print("❌ checkpoint.json not found")
        return False
    
    try:
        with open('test_basic/checkpoint.json') as f:
            checkpoint = json.load(f)
        
        required_fields = ['last_processed_row', 'chunks_completed', 'statistics']
        for field in required_fields:
            if field not in checkpoint:
                print(f"❌ Missing field in checkpoint: {field}")
                return False
        
        print(f"✅ Checkpoint created with {checkpoint['chunks_completed']} chunks")
        return True
    except Exception as e:
        print(f"❌ Failed to read checkpoint: {e}")
        return False

def test_model_segregation():
    """Test that data is correctly segregated by model."""
    print(f"\n{'='*70}")
    print("TEST: Model Segregation")
    print(f"{'='*70}")
    
    ai_dir = Path('test_basic/ai')
    if not ai_dir.exists():
        print("❌ AI directory not found")
        return False
    
    # Get all model directories
    model_dirs = [d for d in ai_dir.iterdir() if d.is_dir()]
    print(f"Found {len(model_dirs)} AI model directories: {[d.name for d in model_dirs]}")
    
    if len(model_dirs) == 0:
        print("❌ No AI model directories found")
        return False
    
    # Verify each model directory contains correct model
    for model_dir in model_dirs:
        parquet_files = list(model_dir.glob('*.parquet'))
        if not parquet_files:
            continue
        
        try:
            df = pd.read_parquet(parquet_files[0])
            unique_models = df['model'].unique()
            
            # Check if model name matches directory
            dir_model = model_dir.name.replace('_', '').replace('.', '')
            df_model = str(unique_models[0]).replace('_', '').replace('.', '').lower()
            
            if dir_model in df_model or df_model in dir_model:
                print(f"✅ {model_dir.name} contains correct model data")
            else:
                print(f"⚠ {model_dir.name} contains: {unique_models}")
        except Exception as e:
            print(f"❌ Error checking {model_dir.name}: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PREPROCESS.PY TEST SUITE")
    print("="*70)
    
    tests = [
        ("Validation Mode", test_validation_mode),
        ("Basic Processing", test_basic_processing),
        ("Compression Options", test_compression_options),
        ("Text Length Filters", test_text_length_filters),
        ("Statistics Accuracy", test_statistics_accuracy),
        ("Metadata Generation", test_metadata_generation),
        ("Checkpoint Creation", test_checkpoint_creation),
        ("Model Segregation", test_model_segregation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    print("="*70 + "\n")
    
    # Cleanup
    print("Cleaning up test directories...")
    for test_dir in ['test_validation', 'test_basic', 'test_gzip', 'test_filter']:
        if Path(test_dir).exists():
            try:
                shutil.rmtree(test_dir)
                print(f"  Removed {test_dir}")
            except:
                pass
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
