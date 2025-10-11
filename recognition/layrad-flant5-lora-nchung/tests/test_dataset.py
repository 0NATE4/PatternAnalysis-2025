#!/usr/bin/env python3
"""
Test script for BioLaySumm dataset loader.

This script tests the dataset loading functionality to ensure everything
works correctly before proceeding with training.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
from pathlib import Path

# Add src directory to path (go up one level from tests/ to find src/)
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility
from dataset import BioLaySummDataset


def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("=" * 60)
    print("Testing BioLaySumm Dataset Loader")
    print("=" * 60)
    
    # Load configuration
    try:
        config = load_config('configs/train_flant5_base_lora.yaml')
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Setup reproducibility
    try:
        setup_reproducibility(config)
        print("✅ Reproducibility setup complete")
    except Exception as e:
        print(f"❌ Failed to setup reproducibility: {e}")
        return False
    
    # Initialize dataset loader
    try:
        loader = BioLaySummDataset(config)
        print("✅ Dataset loader initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize dataset loader: {e}")
        return False
    
    # Test loading validation split (smaller, faster)
    try:
        print("\nLoading validation split...")
        val_data = loader.load_data('validation')
        print(f"✅ Validation data loaded: {len(val_data)} samples")
        
        # Check data structure
        sample = val_data[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"✅ Input text length: {len(sample['input_text'])} chars")
        print(f"✅ Target text length: {len(sample['target_text'])} chars")
        
        # Print a sample
        print("\n" + "=" * 40)
        print("SAMPLE DATA:")
        print("=" * 40)
        print("INPUT TEXT:")
        print(sample['input_text'][:200] + "..." if len(sample['input_text']) > 200 else sample['input_text'])
        print("\nTARGET TEXT:")
        print(sample['target_text'])
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Failed to load validation data: {e}")
        return False
    
    print("\n🎉 All tests passed! Dataset loader is working correctly.")
    return True


if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
