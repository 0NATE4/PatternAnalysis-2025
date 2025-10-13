#!/usr/bin/env python3
"""
Local Testing Script for CPU-based validation

This script tests the training pipeline locally without GPU requirements.
It catches most issues before running on the cluster.

Usage:
    python test_local.py

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all imports work correctly."""
    print("🔍 Testing imports...")
    try:
        from utils import load_config, setup_reproducibility, get_device
        from dataset import BioLaySummDataset
        from modules import FLANT5LoRAModel, build_model_with_lora
        from train import BioLaySummTrainer
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🔍 Testing config loading...")
    try:
        from utils import load_config
        config = load_config("configs/test_local_cpu.yaml")
        print(f"✅ Config loaded successfully!")
        print(f"   Model: {config['model']['name']}")
        print(f"   Strategy: {config['model']['strategy']}")
        print(f"   Max samples: {config['dataset']['max_samples']}")
        return True, config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_dataset_loading(config):
    """Test dataset loading and tokenization."""
    print("\n🔍 Testing dataset loading...")
    try:
        from dataset import BioLaySummDataset
        
        # Create dataset with small sample
        dataset = BioLaySummDataset(config)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train samples: {len(dataset.train_dataset)}")
        print(f"   Val samples: {len(dataset.val_dataset)}")
        
        # Test a single sample
        sample = dataset.train_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Input length: {len(sample['input_ids'])}")
        print(f"   Label length: {len(sample['labels'])}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation(config):
    """Test model creation and basic forward pass."""
    print("\n🔍 Testing model creation...")
    try:
        from modules import build_model_with_lora
        import torch
        
        # Create model
        model, tokenizer = build_model_with_lora(config)
        print(f"✅ Model created successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        
        # Test basic forward pass with dummy data
        dummy_input = tokenizer("Test input", return_tensors="pt", max_length=64, truncation=True)
        dummy_labels = tokenizer("Test output", return_tensors="pt", max_length=64, truncation=True)
        
        with torch.no_grad():
            outputs = model(**dummy_input, labels=dummy_labels["input_ids"])
            print(f"   Forward pass successful! Loss: {outputs.loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_trainer_creation(config):
    """Test trainer creation."""
    print("\n🔍 Testing trainer creation...")
    try:
        from train import BioLaySummTrainer
        
        # Create trainer
        trainer = BioLaySummTrainer(config)
        print(f"✅ Trainer created successfully!")
        print(f"   Trainer type: {type(trainer).__name__}")
        print(f"   Model type: {type(trainer.model).__name__}")
        print(f"   Dataset size: {len(trainer.train_dataset)}")
        
        return True
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        traceback.print_exc()
        return False

def test_training_step(config):
    """Test a single training step (without actual training)."""
    print("\n🔍 Testing training step preparation...")
    try:
        from train import BioLaySummTrainer
        
        trainer = BioLaySummTrainer(config)
        
        # Test dataloader creation
        dataloader = trainer.trainer.get_train_dataloader()
        print(f"✅ Dataloader created successfully!")
        print(f"   Batch size: {len(next(iter(dataloader))['input_ids'])}")
        
        # Test optimizer creation
        optimizer = trainer.trainer.get_optimizer()
        print(f"✅ Optimizer created successfully!")
        print(f"   Optimizer type: {type(optimizer).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Training step preparation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all local tests."""
    print("🚀 Starting Local CPU Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Config Loading", lambda: test_config_loading()[0]),
        ("Dataset Loading", lambda: test_dataset_loading(test_config_loading()[1]) if test_config_loading()[0] else False),
        ("Model Creation", lambda: test_model_creation(test_config_loading()[1]) if test_config_loading()[0] else False),
        ("Trainer Creation", lambda: test_trainer_creation(test_config_loading()[1]) if test_config_loading()[0] else False),
        ("Training Step", lambda: test_training_step(test_config_loading()[1]) if test_config_loading()[0] else False),
    ]
    
    results = []
    config = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Config Loading":
                success, config = test_config_loading()
                results.append((test_name, success))
            elif config is not None:
                success = test_func()
                results.append((test_name, success))
            else:
                print(f"⏭️  Skipping {test_name} (config not loaded)")
                results.append((test_name, False))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for cluster deployment!")
    else:
        print("⚠️  Some tests failed. Fix issues before cluster deployment.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
