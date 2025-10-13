#!/usr/bin/env python3
"""
Quick Local Test - Just test the core functionality

This is a minimal test to quickly validate the pipeline works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def quick_test():
    print("Quick Local Test")
    print("-" * 30)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from utils import load_config
        from dataset import BioLaySummDataset
        from modules import build_model_with_lora
        from train import BioLaySummTrainer
        print("   ✅ Imports OK")
        
        # Test config
        print("2. Testing config...")
        config = load_config("configs/test_local_cpu.yaml")
        print(f"   ✅ Config OK - {config['model']['name']}")
        
        # Test dataset (very small)
        print("3. Testing dataset...")
        dataset = BioLaySummDataset(config)
        train_data = dataset.load_data('train')
        print(f"   ✅ Dataset OK - {len(train_data)} samples")
        
        # Test model
        print("4. Testing model...")
        model, tokenizer = build_model_with_lora(config)
        print(f"   ✅ Model OK - {type(model).__name__}")
        
        # Test trainer
        print("5. Testing trainer...")
        trainer = BioLaySummTrainer(config)
        print(f"   ✅ Trainer OK - {type(trainer).__name__}")
        
        print("\n🎉 All quick tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
