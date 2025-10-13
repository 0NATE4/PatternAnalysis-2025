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
        print(f"   ✅ Dataset loaded - {len(train_data)} samples")
        
        # Test tokenization
        print("4. Testing tokenization...")
        model_wrapper = build_model_with_lora(config)
        model, tokenizer = model_wrapper.get_model_and_tokenizer()
        tokenized_data = train_data.map(
            lambda examples: dataset.preprocess_function(examples, tokenizer),
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            remove_columns=["input_text", "target_text", "source", "images_path"],
            desc="Tokenizing dataset"
        )
        print(f"   ✅ Tokenization OK - {len(tokenized_data)} samples")
        print(f"   Sample keys: {list(tokenized_data[0].keys())}")
        
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
