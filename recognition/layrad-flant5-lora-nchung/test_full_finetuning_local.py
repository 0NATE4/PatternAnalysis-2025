#!/usr/bin/env python3
"""
Local Test for Full Fine-Tuning
Test T5-small full fine-tuning locally before cluster deployment
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_full_finetuning():
    print("Testing Full Fine-Tuning Locally")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from utils import load_config
        from dataset import BioLaySummDataset
        from modules import FLANT5LoRAModel
        from train import BioLaySummTrainer
        print("   ✅ Imports OK")
        
        # Test config
        print("2. Testing config...")
        config = load_config("configs/train_test_full.yaml")
        print(f"   ✅ Config OK - {config['model']['name']}")
        print(f"   Strategy: {config['training']['strategy']}")
        
        # Test dataset (small sample)
        print("3. Testing dataset...")
        dataset = BioLaySummDataset(config)
        train_data = dataset.load_data('train')
        print(f"   ✅ Dataset loaded - {len(train_data)} samples")
        
        # Test model creation
        print("4. Testing model creation...")
        model_wrapper = FLANT5LoRAModel(config)
        model, tokenizer = model_wrapper.get_model_and_tokenizer()
        print(f"   ✅ Model OK - {type(model).__name__}")
        print(f"   Model parameters: {model_wrapper.count_params()}")
        
        # Test tokenization
        print("5. Testing tokenization...")
        tokenized_data = train_data.map(
            lambda examples: dataset.preprocess_function(examples, tokenizer),
            batched=True,
            num_proc=0,  # No multiprocessing for local test
            load_from_cache_file=False,
            remove_columns=["input_text", "target_text", "source", "images_path"],
            desc="Tokenizing dataset"
        )
        print(f"   ✅ Tokenization OK - {len(tokenized_data)} samples")
        print(f"   Sample keys: {list(tokenized_data[0].keys())}")
        
        # Test trainer creation
        print("6. Testing trainer creation...")
        trainer = BioLaySummTrainer(config)
        print(f"   ✅ Trainer OK - {type(trainer).__name__}")
        
        print("\nAll full fine-tuning tests passed!")
        print("Ready for cluster deployment!")
        return True
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_finetuning()
    sys.exit(0 if success else 1)
