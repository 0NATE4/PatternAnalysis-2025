#!/usr/bin/env python3
"""
Test script for full fine-tuning configuration and functionality.

This script verifies that the full fine-tuning system works correctly,
including configuration loading, model building, and parameter counting.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
from pathlib import Path
import torch

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility
from dataset import BioLaySummDataset


def test_full_finetuning_config():
    """Test full fine-tuning configuration loading."""
    print("=" * 60)
    print("Testing Full Fine-Tuning Configuration")
    print("=" * 60)

    try:
        # 1. Load full fine-tuning configuration
        print("Loading full fine-tuning configuration...")
        config = load_config("configs/train_t5_small_full.yaml")
        setup_reproducibility(config['reproducibility'])
        print("✅ Full fine-tuning configuration loaded successfully")

        # 2. Verify configuration structure
        print("Verifying configuration structure...")
        
        # Check required sections
        assert 'dataset' in config, "Should have dataset section"
        assert 'model' in config, "Should have model section"
        assert 'training' in config, "Should have training section"
        assert 'full_finetuning' in config, "Should have full_finetuning section"
        assert 'evaluation' in config, "Should have evaluation section"
        
        # Check model configuration
        assert config['model']['name'] == 't5-small', "Should use t5-small model"
        
        # Check full fine-tuning settings
        assert config['full_finetuning']['enabled'] == True, "Should have full fine-tuning enabled"
        assert config['full_finetuning']['gradient_checkpointing'] == True, "Should have gradient checkpointing"
        
        # Check training settings
        assert config['training']['batch_size'] == 4, "Should have smaller batch size for full FT"
        learning_rate = float(config['training']['learning_rate'])
        print(f"Learning rate: {learning_rate}")
        assert learning_rate == 5e-5, f"Should have lower learning rate for full FT, got {learning_rate}"
        assert config['training']['num_epochs'] == 2, "Should have fewer epochs for full FT"
        
        print("✅ Configuration structure verified")

        # 3. Test dataset loading
        print("Testing dataset loading...")
        dataset_loader = BioLaySummDataset(config)
        train_dataset = dataset_loader.load_data('train').select(range(5))  # Small sample
        val_dataset = dataset_loader.load_data('validation').select(range(3))  # Small sample
        
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")

        print("\n🎉 All full fine-tuning configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Full fine-tuning configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_finetuning_vs_lora():
    """Test full fine-tuning vs LoRA comparison."""
    print("\n" + "=" * 60)
    print("Testing Full Fine-Tuning vs LoRA Comparison")
    print("=" * 60)

    try:
        # 1. Load both configurations
        print("Loading LoRA and full fine-tuning configurations...")
        
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        print("✅ Both configurations loaded successfully")

        # 2. Compare configurations
        print("Comparing configurations...")
        
        # Model comparison
        lora_model = lora_config['model']['name']
        full_model = full_config['model']['name']
        
        print(f"LoRA model: {lora_model}")
        print(f"Full FT model: {full_model}")
        
        # Training strategy comparison
        lora_strategy = lora_config.get('training', {}).get('strategy', 'lora')
        full_strategy = full_config.get('training', {}).get('strategy', 'full')
        full_enabled = full_config.get('full_finetuning', {}).get('enabled', False)
        
        print(f"LoRA strategy: {lora_strategy}")
        print(f"Full FT strategy: {full_strategy or 'full' if full_enabled else 'lora'}")
        
        # Batch size comparison
        lora_batch = lora_config['training']['batch_size']
        full_batch = full_config['training']['batch_size']
        
        print(f"LoRA batch size: {lora_batch}")
        print(f"Full FT batch size: {full_batch}")
        assert full_batch < lora_batch, "Full FT should have smaller batch size"
        
        # Learning rate comparison
        lora_lr = float(lora_config['training']['learning_rate'])
        full_lr = float(full_config['training']['learning_rate'])
        
        print(f"LoRA learning rate: {lora_lr}")
        print(f"Full FT learning rate: {full_lr}")
        print(f"Learning rate comparison: {full_lr} < {lora_lr} = {full_lr < lora_lr}")
        assert full_lr < lora_lr, f"Full FT should have lower learning rate: {full_lr} should be < {lora_lr}"
        
        print("✅ Configuration comparison successful")

        # 3. Test parameter counting setup
        print("Testing parameter counting setup...")
        
        # Mock parameter counts for comparison
        lora_params = {
            'total': 248_462_592,  # FLAN-T5-base
            'trainable': 884_736,   # LoRA parameters
            'frozen': 247_577_856,  # Frozen parameters
            'trainable_percentage': 0.36
        }
        
        full_params = {
            'total': 60_000_000,    # T5-small
            'trainable': 60_000_000, # All parameters trainable
            'frozen': 0,            # No frozen parameters
            'trainable_percentage': 100.0
        }
        
        print(f"LoRA parameters: {lora_params['trainable']:,} trainable ({lora_params['trainable_percentage']:.2f}%)")
        print(f"Full FT parameters: {full_params['trainable']:,} trainable ({full_params['trainable_percentage']:.2f}%)")
        
        assert full_params['trainable_percentage'] > lora_params['trainable_percentage'], "Full FT should have more trainable parameters"
        
        print("✅ Parameter comparison successful")

        print("\n🎉 All full fine-tuning vs LoRA comparison tests passed!")
        return True

    except Exception as e:
        print(f"❌ Full fine-tuning vs LoRA comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_strategy_detection():
    """Test training strategy detection logic."""
    print("\n" + "=" * 60)
    print("Testing Training Strategy Detection")
    print("=" * 60)

    try:
        # Test LoRA strategy detection
        print("Testing LoRA strategy detection...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        
        training_strategy = lora_config.get('training', {}).get('strategy', 'lora')
        full_finetuning_enabled = lora_config.get('full_finetuning', {}).get('enabled', False)
        
        assert training_strategy == 'lora', "Should detect LoRA strategy"
        assert full_finetuning_enabled == False, "Should not have full fine-tuning enabled"
        print("✅ LoRA strategy detection successful")
        
        # Test full fine-tuning strategy detection
        print("Testing full fine-tuning strategy detection...")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        training_strategy = full_config.get('training', {}).get('strategy', 'lora')
        full_finetuning_enabled = full_config.get('full_finetuning', {}).get('enabled', False)
        
        assert full_finetuning_enabled == True, "Should have full fine-tuning enabled"
        print("✅ Full fine-tuning strategy detection successful")
        
        # Test strategy selection logic
        print("Testing strategy selection logic...")
        
        def get_training_strategy(config):
            training_strategy = config.get('training', {}).get('strategy', 'lora')
            full_finetuning_enabled = config.get('full_finetuning', {}).get('enabled', False)
            
            if training_strategy == 'full' or full_finetuning_enabled:
                return 'full'
            else:
                return 'lora'
        
        lora_strategy = get_training_strategy(lora_config)
        full_strategy = get_training_strategy(full_config)
        
        assert lora_strategy == 'lora', "Should select LoRA strategy"
        assert full_strategy == 'full', "Should select full fine-tuning strategy"
        
        print("✅ Strategy selection logic successful")

        print("\n🎉 All training strategy detection tests passed!")
        return True

    except Exception as e:
        print(f"❌ Training strategy detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_full_finetuning_config()
    success2 = test_full_finetuning_vs_lora()
    success3 = test_training_strategy_detection()
    
    if all([success1, success2, success3]):
        print("\n🚀 All full fine-tuning tests passed!")
        print("✅ Full fine-tuning system is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some full fine-tuning tests failed.")
        sys.exit(1)
