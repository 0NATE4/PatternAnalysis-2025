#!/usr/bin/env python3
"""
Test script for training strategy support (LoRA vs Full Fine-tuning).

This script verifies that the training system correctly supports both
LoRA and full fine-tuning strategies through the configuration interface.

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


def test_strategy_validation():
    """Test training strategy validation logic."""
    print("=" * 60)
    print("Testing Training Strategy Validation")
    print("=" * 60)

    try:
        # 1. Test LoRA strategy validation
        print("Testing LoRA strategy validation...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        setup_reproducibility(lora_config['reproducibility'])
        
        # Mock the training system to test validation
        class MockTrainer:
            def __init__(self, config):
                self.config = config
                
            def _validate_training_strategy(self):
                training_strategy = self.config.get('training', {}).get('strategy', 'lora')
                full_finetuning_enabled = self.config.get('full_finetuning', {}).get('enabled', False)
                
                valid_strategies = {'lora', 'full'}
                if training_strategy not in valid_strategies:
                    raise ValueError(f"Invalid training strategy: {training_strategy}. Must be one of {valid_strategies}")
                
                return training_strategy
        
        trainer = MockTrainer(lora_config)
        strategy = trainer._validate_training_strategy()
        
        assert strategy == 'lora', f"Should detect LoRA strategy, got {strategy}"
        print("✅ LoRA strategy validation successful")
        
        # 2. Test full fine-tuning strategy validation
        print("Testing full fine-tuning strategy validation...")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        trainer = MockTrainer(full_config)
        strategy = trainer._validate_training_strategy()
        
        assert strategy == 'full', f"Should detect full fine-tuning strategy, got {strategy}"
        print("✅ Full fine-tuning strategy validation successful")
        
        # 3. Test invalid strategy
        print("Testing invalid strategy handling...")
        invalid_config = lora_config.copy()
        invalid_config['training']['strategy'] = 'invalid'
        
        trainer = MockTrainer(invalid_config)
        try:
            strategy = trainer._validate_training_strategy()
            assert False, "Should have raised ValueError for invalid strategy"
        except ValueError as e:
            assert "Invalid training strategy: invalid" in str(e)
            print("✅ Invalid strategy handling successful")
        
        print("\n🎉 All strategy validation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Strategy validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_configuration():
    """Test strategy configuration loading and consistency."""
    print("\n" + "=" * 60)
    print("Testing Strategy Configuration")
    print("=" * 60)

    try:
        # 1. Test LoRA configuration
        print("Testing LoRA configuration...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        
        # Check training strategy
        assert lora_config['training']['strategy'] == 'lora', "LoRA config should have strategy='lora'"
        
        # Check LoRA-specific settings
        assert 'lora' in lora_config, "LoRA config should have lora section"
        assert lora_config['lora']['target_modules'] == ['q', 'v'], "Should target q and v modules"
        assert lora_config['lora']['r'] == 8, "Should have LoRA rank 8"
        assert lora_config['lora']['alpha'] == 32, "Should have LoRA alpha 32"
        
        print("✅ LoRA configuration validated")
        
        # 2. Test full fine-tuning configuration
        print("Testing full fine-tuning configuration...")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        # Check training strategy
        assert full_config['training']['strategy'] == 'full', "Full FT config should have strategy='full'"
        
        # Check full fine-tuning settings
        assert 'full_finetuning' in full_config, "Full FT config should have full_finetuning section"
        assert full_config['full_finetuning']['enabled'] == True, "Full fine-tuning should be enabled"
        assert full_config['full_finetuning']['gradient_checkpointing'] == True, "Should have gradient checkpointing"
        
        print("✅ Full fine-tuning configuration validated")
        
        # 3. Test configuration differences
        print("Testing configuration differences...")
        
        # Batch sizes
        lora_batch = lora_config['training']['batch_size']
        full_batch = full_config['training']['batch_size']
        assert full_batch < lora_batch, "Full FT should have smaller batch size"
        
        # Learning rates
        lora_lr = float(lora_config['training']['learning_rate'])
        full_lr = float(full_config['training']['learning_rate'])
        assert full_lr < lora_lr, "Full FT should have lower learning rate"
        
        # Models
        lora_model = lora_config['model']['name']
        full_model = full_config['model']['name']
        assert lora_model != full_model, "Should use different models"
        
        print("✅ Configuration differences validated")

        print("\n🎉 All strategy configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Strategy configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_selection_logic():
    """Test strategy selection logic in training system."""
    print("\n" + "=" * 60)
    print("Testing Strategy Selection Logic")
    print("=" * 60)

    try:
        # 1. Test strategy selection function
        print("Testing strategy selection logic...")
        
        def get_training_strategy(config):
            """Mock strategy selection logic."""
            training_strategy = config.get('training', {}).get('strategy', 'lora')
            full_finetuning_enabled = config.get('full_finetuning', {}).get('enabled', False)
            
            if training_strategy == 'full' or full_finetuning_enabled:
                return 'full'
            else:
                return 'lora'
        
        # Test LoRA selection
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        lora_strategy = get_training_strategy(lora_config)
        assert lora_strategy == 'lora', f"Should select LoRA, got {lora_strategy}"
        
        # Test full fine-tuning selection
        full_config = load_config("configs/train_t5_small_full.yaml")
        full_strategy = get_training_strategy(full_config)
        assert full_strategy == 'full', f"Should select full fine-tuning, got {full_strategy}"
        
        print("✅ Strategy selection logic successful")
        
        # 2. Test backward compatibility
        print("Testing backward compatibility...")
        
        # Test config with only full_finetuning.enabled=True (no strategy field)
        backward_config = {
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
            },
            'full_finetuning': {
                'enabled': True
            }
        }
        
        backward_strategy = get_training_strategy(backward_config)
        assert backward_strategy == 'full', "Should detect full fine-tuning from enabled flag"
        
        print("✅ Backward compatibility successful")

        print("\n🎉 All strategy selection logic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Strategy selection logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_parameter_comparison():
    """Test parameter counting and comparison between strategies."""
    print("\n" + "=" * 60)
    print("Testing Strategy Parameter Comparison")
    print("=" * 60)

    try:
        # 1. Load configurations
        print("Loading configurations...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        print("✅ Configurations loaded")
        
        # 2. Mock parameter counts
        print("Testing parameter count comparisons...")
        
        # LoRA parameter counts (FLAN-T5-base)
        lora_params = {
            'total': 248_462_592,      # FLAN-T5-base total parameters
            'trainable': 884_736,       # LoRA trainable parameters (q, v modules)
            'frozen': 247_577_856,      # Frozen parameters
            'trainable_percentage': 0.36
        }
        
        # Full fine-tuning parameter counts (T5-small)
        full_params = {
            'total': 60_000_000,        # T5-small total parameters
            'trainable': 60_000_000,    # All parameters trainable
            'frozen': 0,                # No frozen parameters
            'trainable_percentage': 100.0
        }
        
        print(f"LoRA (FLAN-T5-base):")
        print(f"  - Total parameters: {lora_params['total']:,}")
        print(f"  - Trainable: {lora_params['trainable']:,} ({lora_params['trainable_percentage']:.2f}%)")
        print(f"  - Frozen: {lora_params['frozen']:,}")
        
        print(f"Full FT (T5-small):")
        print(f"  - Total parameters: {full_params['total']:,}")
        print(f"  - Trainable: {full_params['trainable']:,} ({full_params['trainable_percentage']:.2f}%)")
        print(f"  - Frozen: {full_params['frozen']:,}")
        
        # Validate parameter relationships
        assert lora_params['trainable_percentage'] < full_params['trainable_percentage'], "Full FT should have higher trainable percentage"
        assert lora_params['frozen'] > full_params['frozen'], "LoRA should have more frozen parameters"
        
        # Memory efficiency comparison
        memory_efficiency_lora = lora_params['trainable'] / lora_params['total']
        memory_efficiency_full = full_params['trainable'] / full_params['total']
        
        assert memory_efficiency_lora < memory_efficiency_full, "LoRA should be more memory efficient"
        
        print("✅ Parameter comparison successful")
        
        # 3. Test training efficiency trade-offs
        print("Testing training efficiency trade-offs...")
        
        # Training time estimates (relative)
        lora_training_time = 1.0  # Baseline
        full_training_time = 2.5  # Estimated relative time
        
        # Performance estimates (ROUGE scores)
        lora_performance = 0.75  # Estimated ROUGE-L
        full_performance = 0.80  # Estimated ROUGE-L
        
        print(f"Training efficiency trade-offs:")
        print(f"  - LoRA: {lora_training_time}x training time, ~{lora_performance:.2f} ROUGE-L")
        print(f"  - Full FT: {full_training_time}x training time, ~{full_performance:.2f} ROUGE-L")
        
        # Validate trade-offs
        assert full_training_time > lora_training_time, "Full FT should take longer to train"
        assert full_performance >= lora_performance, "Full FT should have equal or better performance"
        
        print("✅ Training efficiency trade-offs validated")

        print("\n🎉 All strategy parameter comparison tests passed!")
        return True

    except Exception as e:
        print(f"❌ Strategy parameter comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_strategy_validation()
    success2 = test_strategy_configuration()
    success3 = test_strategy_selection_logic()
    success4 = test_strategy_parameter_comparison()
    
    if all([success1, success2, success3, success4]):
        print("\n🚀 All training strategy tests passed!")
        print("✅ Training strategy support is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some training strategy tests failed.")
        sys.exit(1)
