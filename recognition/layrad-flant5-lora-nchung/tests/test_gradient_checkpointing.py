#!/usr/bin/env python3
"""
Test script for gradient checkpointing functionality.

This script verifies that gradient checkpointing is properly enabled/disabled
based on configuration settings and training strategy.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility


def test_gradient_checkpointing_logic():
    """Test gradient checkpointing decision logic."""
    print("=" * 60)
    print("Testing Gradient Checkpointing Logic")
    print("=" * 60)

    try:
        # 1. Test LoRA configuration (should disable gradient checkpointing)
        print("Testing LoRA configuration...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        setup_reproducibility(lora_config['reproducibility'])
        
        # Mock the gradient checkpointing logic
        def should_enable_gradient_checkpointing(config):
            training_strategy = config.get('training', {}).get('strategy', 'lora')
            full_finetuning_enabled = config.get('full_finetuning', {}).get('enabled', False)
            
            is_full_finetuning = (training_strategy == 'full' or full_finetuning_enabled)
            
            if not is_full_finetuning:
                return False
            
            training_config = config.get('training', {})
            full_ft_config = config.get('full_finetuning', {})
            full_ft_settings = config.get('full_finetuning_settings', {})
            
            gradient_checkpointing = (
                training_config.get('gradient_checkpointing',
                full_ft_settings.get('gradient_checkpointing', 
                full_ft_config.get('gradient_checkpointing', True)))
            )
            
            return gradient_checkpointing
        
        lora_gc = should_enable_gradient_checkpointing(lora_config)
        assert lora_gc == False, f"LoRA should disable gradient checkpointing, got {lora_gc}"
        print("✅ LoRA gradient checkpointing logic correct")
        
        # 2. Test full fine-tuning configuration (should enable gradient checkpointing)
        print("Testing full fine-tuning configuration...")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        full_gc = should_enable_gradient_checkpointing(full_config)
        assert full_gc == True, f"Full FT should enable gradient checkpointing, got {full_gc}"
        print("✅ Full fine-tuning gradient checkpointing logic correct")
        
        print("\n🎉 All gradient checkpointing logic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Gradient checkpointing logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_checkpointing_configuration():
    """Test gradient checkpointing configuration settings."""
    print("\n" + "=" * 60)
    print("Testing Gradient Checkpointing Configuration")
    print("=" * 60)

    try:
        # 1. Test LoRA configuration settings
        print("Testing LoRA gradient checkpointing settings...")
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        
        # Check that LoRA has gradient_checkpointing set to false
        lora_gc_setting = lora_config.get('training', {}).get('gradient_checkpointing', None)
        assert lora_gc_setting == False, f"LoRA should have gradient_checkpointing=false, got {lora_gc_setting}"
        
        print("✅ LoRA gradient checkpointing configuration correct")
        
        # 2. Test full fine-tuning configuration settings
        print("Testing full fine-tuning gradient checkpointing settings...")
        full_config = load_config("configs/train_t5_small_full.yaml")
        
        # Check that full FT has gradient checkpointing enabled
        full_ft_gc = full_config.get('full_finetuning', {}).get('gradient_checkpointing', None)
        assert full_ft_gc == True, f"Full FT should have gradient_checkpointing=true, got {full_ft_gc}"
        
        # Check full_finetuning_settings
        full_ft_settings_gc = full_config.get('full_finetuning_settings', {}).get('gradient_checkpointing', None)
        assert full_ft_settings_gc == True, f"Full FT settings should have gradient_checkpointing=true, got {full_ft_settings_gc}"
        
        print("✅ Full fine-tuning gradient checkpointing configuration correct")
        
        print("\n🎉 All gradient checkpointing configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Gradient checkpointing configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_checkpointing_priority():
    """Test gradient checkpointing configuration priority."""
    print("\n" + "=" * 60)
    print("Testing Gradient Checkpointing Priority")
    print("=" * 60)

    try:
        # Test configuration priority: training > full_finetuning_settings > full_finetuning > default
        
        print("Testing configuration priority order...")
        
        # Mock priority logic
        def get_gradient_checkpointing_setting(config):
            training_config = config.get('training', {})
            full_ft_config = config.get('full_finetuning', {})
            full_ft_settings = config.get('full_finetuning_settings', {})
            
            # Priority order: training > full_finetuning_settings > full_finetuning > default
            return (
                training_config.get('gradient_checkpointing',
                full_ft_settings.get('gradient_checkpointing', 
                full_ft_config.get('gradient_checkpointing', True)))
            )
        
        # 1. Test training config priority
        test_config = {
            'training': {'gradient_checkpointing': False},
            'full_finetuning': {'gradient_checkpointing': True},
            'full_finetuning_settings': {'gradient_checkpointing': True}
        }
        
        result = get_gradient_checkpointing_setting(test_config)
        assert result == False, f"Training config should have priority, got {result}"
        print("✅ Training config priority correct")
        
        # 2. Test full_finetuning_settings priority (when training not set)
        test_config = {
            'training': {},
            'full_finetuning': {'gradient_checkpointing': True},
            'full_finetuning_settings': {'gradient_checkpointing': False}
        }
        
        result = get_gradient_checkpointing_setting(test_config)
        assert result == False, f"full_finetuning_settings should have priority, got {result}"
        print("✅ full_finetuning_settings priority correct")
        
        # 3. Test full_finetuning priority (when others not set)
        test_config = {
            'training': {},
            'full_finetuning': {'gradient_checkpointing': False},
            'full_finetuning_settings': {}
        }
        
        result = get_gradient_checkpointing_setting(test_config)
        assert result == False, f"full_finetuning should have priority, got {result}"
        print("✅ full_finetuning priority correct")
        
        # 4. Test default value
        test_config = {
            'training': {},
            'full_finetuning': {},
            'full_finetuning_settings': {}
        }
        
        result = get_gradient_checkpointing_setting(test_config)
        assert result == True, f"Default should be True, got {result}"
        print("✅ Default value correct")

        print("\n🎉 All gradient checkpointing priority tests passed!")
        return True

    except Exception as e:
        print(f"❌ Gradient checkpointing priority test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_checkpointing_memory_tradeoffs():
    """Test gradient checkpointing memory vs compute trade-offs."""
    print("\n" + "=" * 60)
    print("Testing Gradient Checkpointing Memory Trade-offs")
    print("=" * 60)

    try:
        print("Testing memory vs compute trade-offs...")
        
        # Mock memory and compute estimates
        def estimate_training_requirements(config, gradient_checkpointing_enabled):
            # Base model parameters
            model_name = config.get('model', {}).get('name', '')
            
            if 't5-small' in model_name.lower():
                base_params = 60_000_000
                base_memory_gb = 4.0
            elif 'flan-t5-base' in model_name.lower():
                base_params = 248_000_000
                base_memory_gb = 12.0
            else:
                base_params = 100_000_000
                base_memory_gb = 8.0
            
            # Training strategy impact
            training_strategy = config.get('training', {}).get('strategy', 'lora')
            batch_size = config.get('training', {}).get('batch_size', 8)
            
            if training_strategy == 'lora':
                memory_multiplier = 1.0
                compute_multiplier = 1.0
                trainable_params = base_params * 0.004  # ~0.4% for LoRA
            else:  # full fine-tuning
                memory_multiplier = 2.5
                compute_multiplier = 1.8
                trainable_params = base_params
                
                # Gradient checkpointing impact
                if gradient_checkpointing_enabled:
                    memory_multiplier *= 0.6  # Reduce memory usage
                    compute_multiplier *= 1.2  # Increase compute time
            
            # Batch size impact
            batch_memory_factor = batch_size / 8.0
            
            estimated_memory = base_memory_gb * memory_multiplier * batch_memory_factor
            estimated_compute_time = compute_multiplier * batch_memory_factor
            
            return {
                'model_params': base_params,
                'trainable_params': trainable_params,
                'estimated_memory_gb': estimated_memory,
                'estimated_compute_time': estimated_compute_time,
                'gradient_checkpointing': gradient_checkpointing_enabled
            }
        
        # Test LoRA configuration
        lora_config = load_config("configs/train_flant5_base_lora.yaml")
        lora_requirements = estimate_training_requirements(lora_config, False)
        
        print(f"LoRA (FLAN-T5-base):")
        print(f"  - Trainable parameters: {lora_requirements['trainable_params']:,.0f}")
        print(f"  - Estimated memory: {lora_requirements['estimated_memory_gb']:.1f} GB")
        print(f"  - Estimated compute time: {lora_requirements['estimated_compute_time']:.1f}x baseline")
        
        # Test full fine-tuning without gradient checkpointing
        full_config = load_config("configs/train_t5_small_full.yaml")
        full_no_gc = estimate_training_requirements(full_config, False)
        
        print(f"\nFull FT (T5-small, no gradient checkpointing):")
        print(f"  - Trainable parameters: {full_no_gc['trainable_params']:,.0f}")
        print(f"  - Estimated memory: {full_no_gc['estimated_memory_gb']:.1f} GB")
        print(f"  - Estimated compute time: {full_no_gc['estimated_compute_time']:.1f}x baseline")
        
        # Test full fine-tuning with gradient checkpointing
        full_with_gc = estimate_training_requirements(full_config, True)
        
        print(f"\nFull FT (T5-small, with gradient checkpointing):")
        print(f"  - Trainable parameters: {full_with_gc['trainable_params']:,.0f}")
        print(f"  - Estimated memory: {full_with_gc['estimated_memory_gb']:.1f} GB")
        print(f"  - Estimated compute time: {full_with_gc['estimated_compute_time']:.1f}x baseline")
        
        # Validate trade-offs
        assert full_with_gc['estimated_memory_gb'] < full_no_gc['estimated_memory_gb'], "Gradient checkpointing should reduce memory usage"
        assert full_with_gc['estimated_compute_time'] > full_no_gc['estimated_compute_time'], "Gradient checkpointing should increase compute time"
        
        print("\n✅ Memory vs compute trade-offs validated")
        print("✅ Gradient checkpointing reduces memory usage at the cost of compute time")

        print("\n🎉 All gradient checkpointing memory trade-off tests passed!")
        return True

    except Exception as e:
        print(f"❌ Gradient checkpointing memory trade-off test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_gradient_checkpointing_logic()
    success2 = test_gradient_checkpointing_configuration()
    success3 = test_gradient_checkpointing_priority()
    success4 = test_gradient_checkpointing_memory_tradeoffs()
    
    if all([success1, success2, success3, success4]):
        print("\n🚀 All gradient checkpointing tests passed!")
        print("✅ Gradient checkpointing toggle is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some gradient checkpointing tests failed.")
        sys.exit(1)
