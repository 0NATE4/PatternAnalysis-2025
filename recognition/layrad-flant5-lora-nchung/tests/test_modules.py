#!/usr/bin/env python3
"""
Test script for FLAN-T5 LoRA modules.

This script tests the model wrapper functionality to ensure everything
works correctly before proceeding with training.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import torch
from pathlib import Path

# Add src directory to path (go up one level from tests/ to find src/)
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility, get_device
from modules import build_model_with_lora, count_model_parameters


def test_model_loading():
    """Test the model loading and LoRA application."""
    print("=" * 60)
    print("Testing FLAN-T5 LoRA Model Wrapper")
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
    
    # Get device
    try:
        device = get_device(config)
        print(f"✅ Device: {device}")
    except Exception as e:
        print(f"❌ Failed to get device: {e}")
        return False
    
    # Build model with LoRA
    try:
        print("\nBuilding FLAN-T5 model with LoRA...")
        model_wrapper = build_model_with_lora(config)
        print("✅ Model wrapper created successfully")
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        return False
    
    # Test parameter counting
    try:
        print("\nCounting model parameters...")
        param_info = model_wrapper.count_params()
        print("✅ Parameter counting completed")
        
        # Print summary
        print(f"\nParameter Summary: {param_info['summary']}")
        
    except Exception as e:
        print(f"❌ Failed to count parameters: {e}")
        return False
    
    # Test model and tokenizer retrieval
    try:
        print("\nTesting model and tokenizer retrieval...")
        model, tokenizer = model_wrapper.get_model_and_tokenizer()
        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Tokenizer type: {type(tokenizer).__name__}")
        print(f"✅ Model device: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ Failed to get model and tokenizer: {e}")
        return False
    
    # Test tokenizer functionality
    try:
        print("\nTesting tokenizer functionality...")
        test_text = "Translate this expert radiology report into layperson terms:\n\nNo infiltrates or consolidations are observed in the study.\n\nLayperson summary:"
        
        # Tokenize
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        print(f"✅ Input tokens: {inputs['input_ids'].shape}")
        
        # Decode
        decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        print(f"✅ Decoded text length: {len(decoded)} characters")
        
    except Exception as e:
        print(f"❌ Failed to test tokenizer: {e}")
        return False
    
    # Test model forward pass (CPU-safe)
    try:
        print("\nTesting model forward pass...")
        model.eval()
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # For T5 models, we need to add labels for forward pass
        # Create dummy labels (same length as input)
        labels = inputs['input_ids'].clone()
        inputs['labels'] = labels
        
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)
            print(f"✅ Forward pass successful")
            print(f"✅ Output logits shape: {outputs.logits.shape}")
            print(f"✅ Loss: {outputs.loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Failed to test forward pass: {e}")
        return False
    
    # Test generation config saving
    try:
        print("\nTesting generation config saving...")
        test_output_dir = Path("./test_output")
        test_output_dir.mkdir(exist_ok=True)
        
        generation_config = model_wrapper.save_generation_config(test_output_dir)
        print("✅ Generation config saved successfully")
        
        # Clean up
        import shutil
        shutil.rmtree(test_output_dir)
        print("✅ Test output cleaned up")
        
    except Exception as e:
        print(f"❌ Failed to test generation config: {e}")
        return False
    
    print("\n🎉 All tests passed! FLAN-T5 LoRA model wrapper is working correctly.")
    return True


def test_standalone_functions():
    """Test standalone utility functions."""
    print("\n" + "=" * 60)
    print("Testing Standalone Functions")
    print("=" * 60)
    
    # Load configuration
    config = load_config('configs/train_flant5_base_lora.yaml')
    
    # Build model
    model_wrapper = build_model_with_lora(config)
    model, tokenizer = model_wrapper.get_model_and_tokenizer()
    
    # Test standalone parameter counting
    try:
        param_string = count_model_parameters(model)
        print(f"✅ Standalone parameter count: {param_string}")
    except Exception as e:
        print(f"❌ Failed standalone parameter count: {e}")
        return False
    
    print("✅ All standalone function tests passed!")
    return True


if __name__ == "__main__":
    success1 = test_model_loading()
    success2 = test_standalone_functions()
    
    if success1 and success2:
        print("\n🚀 All module tests passed! Ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
