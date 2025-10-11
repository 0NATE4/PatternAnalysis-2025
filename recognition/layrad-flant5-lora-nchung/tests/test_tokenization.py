#!/usr/bin/env python3
"""
Test script for tokenization pipeline.

This script tests the pairwise mapper with truncation and label padding
to ensure the tokenization pipeline works correctly.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import torch
from pathlib import Path

# Add src directory to path (go up one level from tests/ to find src/)
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility
from dataset import BioLaySummDataset
from modules import build_model_with_lora


def test_tokenization_pipeline():
    """Test the tokenization pipeline with truncation and label padding."""
    print("=" * 60)
    print("Testing Tokenization Pipeline")
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
        dataset_loader = BioLaySummDataset(config)
        print("✅ Dataset loader initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize dataset loader: {e}")
        return False
    
    # Build model to get tokenizer
    try:
        model_wrapper = build_model_with_lora(config)
        model, tokenizer = model_wrapper.get_model_and_tokenizer()
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model and tokenizer: {e}")
        return False
    
    # Load a small sample of validation data
    try:
        print("\nLoading validation data sample...")
        val_data = dataset_loader.load_data('validation')
        # Take just a few samples for testing
        test_data = val_data.select(range(5))
        print(f"✅ Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return False
    
    # Test tokenization pipeline
    try:
        print("\nTesting tokenization pipeline...")
        
        # Create a small batch for testing
        test_batch = {
            'input_text': [test_data[i]['input_text'] for i in range(3)],
            'target_text': [test_data[i]['target_text'] for i in range(3)]
        }
        
        # Apply tokenization
        tokenized_batch = dataset_loader.preprocess_function(test_batch, tokenizer)
        
        print("✅ Tokenization completed successfully")
        
        # Check input tokenization
        input_ids = tokenized_batch['input_ids']
        attention_mask = tokenized_batch['attention_mask']
        labels = tokenized_batch['labels']
        
        print(f"✅ Input shape: {input_ids.shape}")
        print(f"✅ Attention mask shape: {attention_mask.shape}")
        print(f"✅ Labels shape: {labels.shape}")
        
        # Verify truncation lengths
        max_source_length = config['dataset']['max_source_length']
        max_target_length = config['dataset']['max_target_length']
        
        assert input_ids.shape[1] == max_source_length, f"Input length mismatch: {input_ids.shape[1]} != {max_source_length}"
        assert labels.shape[1] == max_target_length, f"Label length mismatch: {labels.shape[1]} != {max_target_length}"
        
        print(f"✅ Truncation verified: inputs to {max_source_length}, targets to {max_target_length}")
        
    except Exception as e:
        print(f"❌ Failed to test tokenization: {e}")
        return False
    
    # Test label padding with -100
    try:
        print("\nTesting label padding with -100...")
        
        # Check that padding tokens are replaced with -100
        pad_token_id = tokenizer.pad_token_id
        num_pad_tokens = (labels == pad_token_id).sum().item()
        num_minus_100 = (labels == -100).sum().item()
        
        print(f"✅ Pad token ID: {pad_token_id}")
        print(f"✅ Number of -100 tokens in labels: {num_minus_100}")
        print(f"✅ Number of pad tokens in labels: {num_pad_tokens}")
        
        # Verify no pad tokens remain in labels
        assert num_pad_tokens == 0, f"Found {num_pad_tokens} pad tokens in labels, should be 0"
        assert num_minus_100 > 0, "No -100 tokens found in labels"
        
        print("✅ Label padding with -100 verified successfully")
        
    except Exception as e:
        print(f"❌ Failed to test label padding: {e}")
        return False
    
    # Test DataLoader creation
    try:
        print("\nTesting DataLoader creation...")
        
        # Create DataLoader
        dataloader = dataset_loader.get_loader(test_data, tokenizer, batch_size=2)
        
        print(f"✅ DataLoader created successfully")
        print(f"✅ DataLoader length: {len(dataloader)}")
        
        # Test one batch
        batch = next(iter(dataloader))
        
        print(f"✅ Batch keys: {list(batch.keys())}")
        print(f"✅ Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"✅ Batch labels shape: {batch['labels'].shape}")
        
        # Verify batch structure
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch
        
        print("✅ DataLoader batch structure verified")
        
    except Exception as e:
        print(f"❌ Failed to test DataLoader: {e}")
        return False
    
    # Test model forward pass with tokenized data
    try:
        print("\nTesting model forward pass with tokenized data...")
        
        model.eval()
        
        # Move batch to model device
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            
        print(f"✅ Forward pass successful")
        print(f"✅ Output logits shape: {outputs.logits.shape}")
        print(f"✅ Loss: {outputs.loss.item():.4f}")
        
        # Verify loss is reasonable (not NaN or infinite)
        assert not torch.isnan(outputs.loss), "Loss is NaN"
        assert not torch.isinf(outputs.loss), "Loss is infinite"
        assert outputs.loss.item() > 0, "Loss should be positive"
        
        print("✅ Loss validation passed")
        
    except Exception as e:
        print(f"❌ Failed to test model forward pass: {e}")
        return False
    
    print("\n🎉 All tokenization pipeline tests passed!")
    return True


def test_edge_cases():
    """Test edge cases in tokenization."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    # Load configuration
    config = load_config('configs/train_flant5_base_lora.yaml')
    dataset_loader = BioLaySummDataset(config)
    model_wrapper = build_model_with_lora(config)
    model, tokenizer = model_wrapper.get_model_and_tokenizer()
    
    # Test with very long text
    try:
        print("\nTesting with very long text...")
        
        long_text = "This is a very long radiology report. " * 100  # Very long text
        short_text = "Short summary."
        
        test_batch = {
            'input_text': [long_text],
            'target_text': [short_text]
        }
        
        tokenized = dataset_loader.preprocess_function(test_batch, tokenizer)
        
        # Should be truncated to max lengths
        assert tokenized['input_ids'].shape[1] == config['dataset']['max_source_length']
        assert tokenized['labels'].shape[1] == config['dataset']['max_target_length']
        
        print("✅ Long text truncation works correctly")
        
    except Exception as e:
        print(f"❌ Failed to test long text: {e}")
        return False
    
    # Test with empty text
    try:
        print("\nTesting with empty text...")
        
        test_batch = {
            'input_text': [""],
            'target_text': [""]
        }
        
        tokenized = dataset_loader.preprocess_function(test_batch, tokenizer)
        
        # Should still produce valid tensors
        assert tokenized['input_ids'].shape[1] == config['dataset']['max_source_length']
        assert tokenized['labels'].shape[1] == config['dataset']['max_target_length']
        
        print("✅ Empty text handling works correctly")
        
    except Exception as e:
        print(f"❌ Failed to test empty text: {e}")
        return False
    
    print("✅ All edge case tests passed!")
    return True


if __name__ == "__main__":
    success1 = test_tokenization_pipeline()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🚀 All tokenization tests passed! Pipeline is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
