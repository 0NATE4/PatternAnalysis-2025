#!/usr/bin/env python3
"""
Test script for ROUGE metrics integration.

This script verifies that the ROUGE metrics computation works correctly
with the training setup, including proper tokenization and metric calculation.

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
from modules import FLANT5LoRAModel
from train import compute_rouge_metrics


def test_rouge_metrics():
    """Test ROUGE metrics computation."""
    print("=" * 60)
    print("Testing ROUGE Metrics Integration")
    print("=" * 60)

    try:
        # 1. Load configuration
        print("Loading configuration...")
        config = load_config("configs/train_flant5_base_lora.yaml")
        setup_reproducibility(config['reproducibility'])
        print("✅ Configuration loaded successfully")

        # 2. Initialize model and tokenizer
        print("\nInitializing model and tokenizer...")
        model_wrapper = FLANT5LoRAModel(config)
        tokenizer = model_wrapper.tokenizer
        print("✅ Model and tokenizer initialized successfully")

        # 3. Set tokenizer for ROUGE computation
        compute_rouge_metrics.tokenizer = tokenizer
        print("✅ Tokenizer set for ROUGE computation")

        # 4. Create sample predictions and references
        print("\nCreating sample predictions and references...")
        
        # Sample expert reports and layperson summaries
        sample_expert_reports = [
            "The patient presents with acute chest pain. Chest X-ray shows consolidation in the right lower lobe.",
            "MRI reveals a 2.5cm mass in the left frontal lobe with surrounding edema."
        ]
        
        sample_layperson_summaries = [
            "The patient has chest pain. An X-ray shows an infection in the right lung.",
            "A brain scan shows a 2.5cm tumor in the left front part of the brain with swelling."
        ]

        # Tokenize the samples with consistent padding
        max_length = 256  # Use consistent max length
        
        tokenized_predictions = []
        tokenized_references = []
        
        for expert, layperson in zip(sample_expert_reports, sample_layperson_summaries):
            # Tokenize expert report (input) - this will be the "prediction" for testing
            expert_tokens = tokenizer.encode(expert, max_length=max_length, truncation=True, padding='max_length')
            
            # Tokenize layperson summary (target) - this will be the "reference"
            layperson_tokens = tokenizer.encode(layperson, max_length=max_length, truncation=True, padding='max_length')
            
            tokenized_predictions.append(expert_tokens)
            tokenized_references.append(layperson_tokens)

        # Convert to numpy arrays (as expected by HuggingFace)
        import numpy as np
        predictions = np.array(tokenized_predictions)
        labels = np.array(tokenized_references)
        
        print(f"✅ Created {len(predictions)} sample predictions")
        print(f"✅ Created {len(labels)} sample references")

        # 5. Test ROUGE metrics computation
        print("\nTesting ROUGE metrics computation...")
        
        eval_preds = (predictions, labels)
        metrics = compute_rouge_metrics(eval_preds)
        
        print("✅ ROUGE metrics computed successfully!")
        print(f"   - rouge1: {metrics['rouge1']:.4f}")
        print(f"   - rouge2: {metrics['rouge2']:.4f}")
        print(f"   - rougeL: {metrics['rougeL']:.4f}")
        print(f"   - rougeLsum: {metrics['rougeLsum']:.4f}")

        # 6. Verify metric values are reasonable
        print("\nVerifying metric values...")
        
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                print(f"❌ {metric_name} is not a number: {type(value)}")
                return False
            if value < 0 or value > 1:
                print(f"❌ {metric_name} value {value:.4f} is outside expected range [0, 1]")
                return False
            print(f"✅ {metric_name}: {value:.4f} (valid range)")

        print("\n🎉 All ROUGE metrics tests passed!")
        return True

    except Exception as e:
        print(f"❌ ROUGE metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rouge_with_dataset():
    """Test ROUGE metrics with actual dataset samples."""
    print("\n" + "=" * 60)
    print("Testing ROUGE Metrics with Dataset Samples")
    print("=" * 60)

    try:
        # 1. Load configuration and dataset
        print("Loading configuration and dataset...")
        config = load_config("configs/train_flant5_base_lora.yaml")
        setup_reproducibility(config['reproducibility'])
        
        dataset_loader = BioLaySummDataset(config)
        val_dataset = dataset_loader.load_data('validation').select(range(5))  # Small sample
        print("✅ Dataset loaded successfully")

        # 2. Initialize model and tokenizer
        print("Initializing model and tokenizer...")
        model_wrapper = FLANT5LoRAModel(config)
        tokenizer = model_wrapper.tokenizer
        compute_rouge_metrics.tokenizer = tokenizer
        print("✅ Model and tokenizer initialized successfully")

        # 3. Create sample predictions (simulate model output)
        print("Creating sample predictions...")
        
        # Get sample from dataset
        sample = val_dataset[0]
        input_text = sample['input_text']
        target_text = sample['target_text']
        
        print(f"Input: {input_text[:100]}...")
        print(f"Target: {target_text[:100]}...")

        # Simulate model prediction (for testing, use a simple truncation)
        predicted_text = target_text[:len(target_text)//2] + "..."
        
        # Tokenize
        pred_tokens = tokenizer.encode(predicted_text, max_length=256, truncation=True, padding=True)
        target_tokens = tokenizer.encode(target_text, max_length=256, truncation=True, padding=True)
        
        # Create eval_preds format
        import numpy as np
        predictions = np.array([pred_tokens])
        labels = np.array([target_tokens])
        eval_preds = (predictions, labels)

        # 4. Compute ROUGE metrics
        print("Computing ROUGE metrics...")
        metrics = compute_rouge_metrics(eval_preds)
        
        print("✅ ROUGE metrics computed successfully!")
        print(f"   - rouge1: {metrics['rouge1']:.4f}")
        print(f"   - rouge2: {metrics['rouge2']:.4f}")
        print(f"   - rougeL: {metrics['rougeL']:.4f}")
        print(f"   - rougeLsum: {metrics['rougeLsum']:.4f}")

        print("\n🎉 Dataset ROUGE metrics test passed!")
        return True

    except Exception as e:
        print(f"❌ Dataset ROUGE metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_rouge_metrics()
    success2 = test_rouge_with_dataset()
    
    if success1 and success2:
        print("\n🚀 All ROUGE metrics tests passed!")
        print("✅ ROUGE integration is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some ROUGE metrics tests failed.")
        sys.exit(1)
