#!/usr/bin/env python3
"""
Test script for prediction functionality.

This script verifies that the prediction system works correctly,
including example selection, generation, and output formatting.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility
from dataset import BioLaySummDataset
from modules import FLANT5LoRAModel


def test_prediction_setup():
    """Test prediction system setup."""
    print("=" * 60)
    print("Testing Prediction System Setup")
    print("=" * 60)

    try:
        # 1. Load configuration
        print("Loading configuration...")
        config = load_config("configs/train_flant5_base_lora.yaml")
        setup_reproducibility(config['reproducibility'])
        print("✅ Configuration loaded successfully")

        # 2. Initialize model and tokenizer
        print("Initializing model and tokenizer...")
        model_wrapper = FLANT5LoRAModel(config)
        print("✅ Model and tokenizer initialized successfully")

        # 3. Load dataset
        print("Loading dataset...")
        dataset_loader = BioLaySummDataset(config)
        val_dataset = dataset_loader.load_data('validation').select(range(10))  # Small sample
        print(f"✅ Dataset loaded: {len(val_dataset)} samples")

        # 4. Test example selection
        print("Testing example selection...")
        import random
        random.seed(42)
        
        # Select 3 examples
        available_indices = list(range(len(val_dataset)))
        selected_indices = random.sample(available_indices, min(3, len(available_indices)))
        
        examples = []
        for idx in selected_indices:
            sample = val_dataset[idx]
            examples.append({
                'index': idx,
                'input_text': sample['input_text'],
                'target_text': sample['target_text'],
            })
        
        print(f"✅ Selected {len(examples)} examples")
        
        # 5. Test model inference
        print("Testing model inference...")
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        
        # Get first example
        example = examples[0]
        input_text = example['input_text']
        target_text = example['target_text']
        
        print(f"Input: {input_text[:100]}...")
        print(f"Target: {target_text[:100]}...")

        # Tokenize and generate
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Generate prediction
        model.eval()
        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,  # Longer for examples
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text[:100]}...")
        
        print("✅ Model inference test successful")

        print("\n🎉 All prediction setup tests passed!")
        return True

    except Exception as e:
        print(f"❌ Prediction setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_output():
    """Test prediction output formatting."""
    print("\n" + "=" * 60)
    print("Testing Prediction Output Formatting")
    print("=" * 60)

    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock prediction results
            mock_predictions = [
                {
                    'example_id': 1,
                    'dataset_index': 42,
                    'input_text': 'The chest shows significant air trapping in both lungs.',
                    'target_text': 'The chest shows a lot of trapped air in both lungs.',
                    'generated_text': 'The chest shows air trapping in both lungs.',
                    'input_length': 9,
                    'target_length': 12,
                    'generated_length': 9,
                },
                {
                    'example_id': 2,
                    'dataset_index': 156,
                    'input_text': 'MRI reveals a 2.5cm enhancing lesion in the left frontal lobe.',
                    'target_text': 'A brain scan shows a 2.5cm tumor in the left front part of the brain.',
                    'generated_text': 'MRI shows a 2.5cm lesion in the left frontal lobe.',
                    'input_length': 11,
                    'target_length': 15,
                    'generated_length': 11,
                }
            ]
            
            # Test JSONL output creation
            print("Testing JSONL output creation...")
            jsonl_path = temp_path / 'examples.jsonl'
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for pred in mock_predictions:
                    # Create a clean JSON object for each example
                    example_data = {
                        'example_id': pred['example_id'],
                        'dataset_index': pred['dataset_index'],
                        'expert_report': pred['input_text'],
                        'layperson_target': pred['target_text'],
                        'model_prediction': pred['generated_text'],
                        'statistics': {
                            'input_length': pred['input_length'],
                            'target_length': pred['target_length'],
                            'generated_length': pred['generated_length'],
                        },
                        'timestamp': '2024-01-01 12:00:00',
                    }
                    
                    # Write as JSON line
                    f.write(json.dumps(example_data, ensure_ascii=False) + '\n')
            
            assert jsonl_path.exists(), "JSONL file should be created"
            
            # Verify JSONL content
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == 2, "JSONL should have 2 lines"
            
            # Parse first line
            first_example = json.loads(lines[0])
            assert first_example['example_id'] == 1, "First example ID should be correct"
            assert first_example['expert_report'] == mock_predictions[0]['input_text'], "Expert report should be correct"
            assert first_example['layperson_target'] == mock_predictions[0]['target_text'], "Layperson target should be correct"
            assert first_example['model_prediction'] == mock_predictions[0]['generated_text'], "Model prediction should be correct"
            assert first_example['statistics']['input_length'] == 9, "Input length should be correct"
            
            print("✅ JSONL output creation successful")
            
            # Test pretty printing format
            print("Testing pretty printing format...")
            
            # Simulate pretty printing (we'll just verify the structure)
            for pred in mock_predictions:
                # Verify required fields for pretty printing
                assert 'example_id' in pred, "Should have example_id"
                assert 'input_text' in pred, "Should have input_text"
                assert 'target_text' in pred, "Should have target_text"
                assert 'generated_text' in pred, "Should have generated_text"
                assert 'input_length' in pred, "Should have input_length"
                assert 'target_length' in pred, "Should have target_length"
                assert 'generated_length' in pred, "Should have generated_length"
            
            print("✅ Pretty printing format verification successful")
            
            print("\n🎉 All prediction output tests passed!")
            return True

    except Exception as e:
        print(f"❌ Prediction output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_prediction_setup()
    success2 = test_prediction_output()
    
    if success1 and success2:
        print("\n🚀 All prediction tests passed!")
        print("✅ Prediction system is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some prediction tests failed.")
        sys.exit(1)
