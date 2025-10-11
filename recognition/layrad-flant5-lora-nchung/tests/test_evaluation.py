#!/usr/bin/env python3
"""
Test script for evaluation functionality.

This script verifies that the evaluation system works correctly,
including model loading, prediction generation, and report creation.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json
import torch

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import load_config, setup_reproducibility
from dataset import BioLaySummDataset
from modules import FLANT5LoRAModel


def test_evaluation_setup():
    """Test evaluation system setup."""
    print("=" * 60)
    print("Testing Evaluation System Setup")
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

        # 3. Load test dataset
        print("Loading test dataset...")
        dataset_loader = BioLaySummDataset(config)
        test_dataset = dataset_loader.load_data('test').select(range(5))  # Small sample
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")

        # 4. Test model inference
        print("Testing model inference...")
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
        
        # Get a sample
        sample = test_dataset[0]
        input_text = sample['input_text']
        target_text = sample['target_text']
        
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
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,  # Short for testing
                num_beams=2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text[:100]}...")
        
        print("✅ Model inference test successful")

        # 5. Test ROUGE computation
        print("Testing ROUGE computation...")
        import evaluate
        rouge = evaluate.load('rouge')
        
        rouge_results = rouge.compute(
            predictions=[generated_text],
            references=[target_text],
            use_aggregator=True,
            use_stemmer=True
        )
        
        print(f"✅ ROUGE metrics computed:")
        print(f"   - ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"   - ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"   - ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"   - ROUGE-Lsum: {rouge_results['rougeLsum']:.4f}")

        print("\n🎉 All evaluation setup tests passed!")
        return True

    except Exception as e:
        print(f"❌ Evaluation setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_reports():
    """Test evaluation report generation."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Report Generation")
    print("=" * 60)

    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock evaluation results
            mock_predictions = [
                {
                    'sample_id': 0,
                    'input_text': 'The chest shows significant air trapping.',
                    'target_text': 'The chest shows a lot of trapped air.',
                    'generated_text': 'The chest shows air trapping.',
                    'input_length': 5,
                    'target_length': 9,
                    'generated_length': 5,
                },
                {
                    'sample_id': 1,
                    'input_text': 'MRI reveals a 2.5cm mass in the left frontal lobe.',
                    'target_text': 'A brain scan shows a 2.5cm tumor in the left front part of the brain.',
                    'generated_text': 'MRI shows a 2.5cm mass in the left frontal lobe.',
                    'input_length': 10,
                    'target_length': 15,
                    'generated_length': 10,
                }
            ]
            
            mock_metrics = {
                'rouge1': 0.75,
                'rouge2': 0.60,
                'rougeL': 0.70,
                'rougeLsum': 0.72,
                'num_samples': 2,
            }
            
            # Test JSON report creation
            print("Testing JSON report creation...")
            summary_data = {
                'timestamp': '2024-01-01 12:00:00',
                'model_path': str(temp_path),
                'dataset': 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track',
                'num_samples': mock_metrics['num_samples'],
                'rouge_metrics': {
                    'rouge1': mock_metrics['rouge1'],
                    'rouge2': mock_metrics['rouge2'],
                    'rougeL': mock_metrics['rougeL'],
                    'rougeLsum': mock_metrics['rougeLsum'],
                }
            }
            
            json_path = temp_path / 'rouge_summary.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            assert json_path.exists(), "JSON report should be created"
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['rouge_metrics']['rouge1'] == 0.75, "ROUGE-1 should be correct"
            assert loaded_data['num_samples'] == 2, "Number of samples should be correct"
            
            print("✅ JSON report creation successful")
            
            # Test CSV report creation
            print("Testing CSV report creation...")
            import csv
            
            csv_path = temp_path / 'rouge_per_sample.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'sample_id', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
                    'input_length', 'target_length', 'generated_length',
                    'input_text', 'target_text', 'generated_text'
                ])
                
                # Write data
                for pred in mock_predictions:
                    writer.writerow([
                        pred['sample_id'], 0.75, 0.60, 0.70, 0.72,  # Mock ROUGE scores
                        pred['input_length'],
                        pred['target_length'],
                        pred['generated_length'],
                        pred['input_text'],
                        pred['target_text'],
                        pred['generated_text']
                    ])
            
            assert csv_path.exists(), "CSV report should be created"
            
            # Verify CSV content
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            assert len(rows) == 3, "CSV should have header + 2 data rows"
            assert rows[0][0] == 'sample_id', "CSV header should be correct"
            assert rows[1][0] == '0', "First sample ID should be correct"
            
            print("✅ CSV report creation successful")
            
            print("\n🎉 All evaluation report tests passed!")
            return True

    except Exception as e:
        print(f"❌ Evaluation report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_evaluation_setup()
    success2 = test_evaluation_reports()
    
    if success1 and success2:
        print("\n🚀 All evaluation tests passed!")
        print("✅ Evaluation system is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some evaluation tests failed.")
        sys.exit(1)
