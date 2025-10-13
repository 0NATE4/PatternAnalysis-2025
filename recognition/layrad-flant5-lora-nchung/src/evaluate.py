"""
Evaluation script for FLAN-T5 LoRA model on BioLaySumm test set.

This module implements comprehensive evaluation of the trained model on held-out
test data, computing ROUGE metrics and generating detailed reports in JSON and CSV formats.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import json
import csv
import time
import torch
import evaluate
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)
from datasets import Dataset
from peft import PeftModel

from .utils import (
    load_config, setup_reproducibility, get_device, 
    create_reports_dir, log_training_arguments
)
from .dataset import BioLaySummDataset
from .modules import FLANT5LoRAModel


class BioLaySummEvaluator:
    """
    Evaluation wrapper for FLAN-T5 LoRA model on BioLaySumm test set.
    
    This class provides comprehensive evaluation capabilities including:
    - Model loading and inference
    - ROUGE metrics computation
    - Detailed per-sample analysis
    - JSON and CSV report generation
    
    Attributes:
        config (dict): Configuration dictionary
        model_wrapper: FLAN-T5 LoRA model wrapper
        dataset_loader: BioLaySumm dataset loader
        reports_dir (Path): Reports directory for output
        device: Device for computation (CPU/GPU)
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        """
        Initialize the BioLaySumm evaluator.
        
        Args:
            config (dict): Configuration dictionary
            model_path (str): Path to the trained model directory
        """
        self.config = config
        self.model_path = Path(model_path)
        
        # Setup reproducibility
        setup_reproducibility(self.config)
        
        # Get device
        self.device = get_device(self.config)
        
        # Create reports directory
        self.reports_dir = create_reports_dir(self.model_path)
        
        print(f"Evaluation setup complete. Model path: {self.model_path}")
        print(f"Reports directory: {self.reports_dir}")
        
    def load_model_and_tokenizer(self) -> None:
        """
        Load the trained model and tokenizer.
        """
        print("\nLoading trained model and tokenizer...")
        
        # Load the base model and tokenizer
        base_model_name = self.config.get('model', {}).get('name', 'google/flan-t5-base')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load the base model
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.bfloat16,
            device_map="auto" if self.device.type == 'cuda' else None
        )
        
        # Load the LoRA adapter
        if self.model_path.exists():
            self.model = PeftModel.from_pretrained(self.base_model, str(self.model_path))
            print(f"✅ LoRA adapter loaded from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        # Move to device if not using device_map
        if self.device.type == 'cpu':
            self.model = self.model.to(self.device)
        
        # Load generation config if available
        generation_config_path = self.model_path / 'generation_config.json'
        if generation_config_path.exists():
            with open(generation_config_path, 'r') as f:
                gen_config_dict = json.load(f)
            self.generation_config = GenerationConfig(**gen_config_dict)
            print(f"✅ Generation config loaded from: {generation_config_path}")
        else:
            # Use default generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=200,
                num_beams=4,
                length_penalty=0.6,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            print("✅ Using default generation config")
        
        print("✅ Model and tokenizer loaded successfully")
        
    def load_test_dataset(self) -> None:
        """
        Load the test dataset for evaluation.
        """
        print("\nLoading test dataset...")
        
        # Initialize dataset loader
        self.dataset_loader = BioLaySummDataset(self.config)
        
        # Load test dataset
        self.test_dataset = self.dataset_loader.load_data('test')
        
        print(f"✅ Test dataset loaded: {len(self.test_dataset)} samples")
        
        # Show sample
        if len(self.test_dataset) > 0:
            sample = self.test_dataset[0]
            print(f"Sample input: {sample['input_text'][:100]}...")
            print(f"Sample target: {sample['target_text'][:100]}...")
        
    def generate_predictions(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Generate predictions on the test dataset.
        
        Args:
            max_samples (int, optional): Maximum number of samples to evaluate
            
        Returns:
            List[Dict]: List of predictions with input, target, and generated text
        """
        print(f"\nGenerating predictions on test set...")
        
        # Limit samples if specified
        eval_dataset = self.test_dataset
        if max_samples is not None:
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
        
        print(f"Evaluating on {len(eval_dataset)} samples")
        
        # Prepare model for inference
        self.model.eval()
        
        predictions = []
        start_time = time.time()
        
        with torch.no_grad():
            for i, sample in enumerate(eval_dataset):
                if i % 100 == 0:
                    print(f"Processing sample {i+1}/{len(eval_dataset)}")
                
                # Tokenize input
                input_text = sample['input_text']
                target_text = sample['target_text']
                
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.config.get('dataset', {}).get('max_source_length', 512),
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate prediction
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                # Decode prediction
                generated_text = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                # Store prediction
                pred_data = {
                    'sample_id': i,
                    'input_text': input_text,
                    'target_text': target_text,
                    'generated_text': generated_text,
                    'input_length': len(input_text.split()),
                    'target_length': len(target_text.split()),
                    'generated_length': len(generated_text.split()),
                }
                predictions.append(pred_data)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"✅ Generated {len(predictions)} predictions in {generation_time:.2f} seconds")
        print(f"Average time per sample: {generation_time/len(predictions):.3f} seconds")
        
        return predictions
    
    def compute_rouge_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute ROUGE metrics on the predictions.
        
        Args:
            predictions (List[Dict]): List of predictions
            
        Returns:
            Dict[str, float]: ROUGE metrics
        """
        print("\nComputing ROUGE metrics...")
        
        # Extract texts
        generated_texts = [pred['generated_text'] for pred in predictions]
        target_texts = [pred['target_text'] for pred in predictions]
        
        # Load ROUGE metric
        rouge = evaluate.load('rouge')
        
        # Compute metrics
        rouge_results = rouge.compute(
            predictions=generated_texts,
            references=target_texts,
            use_aggregator=True,
            use_stemmer=True
        )
        
        # Extract individual scores
        metrics = {
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL'],
            'rougeLsum': rouge_results['rougeLsum'],
            'num_samples': len(predictions),
        }
        
        print("✅ ROUGE metrics computed:")
        print(f"   - ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"   - ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"   - ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"   - ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
        
        return metrics
    
    def save_rouge_summary(self, metrics: Dict[str, float]) -> None:
        """
        Save ROUGE metrics summary to JSON.
        
        Args:
            metrics (Dict[str, float]): ROUGE metrics
        """
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': str(self.model_path),
            'dataset': self.config.get('dataset', {}).get('name', 'unknown'),
            'num_samples': metrics.get('num_samples', 0),
            'rouge_metrics': {
                'rouge1': metrics['rouge1'],
                'rouge2': metrics['rouge2'],
                'rougeL': metrics['rougeL'],
                'rougeLsum': metrics['rougeLsum'],
            },
            'generation_config': {
                'max_new_tokens': self.generation_config.max_new_tokens,
                'num_beams': self.generation_config.num_beams,
                'length_penalty': self.generation_config.length_penalty,
                'no_repeat_ngram_size': self.generation_config.no_repeat_ngram_size,
                'early_stopping': self.generation_config.early_stopping,
                'do_sample': self.generation_config.do_sample,
            },
            'model_config': {
                'base_model': self.config.get('model', {}).get('name', 'unknown'),
                'lora_config': self.config.get('lora', {}),
            }
        }
        
        # Save to JSON
        summary_path = self.reports_dir / 'rouge_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ ROUGE summary saved to: {summary_path}")
    
    def save_per_sample_results(self, predictions: List[Dict[str, Any]], metrics: Dict[str, float]) -> None:
        """
        Save per-sample results to CSV.
        
        Args:
            predictions (List[Dict]): List of predictions
            metrics (Dict[str, float]): ROUGE metrics
        """
        csv_path = self.reports_dir / 'rouge_per_sample.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'sample_id', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
                'input_length', 'target_length', 'generated_length',
                'input_text', 'target_text', 'generated_text'
            ])
            
            # Compute per-sample ROUGE scores
            rouge = evaluate.load('rouge')
            
            for pred in predictions:
                # Compute ROUGE for this sample
                sample_rouge = rouge.compute(
                    predictions=[pred['generated_text']],
                    references=[pred['target_text']],
                    use_aggregator=True,
                    use_stemmer=True
                )
                
                # Write row
                writer.writerow([
                    pred['sample_id'],
                    sample_rouge['rouge1'],
                    sample_rouge['rouge2'],
                    sample_rouge['rougeL'],
                    sample_rouge['rougeLsum'],
                    pred['input_length'],
                    pred['target_length'],
                    pred['generated_length'],
                    pred['input_text'],
                    pred['target_text'],
                    pred['generated_text']
                ])
        
        print(f"✅ Per-sample results saved to: {csv_path}")
    
    def save_generation_config(self) -> None:
        """
        Save the generation configuration used for evaluation.
        """
        config_path = self.reports_dir / 'generation_config.json'
        
        gen_config_dict = {
            'max_new_tokens': self.generation_config.max_new_tokens,
            'num_beams': self.generation_config.num_beams,
            'length_penalty': self.generation_config.length_penalty,
            'no_repeat_ngram_size': self.generation_config.no_repeat_ngram_size,
            'early_stopping': self.generation_config.early_stopping,
            'do_sample': self.generation_config.do_sample,
            'pad_token_id': self.generation_config.pad_token_id,
            'eos_token_id': self.generation_config.eos_token_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(gen_config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generation config saved to: {config_path}")
    
    def evaluate(self, max_samples: int = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on the test set.
        
        Args:
            max_samples (int, optional): Maximum number of samples to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        print("\n" + "="*60)
        print("STARTING EVALUATION")
        print("="*60)
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_test_dataset()
        
        # Generate predictions
        predictions = self.generate_predictions(max_samples=max_samples)
        
        # Compute metrics
        metrics = self.compute_rouge_metrics(predictions)
        
        # Save results
        self.save_rouge_summary(metrics)
        self.save_per_sample_results(predictions, metrics)
        self.save_generation_config()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.reports_dir}")
        print(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'reports_dir': self.reports_dir
        }


def main():
    """
    Main evaluation function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate FLAN-T5 LoRA model on BioLaySumm test set')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model directory')
    parser.add_argument('--config', type=str, default='configs/train_flant5_base_lora.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create evaluator and run evaluation
    evaluator = BioLaySummEvaluator(config, args.model_path)
    results = evaluator.evaluate(max_samples=args.max_samples)
    
    return results


if __name__ == "__main__":
    main()
