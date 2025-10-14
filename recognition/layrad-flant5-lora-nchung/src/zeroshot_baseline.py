"""
Zero-shot baseline evaluation for FLAN-T5 on BioLaySumm dataset.

This module implements a zero-shot baseline using the untrained FLAN-T5 model
to establish a performance baseline before fine-tuning. It uses the same
prompting strategy as the training data but without any fine-tuning.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import json
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

from utils import (
    load_config, setup_reproducibility, get_device, 
    create_reports_dir
)
from dataset import BioLaySummDataset


class ZeroShotBaseline:
    """
    Zero-shot baseline evaluator for FLAN-T5 on BioLaySumm dataset.
    
    This class provides zero-shot evaluation capabilities including:
    - Untrained model loading and inference
    - Same prompting as training data
    - ROUGE metrics computation
    - Baseline performance reporting
    
    Attributes:
        config (dict): Configuration dictionary
        model: Untrained FLAN-T5 model
        tokenizer: Tokenizer for the model
        reports_dir (Path): Reports directory for output
        device: Device for computation (CPU/GPU)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the zero-shot baseline evaluator.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Setup reproducibility
        setup_reproducibility(self.config)
        
        # Get device
        self.device = get_device(self.config)
        
        # Create reports directory
        output_dir = Path("./checkpoints/zeroshot_baseline")
        self.reports_dir = create_reports_dir(output_dir)
        
        print(f"Zero-shot baseline setup complete.")
        print(f"Reports directory: {self.reports_dir}")
        
    def load_untrained_model(self) -> None:
        """
        Load the untrained FLAN-T5 model (no LoRA, no fine-tuning).
        """
        print("\nLoading untrained FLAN-T5 model...")
        
        # Load the base model and tokenizer (no LoRA, no fine-tuning)
        base_model_name = self.config.get('model', {}).get('name', 'google/flan-t5-base')
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print(f"✅ Tokenizer loaded: {base_model_name}")
        
        # Load the base model without any adapters
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            dtype=torch.float32 if self.device.type == 'cpu' else torch.bfloat16,
            device_map="auto" if self.device.type == 'cuda' else None
        )
        
        # Move to device if not using device_map
        if self.device.type == 'cpu':
            self.model = self.model.to(self.device)
        
        print(f"✅ Untrained model loaded: {base_model_name}")
        print("⚠️  Note: This is the base model with NO fine-tuning or LoRA adapters")
        
        # Use generation config similar to training
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            num_beams=4,
            length_penalty=0.6,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("✅ Generation config configured")
        
    def load_test_dataset(self) -> None:
        """
        Load the test dataset for zero-shot evaluation.
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
        
    def generate_zeroshot_predictions(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Generate zero-shot predictions on the test dataset.
        
        Args:
            max_samples (int, optional): Maximum number of samples to evaluate
            
        Returns:
            List[Dict]: List of predictions with input, target, and generated text
        """
        print(f"\nGenerating zero-shot predictions on test set...")
        
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
                
                # Use the same prompting as training data
                input_text = sample['input_text']  # Already has the prompt
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
        
        print(f"✅ Generated {len(predictions)} zero-shot predictions in {generation_time:.2f} seconds")
        print(f"Average time per sample: {generation_time/len(predictions):.3f} seconds")
        
        return predictions
    
    def compute_rouge_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute ROUGE metrics on the zero-shot predictions.
        
        Args:
            predictions (List[Dict]): List of predictions
            
        Returns:
            Dict[str, float]: ROUGE metrics
        """
        print("\nComputing ROUGE metrics for zero-shot baseline...")
        
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
        
        print("✅ Zero-shot ROUGE metrics computed:")
        print(f"   - ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"   - ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"   - ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"   - ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
        
        return metrics
    
    def save_zeroshot_results(self, metrics: Dict[str, float], predictions: List[Dict[str, Any]]) -> None:
        """
        Save zero-shot baseline results to JSON.
        
        Args:
            metrics (Dict[str, float]): ROUGE metrics
            predictions (List[Dict]): List of predictions
        """
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'baseline_type': 'zero_shot',
            'model_name': self.config.get('model', {}).get('name', 'google/flan-t5-base'),
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
                'fine_tuning': 'none',  # No fine-tuning for zero-shot
                'lora_adapters': 'none',  # No LoRA for zero-shot
            },
            'sample_predictions': predictions[:5]  # Include first 5 predictions as examples
        }
        
        # Save to JSON
        results_path = self.reports_dir / 'zeroshot_baseline_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Zero-shot baseline results saved to: {results_path}")
    
    def print_baseline_summary(self, metrics: Dict[str, float]) -> None:
        """
        Print a summary of the zero-shot baseline performance.
        
        Args:
            metrics (Dict[str, float]): ROUGE metrics
        """
        print("\n" + "="*60)
        print("ZERO-SHOT BASELINE PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Model: {self.config.get('model', {}).get('name', 'google/flan-t5-base')}")
        print(f"Fine-tuning: None (zero-shot)")
        print(f"LoRA adapters: None")
        print(f"Dataset: {self.config.get('dataset', {}).get('name', 'unknown')}")
        print(f"Samples evaluated: {metrics.get('num_samples', 0)}")
        print("\nROUGE Metrics:")
        print(f"  ROUGE-1:  {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:  {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:  {metrics['rougeL']:.4f}")
        print(f"  ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
        print("\nThis represents the baseline performance before any fine-tuning.")
        print("Compare these scores with your fine-tuned model results.")
        print("="*60)
    
    def evaluate_zeroshot(self, max_samples: int = None) -> Dict[str, Any]:
        """
        Run comprehensive zero-shot evaluation.
        
        Args:
            max_samples (int, optional): Maximum number of samples to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        print("\n" + "="*60)
        print("STARTING ZERO-SHOT BASELINE EVALUATION")
        print("="*60)
        
        # Load model and dataset
        self.load_untrained_model()
        self.load_test_dataset()
        
        # Generate predictions
        predictions = self.generate_zeroshot_predictions(max_samples=max_samples)
        
        # Compute metrics
        metrics = self.compute_rouge_metrics(predictions)
        
        # Save results
        self.save_zeroshot_results(metrics, predictions)
        
        # Print summary
        self.print_baseline_summary(metrics)
        
        print(f"\n✅ Zero-shot baseline evaluation complete!")
        print(f"Results saved to: {self.reports_dir}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'reports_dir': self.reports_dir
        }


def main():
    """
    Main zero-shot baseline evaluation function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run zero-shot baseline evaluation on BioLaySumm test set')
    parser.add_argument('--config', type=str, default='configs/train_flant5_base_lora.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create evaluator and run evaluation
    evaluator = ZeroShotBaseline(config)
    results = evaluator.evaluate_zeroshot(max_samples=args.max_samples)
    
    return results


if __name__ == "__main__":
    main()
