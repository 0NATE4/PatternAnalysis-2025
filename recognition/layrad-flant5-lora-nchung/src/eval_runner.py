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

# Handle imports for both direct execution and module import
try:
    from .utils import (
        load_config, setup_reproducibility, get_device, 
        create_reports_dir, log_training_arguments
    )
    from .dataset import BioLaySummDataset
    from .modules import FLANT5LoRAModel
except ImportError:
    # Direct execution - add current directory to path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import (
        load_config, setup_reproducibility, get_device, 
        create_reports_dir, log_training_arguments
    )
    from dataset import BioLaySummDataset
    from modules import FLANT5LoRAModel


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
        model_path (Path): Path to trained model directory
        reports_dir (Path): Output reports directory
        device: Torch device
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        self.config = config
        # Resolve model path to final_model if it exists
        self.model_path = self._resolve_model_path(Path(model_path))
        
        setup_reproducibility(self.config)
        self.device = get_device(self.config)
        self.reports_dir = create_reports_dir(self.model_path)
        
        print(f"Evaluation setup complete. Model path: {self.model_path}")
        print(f"Reports directory: {self.reports_dir}")
        
    def _resolve_model_path(self, model_path: Path) -> Path:
        """
        Resolve model path, preferring final_model/ subdirectory if it exists.
        
        Training saves to output_dir/final_model/, but config points to output_dir.
        This method auto-detects the correct path.
        """
        # If path doesn't exist, try final_model subdirectory
        if not model_path.exists():
            final_model_path = model_path / 'final_model'
            if final_model_path.exists():
                print(f"✅ Resolved model path: {model_path} → {final_model_path}")
                return final_model_path
        
        # If path exists but doesn't have model files, check final_model
        if model_path.exists():
            has_lora = (model_path / 'adapter_config.json').exists()
            has_full = (model_path / 'model.safetensors').exists() or (model_path / 'pytorch_model.bin').exists()
            
            if not has_lora and not has_full:
                final_model_path = model_path / 'final_model'
                if final_model_path.exists():
                    print(f"✅ Model files found in subdirectory: {final_model_path}")
                    return final_model_path
        
        return model_path
        
    def load_model_and_tokenizer(self) -> None:
        print("\nLoading trained model and tokenizer...")
        
        # Detect training strategy from config
        strategy = self.config.get('training', {}).get('strategy', 'lora')
        base_model_name = self.config.get('model', {}).get('name', 'google/flan-t5-base')
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        if strategy == 'full':
            # Load full fine-tuned model directly
            print("Loading full fine-tuned model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                dtype=torch.float32 if self.device.type == 'cpu' else torch.bfloat16,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            print(f"✅ Full fine-tuned model loaded from: {self.model_path}")
        else:
            # Load LoRA adapter (existing code)
            print("Loading LoRA adapter...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                dtype=torch.float32 if self.device.type == 'cpu' else torch.bfloat16,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            self.model = PeftModel.from_pretrained(self.base_model, str(self.model_path))
            print(f"✅ LoRA adapter loaded from: {self.model_path}")
        
        if self.device.type == 'cpu':
            self.model = self.model.to(self.device)
        generation_config_path = self.model_path / 'generation_config.json'
        if generation_config_path.exists():
            with open(generation_config_path, 'r') as f:
                gen_config_dict = json.load(f)
            self.generation_config = GenerationConfig(**gen_config_dict)
            print(f"✅ Generation config loaded from: {generation_config_path}")
        else:
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
        print("\nLoading test dataset...")
        self.dataset_loader = BioLaySummDataset(self.config)
        self.test_dataset = self.dataset_loader.load_data('test')
        print(f"✅ Test dataset loaded: {len(self.test_dataset)} samples")
        if len(self.test_dataset) > 0:
            sample = self.test_dataset[0]
            print(f"Sample input: {sample['input_text'][:100]}...")
            print(f"Sample target: {sample['target_text'][:100]}...")
        
    def generate_predictions(self, max_samples: int = None) -> List[Dict[str, Any]]:
        print(f"\nGenerating predictions on test set...")
        eval_dataset = self.test_dataset
        if max_samples is not None:
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
        print(f"Evaluating on {len(eval_dataset)} samples")
        self.model.eval()
        predictions = []
        start_time = time.time()
        with torch.no_grad():
            for i, sample in enumerate(eval_dataset):
                if i % 100 == 0:
                    print(f"Processing sample {i+1}/{len(eval_dataset)}")
                input_text = sample['input_text']
                target_text = sample['target_text']
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.config.get('dataset', {}).get('max_source_length', 512),
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        print("\nComputing ROUGE metrics...")
        generated_texts = [pred['generated_text'] for pred in predictions]
        target_texts = [pred['target_text'] for pred in predictions]
        rouge = evaluate.load('rouge')
        rouge_results = rouge.compute(
            predictions=generated_texts,
            references=target_texts,
            use_aggregator=True,
            use_stemmer=True
        )
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
        summary_path = self.reports_dir / 'rouge_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"✅ ROUGE summary saved to: {summary_path}")
    
    def save_per_sample_results(self, predictions: List[Dict[str, Any]], metrics: Dict[str, float]) -> None:
        csv_path = self.reports_dir / 'rouge_per_sample.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum',
                'input_length', 'target_length', 'generated_length',
                'input_text', 'target_text', 'generated_text'
            ])
            rouge = evaluate.load('rouge')
            for pred in predictions:
                sample_rouge = rouge.compute(
                    predictions=[pred['generated_text']],
                    references=[pred['target_text']],
                    use_aggregator=True,
                    use_stemmer=True
                )
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
        print("\n" + "="*60)
        print("STARTING EVALUATION")
        print("="*60)
        self.load_model_and_tokenizer()
        self.load_test_dataset()
        predictions = self.generate_predictions(max_samples=max_samples)
        metrics = self.compute_rouge_metrics(predictions)
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
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/train_flant5_base_lora.yaml'
    config = load_config(config_file)
    model_path = config.get('output', {}).get('output_dir', './checkpoints/flan-t5-base-lora-biolaysumm')
    evaluator = BioLaySummEvaluator(config, model_path)
    results = evaluator.evaluate()
    return results


if __name__ == "__main__":
    main()


