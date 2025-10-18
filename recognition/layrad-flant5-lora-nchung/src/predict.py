"""
Prediction script for FLAN-T5 LoRA model on BioLaySumm examples.

This module generates sample expert-to-layperson translations and saves them
in a readable format for analysis and demonstration purposes.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import json
import time
import torch
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)
from datasets import Dataset
from peft import PeftModel

from utils import (
    load_config, setup_reproducibility, get_device, 
    create_reports_dir
)
from dataset import BioLaySummDataset


class BioLaySummPredictor:
    """
    Prediction wrapper for FLAN-T5 LoRA model on BioLaySumm examples.
    
    This class provides sample generation capabilities including:
    - Model loading and inference
    - Example selection and generation
    - Pretty printing to console
    - JSONL output for analysis
    
    Attributes:
        config (dict): Configuration dictionary
        model: Trained FLAN-T5 LoRA model
        tokenizer: Tokenizer for the model
        reports_dir (Path): Reports directory for output
        device: Device for computation (CPU/GPU)
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        """
        Initialize the BioLaySumm predictor.
        
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
        
        print(f"Prediction setup complete. Model path: {self.model_path}")
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
            dtype=torch.float32 if self.device.type == 'cpu' else torch.bfloat16,
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
            # Use default generation config with better parameters for examples
            self.generation_config = GenerationConfig(
                max_new_tokens=256,  # Longer for better examples
                num_beams=4,         # Beam search for better quality (vs greedy)
                length_penalty=0.6,  # Slightly penalize longer sequences
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                early_stopping=True,     # Stop when EOS token is generated
                do_sample=False,         # Deterministic generation for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            print("✅ Using default generation config")
        
        print("✅ Model and tokenizer loaded successfully")
        
    def load_dataset(self) -> None:
        """
        Load the dataset for example selection.
        """
        print("\nLoading dataset for examples...")
        
        # Initialize dataset loader
        self.dataset_loader = BioLaySummDataset(self.config)
        
        # Load validation dataset (good for examples)
        self.dataset = self.dataset_loader.load_data('validation')
        
        print(f"✅ Dataset loaded: {len(self.dataset)} samples")
        
    def select_examples(self, num_examples: int = 5, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Select random examples from the dataset.
        
        Args:
            num_examples (int): Number of examples to select
            random_seed (int): Random seed for reproducible selection
            
        Returns:
            List[Dict]: Selected examples
        """
        print(f"\nSelecting {num_examples} examples...")
        
        # Set random seed for reproducible selection
        random.seed(random_seed)
        
        # Select random indices
        available_indices = list(range(len(self.dataset)))
        selected_indices = random.sample(available_indices, min(num_examples, len(available_indices)))
        
        # Get selected examples
        examples = []
        for idx in selected_indices:
            sample = self.dataset[idx]
            examples.append({
                'index': idx,
                'input_text': sample['input_text'],
                'target_text': sample['target_text'],
            })
        
        print(f"✅ Selected {len(examples)} examples")
        return examples
        
    def generate_predictions(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate predictions for the selected examples.
        
        Args:
            examples (List[Dict]): List of examples
            
        Returns:
            List[Dict]: Examples with generated predictions
        """
        print(f"\nGenerating predictions for {len(examples)} examples...")
        
        # Prepare model for inference
        self.model.eval()
        
        predictions = []
        start_time = time.time()
        
        with torch.no_grad():
            for i, example in enumerate(examples):
                print(f"Generating prediction {i+1}/{len(examples)}...")
                
                # Tokenize input
                input_text = example['input_text']
                
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.config.get('dataset', {}).get('max_source_length', 512),
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate prediction using beam search
                # Beam search explores multiple sequence possibilities and selects the best one
                # This produces higher quality outputs than greedy decoding
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
                
                # Store result
                prediction_data = {
                    'example_id': i + 1,
                    'dataset_index': example['index'],
                    'input_text': input_text,
                    'target_text': example['target_text'],
                    'generated_text': generated_text,
                    'input_length': len(input_text.split()),
                    'target_length': len(example['target_text'].split()),
                    'generated_length': len(generated_text.split()),
                }
                predictions.append(prediction_data)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"✅ Generated {len(predictions)} predictions in {generation_time:.2f} seconds")
        
        return predictions
    
    def pretty_print_examples(self, predictions: List[Dict[str, Any]]) -> None:
        """
        Pretty print examples to console.
        
        Args:
            predictions (List[Dict]): List of predictions with input, target, and generated text
        """
        print("\n" + "="*80)
        print("EXPERT-TO-LAYPERSON TRANSLATION EXAMPLES")
        print("="*80)
        
        for pred in predictions:
            print(f"\n📋 EXAMPLE {pred['example_id']} (Dataset Index: {pred['dataset_index']})")
            print("-" * 60)
            
            print(f"\n🔬 EXPERT REPORT:")
            print(f"{pred['input_text']}")
            
            print(f"\n👥 LAYPERSON TARGET:")
            print(f"{pred['target_text']}")
            
            print(f"\n🤖 MODEL PREDICTION:")
            print(f"{pred['generated_text']}")
            
            print(f"\n📊 STATISTICS:")
            print(f"   Input length: {pred['input_length']} words")
            print(f"   Target length: {pred['target_length']} words")
            print(f"   Generated length: {pred['generated_length']} words")
            
            print("\n" + "="*80)
    
    def save_examples_to_jsonl(self, predictions: List[Dict[str, Any]]) -> None:
        """
        Save examples to JSONL file.
        
        Args:
            predictions (List[Dict]): List of predictions
        """
        jsonl_path = self.reports_dir / 'examples.jsonl'
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
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
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                
                # Write as JSON line
                f.write(json.dumps(example_data, ensure_ascii=False) + '\n')
        
        print(f"✅ Examples saved to: {jsonl_path}")
    
    def predict_examples(self, num_examples: int = 5, random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate example predictions.
        
        Args:
            num_examples (int): Number of examples to generate
            random_seed (int): Random seed for reproducible selection
            
        Returns:
            List[Dict[str, Any]]: Generated predictions
        """
        print("\n" + "="*60)
        print("GENERATING EXAMPLE PREDICTIONS")
        print("="*60)
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Select examples
        examples = self.select_examples(num_examples=num_examples, random_seed=random_seed)
        
        # Generate predictions
        predictions = self.generate_predictions(examples)
        
        # Pretty print to console
        self.pretty_print_examples(predictions)
        
        # Save to JSONL
        self.save_examples_to_jsonl(predictions)
        
        print(f"\n✅ Example predictions complete!")
        print(f"Results saved to: {self.reports_dir / 'examples.jsonl'}")
        
        return predictions


def main():
    """
    Main prediction function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate example predictions from FLAN-T5 LoRA model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model directory')
    parser.add_argument('--config', type=str, default='configs/train_flant5_base_lora.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to generate (default: 5)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for example selection (default: 42)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create predictor and generate examples
    predictor = BioLaySummPredictor(config, args.model_path)
    predictions = predictor.predict_examples(
        num_examples=args.num_examples,
        random_seed=args.random_seed
    )
    
    return predictions


if __name__ == "__main__":
    main()
