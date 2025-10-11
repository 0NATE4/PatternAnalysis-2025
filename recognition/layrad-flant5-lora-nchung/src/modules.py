"""
FLAN-T5 Model Wrapper with LoRA Support for BioLaySumm Translation

This module provides a comprehensive wrapper for FLAN-T5 models with LoRA
(Low-Rank Adaptation) support for parameter-efficient fine-tuning on the
BioLaySumm expert-to-layperson translation task.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

from utils import count_parameters, format_parameter_count


class FLANT5LoRAModel:
    """
    FLAN-T5 model wrapper with LoRA support for parameter-efficient fine-tuning.
    
    This class provides a unified interface for loading, configuring, and managing
    FLAN-T5 models with LoRA adaptations for the BioLaySumm translation task.
    
    Attributes:
        config (dict): Configuration dictionary
        model (AutoModelForSeq2SeqLM): Base FLAN-T5 model
        tokenizer (AutoTokenizer): Model tokenizer
        lora_config (LoraConfig): LoRA configuration
        device (torch.device): Device for model placement
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FLAN-T5 model with LoRA support.
        
        Args:
            config (dict): Configuration dictionary containing model and LoRA settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        # Determine device - use CUDA if available, otherwise CPU
        device_name = config.get('hardware', {}).get('device', 'cuda')
        if device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print(f"CUDA not available, using CPU instead")
        
        # Initialize model and tokenizer
        self._build_model()
        
    def _build_model(self) -> None:
        """
        Build FLAN-T5 model and tokenizer from configuration.
        
        This method loads the base FLAN-T5 model and tokenizer, then applies
        LoRA configuration for parameter-efficient fine-tuning.
        """
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'google/flan-t5-base')
        torch_dtype = getattr(torch, model_config.get('torch_dtype', 'bfloat16'))
        
        print(f"Loading FLAN-T5 model: {model_name}")
        print(f"Using torch dtype: {torch_dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if torch.cuda.is_available():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        else:
            # CPU-only loading
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Use float32 for CPU
            )
        
        # Apply LoRA configuration
        self._apply_lora()
        
        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        elif hasattr(self.model, 'device') and str(self.model.device) == 'cpu':
            # Model wasn't moved by device_map, move it manually
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully on device: {self.device}")
        
    def _apply_lora(self) -> None:
        """
        Apply LoRA (Low-Rank Adaptation) configuration to the model.
        
        This method configures LoRA for parameter-efficient fine-tuning by
        adding low-rank matrices to specific transformer modules.
        """
        lora_config = self.config.get('lora', {})
        
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['q', 'v']),
            bias=lora_config.get('bias', 'none')
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        
        print("LoRA configuration applied successfully")
        print(f"LoRA rank (r): {self.lora_config.r}")
        print(f"LoRA alpha: {self.lora_config.lora_alpha}")
        print(f"LoRA dropout: {self.lora_config.lora_dropout}")
        print(f"Target modules: {self.lora_config.target_modules}")
        
    def count_params(self) -> Dict[str, Any]:
        """
        Count and analyze model parameters.
        
        Returns:
            dict: Dictionary containing parameter counts and statistics
        """
        param_counts = count_parameters(self.model)
        
        # Calculate percentages
        total_params = param_counts['total']
        trainable_params = param_counts['trainable']
        frozen_params = param_counts['frozen']
        
        trainable_percentage = (trainable_params / total_params) * 100
        frozen_percentage = (frozen_params / total_params) * 100
        
        # Format parameter counts
        formatted_counts = {
            'total': format_parameter_count(total_params),
            'trainable': format_parameter_count(trainable_params),
            'frozen': format_parameter_count(frozen_params),
            'trainable_percentage': f"{trainable_percentage:.2f}%",
            'frozen_percentage': f"{frozen_percentage:.2f}%"
        }
        
        # Print parameter summary
        print("\n" + "="*50)
        print("MODEL PARAMETER SUMMARY")
        print("="*50)
        print(f"Total parameters: {formatted_counts['total']} ({total_params:,})")
        print(f"Trainable parameters: {formatted_counts['trainable']} ({trainable_params:,})")
        print(f"Frozen parameters: {formatted_counts['frozen']} ({frozen_params:,})")
        print(f"Trainable percentage: {formatted_counts['trainable_percentage']}")
        print(f"Frozen percentage: {formatted_counts['frozen_percentage']}")
        print("="*50)
        
        return {
            'raw_counts': param_counts,
            'formatted_counts': formatted_counts,
            'summary': f"FLAN-T5 with LoRA: {formatted_counts['trainable']} trainable ({formatted_counts['trainable_percentage']}) of {formatted_counts['total']} total parameters"
        }
    
    def save_generation_config(self, output_dir: Path) -> None:
        """
        Save generation configuration for evaluation.
        
        This method saves the generation parameters used for evaluation
        to ensure reproducibility and proper documentation of results.
        
        Args:
            output_dir (Path): Directory to save the generation config
        """
        eval_config = self.config.get('evaluation', {})
        
        # Create generation config
        generation_config = {
            'max_new_tokens': eval_config.get('max_new_tokens', 200),
            'num_beams': eval_config.get('num_beams', 4),
            'length_penalty': eval_config.get('length_penalty', 0.6),
            'no_repeat_ngram_size': eval_config.get('no_repeat_ngram_size', 3),
            'early_stopping': eval_config.get('early_stopping', True),
            'do_sample': False,  # Deterministic generation for evaluation
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
        }
        
        # Save to JSON file
        config_path = output_dir / 'generation_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(generation_config, f, indent=2, ensure_ascii=False)
        
        print(f"Generation configuration saved to: {config_path}")
        
        # Also create HuggingFace GenerationConfig object
        hf_generation_config = GenerationConfig(
            max_new_tokens=generation_config['max_new_tokens'],
            num_beams=generation_config['num_beams'],
            length_penalty=generation_config['length_penalty'],
            no_repeat_ngram_size=generation_config['no_repeat_ngram_size'],
            early_stopping=generation_config['early_stopping'],
            do_sample=generation_config['do_sample'],
            pad_token_id=generation_config['pad_token_id'],
            eos_token_id=generation_config['eos_token_id']
        )
        
        # Save HuggingFace config
        hf_config_path = output_dir / 'generation_config_hf'
        hf_generation_config.save_pretrained(hf_config_path)
        
        return hf_generation_config
    
    def get_model_and_tokenizer(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Get the model and tokenizer for training/inference.
        
        Returns:
            tuple: (model, tokenizer) for use in training loops
        """
        return self.model, self.tokenizer
    
    def save_model(self, output_dir: Path) -> None:
        """
        Save the trained model and tokenizer.
        
        Args:
            output_dir (Path): Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save generation config
        self.save_generation_config(output_dir)
        
        print(f"Model saved to: {output_dir}")
    
    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (Path): Path to the saved model directory
        """
        model_path = Path(model_path)
        
        # Load base model first
        base_model_name = self.config.get('model', {}).get('name', 'google/flan-t5-base')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print(f"Model loaded from: {model_path}")


def build_model_with_lora(config: Dict[str, Any]) -> FLANT5LoRAModel:
    """
    Build FLAN-T5 model with LoRA configuration.
    
    This is the main factory function for creating FLAN-T5 models with LoRA
    support for the BioLaySumm translation task.
    
    Args:
        config (dict): Configuration dictionary containing model and LoRA settings
        
    Returns:
        FLANT5LoRAModel: Configured model wrapper
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> model_wrapper = build_model_with_lora(config)
        >>> model, tokenizer = model_wrapper.get_model_and_tokenizer()
        >>> param_info = model_wrapper.count_params()
    """
    return FLANT5LoRAModel(config)


def apply_lora_to_model(model: AutoModelForSeq2SeqLM, lora_config: Dict[str, Any]) -> AutoModelForSeq2SeqLM:
    """
    Apply LoRA configuration to an existing model.
    
    This function provides a standalone way to apply LoRA to any FLAN-T5 model
    without creating a full wrapper instance.
    
    Args:
        model (AutoModelForSeq2SeqLM): Base FLAN-T5 model
        lora_config (dict): LoRA configuration dictionary
        
    Returns:
        AutoModelForSeq2SeqLM: Model with LoRA applied
    """
    # Create LoRA configuration
    lora_config_obj = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.1),
        target_modules=lora_config.get('target_modules', ['q', 'v']),
        bias=lora_config.get('bias', 'none')
    )
    
    # Apply LoRA
    model_with_lora = get_peft_model(model, lora_config_obj)
    
    return model_with_lora


def count_model_parameters(model: torch.nn.Module) -> str:
    """
    Count and format model parameters in a human-readable string.
    
    This function provides a simple interface for parameter counting that
    returns a formatted string suitable for logging or display.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        str: Formatted parameter count string
    """
    param_counts = count_parameters(model)
    
    total_params = param_counts['total']
    trainable_params = param_counts['trainable']
    trainable_percentage = (trainable_params / total_params) * 100
    
    return (f"Model parameters: {format_parameter_count(trainable_params)} trainable "
            f"({trainable_percentage:.2f}%) of {format_parameter_count(total_params)} total")
