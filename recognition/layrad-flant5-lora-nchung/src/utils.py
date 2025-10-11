"""
Utility functions for configuration loading and common operations.

This module provides utilities for loading YAML configurations, setting up
reproducibility, and other common functions used throughout the project.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import random
import yaml
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    This function loads a YAML configuration file and returns it as a dictionary.
    It also handles path resolution and provides helpful error messages.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> print(config['model']['name'])
        'google/flan-t5-base'
    """
    # Convert to Path object for better path handling
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"Successfully loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


def setup_reproducibility(config: Dict[str, Any]) -> None:
    """
    Set up reproducibility by fixing all random seeds.
    
    This function sets random seeds for Python's random module, NumPy, PyTorch,
    and CUDA to ensure reproducible results across runs.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing seed values
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> setup_reproducibility(config)
    """
    # Get seed values from config (with fallbacks)
    seed = config.get('reproducibility', {}).get('seed', 42)
    data_seed = config.get('reproducibility', {}).get('data_seed', seed)
    model_seed = config.get('reproducibility', {}).get('model_seed', seed)
    
    # Set Python random seed
    random.seed(data_seed)
    
    # Set NumPy random seed
    np.random.seed(data_seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    
    # Set PyTorch to deterministic mode (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Reproducibility setup complete:")
    print(f"  - Global seed: {seed}")
    print(f"  - Data seed: {data_seed}")
    print(f"  - Model seed: {model_seed}")


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the appropriate device (CPU/GPU) based on configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        torch.device: PyTorch device object
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> device = get_device(config)
        >>> print(device)
        device(type='cuda')
    """
    device_name = config.get('hardware', {}).get('device', 'cuda')
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_output_dir(config: Dict[str, Any]) -> Path:
    """
    Create output directory for checkpoints and logs.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Path: Path to the created output directory
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> output_dir = create_output_dir(config)
        >>> print(output_dir)
        PosixPath('./checkpoints/flan-t5-base-lora-biolaysumm')
    """
    output_dir = Path(config.get('output', {}).get('output_dir', './checkpoints/default'))
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    return output_dir


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        Dict[str, int]: Dictionary with total and trainable parameter counts
        
    Example:
        >>> model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
        >>> param_counts = count_parameters(model)
        >>> print(f"Total parameters: {param_counts['total']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def format_parameter_count(count: int) -> str:
    """
    Format parameter count in human-readable format.
    
    Args:
        count (int): Number of parameters
        
    Returns:
        str: Formatted parameter count (e.g., "248M", "1.2B")
        
    Example:
        >>> count = 248000000
        >>> formatted = format_parameter_count(count)
        >>> print(formatted)
        '248M'
    """
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.0f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.0f}K"
    else:
        return str(count)


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        output_path (Path): Path to save the configuration
        
    Example:
        >>> config = load_config('configs/train_flant5_base_lora.yaml')
        >>> save_config(config, Path('saved_config.yaml'))
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary for required fields.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['dataset', 'model', 'training', 'lora', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset section
    dataset = config['dataset']
    if 'name' not in dataset:
        raise ValueError("Missing required field: dataset.name")
    
    # Validate model section
    model = config['model']
    if 'name' not in model:
        raise ValueError("Missing required field: model.name")
    
    # Validate LoRA section
    lora = config['lora']
    required_lora_fields = ['r', 'alpha', 'dropout', 'target_modules']
    for field in required_lora_fields:
        if field not in lora:
            raise ValueError(f"Missing required field: lora.{field}")
    
    print("Configuration validation passed")
    return True


def create_reports_dir(output_dir: Path) -> Path:
    """
    Create reports directory structure for logging training information.
    
    Args:
        output_dir (Path): Base output directory
        
    Returns:
        Path: Path to the reports directory
    """
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (reports_dir / 'logs').mkdir(exist_ok=True)
    (reports_dir / 'metrics').mkdir(exist_ok=True)
    (reports_dir / 'configs').mkdir(exist_ok=True)
    
    print(f"Reports directory created: {reports_dir}")
    return reports_dir


def log_training_arguments(training_args, reports_dir: Path) -> None:
    """
    Log training arguments to reports directory.
    
    Args:
        training_args: HuggingFace TrainingArguments object
        reports_dir (Path): Reports directory path
    """
    # Convert training arguments to dictionary
    train_args_dict = training_args.to_dict()
    
    # Add additional metadata
    train_args_info = {
        'training_arguments': train_args_dict,
        'timestamp': datetime.now().isoformat(),
        'output_dir': str(training_args.output_dir),
        'run_name': training_args.run_name,
        'num_train_epochs': training_args.num_train_epochs,
        'per_device_train_batch_size': training_args.per_device_train_batch_size,
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'learning_rate': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'max_grad_norm': training_args.max_grad_norm,
        'warmup_steps': training_args.warmup_steps,
        'eval_strategy': training_args.eval_strategy,
        'save_strategy': training_args.save_strategy,
        'metric_for_best_model': training_args.metric_for_best_model,
        'greater_is_better': training_args.greater_is_better,
        'load_best_model_at_end': training_args.load_best_model_at_end,
        'fp16': training_args.fp16,
        'bf16': training_args.bf16,
        'seed': training_args.seed,
        'data_seed': training_args.data_seed,
    }
    
    # Save to JSON file
    train_args_path = reports_dir / 'configs' / 'training_arguments.json'
    with open(train_args_path, 'w', encoding='utf-8') as f:
        json.dump(train_args_info, f, indent=2, ensure_ascii=False)
    
    print(f"Training arguments logged to: {train_args_path}")


def log_trainer_state(trainer, reports_dir: Path) -> None:
    """
    Log trainer state and metrics to reports directory.
    
    Args:
        trainer: HuggingFace Trainer object
        reports_dir (Path): Reports directory path
    """
    try:
        # Get trainer state
        state = trainer.state
        
        # Create trainer state info
        trainer_state_info = {
            'timestamp': datetime.now().isoformat(),
            'global_step': state.global_step,
            'epoch': state.epoch,
            'max_steps': state.max_steps,
            'num_train_epochs': state.num_train_epochs,
            'total_flos': state.total_flos,
            'log_history': state.log_history[-10:] if state.log_history else [],  # Last 10 logs
            'best_metric': getattr(state, 'best_metric', None),
            'best_model_checkpoint': getattr(state, 'best_model_checkpoint', None),
            'is_local_process_zero': state.is_local_process_zero,
            'is_world_process_zero': state.is_world_process_zero,
            'is_hyper_param_search': state.is_hyper_param_search,
        }
        
        # Save trainer state
        state_path = reports_dir / 'logs' / 'trainer_state.json'
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(trainer_state_info, f, indent=2, ensure_ascii=False)
        
        print(f"Trainer state logged to: {state_path}")
        
        # Log metrics if available
        if hasattr(trainer, 'log_history') and trainer.log_history:
            metrics_path = reports_dir / 'metrics' / 'training_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(trainer.log_history, f, indent=2, ensure_ascii=False)
            
            print(f"Training metrics logged to: {metrics_path}")
            
    except Exception as e:
        print(f"Warning: Could not log trainer state: {e}")


def log_training_summary(config: Dict[str, Any], model_info: Dict[str, Any], 
                        training_time: float, reports_dir: Path) -> None:
    """
    Log a comprehensive training summary to reports directory.
    
    Args:
        config (Dict[str, Any]): Training configuration
        model_info (Dict[str, Any]): Model information (parameters, etc.)
        training_time (float): Total training time in seconds
        reports_dir (Path): Reports directory path
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_summary': {
            'total_training_time_seconds': training_time,
            'total_training_time_hours': training_time / 3600,
            'model_info': model_info,
            'dataset_info': {
                'name': config.get('dataset', {}).get('name', 'unknown'),
                'max_source_length': config.get('dataset', {}).get('max_source_length', 'unknown'),
                'max_target_length': config.get('dataset', {}).get('max_target_length', 'unknown'),
            },
            'model_config': {
                'name': config.get('model', {}).get('name', 'unknown'),
                'torch_dtype': config.get('model', {}).get('torch_dtype', 'unknown'),
            },
            'lora_config': config.get('lora', {}),
            'training_config': config.get('training', {}),
            'evaluation_config': config.get('evaluation', {}),
            'hardware_config': config.get('hardware', {}),
        }
    }
    
    # Save summary
    summary_path = reports_dir / 'training_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Training summary logged to: {summary_path}")


def setup_logging(output_dir: Path) -> Path:
    """
    Setup comprehensive logging for training.
    
    Args:
        output_dir (Path): Base output directory
        
    Returns:
        Path: Path to the reports directory
    """
    reports_dir = create_reports_dir(output_dir)
    
    # Create a simple log file for stdout/stderr capture
    log_file = reports_dir / 'logs' / 'training.log'
    
    print(f"Logging setup complete. Reports directory: {reports_dir}")
    print(f"Training log file: {log_file}")
    
    return reports_dir
