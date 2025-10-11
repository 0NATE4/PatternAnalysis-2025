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
