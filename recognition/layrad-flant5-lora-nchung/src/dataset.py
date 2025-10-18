"""
BioLaySumm Dataset Loader for Expert-to-Layperson Radiology Report Translation

This module implements a comprehensive dataset loader for the BioLaySumm dataset,
which contains expert radiology reports paired with layperson summaries.
The loader supports both HuggingFace hub and local file loading with proper
train/validation/test splits and reproducible shuffling.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import random
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
import torch


class BioLaySummDataset:
    """
    Dataset loader for BioLaySumm expert-to-layperson radiology report translation.
    
    This class handles loading, preprocessing, and tokenization of the BioLaySumm dataset
    for fine-tuning FLAN-T5 models to translate expert radiology reports into
    layperson-friendly summaries.
    
    Attributes:
        config (dict): Configuration dictionary containing dataset parameters
        dataset_name (str): Name of the dataset (HuggingFace hub or local path)
        max_source_length (int): Maximum length for input radiology reports
        max_target_length (int): Maximum length for output layperson summaries
        seed (int): Random seed for reproducible data shuffling
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the BioLaySumm dataset loader.
        
        Args:
            config (dict): Configuration dictionary containing dataset section:
                - dataset.name: HuggingFace dataset name or local path
                - dataset.max_source_length: Maximum input sequence length
                - dataset.max_target_length: Maximum output sequence length
                - dataset.seed: Random seed for reproducibility
                - dataset.local_data_path: Optional local data path override
        """
        self.config = config
        dataset_config = config.get('dataset', {})
        
        self.dataset_name = dataset_config.get('name', 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track')
        # 512 tokens for source and 256 tokens for target
        self.max_source_length = dataset_config.get('max_source_length', 512)
        self.max_target_length = dataset_config.get('max_target_length', 256)
        # 42 is a common seed for reproducibility
        self.seed = dataset_config.get('seed', 42)
        self.local_data_path = dataset_config.get('local_data_path', None)
        
        # Set random seed for reproducible shuffling
        random.seed(self.seed)
        
    def load_data(self, split: str) -> Dataset:
        """
        Load BioLaySumm dataset for the specified split with proper preprocessing.
        
        This method loads the dataset from either HuggingFace hub or local files,
        applies expert-to-layperson prompting, and returns a processed Dataset
        object ready for tokenization and training.
        
        Args:
            split (str): Dataset split to load. Must be one of:
                - 'train': Training split (150k samples)
                - 'validation': Validation split (10k samples) 
                - 'test': Test split (10.5k samples)
        
        Returns:
            Dataset: Processed dataset with 'input_text' and 'target_text' fields
            
        Raises:
            ValueError: If split is not one of ['train', 'validation', 'test']
            FileNotFoundError: If local data path is specified but doesn't exist
            
        """
        # Validate split parameter
        valid_splits = ['train', 'validation', 'test']
        if split not in valid_splits:
            raise ValueError(f"Split must be one of {valid_splits}, got '{split}'")
        
        # Load dataset from HuggingFace hub or local files
        if self.local_data_path and os.path.exists(self.local_data_path):
            # Load from local files (if available)
            print(f"Loading {split} data from local path: {self.local_data_path}")
            dataset = self._load_from_local(split)
        else:
            # Load from HuggingFace hub (default)
            print(f"Loading {split} data from HuggingFace: {self.dataset_name}")
            dataset = self._load_from_hub(split)
        
        # Apply expert-to-layperson prompting and preprocessing
        dataset = self._apply_prompting(dataset)
        
        # Shuffle data with reproducible seed (important for consistent splits)
        if split == 'train':
            dataset = dataset.shuffle(seed=self.seed)
        
        print(f"Successfully loaded {len(dataset)} {split} samples")
        return dataset
    
    def _load_from_hub(self, split: str) -> Dataset:
        """
        Load dataset from HuggingFace hub.
        
        Args:
            split (str): Dataset split to load
            
        Returns:
            Dataset: Raw dataset from HuggingFace
        """
        try:
            # Load dataset from HuggingFace hub
            dataset = load_dataset(
                self.dataset_name,
                split=split,
                trust_remote_code=False  # Disabled to avoid deprecation warnings
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from HuggingFace hub: {e}")
    
    def _load_from_local(self, split: str) -> Dataset:
        """
        Load dataset from local files (future implementation).
        
        Args:
            split (str): Dataset split to load
            
        Returns:
            Dataset: Dataset loaded from local files
            
        Raises:
            NotImplementedError: Local loading not yet implemented
        """
        # TODO: Implement local file loading for offline usage
        raise NotImplementedError("Local file loading not yet implemented. Use HuggingFace hub.")
    
    def _apply_prompting(self, dataset: Dataset) -> Dataset:
        """
        Apply expert-to-layperson prompting to the dataset.
        
        This method transforms the raw dataset by adding appropriate prompts
        that instruct the model to translate expert radiology reports into
        layperson-friendly summaries.
        
        Args:
            dataset (Dataset): Raw dataset with 'radiology_report' and 'layman_report' fields
            
        Returns:
            Dataset: Dataset with 'input_text' and 'target_text' fields
        """
        def add_prompts(example):
            """
            Add expert-to-layperson translation prompts to each example.
            
            Args:
                example (dict): Single dataset example with radiology_report and layman_report
                
            Returns:
                dict: Example with input_text and target_text fields
            """
            # Extract expert radiology report and layperson summary
            expert_report = example['radiology_report'].strip()
            layperson_summary = example['layman_report'].strip()
            
            # Create expert-to-layperson translation prompt
            # This prompt instructs the model to translate medical jargon into plain language
            # The format follows instruction-tuning patterns for better model understanding
            input_text = f"Translate this expert radiology report into layperson terms:\n\n{expert_report}\n\nLayperson summary:"
            
            return {
                'input_text': input_text,
                'target_text': layperson_summary,
                'source': example.get('source', 'unknown'),  # Preserve source info
                'images_path': example.get('images_path', '')  # Preserve image path for reference
            }
        
        # Apply prompting to all examples in the dataset
        dataset = dataset.map(
            add_prompts,
            remove_columns=['radiology_report', 'layman_report'],  # Remove original columns
            desc=f"Applying expert-to-layperson prompts"
        )
        
        return dataset
    
    def preprocess_function(self, examples: Dict, tokenizer: AutoTokenizer) -> Dict:
        """
        Tokenize and preprocess dataset examples for training.
        
        This method handles the tokenization of input and target texts with proper
        padding, truncation, and label preparation for sequence-to-sequence training.
        Implements the critical -100 padding for labels to ensure proper loss calculation.
        
        Args:
            examples (dict): Batch of examples with 'input_text' and 'target_text' fields
            tokenizer (AutoTokenizer): HuggingFace tokenizer for the model
            
        Returns:
            dict: Tokenized examples with 'input_ids', 'attention_mask', and 'labels'
            
        Note:
            The -100 padding in labels is crucial for PyTorch's CrossEntropyLoss.
            Tokens with -100 are ignored during loss calculation, allowing proper
            handling of variable-length sequences with padding.
        """
        # Tokenize input texts (expert reports with prompts)
        # Truncate to max_source_length (512 tokens) - sufficient for most radiology reports
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize target texts (layperson summaries)
        # Truncate to max_target_length (256 tokens) - layperson summaries are typically shorter
        labels = tokenizer(
            examples["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract label input_ids and replace padding tokens with -100
        # This is CRITICAL: -100 tokens are ignored by the loss function
        # Without this, the model would try to predict padding tokens, which would
        # artificially inflate loss and hurt training. PyTorch's CrossEntropyLoss
        # specifically ignores -100 labels during loss computation.
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Add labels to model inputs
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def get_loader(self, dataset: Dataset, tokenizer: AutoTokenizer, batch_size: int) -> DataLoader:
        """
        Create a DataLoader for the processed dataset.
        
        This method applies tokenization to the dataset and creates a DataLoader
        with proper batching, shuffling, and collation for training.
        
        Args:
            dataset (Dataset): Processed dataset with 'input_text' and 'target_text'
            tokenizer (AutoTokenizer): Model tokenizer
            batch_size (int): Batch size for training
            
        Returns:
            DataLoader: Ready-to-use DataLoader for training
        """
        # Apply tokenization to the dataset
        processed_dataset = dataset.map(
            lambda examples: self.preprocess_function(examples, tokenizer),
            batched=True,
            num_proc=1,  # Single process for consistency
            load_from_cache_file=False,  # Always reprocess for consistency
            remove_columns=["input_text", "target_text", "source", "images_path"],
            desc="Tokenizing dataset"
        )
        
        # Create DataLoader with proper settings
        loader = DataLoader(
            processed_dataset,
            collate_fn=default_data_collator,  # Standard collation for transformers
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            pin_memory=True,  # Faster GPU transfer
            drop_last=False,  # Keep all samples
        )
        
        return loader