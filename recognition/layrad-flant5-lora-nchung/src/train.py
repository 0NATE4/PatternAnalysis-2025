"""
Training script for FLAN-T5 LoRA on BioLaySumm dataset.

This module implements the training loop using HuggingFace's Seq2SeqTrainer
with proper configuration, metrics, and checkpointing for the expert-to-layperson
radiology report translation task.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import os
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from datasets import Dataset

from utils import load_config, setup_reproducibility, get_device, create_output_dir, save_config
from dataset import BioLaySummDataset
from modules import build_model_with_lora


class BioLaySummTrainer:
    """
    Training wrapper for FLAN-T5 LoRA on BioLaySumm dataset.
    
    This class provides a unified interface for training FLAN-T5 models with LoRA
    on the BioLaySumm expert-to-layperson translation task using HuggingFace's
    Seq2SeqTrainer with proper configuration and metrics.
    
    Attributes:
        config (dict): Configuration dictionary
        model_wrapper: FLAN-T5 LoRA model wrapper
        dataset_loader: BioLaySumm dataset loader
        trainer: HuggingFace Seq2SeqTrainer
        output_dir (Path): Output directory for checkpoints and logs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BioLaySumm trainer.
        
        Args:
            config (dict): Configuration dictionary containing all training settings
        """
        self.config = config
        self.model_wrapper = None
        self.dataset_loader = None
        self.trainer = None
        self.output_dir = None
        
        # Setup training environment
        self._setup_training()
        
    def _setup_training(self) -> None:
        """
        Setup training environment including reproducibility, device, and output directory.
        """
        # Setup reproducibility
        setup_reproducibility(self.config)
        
        # Get device
        self.device = get_device(self.config)
        
        # Create output directory
        self.output_dir = create_output_dir(self.config)
        
        # Save configuration
        save_config(self.config, self.output_dir / 'training_config.yaml')
        
        print(f"Training setup complete. Output directory: {self.output_dir}")
        
    def _build_model_and_data(self) -> None:
        """
        Build model and load datasets for training.
        """
        print("\nBuilding model and loading datasets...")
        
        # Initialize model wrapper
        self.model_wrapper = build_model_with_lora(self.config)
        model, tokenizer = self.model_wrapper.get_model_and_tokenizer()
        
        # Print parameter information
        self.model_wrapper.count_params()
        
        # Initialize dataset loader
        self.dataset_loader = BioLaySummDataset(self.config)
        
        # Load datasets
        print("Loading training dataset...")
        train_dataset = self.dataset_loader.load_data('train')
        
        print("Loading validation dataset...")
        val_dataset = self.dataset_loader.load_data('validation')
        
        # Create data loaders
        print("Creating data loaders...")
        train_dataloader = self.dataset_loader.get_loader(
            train_dataset, tokenizer, self.config['training']['batch_size']
        )
        val_dataloader = self.dataset_loader.get_loader(
            val_dataset, tokenizer, self.config['training']['batch_size']
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
    def _create_data_collator(self) -> DataCollatorForSeq2Seq:
        """
        Create data collator for sequence-to-sequence training.
        
        Returns:
            DataCollatorForSeq2Seq: Data collator for proper batching
        """
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
    
    def _create_generation_config(self) -> GenerationConfig:
        """
        Create generation configuration for evaluation.
        
        Returns:
            GenerationConfig: Configuration for text generation during evaluation
        """
        eval_config = self.config.get('evaluation', {})
        
        return GenerationConfig(
            max_new_tokens=eval_config.get('max_new_tokens', 200),
            num_beams=eval_config.get('num_beams', 4),
            length_penalty=eval_config.get('length_penalty', 0.6),
            no_repeat_ngram_size=eval_config.get('no_repeat_ngram_size', 3),
            early_stopping=eval_config.get('early_stopping', True),
            do_sample=False,  # Deterministic generation for evaluation
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        """
        Create training arguments from configuration.
        
        Returns:
            Seq2SeqTrainingArguments: Training arguments for Seq2SeqTrainer
        """
        training_config = self.config.get('training', {})
        output_config = self.config.get('output', {})
        
        # Calculate total training steps
        num_epochs = training_config.get('num_epochs', 3)
        batch_size = training_config.get('batch_size', 8)
        grad_accum_steps = training_config.get('gradient_accumulation_steps', 4)
        
        # Estimate steps per epoch (approximate)
        steps_per_epoch = len(self.train_dataset) // (batch_size * grad_accum_steps)
        total_steps = steps_per_epoch * num_epochs
        
        print(f"Estimated training steps: {total_steps} ({steps_per_epoch} per epoch)")
        
        return Seq2SeqTrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            run_name=output_config.get('run_name', 'flan-t5-base-lora-biolaysumm'),
            report_to=output_config.get('report_to', ['tensorboard']),
            
            # Training parameters
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # Learning rate scheduling
            warmup_steps=training_config.get('warmup_steps', 500),
            lr_scheduler_type="linear",
            
            # Mixed precision
            fp16=False,  # Use bf16 instead
            bf16=self.config.get('training', {}).get('bf16', True),
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=training_config.get('eval_steps', 1000),
            save_strategy="steps",
            save_steps=training_config.get('save_steps', 1000),
            save_total_limit=training_config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_rougeLsum",
            greater_is_better=True,
            
            # Logging
            logging_steps=training_config.get('logging_steps', 100),
            logging_first_step=True,
            logging_dir=str(self.output_dir / 'logs'),
            
            # Reproducibility
            seed=self.config.get('reproducibility', {}).get('seed', 42),
            data_seed=self.config.get('reproducibility', {}).get('data_seed', 42),
            
            # Performance
            dataloader_num_workers=self.config.get('hardware', {}).get('dataloader_num_workers', 4),
            dataloader_pin_memory=self.config.get('hardware', {}).get('pin_memory', True),
            
            # Generation for evaluation
            predict_with_generate=True,  # Use generation for evaluation
            generation_config=self._create_generation_config(),
            
            # Note: Early stopping parameters not supported in transformers 4.30.0
            # early_stopping_patience=training_config.get('early_stopping_patience', 3),
            # early_stopping_threshold=training_config.get('early_stopping_threshold', 0.001),
            
            # Remove unused columns
            remove_unused_columns=True,
        )
    
    def _create_trainer(self) -> Seq2SeqTrainer:
        """
        Create HuggingFace Seq2SeqTrainer.
        
        Returns:
            Seq2SeqTrainer: Configured trainer for sequence-to-sequence training
        """
        print("\nCreating Seq2SeqTrainer...")
        
        # Create training arguments
        training_args = self._create_training_arguments()
        
        # Create data collator
        data_collator = self._create_data_collator()
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            # Note: compute_metrics will be added in the next commit
        )
        
        print("✅ Seq2SeqTrainer created successfully")
        return trainer
    
    def train(self) -> None:
        """
        Execute the training process.
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Build model and data
        self._build_model_and_data()
        
        # Create trainer
        self.trainer = self._create_trainer()
        
        # Record training start time
        start_time = time.time()
        
        # Start training
        print("\n🚀 Starting training...")
        train_result = self.trainer.train()
        
        # Record training end time
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Save final model
        print("Saving final model...")
        final_model_path = self.output_dir / 'final_model'
        self.trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training results
        training_info = {
            'training_time_seconds': training_time,
            'training_time_hours': training_time / 3600,
            'train_loss': train_result.training_loss,
            'train_steps': train_result.global_step,
            'model_path': str(final_model_path),
            'config': self.config
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training results saved to: {self.output_dir / 'training_results.json'}")
        print(f"Final model saved to: {final_model_path}")
        
        return train_result


def main():
    """
    Main training function.
    """
    # Load configuration
    config = load_config('configs/train_flant5_base_lora.yaml')
    
    # Create and run trainer
    trainer = BioLaySummTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
