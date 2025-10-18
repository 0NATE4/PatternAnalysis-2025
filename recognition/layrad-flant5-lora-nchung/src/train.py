"""
Training script for FLAN-T5 LoRA on BioLaySumm dataset.

This module implements the training loop using HuggingFace's Seq2SeqTrainer
with proper configuration, metrics, and checkpointing for the expert-to-layperson
radiology report translation task.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

# Set multiprocessing start method first thing to avoid CUDA fork issues
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import time
import json
import torch

# Disable HF datasets multiprocessing entirely
os.environ["HF_DATASETS_DISABLE_MP"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# A100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import evaluate as evaluate_lib
import numpy as np

# Preflight check: ensure we have the real Hugging Face evaluate package
import evaluate as _ev
import sys
ev_path = getattr(_ev, "__file__", None)
if not hasattr(_ev, "load"):
    raise ImportError(
        f"'evaluate' resolved to {ev_path}. "
        f"This is not Hugging Face evaluate. "
        f"Rename any local file or folder named 'evaluate'. "
        f"sys.path[0] is {sys.path[0]}"
    )
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from datasets import Dataset

# Handle imports for both direct execution and module import
try:
    from .utils import (
        load_config, setup_reproducibility, get_device, create_output_dir, save_config,
        setup_logging, log_training_arguments, log_trainer_state, log_training_summary
    )
    from .dataset import BioLaySummDataset
    from .modules import build_model_with_lora
except ImportError:
    # Direct execution - add current directory to path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from utils import (
        load_config, setup_reproducibility, get_device, create_output_dir, save_config,
        setup_logging, log_training_arguments, log_trainer_state, log_training_summary
    )
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
        
        # Setup logging and reports directory
        self.reports_dir = setup_logging(self.output_dir)
        
        # Save configuration
        save_config(self.config, self.output_dir / 'training_config.yaml')
        
        print(f"Training setup complete. Output directory: {self.output_dir}")
    
    def _validate_training_strategy(self) -> str:
        """
        Validate and determine the training strategy from configuration.
        
        Returns:
            str: 'lora' or 'full'
            
        Raises:
            ValueError: If strategy is invalid or configuration is inconsistent
        """
        # Get strategy from training config
        training_strategy = self.config.get('training', {}).get('strategy', 'lora')
        
        # Get full fine-tuning flag (backward compatibility)
        full_finetuning_enabled = self.config.get('full_finetuning', {}).get('enabled', False)
        
        # Validate strategy
        valid_strategies = {'lora', 'full'}
        if training_strategy not in valid_strategies:
            raise ValueError(f"Invalid training strategy: {training_strategy}. Must be one of {valid_strategies}")
        
        # Check for configuration consistency
        if training_strategy == 'full' and not full_finetuning_enabled:
            print("⚠️  Warning: training.strategy='full' but full_finetuning.enabled=False")
            print("   Setting full_finetuning.enabled=True for consistency")
            self.config.setdefault('full_finetuning', {})['enabled'] = True
            
        elif training_strategy == 'lora' and full_finetuning_enabled:
            print("⚠️  Warning: training.strategy='lora' but full_finetuning.enabled=True")
            print("   Setting full_finetuning.enabled=False for consistency")
            self.config.setdefault('full_finetuning', {})['enabled'] = False
        
        # Strategy validation based on model
        model_name = self.config.get('model', {}).get('name', '')
        
        if training_strategy == 'full':
            # Full fine-tuning recommendations
            if 'flan-t5-base' in model_name.lower():
                print("⚠️  Warning: Full fine-tuning FLAN-T5-base requires significant memory")
                print("   Consider using T5-small or enabling gradient checkpointing")
            
            # Check for gradient checkpointing
            gradient_checkpointing = self.config.get('full_finetuning_settings', {}).get('gradient_checkpointing', False)
            if not gradient_checkpointing:
                print("⚠️  Warning: Full fine-tuning without gradient checkpointing may cause OOM")
                print("   Consider enabling gradient_checkpointing in full_finetuning_settings")
        
        print(f"✅ Training strategy validated: {training_strategy}")
        return training_strategy
        
    def _build_model_and_data(self) -> None:
        """
        Build model and load datasets for training.
        """
        print("\nBuilding model and loading datasets...")
        
        # Validate and determine training strategy
        training_strategy = self._validate_training_strategy()
        
        # Initialize dataset loader first (before loading model to avoid CUDA fork issues)
        self.dataset_loader = BioLaySummDataset(self.config)
        
        # Load datasets
        print("Loading training dataset...")
        train_dataset = self.dataset_loader.load_data('train')
        
        print("Loading validation dataset...")
        val_dataset = self.dataset_loader.load_data('validation')
        
        # Load model and tokenizer after dataset loading (to avoid CUDA fork issues)
        if training_strategy == 'full':
            print("🔧 Using FULL FINE-TUNING strategy")
            self.model_wrapper = self._build_full_finetuning_model()
        else:
            print("🔧 Using LoRA strategy")
            self.model_wrapper = build_model_with_lora(self.config)
        
        model, tokenizer = self.model_wrapper.get_model_and_tokenizer()
        
        # Print parameter information
        self.model_wrapper.count_params()
        
        # Tokenize datasets for training
        print("Tokenizing training dataset...")
        train_dataset = train_dataset.map(
            lambda examples: self.dataset_loader.preprocess_function(examples, tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=["input_text", "target_text", "source", "images_path"],
            desc="Tokenizing training dataset"
        )
        
        print("Tokenizing validation dataset...")
        val_dataset = val_dataset.map(
            lambda examples: self.dataset_loader.preprocess_function(examples, tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=["input_text", "target_text", "source", "images_path"],
            desc="Tokenizing validation dataset"
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Diagnostic probe to verify spawn method and CUDA initialization order
        print("Start method:", mp.get_start_method())
        print("About to load model. CUDA initialised:", torch.cuda.is_initialized())
        
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
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id  # Required for T5 encoder-decoder generation
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
            learning_rate=float(training_config.get('learning_rate', 1e-4)),
            weight_decay=float(training_config.get('weight_decay', 0.01)),
            max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
            
            # Learning rate scheduling
            warmup_steps=training_config.get('warmup_steps', 500),
            lr_scheduler_type="linear",
            
            # Mixed precision
            fp16=False,  # Use bf16 instead
            bf16=self.config.get('training', {}).get('bf16', True),
            
            # Gradient checkpointing (memory optimization for full fine-tuning)
            gradient_checkpointing=self._should_enable_gradient_checkpointing(),
            
            # Evaluation
            eval_strategy="steps",
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
            
            # Dataset handling
            remove_unused_columns=False,  # Keep custom dataset columns
            
            # Performance
            dataloader_num_workers=0,  # Disable multiprocessing to avoid CUDA fork issues
            dataloader_pin_memory=self.config.get('hardware', {}).get('pin_memory', True),
            
            # Generation for evaluation
            predict_with_generate=True,  # Use generation for evaluation
            generation_config=self._create_generation_config(),
            
            # Note: Early stopping parameters not supported in transformers 4.30.0
            # early_stopping_patience=training_config.get('early_stopping_patience', 3),
            # early_stopping_threshold=training_config.get('early_stopping_threshold', 0.001),
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
        
        # Set tokenizer for ROUGE computation
        compute_rouge_metrics.tokenizer = self.tokenizer
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_rouge_metrics,
        )
        
        print("✅ Seq2SeqTrainer created successfully")
        print("✅ ROUGE metrics integration enabled")
        print("   - rouge1, rouge2, rougeL, rougeLsum")
        print(f"   - Best model metric: eval_rougeLsum")
        
        # Log training arguments with strategy information
        log_training_arguments(training_args, self.reports_dir)
        self._log_strategy_info()
        
        return trainer
    
    def _build_full_finetuning_model(self):
        """
        Build model for full fine-tuning (no LoRA).
        
        Returns:
            Model wrapper for full fine-tuning
        """
        from modules import build_model_with_full_finetuning
        
        # Create a proper full fine-tuning model wrapper
        model_wrapper = build_model_with_full_finetuning(self.config)
        
        # Enable gradient checkpointing if specified
        full_ft_config = self.config.get('full_finetuning', {})
        if full_ft_config.get('gradient_checkpointing', False):
            model_wrapper.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
        
        return model_wrapper
    
    def _log_strategy_info(self) -> None:
        """
        Log training strategy information to reports directory.
        """
        import json
        import pandas as pd
        from pathlib import Path
        
        strategy_info = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_strategy': self.config.get('training', {}).get('strategy', 'lora'),
            'full_finetuning_enabled': self.config.get('full_finetuning', {}).get('enabled', False),
            'model_name': self.config.get('model', {}).get('name', 'unknown'),
            'model_config': {
                'torch_dtype': self.config.get('model', {}).get('torch_dtype', 'unknown'),
            },
            'training_config': {
                'batch_size': self.config.get('training', {}).get('batch_size', 'unknown'),
                'learning_rate': self.config.get('training', {}).get('learning_rate', 'unknown'),
                'num_epochs': self.config.get('training', {}).get('num_epochs', 'unknown'),
                'gradient_accumulation_steps': self.config.get('training', {}).get('gradient_accumulation_steps', 'unknown'),
            },
            'lora_config': self.config.get('lora', {}),
            'full_finetuning_config': self.config.get('full_finetuning', {}),
            'full_finetuning_settings': self.config.get('full_finetuning_settings', {}),
        }
        
        strategy_path = self.reports_dir / 'training_strategy.json'
        with open(strategy_path, 'w', encoding='utf-8') as f:
            json.dump(strategy_info, f, indent=2, ensure_ascii=False)
        
        print(f"Training strategy logged to: {strategy_path}")
    
    def _should_enable_gradient_checkpointing(self) -> bool:
        """
        Determine if gradient checkpointing should be enabled based on configuration.
        
        Gradient checkpointing trades computation for memory by recomputing activations
        during backward pass instead of storing them. Essential for full fine-tuning
        large models on limited GPU memory.
        
        Returns:
            bool: True if gradient checkpointing should be enabled
        """
        # Check if full fine-tuning is enabled
        training_strategy = self.config.get('training', {}).get('strategy', 'lora')
        full_finetuning_enabled = self.config.get('full_finetuning', {}).get('enabled', False)
        
        is_full_finetuning = (training_strategy == 'full' or full_finetuning_enabled)
        
        if not is_full_finetuning:
            # LoRA doesn't need gradient checkpointing - only trains adapter weights
            return False
        
        # Check explicit gradient checkpointing setting
        training_config = self.config.get('training', {})
        full_ft_config = self.config.get('full_finetuning', {})
        full_ft_settings = self.config.get('full_finetuning_settings', {})
        
        # Priority order: training > full_finetuning_settings > full_finetuning > default
        # Default to True for full FT to prevent OOM errors
        gradient_checkpointing = (
            training_config.get('gradient_checkpointing',
            full_ft_settings.get('gradient_checkpointing', 
            full_ft_config.get('gradient_checkpointing', True)))
        )
        
        if gradient_checkpointing:
            print("✅ Gradient checkpointing enabled for full fine-tuning")
            print("   - Memory usage reduced (trades compute for memory)")
            print("   - Training will be ~20% slower but use less VRAM")
        else:
            print("⚠️  Gradient checkpointing disabled for full fine-tuning")
            print("   - Higher memory usage but faster training")
            print("   - May cause OOM errors with large models")
        
        return gradient_checkpointing
    
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
        
        # Log trainer state after training
        log_trainer_state(self.trainer, self.reports_dir)
        
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
        
        # Log comprehensive training summary
        model_info = self.model_wrapper.count_params()
        log_training_summary(self.config, model_info, training_time, self.reports_dir)
        
        print(f"Training results saved to: {self.output_dir / 'training_results.json'}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Reports and logs saved to: {self.reports_dir}")
        
        return train_result


# Global ROUGE metric (loaded once to avoid repeated loading during evaluation)
_ROUGE_METRIC = None

def _get_rouge_metric():
    """
    Lazy load ROUGE metric to avoid repeated loading and scope issues.
    
    This function ensures the ROUGE metric is loaded only once and reused
    across all evaluation calls, preventing AttributeError with torchrun.
    """
    global _ROUGE_METRIC
    if _ROUGE_METRIC is None:
        from evaluate import load as hf_load
        _ROUGE_METRIC = hf_load('rouge')
    return _ROUGE_METRIC


def compute_rouge_metrics(eval_preds) -> Dict[str, float]:
    """
    Compute ROUGE metrics for evaluation.
    
    This function implements the standard ROUGE evaluation protocol for sequence-to-sequence
    models, handling token ID validation, label masking, and metric computation.
    
    Args:
        eval_preds: Evaluation predictions from HuggingFace Trainer
            - predictions: Generated token IDs (or logits if predict_with_generate=False)
            - label_ids: Reference token IDs with -100 for padding
    
    Returns:
        Dict containing ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores
    """
    import numpy as np
    
    predictions, labels = eval_preds
    
    # Get tokenizer from global scope (will be set by trainer)
    tokenizer = getattr(compute_rouge_metrics, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("Tokenizer not set for ROUGE computation")
    
    # Some trainers return a tuple (predictions, past_key_values)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy arrays for robust handling
    preds = np.asarray(predictions)
    
    # Debug log on rank 0
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Predictions shape/dtype: {preds.shape}, {preds.dtype}", flush=True)
    
    # If predictions are logits (3D) or floats, convert to token IDs via argmax
    # This handles cases where the model outputs probability distributions
    if preds.ndim == 3 or not np.issubdtype(preds.dtype, np.integer):
        preds = preds.argmax(axis=-1)
    
    # Ensure we have int64 for safe operations
    pred_ids = preds.astype(np.int64, copy=False)
    
    # Get pad token ID and vocab size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        vocab_size = int(pred_ids.max() + 1)
    
    # Clamp invalid token IDs to pad_id (preserves sequence length, avoids OverflowError)
    # This prevents crashes from out-of-vocabulary tokens that can occur during generation
    pred_ids = np.where((pred_ids >= 0) & (pred_ids < vocab_size), pred_ids, pad_id)
    
    # Handle labels: replace -100 with pad_id
    # -100 is PyTorch's special token for ignored positions in loss computation
    # We replace it with pad_id for proper text decoding
    labels = np.asarray(labels)
    labels = np.where(labels != -100, labels, pad_id)
    
    # Batch decode for efficiency
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    # Use pre-loaded ROUGE metric
    rouge = _get_rouge_metric()
    
    # Compute ROUGE metrics (following radadapt pattern)
    # ROUGE measures n-gram overlap between generated and reference text
    rouge_results = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True  # Use stemming for better word matching
    )
    
    # Extract and scale scores to percentages
    metrics = {
        'rouge1': round(rouge_results['rouge1'] * 100, 4),
        'rouge2': round(rouge_results['rouge2'] * 100, 4),
        'rougeL': round(rouge_results['rougeL'] * 100, 4),
        'rougeLsum': round(rouge_results['rougeLsum'] * 100, 4)
    }
    
    # Add average generation length as diagnostic
    metrics['gen_len'] = float((pred_ids != pad_id).sum(axis=1).mean())
    
    return metrics


def main():
    """
    Main training function.
    """
    import sys
    
    # Get config file from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/train_flant5_base_lora.yaml'
    
    # Load configuration
    config = load_config(config_file)
    
    # Log evaluate package location on rank 0 for debugging
    if int(os.environ.get("RANK", "0")) == 0:
        import evaluate as _ev
        print(f"Using evaluate from: {getattr(_ev, '__file__', None)}", flush=True)
    
    # Create and run trainer
    trainer = BioLaySummTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
