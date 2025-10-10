# LaYRaD-FlanT5-LoRA

Fine-tuning FlanT5 models using LoRA (Low-Rank Adaptation) for recognition tasks.

## Overview

This project implements LoRA-based fine-tuning for Google's FlanT5 models, providing efficient parameter-efficient training for text generation and recognition tasks.

## Project Structure

```
recognition/
  layrad-flant5-lora-nchung/
    README.md                    # Project documentation
    requirements.txt             # Python dependencies
    configs/                     # Configuration files
      train_flant5_base_lora.yaml
      train_t5_small_full.yaml
      rouge_eval.yaml
    src/                         # Source code
      __init__.py
      modules.py                 # Model components and LoRA modules
      dataset.py                 # Data loading and preprocessing
      train.py                   # Training script
      predict.py                 # Inference script
      utils.py                   # Utility functions
      metrics.py                 # Evaluation metrics
    scripts/                     # Execution scripts
      run_train_local.sh         # Local training script
      run_eval_local.sh          # Local evaluation script
      slurm/                     # SLURM cluster scripts
        train_flant5_base_lora.sbatch
        eval_rouge.sbatch
    tests/                       # Unit tests
      test_dataset.py
      test_modules.py
      test_inference.py
    reports/                     # Results and outputs
      examples.jsonl             # Example predictions
      curves/                    # Training curves and plots
```
