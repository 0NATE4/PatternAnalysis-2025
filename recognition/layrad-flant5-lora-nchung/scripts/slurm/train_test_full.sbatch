#!/bin/bash -l

#SBATCH --job-name=t5_small_full_test
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.chung@student.uq.edu.au

set -euo pipefail

# Config for full fine-tuning test (override via: sbatch --export=ALL,EPOCHS=1,BS=2,... scripts/slurm/train_test_full.sbatch)
EPOCHS=${EPOCHS:-1}
BS=${BS:-2}  # Smaller batch size for full FT
LR=${LR:-5e-5}
STRATEGY=${STRATEGY:-full}
CONFIG=${CONFIG:-configs/train_t5_small_full.yaml}

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
OUT_ROOT="$PROJECT_ROOT/reports/test_run_full"

# Ensure directories exist
mkdir -p "$PROJECT_ROOT/logs" "$OUT_ROOT"/{curves,checkpoints}

# Set up HuggingFace cache on node local scratch for faster I/O
export HF_HOME="/scratch/$USER/hf_cache_test_full"
mkdir -p "$HF_HOME"

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set random seeds for reproducibility
export RANDOM_SEED=42
export PYTHONHASHSEED=42

# Debug: Check GPU and environment
echo "=== FULL FINE-TUNING TEST - Environment Check ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "HF Cache: $HF_HOME"
echo ""

# Install required packages (if needed)
echo "=== Installing Dependencies ==="
conda run -n torch pip install -q transformers datasets accelerate evaluate rouge-score

# Print configuration
echo "=== FULL FINE-TUNING TEST - Training Configuration ==="
echo "  Strategy: $STRATEGY"
echo "  Epochs: $EPOCHS (TEST MODE)"
echo "  Batch Size: $BS"
echo "  Learning Rate: $LR"
echo "  Config File: $CONFIG"
echo "  Random Seed: $RANDOM_SEED"
echo "  Output Root: $OUT_ROOT"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Test dataset loading and model initialization for full fine-tuning
echo "=== Testing Full Fine-Tuning Setup ==="
conda run -n torch python -c "
from src.dataset import BioLaySummDataset
from src.modules import FlanT5WithLoRA
import torch

print('Testing dataset loading...')
dataset = BioLaySummDataset('train', max_samples=50)  # Very small sample for speed
print(f'Dataset loaded: {len(dataset)} samples')

print('Testing T5-small model initialization...')
model = FlanT5WithLoRA('google/t5-small', task_type='SEQ_2_SEQ_LM')
print('T5-small model initialized successfully')

print('Testing full fine-tuning forward pass...')
sample = dataset[0]
inputs = model.tokenizer(sample['radiology_report'], return_tensors='pt', max_length=256, truncation=True)  # Shorter sequences
labels = model.tokenizer(sample['layman_report'], return_tensors='pt', max_length=256, truncation=True)

with torch.no_grad():
    outputs = model(**inputs, labels=labels['input_ids'])
    print(f'Forward pass successful - Loss: {outputs.loss.item():.4f}')

# Test gradient checkpointing
print('Testing gradient checkpointing setup...')
model.gradient_checkpointing_enable()
print('Gradient checkpointing enabled successfully')

print('✅ All full fine-tuning tests passed!')
"

# Run training with torchrun for better distributed training support
echo "=== Starting FULL FINE-TUNING TEST - T5-small Training ==="
conda run -n torch torchrun \
  --standalone \
  --nproc_per_node=1 \
  src/train.py \
  --config "$CONFIG" \
  --training.epochs "$EPOCHS" \
  --training.batch_size "$BS" \
  --training.learning_rate "$LR" \
  --training.strategy "$STRATEGY" \
  --training.output_dir "$OUT_ROOT/checkpoints" \
  --training.seed "$RANDOM_SEED" \
  --training.logging_steps 5 \
  --training.eval_steps 20 \
  --training.save_steps 50 \
  --training.evaluation_strategy steps \
  --training.save_strategy steps \
  --training.load_best_model_at_end false \
  --training.report_to none \
  --training.max_steps 25 \
  --training.gradient_checkpointing true

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ FULL FINE-TUNING TEST completed successfully!"
    
    # List output files
    echo "=== Output Files ==="
    ls -la "$OUT_ROOT/checkpoints/"
    
    # Check if model exists (should be pytorch_model.bin for full FT)
    if [ -f "$OUT_ROOT/checkpoints/pytorch_model.bin" ]; then
        echo "✅ Full model saved successfully (pytorch_model.bin)"
    else
        echo "⚠️  Warning: Model files not found"
        echo "Available files:"
        ls -la "$OUT_ROOT/checkpoints/"
    fi
    
    # Check memory usage and training efficiency
    echo "=== Training Efficiency Check ==="
    python -c "
import json
import os
try:
    if os.path.exists('$OUT_ROOT/checkpoints/trainer_state.json'):
        with open('$OUT_ROOT/checkpoints/trainer_state.json') as f:
            state = json.load(f)
            if 'log_history' in state:
                final_log = state['log_history'][-1]
                print(f'  Final Loss: {final_log.get(\"train_loss\", \"N/A\")}')
                print(f'  Final ROUGE-Lsum: {final_log.get(\"eval_rougeLsum\", \"N/A\")}')
                print(f'  Training Steps: {len(state.get(\"log_history\", []))}')
    else:
        print('  Trainer state not found')
except Exception as e:
    print(f'  Could not parse trainer state: {e}')
"
    
    # Skip evaluation for speed - just verify model files exist
    echo "✅ FULL FINE-TUNING TEST VALIDATION COMPLETE!"
    echo "Ready for full fine-tuning run on a100 partition"
    echo "Note: Evaluation skipped for speed - run eval_rouge.sbatch separately if needed"
    
else
    echo "❌ FULL FINE-TUNING TEST failed!"
    echo "Check the logs for errors before running full fine-tuning"
    exit 1
fi

echo "Full fine-tuning test job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
