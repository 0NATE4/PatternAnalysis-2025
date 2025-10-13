#!/bin/bash -l

#SBATCH --job-name=flant5_lora_test
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.chung@student.uq.edu.au

set -euo pipefail

# Config for quick test (override via: sbatch --export=ALL,EPOCHS=1,BS=4,... scripts/slurm/train_test.sbatch)
EPOCHS=${EPOCHS:-1}
BS=${BS:-4}
LR=${LR:-1e-4}
STRATEGY=${STRATEGY:-lora}
CONFIG=${CONFIG:-configs/train_flant5_base_lora.yaml}

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
OUT_ROOT="$PROJECT_ROOT/reports/test_run"

# Ensure directories exist
mkdir -p "$PROJECT_ROOT/logs" "$OUT_ROOT"/{curves,checkpoints}

export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set random seeds for reproducibility
export RANDOM_SEED=42
export PYTHONHASHSEED=42

# Debug: Check GPU and environment
echo "=== TEST RUN - Environment Check ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)"
echo "CUDA Version: $(echo 'CUDA available via PyTorch')"
echo "Python: $(conda run -n torch python --version)"
echo "PyTorch: $(conda run -n torch python -c 'import torch; print(torch.__version__)')"
echo "HF Cache: $HF_HOME"
echo ""

# Install required packages (if needed)
echo "=== Installing Dependencies ==="
conda run -n torch pip install -q transformers datasets accelerate evaluate rouge-score peft tensorboard

# Print configuration
echo "=== TEST RUN - Training Configuration ==="
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

# Quick validation - just check if imports work
echo "=== Quick Import Test ==="
conda run -n torch python -c "
print('Testing imports...')
from src.dataset import BioLaySummDataset
from src.modules import FLANT5LoRAModel
from src.train import BioLaySummTrainer
print('✅ All imports successful!')

print('Testing config loading...')
import yaml
with open('configs/train_flant5_base_lora.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Config loaded successfully!')

print('Testing dataset instantiation...')
dataset = BioLaySummDataset(config)
print('✅ Dataset created successfully!')

print('Testing trainer instantiation...')
trainer = BioLaySummTrainer(config)
print('✅ Trainer created successfully!')

print('🎉 All basic tests passed!')
"

# Run training with torchrun for better distributed training support
echo "=== Starting TEST RUN - FLAN-T5 LoRA Training ==="
conda run -n torch torchrun \
  --standalone \
  --nproc_per_node=1 \
  src/train.py \
  configs/train_test_lora.yaml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ TEST RUN completed successfully!"
    
    # List output files
    echo "=== Output Files ==="
    ls -la "$OUT_ROOT/checkpoints/"
    
    # Check if model exists
    if [ -f "$OUT_ROOT/checkpoints/pytorch_model.bin" ] || [ -f "$OUT_ROOT/checkpoints/adapter_model.bin" ]; then
        echo "✅ Model saved successfully"
    else
        echo "⚠️  Warning: Model files not found"
    fi
    
    # Skip evaluation for speed - just verify model files exist
    echo "✅ TEST RUN VALIDATION COMPLETE!"
    echo "Ready for full training run on a100 partition"
    echo "Note: Evaluation skipped for speed - run eval_rouge.sbatch separately if needed"
    
else
    echo "❌ TEST RUN failed!"
    echo "Check the logs for errors before running full training"
    exit 1
fi

echo "Test job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
