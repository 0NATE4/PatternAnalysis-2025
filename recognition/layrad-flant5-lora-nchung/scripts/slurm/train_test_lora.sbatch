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
conda run -n torch pip install -q transformers datasets accelerate evaluate rouge-score peft

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

# Test dataset loading and model initialization
echo "=== Testing Dataset Loading ==="
conda run -n torch python -c "
from src.dataset import BioLaySummDataset
from src.modules import FlanT5WithLoRA
import torch

print('Testing dataset loading...')
dataset = BioLaySummDataset('train', max_samples=50)  # Very small sample for speed
print(f'Dataset loaded: {len(dataset)} samples')

print('Testing model initialization...')
model = FlanT5WithLoRA('google/flan-t5-base', task_type='SEQ_2_SEQ_LM')
print('Model initialized successfully')

print('Testing forward pass...')
sample = dataset[0]
inputs = model.tokenizer(sample['radiology_report'], return_tensors='pt', max_length=256, truncation=True)  # Shorter sequences
labels = model.tokenizer(sample['layman_report'], return_tensors='pt', max_length=256, truncation=True)

with torch.no_grad():
    outputs = model(**inputs, labels=labels['input_ids'])
    print(f'Forward pass successful - Loss: {outputs.loss.item():.4f}')

print('✅ All tests passed!')
"

# Run training with torchrun for better distributed training support
echo "=== Starting TEST RUN - FLAN-T5 LoRA Training ==="
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
  --training.max_steps 25

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
