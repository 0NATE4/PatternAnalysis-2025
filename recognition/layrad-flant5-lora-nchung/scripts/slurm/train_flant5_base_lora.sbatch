#!/bin/bash -l

#SBATCH --job-name=flant5_lora_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.chung@student.uq.edu.au

set -euo pipefail

# Config (override via: sbatch --export=ALL,EPOCHS=3,BS=8,LR=1e-4,... scripts/slurm/train_flant5_base_lora.sbatch)
EPOCHS=${EPOCHS:-3}
BS=${BS:-8}
LR=${LR:-1e-4}
STRATEGY=${STRATEGY:-lora}
CONFIG=${CONFIG:-configs/train_flant5_base_lora.yaml}

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
OUT_ROOT="$PROJECT_ROOT/reports"

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
echo "=== Environment Check ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)"
echo "CUDA Version: $(echo 'CUDA available via PyTorch')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "HF Cache: $HF_HOME"
echo ""

# Install required packages (if needed)
echo "=== Installing Dependencies ==="
conda run -n torch pip install -q transformers datasets accelerate evaluate rouge-score peft tensorboard

# Print configuration
echo "=== Training Configuration ==="
echo "  Strategy: $STRATEGY"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BS"
echo "  Learning Rate: $LR"
echo "  Config File: $CONFIG"
echo "  Random Seed: $RANDOM_SEED"
echo "  Output Root: $OUT_ROOT"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Run training with torchrun for better distributed training support
echo "=== Starting FLAN-T5 LoRA Training ==="
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
  --training.logging_steps 100 \
  --training.eval_steps 500 \
  --training.save_steps 1000 \
  --training.evaluation_strategy steps \
  --training.save_strategy steps \
  --training.load_best_model_at_end true \
  --training.metric_for_best_model rougeLsum \
  --training.greater_is_better true \
  --training.report_to none

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # List output files
    echo "=== Output Files ==="
    ls -la "$OUT_ROOT/checkpoints/"
    
    # Check if best model exists
    if [ -f "$OUT_ROOT/checkpoints/pytorch_model.bin" ] || [ -f "$OUT_ROOT/checkpoints/adapter_model.bin" ]; then
        echo "✅ Model saved successfully"
    else
        echo "⚠️  Warning: Model files not found"
    fi
    
    # Check training logs
    if [ -f "$OUT_ROOT/checkpoints/trainer_state.json" ]; then
        echo "✅ Training state saved: trainer_state.json"
        echo "Final metrics:"
        python -c "
import json
try:
    with open('$OUT_ROOT/checkpoints/trainer_state.json') as f:
        state = json.load(f)
        if 'log_history' in state:
            final_log = state['log_history'][-1]
            print(f'  Final Loss: {final_log.get(\"train_loss\", \"N/A\")}')
            print(f'  Final ROUGE-Lsum: {final_log.get(\"eval_rougeLsum\", \"N/A\")}')
except Exception as e:
    print(f'Could not parse trainer state: {e}')
"
    fi
    
    # Run evaluation on test set
    echo "=== Running Final Evaluation ==="
    conda run -n torch python src/evaluate.py \
      --config "$CONFIG" \
      --model_path "$OUT_ROOT/checkpoints" \
      --output_dir "$OUT_ROOT" \
      --split test
    
else
    echo "❌ Training failed!"
    exit 1
fi

echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
