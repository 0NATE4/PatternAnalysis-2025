#!/bin/bash -l

#SBATCH --job-name=t5_small_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=50:00:00

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.chung@student.uq.edu.au

set -euo pipefail

# Config (override via: sbatch --export=ALL,MODEL_PATH=reports/checkpoints,... scripts/slurm/eval_rouge.sbatch)
MODEL_PATH=${MODEL_PATH:-reports/checkpoints}
CONFIG=${CONFIG:-configs/train_t5_small_full.yaml}
SPLIT=${SPLIT:-test}
MAX_SAMPLES=${MAX_SAMPLES:-1000}

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
OUT_ROOT="$MODEL_PATH/reports"

# Ensure directories exist
mkdir -p "$PROJECT_ROOT/logs" "$OUT_ROOT"

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
echo "=== Evaluation Environment Check ==="
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
echo "=== Evaluation Configuration ==="
echo "  Model Path: $MODEL_PATH"
echo "  Config File: $CONFIG"
echo "  Split: $SPLIT"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Output Root: $OUT_ROOT"
echo "  Random Seed: $RANDOM_SEED"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Resolve MODEL_PATH from CONFIG if not provided or missing
if [ ! -d "$MODEL_PATH" ]; then
    # Try to parse output.output_dir from YAML (simple grep/awk for this config structure)
    YAML_OUTPUT_DIR=$(awk '/^output:/{f=1;next} f && /output_dir:/{print $2; exit}' "$CONFIG" | tr -d '"')
    if [ -n "$YAML_OUTPUT_DIR" ] && [ -d "$YAML_OUTPUT_DIR" ]; then
        MODEL_PATH="$YAML_OUTPUT_DIR"
        echo "Resolved MODEL_PATH from CONFIG: $MODEL_PATH"
    else
        echo "❌ Model path not found and could not resolve from CONFIG."
        echo "MODEL_PATH attempted: $MODEL_PATH"
        exit 1
    fi
fi

echo "=== Model Check ==="
if [ -f "$MODEL_PATH/adapter_model.bin" ]; then
    echo "✅ LoRA adapter model found"
elif [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    echo "✅ Full model found"
else
    echo "⚠️  No model files found in $MODEL_PATH"
    echo "Available files:"
    ls -la "$MODEL_PATH/"
fi

# Promote saved weights from final_model/ if root is missing files
if [ ! -f "$MODEL_PATH/adapter_config.json" ] && [ -f "$MODEL_PATH/final_model/adapter_config.json" ]; then
    echo "Promoting LoRA adapter from $MODEL_PATH/final_model to $MODEL_PATH"
    cp -r "$MODEL_PATH/final_model/"* "$MODEL_PATH"/
elif [ ! -f "$MODEL_PATH/pytorch_model.bin" ] && [ -f "$MODEL_PATH/final_model/pytorch_model.bin" ]; then
    echo "Promoting full model from $MODEL_PATH/final_model to $MODEL_PATH"
    cp -r "$MODEL_PATH/final_model/"* "$MODEL_PATH"/
fi

# Run evaluation
echo "=== Starting ROUGE Evaluation ==="
conda run -n torch python src/eval_runner.py \
  "$CONFIG"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    
    # List output files
    echo "=== Evaluation Results ==="
    ls -la "$OUT_ROOT/"
    
    # Display key metrics
    if [ -f "$OUT_ROOT/evaluation_results.json" ]; then
        echo "✅ Results saved: evaluation_results.json"
        echo "Key metrics:"
        python -c "
import json
try:
    with open('$OUT_ROOT/evaluation_results.json') as f:
        results = json.load(f)
        print(f'  ROUGE-1: {results.get(\"rouge1\", \"N/A\")}')
        print(f'  ROUGE-2: {results.get(\"rouge2\", \"N/A\")}')
        print(f'  ROUGE-L: {results.get(\"rougeL\", \"N/A\")}')
        print(f'  ROUGE-Lsum: {results.get(\"rougeLsum\", \"N/A\")}')
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
    # Check predictions
    if [ -f "$OUT_ROOT/predictions.jsonl" ]; then
        echo "✅ Predictions saved: predictions.jsonl"
        echo "Sample predictions:"
        head -3 "$OUT_ROOT/predictions.jsonl"
    fi
    
else
    echo "❌ Evaluation failed!"
    exit 1
fi

echo "Evaluation job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
