#!/bin/bash -l

#SBATCH --job-name=zeroshot_baseline
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

# Config (override via: sbatch --export=ALL,CONFIG=...,MAX_SAMPLES=... scripts/slurm/eval_zeroshot_baseline.sbatch)
CONFIG=${CONFIG:-configs/train_flant5_base_lora.yaml}
MAX_SAMPLES=${MAX_SAMPLES:-1000}  # Default to 1000 for faster testing
SPLIT=${SPLIT:-test}

# Project paths
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
OUT_ROOT="$PROJECT_ROOT/checkpoints/zeroshot_baseline"

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
echo "=== Zero-Shot Baseline Environment Check ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)"
echo "CUDA Version: $(echo 'CUDA available via PyTorch')"
echo "Python: $(conda run -n torch python --version)"
echo "PyTorch: $(conda run -n torch python -c 'import torch; print(torch.__version__)')"
echo "HF Cache: $HF_HOME"
echo ""

# Install required packages (if needed)
echo "=== Installing Dependencies ==="
conda run -n torch pip install -q transformers datasets accelerate evaluate rouge-score tensorboard

# Print configuration
echo "=== Zero-Shot Baseline Configuration ==="
echo "  Config File: $CONFIG"
echo "  Split: $SPLIT"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Output Root: $OUT_ROOT"
echo "  Random Seed: $RANDOM_SEED"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Run zero-shot baseline evaluation
echo "=== Starting Zero-Shot Baseline Evaluation ==="
conda run -n torch python src/zeroshot_baseline.py \
  --config "$CONFIG" \
  --max_samples "$MAX_SAMPLES"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Zero-shot baseline evaluation completed successfully!"
    
    # List output files
    echo "=== Zero-Shot Baseline Results ==="
    ls -la "$OUT_ROOT/reports/"
    
    # Display key metrics
    if [ -f "$OUT_ROOT/reports/zeroshot_baseline_results.json" ]; then
        echo "✅ Results saved: zeroshot_baseline_results.json"
        echo "Zero-shot baseline metrics:"
        python -c "
import json
try:
    with open('$OUT_ROOT/reports/zeroshot_baseline_results.json') as f:
        results = json.load(f)
        rouge_metrics = results.get('rouge_metrics', {})
        print(f'  ROUGE-1:  {rouge_metrics.get(\"rouge1\", \"N/A\")}')
        print(f'  ROUGE-2:  {rouge_metrics.get(\"rouge2\", \"N/A\")}')
        print(f'  ROUGE-L:  {rouge_metrics.get(\"rougeL\", \"N/A\")}')
        print(f'  ROUGE-Lsum: {rouge_metrics.get(\"rougeLsum\", \"N/A\")}')
        print(f'  Samples:  {results.get(\"num_samples\", \"N/A\")}')
        print(f'  Model:    {results.get(\"model_name\", \"N/A\")}')
        print(f'  Baseline: {results.get(\"baseline_type\", \"N/A\")}')
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
    echo ""
    echo "📊 COMPARISON NOTE:"
    echo "Compare these zero-shot baseline scores with your fine-tuned model results"
    echo "to measure the improvement from training. Run eval_rouge.sbatch to get"
    echo "fine-tuned model performance for comparison."
    
else
    echo "❌ Zero-shot baseline evaluation failed!"
    exit 1
fi

echo "Zero-shot baseline job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
