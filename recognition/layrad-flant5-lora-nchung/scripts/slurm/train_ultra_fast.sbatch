#!/bin/bash
#SBATCH --job-name=ultra_fast_test
#SBATCH --output=logs/ultra_fast_test_%j.out
#SBATCH --error=logs/ultra_fast_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Ultra-Fast Test Script (15 minutes max)
# This script does the absolute minimum to validate the training pipeline

echo "=== ULTRA-FAST TEST RUN ==="
echo "Time limit: 15 minutes"
echo "Samples: 20 (train) + 10 (val)"
echo "Steps: 5 training steps only"
echo ""

# Set environment variables
export HF_HOME=$HOME/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Environment check
echo "=== Environment Check ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)"
echo "CUDA Version: $(echo 'CUDA available via PyTorch')"
echo "Python: $(conda run -n torch python --version)"
echo "PyTorch: $(conda run -n torch python -c 'import torch; print(torch.__version__)')"
echo "HF Cache: $HF_HOME"
echo ""

# Change to project directory
cd /home/Student/s4800977/comp3710/a3

# Ultra-quick validation - just check if everything loads
echo "=== Ultra-Quick Validation ==="
conda run -n torch python -c "
print('Testing imports...')
from src.dataset import BioLaySummDataset
from src.modules import FLANT5LoRAModel
from src.train import BioLaySummTrainer
print('✅ All imports successful!')

print('Testing config loading...')
import yaml
with open('configs/train_ultra_fast.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Config loaded successfully!')

print('Testing dataset instantiation...')
dataset = BioLaySummDataset(config)
print('✅ Dataset created successfully!')

print('Testing trainer instantiation...')
trainer = BioLaySummTrainer(config)
print('✅ Trainer created successfully!')

print('🎉 All ultra-fast tests passed!')
"

# Run ultra-fast training
echo "=== Starting ULTRA-FAST Training ==="
conda run -n torch torchrun \
  --standalone \
  --nproc_per_node=1 \
  src/train.py \
  configs/train_ultra_fast.yaml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ ULTRA-FAST TRAINING completed successfully!"
    echo "🎉 Pipeline validation complete!"
    echo ""
    echo "Ready for full training runs!"
else
    echo "❌ ULTRA-FAST TRAINING failed"
    echo "Check the error logs above"
fi
