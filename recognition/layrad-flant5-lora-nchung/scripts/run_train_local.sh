#!/bin/bash

# Training script for FLAN-T5 LoRA on BioLaySumm dataset
# Author: Nathan Chung
# Course: COMP3710 Pattern Analysis

set -e  # Exit on any error

echo "============================================================"
echo "FLAN-T5 LoRA Training on BioLaySumm Dataset"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "src/train.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected structure: recognition/layrad-flant5-lora-nchung/"
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   Please activate your conda environment: conda activate biolaysumm"
    echo "   Continuing anyway..."
fi

# Check if config file exists
if [ ! -f "configs/train_flant5_base_lora.yaml" ]; then
    echo "❌ Error: Configuration file not found: configs/train_flant5_base_lora.yaml"
    exit 1
fi

# Display configuration
echo "📋 Configuration:"
echo "   - Model: FLAN-T5-Base with LoRA"
echo "   - Dataset: BioLaySumm Expert-to-Layperson Translation"
echo "   - Config: configs/train_flant5_base_lora.yaml"
echo "   - Output: ./checkpoints/flan-t5-base-lora-biolaysumm/"

# Check available resources
echo ""
echo "🖥️  System Resources:"
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
else
    echo "   GPU: Not available (using CPU)"
fi

echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Check if CUDA is available
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "   CUDA: Available ✅"
else
    echo "   CUDA: Not available (CPU training) ⚠️"
fi

echo ""
echo "🚀 Starting training..."

# Run training
python src/train.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📁 Output files:"
    echo "   - Model checkpoints: ./checkpoints/flan-t5-base-lora-biolaysumm/"
    echo "   - Training logs: ./checkpoints/flan-t5-base-lora-biolaysumm/logs/"
    echo "   - Final model: ./checkpoints/flan-t5-base-lora-biolaysumm/final_model/"
    echo "   - Results: ./checkpoints/flan-t5-base-lora-biolaysumm/training_results.json"
    echo ""
    echo "Next steps:"
    echo "   1. Run evaluation: bash scripts/run_eval_local.sh"
    echo "   2. Generate predictions: bash scripts/run_predict_local.sh"
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi
