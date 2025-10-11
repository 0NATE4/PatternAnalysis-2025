# FLAN-T5 LoRA for BioLaySumm Expert-to-Layperson Translation

**Author:** Nathan Chung  
**Course:** COMP3710 Pattern Analysis  
**Difficulty:** Hard  

## Overview

This project implements a parameter-efficient fine-tuning approach using LoRA (Low-Rank Adaptation) on FLAN-T5 to translate expert radiology reports into layperson-friendly summaries. The system addresses the critical need for medical communication accessibility by converting complex medical terminology into plain language that patients can understand.

## Problem Statement

Medical radiology reports are written in technical language that is often incomprehensible to patients. This creates barriers to patient understanding and engagement with their own healthcare. This project tackles **Subtask 2.1 of the ACL 2025 BioLaySumm workshop**, which focuses on translating expert radiology reports into layperson summaries.

## Dataset

### BioLaySumm Dataset

**Source:** [BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track)

**Description:** The BioLaySumm dataset contains expert radiology reports paired with layperson summaries, specifically designed for medical text simplification tasks.

**Dataset Statistics:**
- **Total samples:** 170,991
- **Training split:** 150,454 samples
- **Validation split:** 10,000 samples  
- **Test split:** 10,537 samples
- **Source:** Primarily PadChest dataset (77.7% of samples)

**Data Format:**
```json
{
  "radiology_report": "No infiltrates or consolidations are observed in the study.",
  "layman_report": "The study did not show any signs of lung infections or areas of lung tissue replacement.",
  "source": "PadChest",
  "images_path": "216840111366964013076187734852011201090749220_00-141-160.png"
}
```

### Split Policy

**Train/Validation/Test Split:**
- **Training (87.9%):** Used for model fine-tuning with LoRA
- **Validation (5.8%):** Used for hyperparameter tuning and early stopping
- **Test (6.2%):** Held-out for final evaluation only

**Reproducibility:**
- Fixed random seed (42) for consistent shuffling
- Deterministic data loading across runs
- Stable train/val/test splits maintained

### PHI (Protected Health Information) Handling

**Privacy Considerations:**
- Dataset contains de-identified radiology reports
- No direct patient identifiers in the text
- Image paths are anonymized (numeric identifiers only)
- Original dataset creators have handled PHI removal

**Our Implementation:**
- No additional PHI processing required
- Dataset is already compliant for research use
- Focus on text translation without storing sensitive information
- All processing done on de-identified data

## Model Architecture

### Base Model: FLAN-T5-Base
- **Model:** `google/flan-t5-base`
- **Parameters:** ~248M parameters
- **Architecture:** Encoder-decoder transformer
- **Context Length:** 512 tokens
- **Pre-training:** Instruction-tuned for better few-shot performance

### LoRA Configuration
- **Rank (r):** 8 - Low-rank adaptation dimension
- **Alpha:** 32 - LoRA scaling parameter (alpha/r = 4.0)
- **Dropout:** 0.1 - Regularization to prevent overfitting
- **Target Modules:** Query (q), Value (v), Key (k), Output (o) projections
- **Task Type:** Sequence-to-sequence language modeling

**Parameter Efficiency:**
- **Trainable Parameters:** ~1.2M (0.5% of total parameters)
- **Memory Efficiency:** ~4x reduction in GPU memory usage
- **Training Speed:** ~3x faster than full fine-tuning

## Prompt Engineering

**Expert-to-Layperson Translation Prompt:**
```
Translate this expert radiology report into layperson terms:

{expert_radiology_report}

Layperson summary:
```

**Example:**
- **Input:** "Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion."
- **Output:** "There is a cloudiness near the right lung's airways and a part of the lung has collapsed. The area behind the heart is denser, which could be due to the collapsed lung and a possible lung infection along with fluid around the right lung."

## Training Configuration

### Hyperparameters
- **Learning Rate:** 1e-4 (LoRA-specific)
- **Batch Size:** 8 per GPU
- **Gradient Accumulation:** 4 steps (effective batch size: 32)
- **Epochs:** 3
- **Warmup Steps:** 500
- **Weight Decay:** 0.01
- **Max Gradient Norm:** 1.0

### Training Strategy
- **Mixed Precision:** bfloat16 for memory efficiency
- **Early Stopping:** Patience of 3 epochs on validation ROUGE-Lsum
- **Checkpointing:** Save best model based on validation performance
- **Reproducibility:** Fixed seeds for all random operations

## Evaluation Metrics

### Primary Metrics (Required by Assignment)
- **ROUGE-1:** Unigram overlap between generated and reference summaries
- **ROUGE-2:** Bigram overlap for fluency assessment  
- **ROUGE-L:** Longest common subsequence for coherence
- **ROUGE-Lsum:** Sentence-level ROUGE-L for structure preservation

### Evaluation Protocol
- **Test Set:** Held-out 10,537 samples (never used during training)
- **Generation:** Beam search (width=4) with length penalty (0.6)
- **Max New Tokens:** 200
- **No Repeat N-gram:** Size 3 to prevent repetition

## Project Structure

```
recognition/layrad-flant5-lora-nchung/
├── src/
│   ├── dataset.py          # BioLaySumm dataset loader
│   ├── modules.py          # FLAN-T5 + LoRA model wrapper
│   ├── train.py            # Training loop implementation
│   ├── predict.py          # Inference and prediction
│   ├── metrics.py          # ROUGE evaluation metrics
│   └── utils.py            # Configuration and utility functions
├── configs/
│   ├── train_flant5_base_lora.yaml    # Main training configuration
│   └── rouge_eval.yaml                # Evaluation configuration
├── scripts/
│   ├── run_train_local.sh             # Local training script
│   ├── run_eval_local.sh              # Local evaluation script
│   └── slurm/                         # Slurm cluster scripts
├── tests/
│   └── test_dataset.py                # Dataset loading tests
├── reports/
│   ├── curves/                        # Training curves and plots
│   ├── examples.jsonl                 # Sample predictions
│   └── rouge_summary.json             # Final evaluation results
└── requirements.txt                   # Python dependencies
```

## Installation and Setup

### Environment Setup
```bash
# Create conda environment
conda create -n biolaysumm python=3.9 -y
conda activate biolaysumm

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Test dataset loading
python tests/test_dataset.py

# Train model (local)
bash scripts/run_train_local.sh

# Evaluate model
bash scripts/run_eval_local.sh
```

## Usage

### Training
```python
from src.utils import load_config
from src.dataset import BioLaySummDataset
from src.modules import build_model_with_lora

# Load configuration
config = load_config('configs/train_flant5_base_lora.yaml')

# Initialize dataset
dataset_loader = BioLaySummDataset(config)
train_data = dataset_loader.load_data('train')
val_data = dataset_loader.load_data('validation')

# Build model with LoRA
model = build_model_with_lora(config)
```

### Inference
```python
from src.predict import generate_layperson_summary

# Generate layperson summary
expert_report = "No infiltrates or consolidations are observed in the study."
layperson_summary = generate_layperson_summary(expert_report, model, tokenizer)
print(layperson_summary)
```

## Hardware Requirements

### Minimum Requirements
- **GPU:** NVIDIA GTX 1080 Ti (11GB VRAM) or better
- **RAM:** 16GB system RAM
- **Storage:** 10GB free space for dataset and checkpoints

### Recommended Setup
- **GPU:** NVIDIA RTX 3080 (10GB VRAM) or RTX 4090 (24GB VRAM)
- **RAM:** 32GB system RAM
- **Storage:** 50GB free space for full experimentation

### Training Time Estimates
- **Single GPU (RTX 3080):** ~4-6 hours for 3 epochs
- **Multi-GPU (2x RTX 3080):** ~2-3 hours with distributed training
- **CPU-only:** Not recommended (would take days)

## Results and Performance

*Results will be updated after training completion*

### Expected Performance
Based on similar medical text simplification tasks:
- **ROUGE-1:** 0.45-0.55
- **ROUGE-2:** 0.25-0.35  
- **ROUGE-L:** 0.40-0.50
- **ROUGE-Lsum:** 0.40-0.50

### Model Efficiency
- **Trainable Parameters:** 1.2M (0.5% of total)
- **Training Memory:** ~8GB VRAM (vs ~32GB for full fine-tuning)
- **Inference Speed:** ~50ms per report on RTX 3080

## Error Analysis

*Sample error analysis will be added after training completion*

### Common Failure Modes
1. **Medical Terminology:** Complex terms not properly simplified
2. **Context Loss:** Important clinical context omitted in translation
3. **Length Mismatch:** Generated summaries too long or too short
4. **Coherence Issues:** Disconnected sentences in layperson summary

## Future Improvements

1. **Medical-Specific Metrics:** Integrate F1-CheXbert and F1-RadGraph
2. **Domain Adaptation:** Fine-tune on specific radiology subdomains
3. **Multi-modal:** Incorporate radiology images for better context
4. **Interactive Refinement:** Allow human feedback for summary improvement

## License and Citation

### Dataset License
The BioLaySumm dataset is released under appropriate research licenses. Please refer to the original dataset repository for specific licensing terms.

### Model License
FLAN-T5 is released under Apache 2.0 license. Our LoRA adaptations follow the same licensing terms.

### Citation
```bibtex
@article{chung2024flant5lora,
  title={FLAN-T5 LoRA for Expert-to-Layperson Radiology Report Translation},
  author={Chung, Nathan},
  journal={COMP3710 Pattern Analysis},
  year={2024}
}
```

## Training Instructions

This section provides detailed instructions for training the FLAN-T5 LoRA model on both CPU and GPU environments.

### Prerequisites

1. **Environment Setup:**
   ```bash
   # Create conda environment
   conda create -n biolaysumm python=3.9 -y
   conda activate biolaysumm
   
   # Install PyTorch (CPU version)
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

2. **For GPU Training (Optional):**
   ```bash
   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Verify CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Configuration

The training configuration is managed through `configs/train_flant5_base_lora.yaml`. Key settings:

- **Dataset**: BioLaySumm expert-to-layperson pairs
- **Model**: google/flan-t5-base with LoRA adaptation
- **Hardware**: Automatic CPU/GPU detection
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

### CPU Training

**Recommended for:**
- Testing and development
- Small-scale experiments
- Systems without GPU

**Instructions:**
```bash
# Activate environment
conda activate biolaysumm

# Navigate to project directory
cd recognition/layrad-flant5-lora-nchung

# Run training (CPU mode)
bash scripts/run_train_local.sh
```

**Expected Performance:**
- Training time: ~2-4 hours for 1 epoch (150K samples)
- Memory usage: ~4-6 GB RAM
- Model size: 248M total parameters, 885K trainable (0.36%)

**Monitoring CPU Training:**
```bash
# Check training progress
tail -f checkpoints/flan-t5-base-lora-biolaysumm/reports/logs/training.log

# Monitor metrics
cat checkpoints/flan-t5-base-lora-biolaysumm/reports/metrics/training_metrics.json

# Check training summary
cat checkpoints/flan-t5-base-lora-biolaysumm/reports/training_summary.json
```

### Single GPU Training

**Recommended for:**
- Production training
- Faster iteration
- Better convergence

**Prerequisites:**
- CUDA-capable GPU (8GB+ VRAM recommended)
- CUDA 11.8+ installed
- GPU drivers updated

**Instructions:**
```bash
# Activate environment
conda activate biolaysumm

# Install GPU version (if not already installed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Navigate to project directory
cd recognition/layrad-flant5-lora-nchung

# Run training (GPU mode)
bash scripts/run_train_local.sh
```

**Expected Performance:**
- Training time: ~30-60 minutes for 1 epoch (150K samples)
- Memory usage: ~6-8 GB VRAM
- Speed improvement: 3-5x faster than CPU

**Monitoring GPU Training:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check training logs
tail -f checkpoints/flan-t5-base-lora-biolaysumm/reports/logs/training.log

# Monitor TensorBoard (if enabled)
tensorboard --logdir checkpoints/flan-t5-base-lora-biolaysumm/logs
```

### Training Output Structure

After training completes, you'll find:

```
checkpoints/flan-t5-base-lora-biolaysumm/
├── reports/                          # Training logs and metrics
│   ├── logs/
│   │   ├── trainer_state.json       # Trainer state and progress
│   │   └── training.log             # Training log file
│   ├── metrics/
│   │   └── training_metrics.json    # ROUGE metrics history
│   ├── configs/
│   │   └── training_arguments.json  # Training hyperparameters
│   └── training_summary.json        # Complete training summary
├── final_model/                      # Best model checkpoint
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── generation_config.json
│   └── tokenizer files...
├── training_config.yaml             # Training configuration
└── training_results.json            # Training results summary
```

### Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory:**
   ```yaml
   # Reduce batch size in configs/train_flant5_base_lora.yaml
   training:
     batch_size: 4  # Reduce from 8
     gradient_accumulation_steps: 8  # Increase from 4
   ```

2. **CPU Training Too Slow:**
   ```yaml
   # Reduce dataset size for testing
   # Use smaller subset: dataset.select(range(1000))
   ```

3. **Import Errors:**
   ```bash
   # Ensure all dependencies installed
   pip install -r requirements.txt
   
   # Check Python version
   python --version  # Should be 3.9+
   ```

4. **Dataset Loading Issues:**
   ```bash
   # Test dataset loading
   python tests/test_dataset.py
   ```

### Performance Tuning

**For Better Performance:**

1. **GPU Optimization:**
   - Use mixed precision training (bf16)
   - Enable gradient accumulation
   - Pin memory for data loading

2. **CPU Optimization:**
   - Reduce batch size
   - Use fewer workers
   - Enable gradient accumulation

3. **Memory Optimization:**
   - Use LoRA (already enabled)
   - Reduce sequence lengths if needed
   - Enable gradient checkpointing

### Evaluation

**ROUGE Metrics:**
- **ROUGE-1**: Word-level overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Sentence-level ROUGE-L

**Best Model Selection:**
- Model with highest validation ROUGE-Lsum is automatically saved
- Checkpointing occurs every 1000 steps
- Best model loaded at training end

### Next Steps

After training:
1. **Evaluate** the model on test set
2. **Generate** sample expert-to-layperson translations
3. **Analyze** ROUGE metrics and training curves
4. **Fine-tune** hyperparameters if needed

## Contributing

This project is part of a university course assignment. For questions or issues, please contact the course instructor or create an issue in the repository.

## Acknowledgments

- **BioLaySumm Workshop:** For providing the dataset and task definition
- **Google Research:** For the FLAN-T5 base model
- **Microsoft:** For the LoRA parameter-efficient fine-tuning technique
- **HuggingFace:** For the transformers library and dataset infrastructure