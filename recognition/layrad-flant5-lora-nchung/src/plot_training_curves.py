#!/usr/bin/env python3
"""
Training Visualization Script for BioLaySumm Models

This script generates training curves and performance visualizations from
checkpoint trainer_state.json files for both LoRA and Full Fine-tuning models.

Usage:
    python src/plot_training_curves.py
    python src/plot_training_curves.py --output_dir reports/curves
    python src/plot_training_curves.py --lora_path checkpoints/flan-t5-base-lora-biolaysumm/checkpoint-14106/trainer_state.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def load_training_history(file_path: str) -> Dict:
    """Load training history from trainer_state.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_training_data(history: Dict) -> Tuple[List, List, List, List, List]:
    """Extract training loss, validation metrics, and learning rates from history."""
    log_history = history.get('log_history', [])
    
    # Extract training data
    train_steps = []
    train_losses = []
    learning_rates = []
    
    # Extract validation data
    val_steps = []
    val_rouge1 = []
    val_rouge2 = []
    val_rougeL = []
    val_rougeLsum = []
    
    for entry in log_history:
        if 'step' in entry:
            step = entry['step']
            
            # Training data (every entry has step)
            if 'loss' in entry and 'eval_loss' not in entry:
                train_steps.append(step)
                train_losses.append(entry['loss'])
                if 'learning_rate' in entry:
                    learning_rates.append(entry['learning_rate'])
            
            # Validation data (only eval entries)
            if 'eval_rouge1' in entry:
                val_steps.append(step)
                val_rouge1.append(entry['eval_rouge1'])
                val_rouge2.append(entry['eval_rouge2'])
                val_rougeL.append(entry['eval_rougeL'])
                val_rougeLsum.append(entry['eval_rougeLsum'])
    
    return (train_steps, train_losses, learning_rates, 
            val_steps, val_rouge1, val_rouge2, val_rougeL, val_rougeLsum)


def plot_training_loss_comparison(lora_data: Tuple, full_ft_data: Tuple, output_dir: str):
    """Plot training loss comparison between LoRA and Full Fine-tuning."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    lora_steps, lora_losses, _, _, _, _, _, _ = lora_data
    full_steps, full_losses, _, _, _, _, _, _ = full_ft_data
    
    # Plot training losses
    ax.plot(lora_steps, lora_losses, 'b-', label='FLAN-T5-base LoRA', linewidth=2, alpha=0.8)
    ax.plot(full_steps, full_losses, 'r-', label='T5-small Full FT', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison: LoRA vs Full Fine-tuning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add final loss values as text
    final_lora_loss = lora_losses[-1] if lora_losses else 0
    final_full_loss = full_losses[-1] if full_losses else 0
    ax.text(0.02, 0.98, f'Final LoRA Loss: {final_lora_loss:.4f}\nFinal Full FT Loss: {final_full_loss:.4f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_validation_rouge_metrics(lora_data: Tuple, full_ft_data: Tuple, output_dir: str):
    """Plot validation ROUGE metrics for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Validation ROUGE Metrics During Training', fontsize=16, fontweight='bold')
    
    # Extract validation data
    lora_val_steps, _, _, _, lora_rouge1, lora_rouge2, lora_rougeL, lora_rougeLsum = lora_data
    full_val_steps, _, _, _, full_rouge1, full_rouge2, full_rougeL, full_rougeLsum = full_ft_data
    
    metrics = [
        ('ROUGE-1', lora_rouge1, full_rouge1, axes[0, 0]),
        ('ROUGE-2', lora_rouge2, full_rouge2, axes[0, 1]),
        ('ROUGE-L', lora_rougeL, full_rougeL, axes[1, 0]),
        ('ROUGE-Lsum', lora_rougeLsum, full_rougeLsum, axes[1, 1])
    ]
    
    for metric_name, lora_scores, full_scores, ax in metrics:
        ax.plot(lora_val_steps, lora_scores, 'b-o', label='FLAN-T5-base LoRA', 
                linewidth=2, markersize=4, alpha=0.8)
        ax.plot(full_val_steps, full_scores, 'r-s', label='T5-small Full FT', 
                linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel(f'{metric_name} Score', fontsize=11)
        ax.set_title(f'{metric_name} During Training', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final scores
        final_lora = lora_scores[-1] if lora_scores else 0
        final_full = full_scores[-1] if full_scores else 0
        ax.text(0.02, 0.98, f'LoRA: {final_lora:.4f}\nFull FT: {final_full:.4f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_rouge_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_rate_schedules(lora_data: Tuple, full_ft_data: Tuple, output_dir: str):
    """Plot learning rate schedules for both models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract learning rate data
    lora_steps, _, lora_lr, _, _, _, _, _ = lora_data
    full_steps, _, full_lr, _, _, _, _, _ = full_ft_data
    
    # Plot learning rates
    ax.plot(lora_steps, lora_lr, 'b-', label='FLAN-T5-base LoRA (1e-4)', linewidth=2, alpha=0.8)
    ax.plot(full_steps, full_lr, 'r-', label='T5-small Full FT (5e-5)', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedules During Training', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add peak learning rates
    peak_lora = max(lora_lr) if lora_lr else 0
    peak_full = max(full_lr) if full_lr else 0
    ax.text(0.02, 0.98, f'Peak LoRA LR: {peak_lora:.2e}\nPeak Full FT LR: {peak_full:.2e}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_rate_schedules.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_performance_comparison(output_dir: str):
    """Plot final performance comparison bar chart."""
    # Final ROUGE scores from evaluation results
    models = ['Zero-shot\nBaseline', 'T5-small\nFull FT', 'FLAN-T5-base\nLoRA']
    rouge1_scores = [0.317, 0.444, 0.696]
    rouge2_scores = [0.116, 0.230, 0.496]
    rougeL_scores = [0.287, 0.397, 0.640]
    rougeLsum_scores = [0.287, 0.397, 0.640]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars
    bars1 = ax.bar(x - 1.5*width, rouge1_scores, width, label='ROUGE-1', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x - 0.5*width, rouge2_scores, width, label='ROUGE-2', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + 0.5*width, rougeL_scores, width, label='ROUGE-L', alpha=0.8, color='lightgreen')
    bars4 = ax.bar(x + 1.5*width, rougeLsum_scores, width, label='ROUGE-Lsum', alpha=0.8, color='gold')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    # Styling
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('ROUGE Score', fontsize=12)
    ax.set_title('Final Performance Comparison: All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.8)
    
    # Add performance improvement annotations
    ax.annotate('+37.9 points\nvs Zero-shot', xy=(2, 0.696), xytext=(1.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training visualizations')
    parser.add_argument('--lora_path', 
                       default='checkpoints/flan-t5-base-lora-biolaysumm/checkpoint-14106/trainer_state.json',
                       help='Path to LoRA trainer_state.json')
    parser.add_argument('--full_ft_path',
                       default='checkpoints/t5-small-full-biolaysumm/checkpoint-9404/trainer_state.json', 
                       help='Path to Full FT trainer_state.json')
    parser.add_argument('--output_dir', default='reports/curves',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading training histories...")
    
    # Load training histories
    try:
        lora_history = load_training_history(args.lora_path)
        full_ft_history = load_training_history(args.full_ft_path)
        print(f"✅ Loaded LoRA history: {len(lora_history.get('log_history', []))} entries")
        print(f"✅ Loaded Full FT history: {len(full_ft_history.get('log_history', []))} entries")
    except FileNotFoundError as e:
        print(f"❌ Error loading training history: {e}")
        return
    
    # Extract training data
    print("Extracting training data...")
    lora_data = extract_training_data(lora_history)
    full_ft_data = extract_training_data(full_ft_history)
    
    print("Generating plots...")
    
    # Generate all plots
    plot_training_loss_comparison(lora_data, full_ft_data, args.output_dir)
    print("✅ Generated training loss comparison")
    
    plot_validation_rouge_metrics(lora_data, full_ft_data, args.output_dir)
    print("✅ Generated validation ROUGE metrics")
    
    plot_learning_rate_schedules(lora_data, full_ft_data, args.output_dir)
    print("✅ Generated learning rate schedules")
    
    plot_final_performance_comparison(args.output_dir)
    print("✅ Generated final performance comparison")
    
    print(f"\n🎉 All plots saved to: {args.output_dir}/")
    print("Generated files:")
    for file in os.listdir(args.output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == '__main__':
    main()
