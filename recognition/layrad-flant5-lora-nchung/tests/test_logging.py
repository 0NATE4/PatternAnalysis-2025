#!/usr/bin/env python3
"""
Test script for logging functionality.

This script verifies that the logging functions work correctly,
including reports directory creation, training arguments logging,
and trainer state logging.

Author: Nathan Chung
Course: COMP3710 Pattern Analysis
"""

import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import (
    setup_logging, create_reports_dir, log_training_arguments, 
    log_trainer_state, log_training_summary
)


def test_reports_directory_creation():
    """Test reports directory creation."""
    print("=" * 60)
    print("Testing Reports Directory Creation")
    print("=" * 60)

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test create_reports_dir
            reports_dir = create_reports_dir(output_dir)
            
            # Verify directory structure
            assert reports_dir.exists(), "Reports directory should exist"
            assert (reports_dir / 'logs').exists(), "Logs subdirectory should exist"
            assert (reports_dir / 'metrics').exists(), "Metrics subdirectory should exist"
            assert (reports_dir / 'configs').exists(), "Configs subdirectory should exist"
            
            print("✅ Reports directory structure created successfully")
            print(f"   - Reports dir: {reports_dir}")
            print(f"   - Logs dir: {reports_dir / 'logs'}")
            print(f"   - Metrics dir: {reports_dir / 'metrics'}")
            print(f"   - Configs dir: {reports_dir / 'configs'}")
            
            return True

    except Exception as e:
        print(f"❌ Reports directory creation test failed: {e}")
        return False


def test_logging_setup():
    """Test logging setup."""
    print("\n" + "=" * 60)
    print("Testing Logging Setup")
    print("=" * 60)

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test setup_logging
            reports_dir = setup_logging(output_dir)
            
            # Verify setup
            assert reports_dir.exists(), "Reports directory should exist"
            assert (reports_dir / 'logs' / 'training.log').parent.exists(), "Training log directory should exist"
            
            print("✅ Logging setup successful")
            print(f"   - Reports directory: {reports_dir}")
            
            return True

    except Exception as e:
        print(f"❌ Logging setup test failed: {e}")
        return False


def test_training_arguments_logging():
    """Test training arguments logging."""
    print("\n" + "=" * 60)
    print("Testing Training Arguments Logging")
    print("=" * 60)

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            reports_dir = create_reports_dir(output_dir)
            
            # Mock training arguments
            class MockTrainingArgs:
                def to_dict(self):
                    return {
                        'output_dir': str(output_dir),
                        'run_name': 'test_run',
                        'num_train_epochs': 3,
                        'per_device_train_batch_size': 8,
                        'gradient_accumulation_steps': 4,
                        'learning_rate': 1e-4,
                        'weight_decay': 0.01,
                        'max_grad_norm': 1.0,
                        'warmup_steps': 500,
                        'eval_strategy': 'steps',
                        'save_strategy': 'steps',
                        'metric_for_best_model': 'eval_rougeLsum',
                        'greater_is_better': True,
                        'load_best_model_at_end': True,
                        'fp16': False,
                        'bf16': True,
                        'seed': 42,
                        'data_seed': 42,
                    }
                
                def __getattr__(self, name):
                    return self.to_dict().get(name, None)
            
            mock_args = MockTrainingArgs()
            
            # Log training arguments
            log_training_arguments(mock_args, reports_dir)
            
            # Verify file was created
            args_file = reports_dir / 'configs' / 'training_arguments.json'
            assert args_file.exists(), "Training arguments file should exist"
            
            # Verify content
            with open(args_file, 'r') as f:
                args_data = json.load(f)
            
            assert 'training_arguments' in args_data, "Should contain training_arguments"
            assert 'timestamp' in args_data, "Should contain timestamp"
            assert args_data['run_name'] == 'test_run', "Should contain correct run_name"
            assert args_data['learning_rate'] == 1e-4, "Should contain correct learning_rate"
            
            print("✅ Training arguments logging successful")
            print(f"   - File: {args_file}")
            print(f"   - Run name: {args_data['run_name']}")
            print(f"   - Learning rate: {args_data['learning_rate']}")
            
            return True

    except Exception as e:
        print(f"❌ Training arguments logging test failed: {e}")
        return False


def test_training_summary_logging():
    """Test training summary logging."""
    print("\n" + "=" * 60)
    print("Testing Training Summary Logging")
    print("=" * 60)

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            reports_dir = create_reports_dir(output_dir)
            
            # Mock config and model info
            config = {
                'dataset': {
                    'name': 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track',
                    'max_source_length': 512,
                    'max_target_length': 256,
                },
                'model': {
                    'name': 'google/flan-t5-base',
                    'torch_dtype': 'bfloat16',
                },
                'lora': {
                    'r': 8,
                    'alpha': 32,
                    'dropout': 0.1,
                },
                'training': {
                    'batch_size': 8,
                    'learning_rate': 1e-4,
                    'num_epochs': 3,
                },
                'evaluation': {
                    'eval_strategy': 'steps',
                    'metric_for_best_model': 'rougeLsum',
                },
                'hardware': {
                    'device': 'cuda',
                },
            }
            
            model_info = {
                'total': '248M (248,462,592)',
                'trainable': '885K (884,736)',
                'frozen': '248M (247,577,856)',
                'trainable_percentage': '0.36%',
                'frozen_percentage': '99.64%',
            }
            
            training_time = 3600.0  # 1 hour
            
            # Log training summary
            log_training_summary(config, model_info, training_time, reports_dir)
            
            # Verify file was created
            summary_file = reports_dir / 'training_summary.json'
            assert summary_file.exists(), "Training summary file should exist"
            
            # Verify content
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            assert 'timestamp' in summary_data, "Should contain timestamp"
            assert 'training_summary' in summary_data, "Should contain training_summary"
            
            ts = summary_data['training_summary']
            assert ts['total_training_time_seconds'] == 3600.0, "Should contain correct training time"
            assert ts['total_training_time_hours'] == 1.0, "Should contain correct training time in hours"
            assert ts['model_info'] == model_info, "Should contain model info"
            assert ts['dataset_info']['name'] == 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track', "Should contain dataset info"
            
            print("✅ Training summary logging successful")
            print(f"   - File: {summary_file}")
            print(f"   - Training time: {ts['total_training_time_hours']} hours")
            print(f"   - Dataset: {ts['dataset_info']['name']}")
            
            return True

    except Exception as e:
        print(f"❌ Training summary logging test failed: {e}")
        return False


if __name__ == "__main__":
    success1 = test_reports_directory_creation()
    success2 = test_logging_setup()
    success3 = test_training_arguments_logging()
    success4 = test_training_summary_logging()
    
    if all([success1, success2, success3, success4]):
        print("\n🚀 All logging tests passed!")
        print("✅ Logging functionality is working correctly")
        sys.exit(0)
    else:
        print("\n❌ Some logging tests failed.")
        sys.exit(1)
