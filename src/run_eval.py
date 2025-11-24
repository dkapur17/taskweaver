#!/usr/bin/env python3
"""
Unified evaluation runner for multiple datasets.
Usage: python run_eval.py <config_file.yaml>
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import Evaluator
from eval.datasets import (
    create_gsm8k_task,
    create_snli_task,
    create_squad_task,
    create_arc_easy_task
)


DATASET_LOADERS = {
    'gsm8k': create_gsm8k_task,
    'snli': create_snli_task,
    'squad_v2': create_squad_task,
    'arc_easy': create_arc_easy_task
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'is_chat_model', 'datasets', 'output_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    return config


def format_output_path(path_template: str, model_name: str) -> Path:
    """Format output path with placeholders."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model = model_name.replace('/', '-')
    
    formatted_path = path_template.format(
        model_name=sanitized_model,
        timestamp=timestamp
    )
    
    return Path(formatted_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eval.py <config_file.yaml>")
        print("\nExample configs:")
        print("  python run_eval.py configs/quick_test.yaml")
        print("  python run_eval.py configs/base_model.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side for batched generation
    tokenizer.padding_side = 'left'
    
    # Create tasks for enabled datasets
    is_chat_model = config['is_chat_model']
    tasks = []
    task_specific_kwargs = {}
    
    print("\nLoading datasets:")
    for dataset_name, dataset_config in config['datasets'].items():
        if not dataset_config.get('enabled', True):
            print(f"  Skipping {dataset_name} (disabled)")
            continue
        
        if dataset_name not in DATASET_LOADERS:
            print(f"  Warning: Unknown dataset {dataset_name}, skipping")
            continue
        
        # Get dataset-specific settings
        split = dataset_config.get('split', 'test[:5%]')
        batch_size = dataset_config.get('batch_size', config.get('batch_size', 8))
        max_new_tokens = dataset_config.get('max_new_tokens', config.get('max_new_tokens', 128))
        
        print(f"  Loading {dataset_name} (split={split})")
        
        # Create task
        loader_fn = DATASET_LOADERS[dataset_name]
        task = loader_fn(split=split, is_chat_task=is_chat_model)
        tasks.append(task)
        
        # Store task-specific kwargs
        task_specific_kwargs[task.task_name] = {
            'batch_size': batch_size,
            'max_new_tokens': max_new_tokens
        }
    
    if not tasks:
        print("\nError: No datasets enabled in config")
        sys.exit(1)
    
    print(f"\nTotal tasks to evaluate: {len(tasks)}")
    
    # Create evaluator and run
    evaluator = Evaluator(tasks=tasks, verbose=True)
    results = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        progress=True,
        task_specific_kwargs=task_specific_kwargs
    )
    
    # Save results
    output_path = format_output_path(config['output_path'], model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    include_predictions = config.get('include_predictions', False)
    evaluator.save_results(output_path, include_predictions=include_predictions)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
