#!/usr/bin/env python3
"""
Evaluation runner for models with LoRA adapters.
Customize load_model_with_lora() function for your model.
Usage: python run_eval_lora.py <config_file.yaml>
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
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


def load_model_with_lora(base_model_name: str, lora_adapter_path: str) -> Tuple[Any, Any]:
    """Customize this function to load your base model + LoRA adapters."""
    from peft import PeftModel
       
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_fields = ['is_chat_model', 'datasets', 'output_path', 'base_model_name', 'lora_adapter_path']
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


def login():
    from huggingface_hub import login
    # Login to Hugging Face to access gated models
    # Get your token from: https://huggingface.co/settings/tokens
    # Ensure your token has 'Read' permissions (or 'Write' if you plan to push models)
    hf_token = "hf_"  # Replace with your actual token
    login(token=hf_token)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_eval_lora.py <config_file.yaml>")
        print("\nExample: python run_eval_lora.py configs/lora_model.yaml")
        print("\nNote: Edit load_model_with_lora() to customize model loading")
        sys.exit(1)
    
    config_path = sys.argv[1]
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    print("\n" + "="*70)
    model, tokenizer = load_model_with_lora(config['base_model_name'], config['lora_adapter_path'])
    print("="*70)
    
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
        
        split = dataset_config.get('split', 'test[:5%]')
        batch_size = dataset_config.get('batch_size', config.get('batch_size', 8))
        max_new_tokens = dataset_config.get('max_new_tokens', config.get('max_new_tokens', 128))
        temperature = dataset_config.get('temperature', config.get('temperature', 1.0))
        
        print(f"  Loading {dataset_name} (split={split})")
        
        loader_fn = DATASET_LOADERS[dataset_name]
        task = loader_fn(split=split, is_chat_task=is_chat_model)
        tasks.append(task)
        
        task_specific_kwargs[task.task_name] = {
            'batch_size': batch_size,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        }
    
    if not tasks:
        print("\nError: No datasets enabled in config")
        sys.exit(1)
    
    print(f"\nTotal tasks to evaluate: {len(tasks)}")
    
    evaluator = Evaluator(tasks=tasks, verbose=True)
    results = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        progress=True,
        task_specific_kwargs=task_specific_kwargs
    )
    
    model_name = config.get('model_name', 'lora-model')
    output_path = format_output_path(config['output_path'], model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    include_predictions = config.get('include_predictions', False)
    evaluator.save_results(output_path, include_predictions=include_predictions)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    login()
    main()
