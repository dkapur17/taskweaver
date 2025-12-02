#!/usr/bin/env python3
"""
Unified evaluation runner for multiple datasets.
Usage: 
  Model evaluation: python run_eval.py <config_file.yaml>
  Re-evaluation:    python run_eval.py --reevaluate <results.json> [--output new.json]
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import Evaluator
from eval.eval_configs import EvaluationResult, NumericConfig
from dsconf.dataset_configs import DatasetConfig


def load_model_with_lora(base_model_name: str, lora_adapter_path: str):
    """Load model with LoRA adapters merged."""
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "LoRA evaluation requires the 'peft' library. "
            "Install it with: pip install peft"
        )
    
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
    
    
    # Validate required fields
    required_fields = ['base_model_name', 'is_chat_model', 'datasets', 'output_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Check if this is a LoRA config
    is_lora = 'lora_adapter_path' in config

    # Validate LoRA-specific fields
    if is_lora and not config.get('lora_adapter_path'):
        raise ValueError("LoRA configs require 'lora_adapter_path' field")
    
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


def get_eval_config_for_task(task_name: str, **kwargs):
    """
    Get the appropriate evaluation config for a task name.
    Uses DatasetConfig to determine the correct eval config.
    """
    # Try to match task name to dataset config
    for (dataset_path, dataset_name) in DatasetConfig.list_available():
        config_cls = DatasetConfig.from_dataset_path(dataset_path, dataset_name)
        if config_cls.id() == task_name:
            return config_cls.get_eval_config()
    
    raise ValueError(f"Cannot determine eval config for task: {task_name}")


def reevaluate_from_json(
    json_filepath: Path,
    output_filepath: Optional[Path] = None,
    verbose: bool = True,
    **eval_kwargs
) -> None:
    """
    Re-evaluate results from a saved JSON file.
    
    Args:
        json_filepath: Path to the JSON file with saved results
        output_filepath: Path to save re-evaluated results (default: <input>_reevaluated.json)
        verbose: Whether to print progress
        **eval_kwargs: Additional kwargs to pass to eval configs
    """
    if not json_filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {json_filepath}")
    
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        raise ValueError("Invalid JSON format: missing 'results' key")
    
    new_results = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Re-evaluating results from: {json_filepath.name}")
        if eval_kwargs:
            print(f"Custom parameters: {eval_kwargs}")
        print(f"{'='*70}\n")
    
    for task_name, task_data in data['results'].items():
        if verbose:
            print(f"Re-evaluating: {task_name}")
            print("-" * 70)
        
        # Extract samples from JSON
        samples = task_data['samples']
        inputs = [s['input'] for s in samples]
        predictions = [s['prediction'] for s in samples]
        references = [s['reference'] for s in samples]
        
        # Get eval config for this task
        try:
            eval_config = get_eval_config_for_task(task_name, **eval_kwargs)
        except ValueError as e:
            if verbose:
                print(f"  Warning: {e}, skipping metric computation\n")
            result = EvaluationResult(
                eval_type=task_data.get('eval_type', 'UNKNOWN'),
                metrics=None,
                inputs=inputs,
                predictions=predictions,
                references=references,
                parsed_predictions=None,
                parsed_references=None,
                num_samples=len(predictions)
            )
            new_results[task_name] = result
            continue
        
        # Re-run evaluation with the config
        result = eval_config(inputs, predictions, references)
        new_results[task_name] = result
        
        # Print new results
        if verbose and result.metrics:
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f} ({value:.2%})")
                else:
                    print(f"  {metric}: {value}")
            print()
    
    if verbose:
        print(f"\n{'='*70}")
        print("Re-evaluation complete!")
        print(f"{'='*70}\n")
        
        # Print summary
        accuracies = []
        for task_name, result in new_results.items():
            if result.metrics and 'accuracy' in result.metrics:
                accuracies.append(result.metrics['accuracy'])
        
        if accuracies:
            print(f"Average accuracy: {sum(accuracies)/len(accuracies):.2%}")
    
    # Save results using Evaluator's save method
    if output_filepath is None:
        output_filepath = json_filepath.parent / f"{json_filepath.stem}_reevaluated.json"
    
    evaluator = Evaluator(tasks=[], verbose=False)
    evaluator.results = new_results
    evaluator.save_results(output_filepath)
    
    print(f"\nRe-evaluated results saved to: {output_filepath}")


def run_model_evaluation(config_path: str) -> None:
    """Run model evaluation from YAML config."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Detect LoRA config and load model accordingly
    is_lora = 'lora_adapter_path' in config
    
    if is_lora:
        print(f"\nDetected LoRA configuration")
        print("=" * 70)
        model, tokenizer = load_model_with_lora(
            config['base_model_name'],
            config['lora_adapter_path']
        )
        print("=" * 70)
    else:
        # Standard model loading
        base_model_name = config['base_model_name']
        print(f"\nLoading model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='auto',
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set padding side for batched generation
        tokenizer.padding_side = 'left'
    
    # Create tasks for enabled datasets
    is_chat_model = config['is_chat_model']
    tasks = []
    task_specific_kwargs = {}
    
    print("\nLoading datasets:")
    for dataset_key, dataset_config in config['datasets'].items():
        if not dataset_config.get('enabled', True):
            print(f"  Skipping {dataset_key} (disabled)")
            continue
        
        # Parse dataset_key to get path and name
        # Format: "path" or "path.name"
        if '.' in dataset_key:
            dataset_path, dataset_name = dataset_key.rsplit('.', 1)
        else:
            dataset_path = dataset_key
            dataset_name = None
        
        # Override with explicit values from config if provided
        dataset_path = dataset_config.get('dataset_path', dataset_path)
        dataset_name = dataset_config.get('dataset_name', dataset_name)
        
        split = dataset_config.get('split', 'test[:5%]')
        batch_size = dataset_config.get('batch_size', config.get('batch_size', 8))
        max_new_tokens = dataset_config.get('max_new_tokens', config.get('max_new_tokens', 128))
        temperature = dataset_config.get('temperature', config.get('temperature', 1.0))
        
        print(f"  Loading {dataset_key} from {dataset_path}" + 
              (f"/{dataset_name}" if dataset_name else "") + 
              f" (split={split})")
        
        try:
            # Get config class and create task
            config_cls = DatasetConfig.from_dataset_path(dataset_path, dataset_name)
            task = config_cls.create_task(split=split, is_chat=is_chat_model)
            tasks.append(task)
            
            # Store task-specific kwargs
            task_specific_kwargs[task.task_name] = {
                'batch_size': batch_size,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature
            }
        except ValueError as e:
            print(f"  Error: {e}")
            print(f"  Available datasets: {DatasetConfig.list_available()}")
            continue
        except Exception as e:
            print(f"  Error loading {dataset_key}: {e}")
            continue
    
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
    # Use model_name for display if provided, otherwise use base_model_name
    display_name = config.get('model_name', config['base_model_name'])
    output_path = format_output_path(config['output_path'], display_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(output_path)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate models (standard or LoRA) or re-evaluate saved results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Standard model evaluation:
    python run_eval.py configs/quick_test.yaml
    python run_eval.py configs/base_model.yaml
  
  LoRA model evaluation:
    python run_eval.py configs/lora_model.yaml
    (Requires 'peft' library: pip install peft)
  
  Re-evaluation from JSON:
    python run_eval.py --reevaluate results/model/results.json
    python run_eval.py --reevaluate results.json
    python run_eval.py --reevaluate results.json --output new.json --quiet

Config file format:
  Standard models require: base_model_name, is_chat_model, datasets, output_path
  LoRA models also require: lora_adapter_path
  Optional: model_name (for display/output naming)
        """
    )
    
    parser.add_argument('config_or_json', nargs='?', help='YAML config file or JSON results file')
    parser.add_argument('--reevaluate', '-r', type=str, metavar='JSON',
                       help='Re-evaluate from saved JSON results file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path for re-evaluated results')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output during re-evaluation')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.reevaluate:
        # Re-evaluation mode
        json_path = Path(args.reevaluate)
        output_path = Path(args.output) if args.output else None
        reevaluate_from_json(
            json_path,
            output_filepath=output_path,
            verbose=not args.quiet,
        )
    elif args.config_or_json:
        # Check if it's a JSON file (re-evaluation) or YAML (model eval)
        filepath = Path(args.config_or_json)
        if filepath.suffix == '.json':
            output_path = Path(args.output) if args.output else None
            reevaluate_from_json(
                filepath,
                output_filepath=output_path,
                verbose=not args.quiet,
            )
        else:
            # Assume YAML config for model evaluation
            run_model_evaluation(args.config_or_json)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
