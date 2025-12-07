#!/usr/bin/env python3
"""
Re-evaluate saved predictions with different evaluation configs.

Usage:
    python reevaluate.py results.json
    python reevaluate.py results.json --output new_results.json
    python reevaluate.py results.json --tolerance 0.01
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from eval import Evaluator
from eval.eval_configs import EvaluationResult
from dsconf import DatasetConfig


def load_saved_results(filepath: Path) -> Dict[str, Any]:
    """Load saved evaluation results from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        raise ValueError("Invalid format: missing 'results' key. Is this a valid evaluation output?")
    
    return data


def get_eval_config_for_task(task_name: str, **kwargs):
    """Get the appropriate evaluation config for a task by matching dataset config"""
    # Try to match task name to dataset config
    for dataset_path, dataset_name in DatasetConfig.list_available():
        config_cls = DatasetConfig.from_dataset_path(dataset_path, dataset_name)
        
        if config_cls.id() == task_name:
            eval_config = config_cls.get_eval_config()
            
            # Apply custom kwargs if applicable
            if hasattr(eval_config, 'tolerance') and 'tolerance' in kwargs:
                eval_config.tolerance = kwargs['tolerance']
            if hasattr(eval_config, 'case_sensitive') and 'case_sensitive' in kwargs:
                eval_config.case_sensitive = kwargs['case_sensitive']
            
            return eval_config
    
    raise ValueError(f"Cannot find eval config for task: {task_name}")


def reevaluate(
    input_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    **eval_kwargs
) -> Dict[str, EvaluationResult]:
    """
    Re-evaluate saved results with potentially different configs
    
    Args:
        input_path: Path to saved results JSON
        output_path: Where to save re-evaluated results
        verbose: Print progress
        **eval_kwargs: Additional parameters for eval configs (tolerance, case_sensitive, etc.)
    
    Returns:
        Dictionary of re-evaluated results
    """
    data = load_saved_results(input_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Re-evaluating: {input_path.name}")
        if eval_kwargs:
            print(f"Custom parameters: {eval_kwargs}")
        print(f"{'='*70}\n")
    
    new_results = {}
    
    for task_name, task_data in data['results'].items():
        if verbose:
            print(f"Task: {task_name}")
        
        # Extract samples
        samples = task_data['samples']
        inputs = [s['input'] for s in samples]
        predictions = [s['prediction'] for s in samples]
        references = [s['reference'] for s in samples]
        
        # Detect K-pass evaluation
        num_pass = task_data.get('metrics', {}).get('num_pass', 1)
        if verbose and num_pass > 1:
            print(f"  (K-pass evaluation detected: K={num_pass})")
        
        # Get eval config
        try:
            eval_config = get_eval_config_for_task(task_name, **eval_kwargs)
        except ValueError as e:
            if verbose:
                print(f"  Warning: {e}, skipping")
            continue
        
        # Re-evaluate
        result = eval_config(inputs, predictions, references)
        new_results[task_name] = result
        
        # Print metrics
        if verbose and result.metrics:
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        print()
    
    # Save if output path provided
    if output_path:
        evaluator = Evaluator(tasks=[], verbose=False)
        evaluator.results = new_results
        
        # Preserve original metadata and add re-evaluation info
        metadata = data.get('metadata', {})
        metadata['reevaluated_from'] = str(input_path)
        metadata['reevaluation_params'] = eval_kwargs
        
        evaluator.save_results(output_path, metadata=metadata)
        
        if verbose:
            print(f"\n✓ Re-evaluated results saved to: {output_path}")
    
    return new_results


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved predictions with different metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reevaluate.py _results/model/eval_20251207_120000.json
  python reevaluate.py results.json --output new_results.json
  python reevaluate.py results.json --tolerance 0.001
        """
    )
    
    parser.add_argument('input_file', type=str, 
                       help='Path to saved results JSON')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path (default: <input>_reevaluated.json)')
    parser.add_argument('--tolerance', type=float,
                       help='Tolerance for numeric comparisons')
    parser.add_argument('--case_sensitive', action='store_true',
                       help='Enable case-sensitive matching')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_reevaluated.json"
    
    # Collect eval kwargs
    eval_kwargs = {}
    if args.tolerance is not None:
        eval_kwargs['tolerance'] = args.tolerance
    if args.case_sensitive:
        eval_kwargs['case_sensitive'] = args.case_sensitive
    
    # Run re-evaluation
    try:
        reevaluate(
            input_path,
            output_path,
            verbose=not args.quiet,
            **eval_kwargs
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
