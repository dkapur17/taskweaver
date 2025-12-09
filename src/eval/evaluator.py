import os
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from .task import Task
from .eval_configs import EvaluationResult

@dataclass
class EvaluatorSummary:
    """Summary statistics across all tasks"""
    num_tasks: int
    total_samples: int
    average_accuracy: float  # Average of pass@K (highest K) across all tasks
    task_accuracies: Dict[str, float]  # Pass@K accuracy per task
    timestamp: str
    max_k: int = 1  # Maximum K value across all tasks
    pass_at_k_metrics: Optional[Dict[str, Dict[str, float]]] = None  # {task: {pass_at_1: 0.4, pass_at_2: 0.5, ...}}
    average_pass_at_k: Optional[Dict[str, float]] = None  # {pass_at_1: 0.35, pass_at_2: 0.48, ...}
    
    def __repr__(self):
        return f"EvaluatorSummary(tasks={self.num_tasks}, avg_acc={self.average_accuracy:.4f})"


class Evaluator:
    """Evaluates a model on multiple tasks"""
    
    def __init__(
        self, 
        tasks: List[Task],
        verbose: bool = True
    ):
        """
        Initialize evaluator with tasks
        
        Args:
            tasks: List of Task objects to evaluate
            verbose: Whether to print progress and results
        """
        self.tasks = tasks
        self.verbose = verbose
        self.results: Optional[Dict[str, EvaluationResult]] = None
    
    def evaluate(
        self,
        model,
        tokenizer,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_pass: int = 1,
        progress: bool = True,
        task_specific_kwargs: Optional[Dict[str, dict]] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model on all tasks
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            batch_size: Default batch size for all tasks
            max_new_tokens: Default max new tokens for all tasks
            temperature: Default temperature for generation (0.0=deterministic, 1.0=standard)
            num_pass: Number of generations per sample (K for pass@K evaluation)
            progress: Show progress bars during evaluation
            task_specific_kwargs: Dict mapping task names to specific kwargs
                Example: {'GSM8K': {'max_new_tokens': 512, 'temperature': 0.7}}
        
        Returns:
            Dictionary mapping task names to EvaluationResults
        """
        task_specific_kwargs = task_specific_kwargs or {}
        results = {}
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting evaluation on {len(self.tasks)} tasks")
            print(f"{'='*70}\n")
        
        for i, task in enumerate(self.tasks, 1):
            if self.verbose:
                print(f"[{i}/{len(self.tasks)}] Evaluating: {task.task_name}")
                print("-" * 70)
            
            # Get task-specific kwargs if provided
            kwargs = {
                'batch_size': batch_size,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'progress': progress,
                'num_pass': num_pass
            }
            if task.task_name in task_specific_kwargs:
                kwargs.update(task_specific_kwargs[task.task_name])
            
            # Evaluate
            result = task.evaluate(model, tokenizer, **kwargs)
            results[task.task_name] = result
            
            # Print results
            if self.verbose and result.metrics:
                self._print_task_result(task.task_name, result)
        
        self.results = results
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("Evaluation complete!")
            print(f"{'='*70}\n")
            self.print_summary()
        
        return results
    
    def _print_task_result(self, task_name: str, result: EvaluationResult):
        """Print results for a single task"""
        if result.metrics:
            for metric, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f} ({value:.2%})")
                else:
                    print(f"  {metric}: {value}")
        print()
    
    def print_summary(self, sort_by: str = 'name') -> None:
        """
        Print summary table of all results
        
        Args:
            sort_by: How to sort results - 'name', 'accuracy', or 'samples'
        """
        if not self.results:
            print("No results available. Run evaluate() first.")
            return
        
        # Get summary to access pass@K metrics
        summary = self.get_summary()
        max_k = summary.max_k
        
        # Prepare data for table
        rows = []
        for task_name, result in self.results.items():
            if not result.metrics:
                continue
            
            row = {
                'name': task_name,
                'samples': result.num_samples,
                'eval_type': result.eval_type,
            }
            
            # Extract pass@k accuracies
            k = result.metrics.get('num_pass', 1)
            for i in range(1, k + 1):
                key = f'pass_at_{i}_accuracy'
                if key in result.metrics:
                    row[f'pass_at_{i}'] = result.metrics[key]
            
            # Use highest K for sorting
            row['accuracy'] = row.get(f'pass_at_{k}', 0.0)
            rows.append(row)
        
        if not rows:
            print("No metrics available for summary.")
            return
        
        # Sort
        if sort_by == 'accuracy':
            rows.sort(key=lambda x: x['accuracy'], reverse=True)
        elif sort_by == 'samples':
            rows.sort(key=lambda x: x['samples'], reverse=True)
        else:  # name
            rows.sort(key=lambda x: x['name'])
        
        # Build header based on max_k
        if max_k == 1:
            # Simple format for K=1
            header_width = 90
            print(f"\n{'='*header_width}")
            print(f"{'Task':<30} {'Type':<18} {'Accuracy':<12} {'Samples':<10}")
            print(f"{'='*header_width}")
            
            for row in rows:
                acc = row.get('pass_at_1', 0.0)
                print(f"{row['name']:<30} {row['eval_type']:<18} "
                      f"{acc:>10.2%}  {row['samples']:>8}")
            
            print(f"{'='*header_width}")
            print(f"{'Average':<30} {'':<18} {summary.average_accuracy:>10.2%}  {summary.total_samples:>8}")
            print(f"{'='*header_width}\n")
        else:
            # Extended format for K>1: show all pass@k columns
            pass_cols = [f'Pass@{i}' for i in range(1, max_k + 1)]
            col_width = 10
            pass_section_width = col_width * max_k
            header_width = 50 + pass_section_width + 10
            
            print(f"\n{'='*header_width}")
            header = f"{'Task':<30} {'Type':<18} "
            for col in pass_cols:
                header += f"{col:>{col_width}}"
            header += f"{'Samples':>10}"
            print(header)
            print(f"{'='*header_width}")
            
            for row in rows:
                line = f"{row['name']:<30} {row['eval_type']:<18} "
                for i in range(1, max_k + 1):
                    acc = row.get(f'pass_at_{i}', 0.0)
                    line += f"{acc:>{col_width}.2%}"
                line += f"{row['samples']:>10}"
                print(line)
            
            # Print average row
            print(f"{'='*header_width}")
            avg_line = f"{'Average':<30} {'':<18} "
            for i in range(1, max_k + 1):
                avg = summary.average_pass_at_k.get(f'pass_at_{i}', 0.0)
                avg_line += f"{avg:>{col_width}.2%}"
            avg_line += f"{summary.total_samples:>10}"
            print(avg_line)
            print(f"{'='*header_width}\n")
    
    def get_summary(self) -> EvaluatorSummary:
        """
        Get summary statistics
        
        Returns:
            EvaluatorSummary object with key statistics
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        task_accuracies = {}
        accuracies = []
        total_samples = 0
        max_k = 1
        pass_at_k_metrics = {}
        
        # Extract pass@K metrics from all tasks
        for task_name, result in self.results.items():
            if not result.metrics:
                continue
                
            total_samples += result.num_samples
            
            # Determine K for this task
            k = result.metrics.get('num_pass', 1)
            max_k = max(max_k, k)
            
            # Extract all pass@k accuracies for this task
            task_pass_metrics = {}
            for i in range(1, k + 1):
                key = f'pass_at_{i}_accuracy'
                if key in result.metrics:
                    task_pass_metrics[f'pass_at_{i}'] = result.metrics[key]
            
            pass_at_k_metrics[task_name] = task_pass_metrics
            
            # Use pass@K (highest k) as the main accuracy
            highest_k_key = f'pass_at_{k}_accuracy'
            if highest_k_key in result.metrics:
                acc = result.metrics[highest_k_key]
                task_accuracies[task_name] = acc
                accuracies.append(acc)
        
        # Compute average for each pass@k level
        average_pass_at_k = {}
        for i in range(1, max_k + 1):
            pass_key = f'pass_at_{i}'
            values = [metrics.get(pass_key, 0.0) for metrics in pass_at_k_metrics.values() if pass_key in metrics]
            if values:
                average_pass_at_k[pass_key] = sum(values) / len(values)
        
        return EvaluatorSummary(
            num_tasks=len(self.tasks),
            total_samples=total_samples,
            average_accuracy=sum(accuracies) / len(accuracies) if accuracies else 0.0,
            task_accuracies=task_accuracies,
            timestamp=datetime.now().isoformat(),
            max_k=max_k,
            pass_at_k_metrics=pass_at_k_metrics,
            average_pass_at_k=average_pass_at_k
        )

    def save_results(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save evaluation results with full I/O data to JSON file
        
        Args:
            output_dir: Directory to save results to
            metadata: Optional metadata dict to include in output
        """

        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        for task_name, result in self.results.items():

            samples = []
            for i in range(result.num_samples):
                is_correct = None
                if result.parsed_predictions and result.parsed_references:
                    parsed_pred = result.parsed_predictions[i]
                    parsed_ref = result.parsed_references[i]

                    if result.num_pass > 1:
                        is_correct = any(p == parsed_ref for p in parsed_pred)
                    else:
                        is_correct = (parsed_pred == parsed_ref)

                sample = {
                    'input': result.inputs[i],
                    'prediction': result.predictions[i],
                    'reference': result.references[i],
                    'parsed_prediction': result.parsed_predictions[i] if result.parsed_predictions else None,
                    'parsed_reference': result.parsed_references[i] if result.parsed_references else None,
                    'correct': is_correct
                }

                samples.append(sample)
            
            result_dict = {
                'task': task_name,
                'eval_type': result.eval_type,
                'metrics': result.metrics,
                'num_samples': result.num_samples,
                'samples': samples
            }

            results_file_name = f"{task_name.replace('/', '_')}.results.json"
            with open(os.path.join(output_dir, results_file_name), 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        if metadata:
            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(asdict(self.get_summary()), f, indent=2)

        if self.verbose:
            print(f"Saving results, metadata and summary to {output_dir}")

    
    def compare_with(
        self, 
        other_results: Dict[str, EvaluationResult],
        other_name: str = "Other"
    ) -> None:
        """
        Compare current results with another set of results
        
        Args:
            other_results: Dictionary of results to compare with
            other_name: Name for the other results in the comparison
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        # Get common tasks
        common_tasks = set(self.results.keys()) & set(other_results.keys())
        
        if not common_tasks:
            print("No common tasks to compare.")
            return
        
        print(f"\n{'='*100}")
        print(f"{'Task':<30} {'Current':<15} {other_name:<15} {'Difference':<15}")
        print(f"{'='*100}")
        
        improvements = []
        
        for task_name in sorted(common_tasks):
            curr_result = self.results[task_name]
            other_result = other_results[task_name]
            
            if curr_result.metrics and other_result.metrics:
                if 'accuracy' in curr_result.metrics and 'accuracy' in other_result.metrics:
                    curr_acc = curr_result.metrics['accuracy']
                    other_acc = other_result.metrics['accuracy']
                    diff = curr_acc - other_acc
                    
                    # Format with color indicators
                    diff_str = f"{diff:+.2%}"
                    if diff > 0:
                        diff_str = f"↑ {diff_str}"
                    elif diff < 0:
                        diff_str = f"↓ {diff_str}"
                    else:
                        diff_str = f"  {diff_str}"
                    
                    print(f"{task_name:<30} {curr_acc:>13.2%}  {other_acc:>13.2%}  {diff_str:>13}")
                    improvements.append(diff)
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"{'='*100}")
            print(f"{'Average Improvement':<30} {'':<15} {'':<15} {avg_improvement:>+13.2%}")
            print(f"{'='*100}\n")
    
    def get_failed_predictions(
        self, 
        task_name: str, 
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get examples where the model failed
        
        Args:
            task_name: Name of the task
            n: Number of failed examples to return
            
        Returns:
            List of dicts with prediction, reference, and parsed values
        """
        if not self.results or task_name not in self.results:
            raise ValueError(f"No results for task: {task_name}")
        
        result = self.results[task_name]
        
        failed = []
        for i, (pred, ref, parsed_pred, parsed_ref) in enumerate(zip(
            result.predictions,
            result.references,
            result.parsed_predictions or [None] * len(result.predictions),
            result.parsed_references or [None] * len(result.references)
        )):
            if parsed_pred != parsed_ref:
                failed.append({
                    'index': i,
                    'prediction': pred,
                    'reference': ref,
                    'parsed_prediction': parsed_pred,
                    'parsed_reference': parsed_ref
                })
                
                if len(failed) >= n:
                    break
        
        return failed
    
    def print_failed_examples(self, task_name: str, n: int = 5) -> None:
        """Print failed examples for inspection"""
        failed = self.get_failed_predictions(task_name, n)
        
        print(f"\n{'='*70}")
        print(f"Failed examples for: {task_name}")
        print(f"{'='*70}\n")
        
        for i, example in enumerate(failed, 1):
            print(f"Example {i}:")
            print(f"  Prediction: {example['prediction'][:100]}...")
            print(f"  Reference:  {example['reference'][:100]}...")
            print(f"  Parsed Pred: {example['parsed_prediction']}")
            print(f"  Parsed Ref:  {example['parsed_reference']}")
            print()
    
    def __repr__(self):
        status = "evaluated" if self.results else "not evaluated"
        return f"Evaluator(tasks={len(self.tasks)}, {status})"