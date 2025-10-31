import sys
import yaml
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from lm_eval.tasks import TaskManager
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
import multiprocessing as mp
import json

@dataclass
class EvalConfig:
    model: str
    tasks: List[str]
    max_batch_size: int
    output_path: str


def get_eval_config() -> EvalConfig:
    if len(sys.argv) < 2:
        print("Provide Config File as Argument")
        exit(1)

    with open(sys.argv[1], 'r') as f:
        config_dict = yaml.safe_load(f)

    return EvalConfig(**config_dict)

def get_tasks_sizes(tasks: List[str]) -> Dict[str, int]:
    task_manager = TaskManager()
    tasks_dict = task_manager.load_task_or_group(tasks)
    return {task:len(tasks_dict[task].eval_docs) for task in tasks}


def get_task_split(tasks: List[str]) -> Dict[str, List[str]]:
    """Gets all available devices to evaluate on and load-balances tasks across devices"""

    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    num_devices = getattr(torch, device_type).device_count()

    task_sizes = get_tasks_sizes(tasks)
    tasks = sorted(tasks, key=lambda task: task_sizes[task], reverse=True)

    devices = [f'{device_type}:{device_id}' for device_id in range(num_devices)]
    task_split = {device:[] for device in devices}
    device_load = {device: 0 for device in devices}
    
    for task in tasks:
        target_device = min(devices, key=lambda device: device_load[device])
        task_split[target_device].append(task)
        device_load[target_device] += task_sizes[task]

    return task_split

def eval_tasks_on_device(tasks: List[str], device: str, eval_config: EvalConfig) -> Dict[str, Dict[str, float]]:
    model = HFLM(
        pretrained=eval_config.model,
        device=device,
        batch_size=eval_config.max_batch_size,
    )

    results = simple_evaluate(
        model=model,
        tasks=tasks,
        device=device
    )

    return results['results']
    
def main():
    # Set multiprocessing start method to 'spawn' to avoid CUDA reinitialization issues
    mp.set_start_method('spawn', force=True)
    
    eval_config = get_eval_config()
    task_split = get_task_split(eval_config.tasks)
    
    print("Task Split:")
    for device, tasks in task_split.items():
        print(f"{device}: {tasks}")

    mp_args = [(tasks, device, eval_config) for device, tasks in task_split.items()]

    with mp.Pool(processes=len(task_split)) as pool:
        results = pool.starmap(eval_tasks_on_device, mp_args)

    combined_results = {}

    for res in results:
        combined_results.update(res)
    
    with open(eval_config.output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)

if __name__ == "__main__":
    main()