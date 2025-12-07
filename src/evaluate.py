import os
import sys
import torch
import json
import subprocess
import time
from jsonargparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Literal, Type, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval import Evaluator
from dsconf import DatasetConfig
from hypernet import TaskWeaver
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv('../.env')


def get_gpu_info() -> Dict[str, Any]:
    """Capture NVIDIA GPU information using nvidia-smi"""
    gpu_info = {}
    try:
        # Get GPU name
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info['gpu_name'] = result.stdout.strip().split('\n')
        
        # Get GPU memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info['gpu_memory'] = result.stdout.strip().split('\n')
        
        # Get driver version
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info['driver_version'] = result.stdout.strip().split('\n')[0]
        
        # Get CUDA version
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    gpu_info['cuda_version'] = line.split('CUDA Version:')[1].split()[0]
                    break
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info


def get_base_model_and_tokenizer(model_path: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    return model, tokenizer

def get_lora_model_and_tokenizer(lora_adapter_path: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    adapter_config = json.load(open(os.path.join(lora_adapter_path, 'adapter_config.json'), 'r'))
    base_model_path = adapter_config['base_model_name_or_path']

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')

    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    lora_model = lora_model.merge_and_unload() # Fuses lora adapter into base model

    return lora_model, tokenizer

def get_hypernet_and_tokenizer(hypernet_path: str, device: str) -> Tuple[TaskWeaver, AutoTokenizer]:
    # Load TaskWeaver hypernetwork from checkpoint
    # device_map parameter is not supported by TaskWeaver.from_pretrained, so we use device directly
    taskweaver = TaskWeaver.from_pretrained(hypernet_path, device=device if device != 'auto' else None)
    
    # Load config to get base model name for tokenizer
    config_path = os.path.join(hypernet_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_model_name = config.get('model_name')
    if not base_model_name:
        raise ValueError(f"No model_name found in config at {config_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
    return taskweaver, tokenizer

def get_model_and_tokenizer(path: str, model_type: Literal['base', 'lora', 'hypernet'], device: str) -> Tuple[Union[AutoModelForCausalLM, TaskWeaver], AutoTokenizer]:
    if model_type == 'base':
        model, tokenizer = get_base_model_and_tokenizer(path, device)
    elif model_type == 'lora':
        model, tokenizer =  get_lora_model_and_tokenizer(path, device)
    elif model_type == 'hypernet':
        model, tokenizer = get_hypernet_and_tokenizer(path, device)
    else:
        raise NotImplementedError(f"Model type {model_type} not supported")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_dataset_configs(datasets: List[str], ignore_list: List[str]) -> List[Type[DatasetConfig]]:

    configs = []

    if 'all' in datasets:
        for path, name in DatasetConfig.list_available():
            if f"{path}.{name}" not in ignore_list:
                configs.append(DatasetConfig.from_dataset_path(path, name))
    else:
        for dataset in datasets:
            if '.' in dataset:
                path, name = dataset.split('.')
            else:
                path, name = dataset, None
            configs.append(DatasetConfig.from_dataset_path(path, name))

    return configs

@dataclass
class EvaluatorConfig:
    batch_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.7
    num_pass: int = 1


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluating Base, Finetuned and Hypernetwork models")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument('--model_type', type=Literal['base', 'lora', 'hypernet'], required=True, help="Model type")
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], help="Dataset to evaluate on")
    parser.add_argument('--ignore_datasets', type=str, nargs='+', default=[], help='Datasets to ignore')
    parser.add_argument('--split', type=str, default='test', help="Split to evaluate on")
    parser.add_argument('--device', type=str, default='auto', help='Device to run evals on')
    parser.add_class_arguments(EvaluatorConfig, 'evaluator', help="Evaluator configuration")
    parser.add_argument('--output_dir', type=str, default='_results', help="Directory to save evaluation outputs")
    parser.add_argument('--no_save', action='store_true', help="Disable automatic saving of results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Capture the command that was run
    command = ' '.join(sys.argv)
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now()

    model, tokenizer = get_model_and_tokenizer(args.model_path, args.model_type, args.device)
    dataset_configs = get_dataset_configs(args.datasets, args.ignore_datasets)

    is_chat = tokenizer.chat_template is not None
    tasks = [config.create_task(is_chat, args.split) for config in dataset_configs]
    
    evaluator = Evaluator(tasks=tasks, verbose=True)
    results = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        progress=True,
        batch_size=args.evaluator.batch_size,
        max_new_tokens=args.evaluator.max_new_tokens,
        temperature=args.evaluator.temperature,
        num_pass=args.evaluator.num_pass
    )

    # Save results by default
    if not args.no_save:
        # End timing
        end_time = time.time()
        end_datetime = datetime.now()
        duration_seconds = end_time - start_time
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.model_path).name.replace('/', '-')
        output_dir = Path(args.output_dir) / f"{model_name}_{args.model_type}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"eval_{timestamp}.json"
        
        # Capture GPU info
        gpu_info = get_gpu_info()
        
        # Prepare metadata
        metadata = {
            'command': command,
            'model_path': args.model_path,
            'model_type': args.model_type,
            'datasets': args.datasets,
            'split': args.split,
            'batch_size': args.evaluator.batch_size,
            'max_new_tokens': args.evaluator.max_new_tokens,
            'temperature': args.evaluator.temperature,
            'num_pass': args.evaluator.num_pass,
            'is_chat': is_chat,
            'timestamp': timestamp,
            'runtime': {
                'start_time': start_datetime.isoformat(),
                'end_time': end_datetime.isoformat(),
                'duration_seconds': round(duration_seconds, 2),
                'duration_formatted': f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m {int(duration_seconds % 60)}s"
            },
            'gpu_info': gpu_info
        }
        
        evaluator.save_results(output_file, metadata=metadata)
        print(f"\n✓ Results saved to: {output_file}")