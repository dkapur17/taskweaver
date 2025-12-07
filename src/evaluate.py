import os
import torch
import json
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
    raise NotImplementedError("Hypernetwork loading not implemented")

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


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluating Base, Finetuned and Hypernetwork models")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument('--model_type', type=Literal['base', 'lora', 'hypernet'], required=True, help="Model type")
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], help="Dataset to evaluate on")
    parser.add_argument('--ignore_datasets', type=str, nargs='+', default=[], help='Datasets to ignore')
    parser.add_argument('--split', type=str, default='test', help="Split to evaluate on")
    parser.add_argument('--device', type=str, default='auto', help='Device to run evals on')
    parser.add_class_arguments(EvaluatorConfig, 'evaluator', help="Evaluator configuration")
    parser.add_argument('--output_dir', type=str, help="Location to write evaluation output to")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

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
        temperature=args.evaluator.temperature
    )

    print(json.dumps(asdict(results['allenai/ai2_arc.ARC-Easy'])))