import os
import json
from jsonargparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from eval import Evaluator
from dsconf import DatasetConfig
from hypernet import TaskWeaver
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv('../.env')


def get_base_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_lora_model_and_tokenizer(lora_adapter_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    adapter_config = json.load(open(os.path.join(lora_adapter_path, 'adapter_config.json'), 'r'))
    base_model_path = adapter_config['base_model_name_or_path']

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    lora_model = lora_model.merge_and_unload() # Fuses lora adapter into base model

    return lora_model, tokenizer

def get_hypernet_and_tokenizer(hypernet_path: str) -> Tuple[TaskWeaver, AutoTokenizer]:
    raise NotImplementedError("Hypernetwork loading not implemented")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluating Base, Finetuned and Hypernetwork models")
    parser.add_argument('--model_path', type=str, help="Path to the model to evaluate")
    parser.add_argument('--mode_type', type=Literal['base', 'lora', 'hypernet'], help="Model type")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()