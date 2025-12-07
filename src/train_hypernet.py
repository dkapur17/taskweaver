"""
Training Script for TaskWaver Hypernetwork

This script provides a complete training pipeline for the TaskWeaver hypernetwork
with TensorBoard logging support.
"""

import os
from jsonargparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
from dotenv import load_dotenv

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer
from datasets import Dataset

from torch.utils.tensorboard import SummaryWriter

from hypernet import TaskWeaver
from hypernet import DataCollatorWithPromptLength

from dsconf import DatasetConfig, DatasetMixer

from dotenv import load_dotenv

load_dotenv('../.env')

class TensorBoardCallback(TrainerCallback):
    """Callback to log training metrics to TensorBoard."""

    def __init__(self, log_dir: str = './runs'):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Log metrics to TensorBoard.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Dictionary of metrics to log
        """
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Close the TensorBoard writer at the end of training."""
        self.writer.close()

@dataclass
class HypernetConfig:
    hidden_dim: int = 256
    lora_rank: int = 2
    lora_alpha: int = 8
    lora_dropout: float = 0.01
    layers_module_name: str = 'layers'

@dataclass
class TrainConfig:
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    bf16: bool = False
    logging_steps: int = 10
    warmup_ratio: float = 0.1
    dataloader_pin_memory: bool = False  # Avoid DataParallel issues

@dataclass
class MixerConfig:
    seed: Optional[int] = None
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'


def parse_args():
    """Parse commandline arguments"""
    parser = ArgumentParser(description="Train TaskWeaver Hypernetwork")

    parser.add_argument('--model', type=str, default='EleutherAI/pythia-70m', help='Pretrained model path')
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], help="Datasets to include in Mixer")
    parser.add_argument('--output_dir', type=str, default='_models/hypernet', help='Base output directory for trained models')
    parser.add_argument('--device', type=str, default='auto', help="Device to train on")
    parser.add_class_arguments(HypernetConfig, 'hypernet', help='TaskWeaver configuration')
    parser.add_class_arguments(TrainConfig, 'train', help='Training configuration parameters')
    parser.add_class_arguments(MixerConfig, 'mixer', help='DatasetMixer configuration parameters')
    parser.add_argument('--hypernet.target_modules', type=str, nargs='+', default=['query_key_value'], help='Modules to generate LoRA weights for')

    return parser.parse_args()


def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load pretrained model and tokenizer.

    Args:
        model_name: Name or path of the pretrained model

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model and tokenizer: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_datasets(datasets: List[str], mixer_conf:MixerConfig, is_chat: bool, train_split: str = "train", test_split="test") -> Tuple[Dataset, Dataset, str]:
    """
    Get train and test datasets
    """
    print(f"Loading {'Chat' if is_chat else 'Non-Chat'} version of the datasets")
    
    datasets = datasets if 'all' not in datasets else None

    config = DatasetMixer(datasets, seed=mixer_conf.seed, stopping_strategy=mixer_conf.stopping_strategy)
    train_dataset = config.load_and_process(is_chat, train_split)
    test_dataset = config.load_and_process(is_chat, test_split)
    return train_dataset, test_dataset, config.id()


def main():

    args = parse_args()

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    # Force single GPU to avoid DataParallel issues
    # TaskWeaver's dynamic LoRA injection stores weights as instance attributes,
    # which don't replicate properly across DataParallel's model copies
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Warning: Multiple GPUs detected ({torch.cuda.device_count()}). "
              "Forcing single GPU mode for TaskWeaver compatibility.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print(f"Using device: {device}")

    lm, tokenizer = load_model_and_tokenizer(args.model)

    train_dataset, test_dataset, dataset_id = prepare_datasets(
                                    datasets=args.datasets, 
                                    mixer_conf=args.mixer,
                                    is_chat=tokenizer.chat_template is not None
                                    )
    
    hypernet = TaskWeaver(
        lm=lm,
        hidden_dim=args.hypernet.hidden_dim,
        lora_rank=args.hypernet.lora_rank,
        lora_alpha=args.hypernet.lora_alpha,
        target_modules=args.hypernet.target_modules,
        lora_dropout=args.hypernet.lora_dropout,
        model_name=args.model
    )

    hypernet.to(device)

    hypernet.print_trainable_parameters()

    pad_token = tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    collator = DataCollatorWithPromptLength(pad_token_id=pad_token_id)

    run_identifier = f"d{args.hypernet.hidden_dim}_r{args.hypernet.lora_rank}_a{args.hypernet.lora_alpha}"
    output_dir = os.path.join(args.output_dir, args.model.replace('/', '_'), run_identifier)

    trainer_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.train.num_train_epochs,
        per_device_train_batch_size=args.train.per_device_train_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        learning_rate=args.train.learning_rate,
        bf16=args.train.bf16,
        logging_steps=args.train.logging_steps,
        warmup_ratio=args.train.warmup_ratio,
        save_strategy='no'
        # Disable DataParallel - TaskWeaver's dynamic LoRA injection doesn't support it
        # The hypernetwork generates weights that are stored as instance attributes,
        # which don't properly replicate across DataParallel's model copies
        dataloader_pin_memory=args.train.dataloader_pin_memory,
    )

    trainer = SFTTrainer(
        model=hypernet,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        args=trainer_args
    )

    trainer.train()

    metrics = trainer.evaluate()
    print(f"Evalation metrics: {metrics}")

    print(f"Saving model")
    hypernet.save_pretrained(output_dir)


if __name__ == "__main__":
    main()