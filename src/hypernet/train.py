"""
Training Script for TaskWeaver Hypernetwork

This script provides a complete training pipeline for the TaskWeaver hypernetwork
with TensorBoard logging support.
"""

import os
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from torch.utils.tensorboard import SummaryWriter

from hypernetwork import TaskWeaver
from collator import DataCollatorWithPromptLengths
from dataset import create_dataset
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train TaskWeaver Hypernetwork'
    )

    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default='EleutherAI/pythia-70M-deduped',
        help='Pretrained model name or path'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=1024,
        help='Hidden dimension for hypernetwork'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=2,
        help='Rank for LoRA decomposition'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=8,
        help='Scaling factor for LoRA'
    )
    parser.add_argument(
        '--lora_dropout',
        type=float,
        default=0.01,
        help='Dropout probability for LoRA'
    )
    parser.add_argument(
        '--lora_target_layers',
        type=str,
        nargs='+',
        default=['query_key_value'],
        help='Names of layers to apply LoRA to'
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset_config',
        type=str,
        default='configs/dataset_config.json',
        help='Path to dataset configuration JSON file'
    )
    parser.add_argument(
        '--use_default_dataset',
        action='store_true',
        help='Use default GSM8K dataset configuration'
    )

    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./taskweaver_output',
        help='Directory for model outputs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_false',
        help='Whether to use FP16 during training'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=10,
        help='Number of steps between logging'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help='Number of steps between checkpoints'
    )
    parser.add_argument(
        '--tensorboard_dir',
        type=str,
        default='./runs',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to run training on (e.g., "cpu", "cuda", "cuda:0", "mps", or "auto" for automatic selection)'
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
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


def prepare_dataset(tokenizer, dataset_config_path: Optional[str] = None):
    """
    Prepare training dataset.

    Args:
        tokenizer: Tokenizer instance
        dataset_config_path: Path to dataset configuration file (optional)

    Returns:
        Processed dataset
    """
    if dataset_config_path and os.path.exists(dataset_config_path):
        import json
        print(f"Loading dataset configuration from: {dataset_config_path}")
        with open(dataset_config_path, 'r') as f:
            dataset_configs = json.load(f)
    else:
        # Default configuration: GSM8K only
        print("Using default dataset configuration (GSM8K)")
        dataset_configs = [
            {
                'name': 'gsm8k',
                'dataset_path': 'openai/gsm8k',
                'dataset_config': 'main'
            }
        ]

    dataset = create_dataset(tokenizer, dataset_configs)
    print(f"Dataset created with {len(dataset)} examples")

    return dataset


def main():
    """Main training function."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_args()

    # Login to HuggingFace if token is available
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model and tokenizer
    lm, tokenizer = load_model_and_tokenizer(args.model_name)

    # Create TaskWeaver hypernetwork
    print("Creating TaskWeaver hypernetwork...")
    hypernet = TaskWeaver(
        lm=lm,
        hidden_dim=args.hidden_dim,
        lora_rank=args.lora_rank,
        lora_target_layers=args.lora_target_layers,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        model_name=args.model_name
    )

    # Move model to specified device
    hypernet = hypernet.to(device)

    hypernet.print_trainable_parameters()

    # Prepare dataset
    dataset_config = None if args.use_default_dataset else args.dataset_config
    train_dataset = prepare_dataset(tokenizer, dataset_config)

    # Create data collator
    data_collator = DataCollatorWithPromptLengths(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        report_to=['tensorboard'],
        logging_dir=args.tensorboard_dir,
        warmup_ratio=0.1
    )

    # Create TensorBoard callback
    tensorboard_callback = TensorBoardCallback(log_dir=args.tensorboard_dir)

    # Create trainer
    trainer = Trainer(
        model=hypernet,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[tensorboard_callback]
    )

    # Print training info
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"LoRA target layers: {args.lora_target_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output dir: {args.output_dir}")
    print(f"TensorBoard dir: {args.tensorboard_dir}")
    print("="*50 + "\n")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model using TaskWeaver's save_pretrained
    print(f"\nSaving final model to {args.output_dir}")
    hypernet.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining complete!")
    print(f"View training logs with: tensorboard --logdir {args.tensorboard_dir}")
    print(f"\nTo load the model later, use:")
    print(f"  from hypernetwork import TaskWeaver")
    print(f"  model = TaskWeaver.from_pretrained('{args.output_dir}')")


if __name__ == '__main__':
    main()
