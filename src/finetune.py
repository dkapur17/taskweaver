import sys
from jsonargparse import ArgumentParser, Namespace
from dataclasses import dataclass

from dsconf import DatasetConfig, DatasetMixer
from lora import LoraFinetuner

from utils import print_args, serialize_args

from typing import List, Type, Optional, Literal
from dotenv import load_dotenv

load_dotenv('../.env')


@dataclass
class LoraConfig:
    rank: int = 2
    alpha: int = 8
    dropout: float = 0.05


@dataclass
class MixerConfig:
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'


@dataclass
class TrainConfig:
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    bf16: bool = False
    logging_steps: int = 10
    save_strategy: str = 'no'


@dataclass
class DatasetSplitConfig:
    train_split: str = 'train'
    test_split: Optional[str] = 'test'


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='google/gemma-3-270m-it', help='HuggingFace model path')
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'], help='Datasets to finetune on (use "all" for all registered datasets)')
    parser.add_argument('--ignore_datasets', type=str, nargs='+', default=[], help='Datasets to ignore when using "all"')
    parser.add_argument('--device_map', type=str, default='auto', help='Device map for model loading')
    parser.add_argument('--output_dir', type=str, default='_models/lora', help='Base output directory for trained models')
    parser.add_argument('--run_eval', action='store_true', help='Run evaluation after training')
    parser.add_class_arguments(LoraConfig, 'lora', help='LoRA configuration parameters')
    parser.add_class_arguments(TrainConfig, 'train', help='Training configuration parameters')
    parser.add_class_arguments(DatasetSplitConfig, 'dataset', help='Dataset split configuration')
    parser.add_class_arguments(MixerConfig, 'mixer', help='Configurations for DatasetMixer. Only used if "mix" is provided as a dataset.')

    # Override target_modules to support nargs='+'
    parser.add_argument('--lora.target_modules', type=str, nargs='+', default=['q_proj', 'v_proj'], help='Target modules for LoRA (can specify multiple)')

    args = parser.parse_args()

    # Print all arguments neatly
    print_args(args)

    return args

def get_dataset_configs(datasets: List[str], ignore_list: List[str], mixer_config: MixerConfig) -> List[Type[DatasetConfig]]:

    configs = []

    if 'all' in datasets:
        for path, name in DatasetConfig.list_available():
            dataset_id = f"{path}.{name}" if name else path
            if dataset_id not in ignore_list:
                configs.append(DatasetConfig.from_dataset_path(path, name))
    elif 'mix' in datasets:
        target_datasets = [path if name is None else f"{path}.{name}" for path, name in DatasetConfig.list_available()]
        target_datasets = [dataset_id for dataset_id in target_datasets if dataset_id not in ignore_list]
        print(f"Mixing datasets: {target_datasets}")
        configs.append(DatasetMixer(target_datasets, stopping_strategy=mixer_config.stopping_strategy))
    else:
        for dataset in datasets:
            if dataset in ignore_list:
                print(f"WARNING: {dataset} in list and ignore list. Will be ignoring.")
                continue
            if '.' in dataset:
                path, name = dataset.split('.')
            else:
                path, name = dataset, None
            configs.append(DatasetConfig.from_dataset_path(path, name))

    return configs

if __name__ == "__main__":

    args = parse_args()
    dataset_configs = get_dataset_configs(args.datasets, args.ignore_datasets, args.mixer)

    for config in dataset_configs:
        print(f"Finetuning for dataset: {config.id()}")
        finetuner = LoraFinetuner(
            model_path=args.model,
            dataset_config=config,
            dataset_train_split=args.dataset.train_split,
            dataset_test_split=args.dataset.test_split,
            lora_rank=args.lora.rank,
            lora_alpha=args.lora.alpha,
            lora_dropout=args.lora.dropout,
            target_modules=args.lora.target_modules,
            device_map=args.device_map,
            output_dir=args.output_dir
        )

        total_params, trainable_params = finetuner.print_trainable_parameters()

        finetuner.train(
            num_train_epochs=args.train.num_train_epochs,
            per_device_train_batch_size=args.train.per_device_train_batch_size,
            gradient_accumulation_steps=args.train.gradient_accumulation_steps,
            learning_rate=args.train.learning_rate,
            bf16=args.train.bf16,
            logging_steps=args.train.logging_steps,
            save_strategy=args.train.save_strategy,
            run_eval=args.run_eval
        )

        finetuner.save(
            metadata = {
                'invocation': 'python ' + ' '.join(sys.argv),
                'configuration': serialize_args(args),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'percent_trainable': trainable_params / total_params
            }
        )
    
