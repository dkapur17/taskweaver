"""
TaskWeaver Hypernetwork Package

This package provides a hypernetwork-based approach to dynamic LoRA adaptation
for language models.
"""

from .dynamic_lora import DynamicLoraLinear
from .hypernetwork import TaskWeaver
from .collator import DataCollatorWithPromptLengths
from .dataset import (
    DatasetProcessor,
    GSM8KProcessor,
    ARCProcessor,
    MultiDatasetCreator,
    create_dataset
)

__all__ = [
    'DynamicLoraLinear',
    'TaskWeaver',
    'DataCollatorWithPromptLengths',
    'DatasetProcessor',
    'GSM8KProcessor',
    'ARCProcessor',
    'MultiDatasetCreator',
    'create_dataset',
]

__version__ = '0.1.0'
