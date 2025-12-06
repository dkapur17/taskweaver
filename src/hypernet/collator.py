"""
Data Collator with Prompt Lengths

This module provides a data collator that handles padding and preserves
prompt lengths for the TaskWeaver hypernetwork.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


@dataclass
class DataCollatorWithPromptLength(DataCollatorForLanguageModeling):

    def __call__(self, examples:List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        batch['prompt_length'] = (batch['labels'] != -100).int().argmax(dim=1)
        return batch