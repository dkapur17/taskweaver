"""
Data Collator with Prompt Lengths

This module provides a data collator that handles padding and preserves
prompt lengths for the TaskWeaver hypernetwork.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorWithPromptLengths(DataCollatorForLanguageModeling):
    """
    Data collator that handles padding and prompt lengths.

    This collator extends DataCollatorForLanguageModeling to preserve
    prompt_length information, which is needed by the hypernetwork to
    extract semantic embeddings from the last prompt token.

    Args:
        tokenizer: Tokenizer instance
        mlm: Whether to use masked language modeling (default: False)
        return_tensors: Format for returned tensors (default: 'pt')
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch with proper padding.

        Args:
            features: List of feature dictionaries

        Returns:
            Batched dictionary with padded tensors
        """
        # Extract prompt_lengths before processing
        prompt_lengths = None
        if features and 'prompt_length' in features[0]:
            prompt_lengths = [f.pop('prompt_length') for f in features]

        # Manual padding
        batch = {}

        # Get max length in batch
        max_length = max(len(f['input_ids']) for f in features)

        # Pad each sequence
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            # Convert to list if needed
            input_id = feature['input_ids']
            if isinstance(input_id, torch.Tensor):
                input_id = input_id.tolist()

            attn_mask = feature['attention_mask']
            if isinstance(attn_mask, torch.Tensor):
                attn_mask = attn_mask.tolist()

            label = feature['labels']
            if isinstance(label, torch.Tensor):
                label = label.tolist()

            # Calculate padding length
            padding_length = max_length - len(input_id)

            # Pad sequences (padding on the right for causal LM)
            input_ids.append(input_id + [self.tokenizer.pad_token_id] * padding_length)
            attention_mask.append(attn_mask + [0] * padding_length)
            labels.append(label + [-100] * padding_length)  # -100 is ignored in loss

        # Convert to tensors
        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        batch['labels'] = torch.tensor(labels, dtype=torch.long)

        # Add prompt_lengths back
        if prompt_lengths is not None:
            batch['prompt_lengths'] = torch.tensor(prompt_lengths, dtype=torch.long)

        return batch
