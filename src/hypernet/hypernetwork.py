"""
TaskWeaver Hypernetwork

This module implements the TaskWeaver hypernetwork that generates task-specific
LoRA weights dynamically based on the input prompt.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Literal
from operator import attrgetter
from functools import partial
from transformers import AutoModelForCausalLM

from dynamic_lora import DynamicLoraLinear


class TaskWeaver(nn.Module):
    """
    TaskWeaver: A hypernetwork that generates task-specific LoRA weights.

    This module wraps a language model and replaces target linear layers with
    DynamicLoraLinear layers. It then uses a hypernetwork to generate instance-level
    LoRA weights based on the input prompt's semantic embedding.

    Args:
        lm: Pre-trained language model
        hidden_dim: Hidden dimension for the hypernetwork
        lora_rank: Rank for LoRA decomposition
        lora_target_layers: Names of layers to replace with LoRA
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA (default: 0.0)
        layers_module_name: Name of the layers module in the LM (default: 'layers')
        model_name: Name of the base LM for saving/loading (default: None)
    """

    def __init__(
        self,
        lm: AutoModelForCausalLM,
        hidden_dim: int,
        lora_rank: int,
        lora_target_layers: List[str],
        lora_alpha: float,
        lora_dropout: float = 0.0,
        layers_module_name: str = 'layers',
        model_name: Optional[str] = None
    ):
        super().__init__()
        self.lm = lm
        self.lora_target_layers = lora_target_layers
        self.lora_rank = lora_rank
        self.hidden_dim = hidden_dim
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.layers_module_name = layers_module_name
        self.model_name = model_name

        # LLM config values
        self.lm_num_layers = self.lm.config.num_hidden_layers
        self.lm_hidden_dim = self.lm.config.hidden_size

        # Get reference to the layers module
        lm_layers_ref = self.get_layers_ref(layers_module_name)
        assert isinstance(lm_layers_ref, nn.ModuleList), "Layers must be an nn.ModuleList"

        # Create partial function for DynamicLoraLinear
        dynamic_lora_fn = partial(
            DynamicLoraLinear,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=self.lm.device
        )

        # Replace target linear layers with DynamicLoraLinear
        self.module_references, self.in_features, self.out_features = self.replace_linears(
            self.lora_target_layers, lm_layers_ref, dynamic_lora_fn
        )

        # Hypernetwork components
        self.semantic_proj = nn.Linear(self.lm_hidden_dim, hidden_dim)

        self.module_embedding = nn.Embedding(len(lora_target_layers), hidden_dim)
        self.matrix_embedding = nn.Embedding(2, hidden_dim)
        self.layer_embedding = nn.Embedding(self.lm_num_layers, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Output heads for each module and matrix (A/B)
        self.heads = nn.ModuleDict({
            module_name: nn.ModuleDict({
                'A': nn.Linear(hidden_dim, self.in_features[module_name] * self.lora_rank),
                'B': nn.Linear(hidden_dim, self.out_features[module_name] * self.lora_rank)
            }) for module_name in self.lora_target_layers
        })

        self._freeze_lm()
        self._init_weights()

    def _freeze_lm(self) -> None:
        for param in self.lm.parameters():
            param.requires_grad = False

    def _init_weights(self):
        # Initialize MLP layers with smaller weights
        for module in [self.semantic_proj, self.mlp]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output heads to produce small initial LoRA weights
        for module_name in self.lora_target_layers:
            for matrix_name in ['A', 'B']:
                head = self.heads[module_name][matrix_name]
                nn.init.zeros_(head.weight)  # Start with zero weights
                
                if matrix_name == 'A':
                    # Small random bias for A matrix
                    if hasattr(head, 'bias') and head.bias is not None:
                        nn.init.uniform_(head.bias, -1/(np.sqrt(2) * self.in_features[module_name]), 
                                                    1/(np.sqrt(2) * self.in_features[module_name]))
                else:  # B matrix
                    # Zero bias for B matrix (standard LoRA init)
                    if hasattr(head, 'bias') and head.bias is not None:
                        nn.init.zeros_(head.bias)

    def get_layers_ref(self, layers_module_name: str) -> nn.Module:
        """
        Get reference to the layers module in the language model.

        Args:
            layers_module_name: Name of the layers module

        Returns:
            Reference to the layers module
        """
        for name, _ in self.lm.named_modules():
            if not name or name.count('.') == 0:
                continue
            path, attribute = name.rsplit(".", 1)
            if attribute == layers_module_name:
                return attrgetter(name)(self.lm)

    def replace_linears(
        self,
        lora_target_layers: List[str],
        lm_layers_ref: nn.ModuleList,
        dynamic_lora_fn: callable
    ) -> Tuple[List[Dict[str, DynamicLoraLinear]], Dict[str, int], Dict[str, int]]:
        """
        Replace target Linear layers with DynamicLoraLinear layers.

        Args:
            lora_target_layers: Names of layers to replace
            lm_layers_ref: Reference to the layers module
            dynamic_lora_fn: Function to create DynamicLoraLinear instances

        Returns:
            Tuple of (module_references, in_features, out_features)
        """
        references = [{} for _ in range(self.lm_num_layers)]
        in_features = {}
        out_features = {}

        for i, layer in enumerate(lm_layers_ref):
            for name, _ in layer.named_modules():
                if not name or name.count('.') == 0:
                    continue

                path, attribute = name.rsplit('.', 1)
                if attribute not in lora_target_layers:
                    continue

                parent_ref = attrgetter(path)(layer)
                linear_ref = getattr(parent_ref, attribute)
                assert isinstance(linear_ref, nn.Linear), "Can only adapt nn.Linear layers"

                in_features[attribute] = linear_ref.in_features
                out_features[attribute] = linear_ref.out_features
                
                dynamic_lora_layer = dynamic_lora_fn(
                                        in_features=linear_ref.in_features, 
                                        out_features=linear_ref.out_features, 
                                        bias=linear_ref.bias is not None
                                    )
                
                dynamic_lora_layer.replicate(linear_ref)
                setattr(parent_ref, attribute, dynamic_lora_layer)

                references[i][attribute] = getattr(parent_ref, attribute)

        return references, in_features, out_features

    def _hypernet_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[Literal['A', 'B'], torch.Tensor]]]:
        """
        Forward pass through the hypernetwork to generate LoRA weights.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_lengths: Length of prompts in each sequence (optional)

        Returns:
            List of LoRA weights for each layer
        """
        self.clear_lora_weights()

        batch_size = input_ids.shape[0]

        # Create prompt mask if prompt_lengths provided
        if prompt_lengths is not None:
            seq_len = attention_mask.shape[1]
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            prompt_lengths_expanded = prompt_lengths.unsqueeze(1)
            prompt_mask = (positions < prompt_lengths_expanded).long()
        else:
            prompt_mask = attention_mask

        # Get semantic embedding from the last prompt token
        with torch.no_grad():
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]

            if prompt_lengths is not None:
                last_prompt_indices = prompt_lengths - 1
                semantic_embedding = last_hidden[
                    torch.arange(batch_size, device=last_hidden.device),
                    last_prompt_indices
                ]
            else:
                last_indices = attention_mask.sum(dim=1) - 1
                semantic_embedding = last_hidden[
                    torch.arange(batch_size, device=last_hidden.device),
                    last_indices
                ]

        # Project semantic embedding
        semantic_embedding = self.semantic_proj(semantic_embedding.detach())

        # Generate LoRA weights for each layer, module, and matrix
        lora_weights = []

        for layer_idx in range(self.lm_num_layers):
            layer_dict = {}
            layer_emb = self.layer_embedding.weight[layer_idx:layer_idx + 1]

            for module_idx, module_name in enumerate(self.lora_target_layers):
                module_dict = {}
                module_emb = self.module_embedding.weight[module_idx:module_idx + 1]

                for matrix_idx, matrix_name in enumerate(['A', 'B']):
                    matrix_emb = self.matrix_embedding.weight[matrix_idx:matrix_idx + 1]

                    # Combine embeddings
                    combined_emb = semantic_embedding + layer_emb + module_emb + matrix_emb
                    combined_emb = self.mlp(combined_emb)

                    # Generate weight matrix
                    flat_weight = self.heads[module_name][matrix_name](combined_emb)

                    if matrix_name == 'A':
                        weight = flat_weight.view(
                            batch_size, self.lora_rank, self.in_features[module_name]
                        )
                    else:
                        weight = flat_weight.view(
                            batch_size, self.out_features[module_name], self.lora_rank
                        )

                    module_dict[matrix_name] = weight

                layer_dict[module_name] = module_dict

            lora_weights.append(layer_dict)

        return lora_weights

    def inject_lora_weights(
        self,
        lora_weights: List[Dict[str, Dict[Literal['A', 'B'], torch.Tensor]]]
    ) -> None:
        """
        Inject LoRA weights into the DynamicLoraLinear layers.

        Args:
            lora_weights: List of LoRA weights for each layer
        """
        for i, layer_dict in enumerate(self.module_references):
            for module_name in layer_dict:
                layer_dict[module_name].set_lora_paramters(**lora_weights[i][module_name])

    def clear_lora_weights(self) -> None:
        """Clear all LoRA weights from the DynamicLoraLinear layers."""
        for layer_dict in self.module_references:
            for module_name in layer_dict:
                layer_dict[module_name].unset_lora_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        skip_hypernet: bool = False
    ):
        """
        Forward pass through TaskWeaver.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training (optional)
            prompt_lengths: Length of prompts in each sequence (optional)
            skip_hypernet: If True, skip hypernetwork and use base model (default: False)

        Returns:
            Model outputs from the language model
        """
        if not skip_hypernet:
            lora_weights = self._hypernet_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths
            )
            self.inject_lora_weights(lora_weights)

        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    @torch.no_grad
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        **generation_kwargs
    ):
        """
        Generate text using the task-adapted model.

        This method first generates LoRA weights using the hypernetwork,
        injects them into the model, and then runs generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            prompt_lengths: Length of prompts in each sequence (optional)
            **generation_kwargs: Additional arguments passed to the LM's generate method

        Returns:
            Generated token IDs
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Generate LoRA weights based on the prompt
        lora_weights = self._hypernet_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths
        )

        # Inject LoRA weights into the model
        self.inject_lora_weights(lora_weights)

        # Generate using the adapted model
        try:
            outputs = self.lm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        finally:
            # Clear LoRA weights after generation
            self.clear_lora_weights()

        return outputs

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the TaskWeaver hypernetwork to a directory.

        This saves the hypernetwork configuration and weights, but NOT the base
        language model. The LM name is stored for later reloading.

        Args:
            save_directory: Directory path to save the model
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config = {
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'lora_rank': self.lora_rank,
            'lora_target_layers': self.lora_target_layers,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'layers_module_name': self.layers_module_name
        }

        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Get state dict excluding the base LM
        state_dict = {}
        for name, param in self.named_parameters():
            # Only save hypernetwork parameters (exclude 'lm.' prefix)
            if not name.startswith('lm.'):
                state_dict[name] = param

        # Save the hypernetwork weights
        weights_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(state_dict, weights_path)

        print(f"TaskWeaver hypernetwork saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> 'TaskWeaver':
        """
        Load a TaskWeaver hypernetwork from a directory.

        This loads the hypernetwork configuration and weights, and initializes
        the base language model using transformers.

        Args:
            save_directory: Directory path containing the saved model
            model_name: Optional model name to override the saved config
            device: Optional device to load the model on

        Returns:
            TaskWeaver instance with loaded weights
        """
        # Load configuration
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Use provided model_name or fall back to saved config
        lm_name = model_name if model_name is not None else config['model_name']

        if lm_name is None:
            raise ValueError(
                "No model_name provided and no model_name found in saved config. "
                "Please provide a model_name parameter."
            )

        # Load the base language model
        print(f"Loading base language model: {lm_name}")
        lm = AutoModelForCausalLM.from_pretrained(lm_name)

        # Create TaskWeaver instance
        taskweaver = cls(
            lm=lm,
            hidden_dim=config['hidden_dim'],
            lora_rank=config['lora_rank'],
            lora_target_layers=config['lora_target_layers'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            layers_module_name=config['layers_module_name'],
            model_name=lm_name
        )

        # Load hypernetwork weights
        weights_path = os.path.join(save_directory, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu')

        # Load state dict (only hypernetwork parameters)
        taskweaver.load_state_dict(state_dict, strict=False)

        # Move to device if specified
        if device is not None:
            taskweaver = taskweaver.to(device)

        print(f"TaskWeaver hypernetwork loaded from {save_directory}")

        return taskweaver

    @property
    def device(self) -> torch.device:
        """Get the device of the language model."""
        return self.lm.device
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        
        print("\n=== Trainable Parameters ===")
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✓ {name}: {param.numel():,}")
        
        print(f"\nTotal params: {all_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
