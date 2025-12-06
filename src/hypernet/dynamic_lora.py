"""
Dynamic LoRA Linear Layer

This module implements a linear layer that can accept dynamic LoRA weights per batch,
allowing for instance-level LoRA adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class DynamicLoraLinear(nn.Linear):
    """
    A linear layer with dynamic LoRA (Low-Rank Adaptation) capability.

    This layer extends nn.Linear to support instance-level LoRA by accepting
    batch-specific A and B matrices that define the low-rank adaptation.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        lora_rank: Rank of the LoRA decomposition
        lora_alpha: Scaling factor for LoRA weights
        lora_dropout: Dropout probability for LoRA (default: 0.0)
        bias: Whether to use bias (default: True)
        device: Device to place the layer on
        dtype: Data type for the layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        assert lora_rank > 0, "Use nn.Linear for Non-Lora Layer"

        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout
        self.lora_scaling = lora_alpha / lora_rank

        self.A = None
        self.B = None
        self.reset_parameters()

    def replicate(self, target: nn.Linear) -> None:
        """
        Replicate the parameters of a target nn.Linear

        Args:
            target (nn.Linear): Linear layer to copy weight and bias from
        """
        assert isinstance(target, nn.Linear), "Can only replicate nn.Linear"

        self.weight.data = target.weight.data
        if self.bias is not None:
            self.bias.data = target.bias.data

    def set_lora_paramters(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """
        Set the LoRA parameters for the current forward pass.

        Args:
            A: Low-rank matrix of shape [batch_size, rank, input_dim]
            B: Low-rank matrix of shape [batch_size, output_dim, rank]
        """
        self.A = A
        self.B = B


    def unset_lora_parameters(self) -> None:
        """Clear the LoRA parameters."""
        self.A = None
        self.B = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional LoRA adaptation.

        Args:
            input: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]

        Raises:
            RuntimeError: If batch size mismatch between input and LoRA weights
        """
        # Standard linear transformation if no LoRA weights
        if self.A is None:
            return F.linear(input, self.weight, self.bias)

        # Batch size sanity check
        batch_size = input.size(0)
        if self.A.size(0) != batch_size:
            raise RuntimeError(
                f"Batch size mismatch! Input batch_size={batch_size}, "
                f"but LoRA A has batch_size={self.A.size(0)}. "
                f"Old LoRA weights are being reused!"
            )
        
        # Instance-level LoRA transformation
        out_delta = einsum(
            self.A, self.B, F.dropout(input, self.lora_dropout),
            'b r i, b o r, b s i -> b s o'
        )

        # Wx + b + s*BAx
        return F.linear(input, self.weight, self.bias) + self.lora_scaling * out_delta

    def extra_repr(self) -> str:
        """String representation including LoRA parameters."""
        out = nn.Linear.extra_repr(self)
        out += f', lora_rank={self.lora_rank}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout}'
        return out
