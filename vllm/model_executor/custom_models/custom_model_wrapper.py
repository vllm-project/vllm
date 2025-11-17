# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Base class for integrating custom model implementations with vLLM.

This module provides an abstract base class that enforces the vLLM model interface,
making it easy to integrate external model implementations (e.g., from TorchTitan,
NanoGPT, etc.) with vLLM.

Example usage:
    ```python
    from some_external_lib.models import ExternalModel
    from vllm.model_executor.models.custom_model_wrapper import VLLMModelForCausalLM


    class MyCustomModelForCausalLM(VLLMModelForCausalLM):
        def __init__(self, vllm_config, parallel_context=None, **kwargs):
            super().__init__()
            self.model = ExternalModel(...)  # Create external model
            # Replace attention layers with vLLM's trainable attention

        def get_input_embeddings(self, input_ids):
            return self.model.tok_embeddings(input_ids)

        def forward(self, input_ids, positions=None, **kwargs):
            # Forward pass
            return hidden_states

        def compute_logits(self, hidden_states, sampling_metadata=None):
            return self.model.output(hidden_states)

        def load_weights(self, weights_iter):
            # Load weights from HuggingFace checkpoint
            pass
    ```
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


class VLLMModelForCausalLM(nn.Module, ABC):
    """
    Abstract base class for integrating custom model implementations with vLLM.

    This class enforces the vLLM model interface that all text generation models
    must implement. Subclasses should:
    1. Import and instantiate the external model in __init__
    2. Replace attention layers with vLLM's trainable attention
    3. Implement the abstract methods below

    Class attributes:
        supports_pp: Whether pipeline parallelism is supported
        supports_multimodal: Whether multimodal inputs are supported
    """

    supports_pp: bool = False
    supports_multimodal: bool = False

    @abstractmethod
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input token IDs to embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len] or [total_tokens]

        Returns:
            Embeddings [batch, seq_len, hidden_size] or [total_tokens, hidden_size]
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len] or [total_tokens]
            positions: Position indices from vLLM for RoPE indexing
            **kwargs: Additional vLLM-specific arguments

        Returns:
            Hidden states before final projection [batch, seq_len, hidden_size]
            or [total_tokens, hidden_size]
        """
        pass

    @abstractmethod
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: Output from forward() [batch, seq_len, hidden_size]
            sampling_metadata: vLLM sampling metadata (optional)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def load_weights(self, weights_iter: Iterator[tuple[str, torch.Tensor]]) -> None:
        """
        Load weights from HuggingFace checkpoint.

        This method should map HuggingFace weight names to model parameter names
        and load them into the model.

        Args:
            weights_iter: Iterator yielding (name, tensor) tuples from HF checkpoint
        """
        pass
