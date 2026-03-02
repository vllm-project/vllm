# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Predictable dummy model for testing extract_hidden_states.

Subclasses LlamaForCausalLM but overrides the model to produce deterministic
hidden states: layer i outputs values equal to (i).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.sequence import IntermediateTensors


class PredictableLlamaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.aux_hidden_state_layers = tuple[int, ...]()

        # Create minimal embed_tokens for embedding
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )

        # Required for pipeline parallelism
        from vllm.model_executor.models.utils import (
            make_empty_intermediate_tensors_factory,
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input IDs."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass that produces predictable outputs.

        Returns:
            If aux_hidden_state_layers is set: (hidden_states, aux_hidden_states)
            Otherwise: hidden_states
        """
        # Determine sequence length
        if inputs_embeds is not None:
            seq_len = inputs_embeds.shape[0]
            device = inputs_embeds.device
        elif input_ids is not None:
            seq_len = input_ids.shape[0] if input_ids.ndim == 1 else input_ids.shape[-1]
            device = input_ids.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Final hidden states (last layer value)
        hidden_states = torch.full(
            (seq_len, self.config.hidden_size),
            fill_value=float(self.config.num_hidden_layers),
            device=device,
            dtype=torch.bfloat16,
        )

        # Check if we need auxiliary hidden states
        if len(self.aux_hidden_state_layers) > 0:
            aux_hidden_states = []
            for layer_idx in self.aux_hidden_state_layers:
                # Fill with (layer_idx) for predictability
                layer_hidden = torch.full(
                    (seq_len, self.config.hidden_size),
                    fill_value=float(layer_idx),
                    device=device,
                    dtype=torch.bfloat16,
                )
                aux_hidden_states.append(layer_hidden)

            return hidden_states, aux_hidden_states

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Skip weight loading."""
        return set()


class PredictableLlamaForCausalLM(LlamaForCausalLM):
    """Predictable Llama model for testing.

    Overrides _init_model to use PredictableLlamaModel instead of LlamaModel.
    """

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] | None = None,
    ):
        """Initialize with predictable model."""
        return PredictableLlamaModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Skip weight loading for dummy model."""
        return set()
