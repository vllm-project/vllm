from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from vllm.adapter_commons.layers import AdapterMapping
from vllm.config import PromptAdapterConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)


@dataclass
class PromptAdapterMapping(AdapterMapping):
    pass


class VocabParallelEmbeddingWithPromptAdapter(nn.Module):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.emb_layer = self.base_layer
        if 'LoRA' in base_layer.__class__.__name__:
            self.emb_layer = self.base_layer.base_layer

    def create_prompt_adapter_weights(
            self, prompt_adapter_config: PromptAdapterConfig):
        self.embeddings_tensors = torch.zeros(
            (
                prompt_adapter_config.max_prompt_adapters,
                prompt_adapter_config.max_prompt_adapter_token,
                self.emb_layer.embedding_dim,
            ),
            dtype=self.emb_layer.weight.dtype,
            device=self.emb_layer.weight.device,
        )
        self.adapter_lengths = torch.zeros(
            prompt_adapter_config.max_prompt_adapters,
            dtype=torch.long,
            device=self.emb_layer.weight.device)

        self.indices_gpu: torch.Tensor
        self.embedding_indices_gpu: torch.Tensor

    def reset_prompt_adapter(self, index: int):
        self.embeddings_tensors[index] = 0

    def set_prompt_adapter(
        self,
        index: int,
        adapter_model: Optional[torch.Tensor],
    ):
        self.reset_prompt_adapter(index)
        if adapter_model is not None:
            length = adapter_model.shape[0]
            self.embeddings_tensors[index, :length] = adapter_model
            self.adapter_lengths[index] = length

    def set_mapping(
        self,
        prompt_indices: torch.Tensor,
        prompt_embedding_indices: torch.Tensor,
    ):
        self.indices_gpu = prompt_indices.to(
            device=self.emb_layer.weight.device)
        self.embedding_indices_gpu = prompt_embedding_indices.to(
            device=self.emb_layer.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base_layer(x)
        if self.embedding_indices_gpu.ndim > 1:
            valid_mask = self.indices_gpu != -1
            gathered_embeddings = self.embeddings_tensors[
                self.embedding_indices_gpu[:, 0],
                self.embedding_indices_gpu[:, 1]]

            # Update hidden states
            hidden_states[valid_mask] = gathered_embeddings
        return hidden_states