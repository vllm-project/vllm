from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy
import torch
from torch import nn

from vllm.adapter_commons.layers import AdapterMapping
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)


@dataclass
class PromptAdapterMapping(AdapterMapping):
    pass


class VocabParallelEmbeddingWithPromptAdapter(nn.Module):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.embedding_tensors: Dict[int, torch.Tensor] = {}
        self.indices: torch.Tensor

    def reset_prompt_adapter(self, index: int):
        self.embedding_tensors[index] = 0

    def set_prompt_adapter(
        self,
        index: int,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_prompt_adapter(index)
        if embeddings_tensor is not None:
            self.embedding_tensors[index] = embeddings_tensor

    def set_mapping(
        self,
        base_indices: List[int],
    ):
        self.indices = base_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base_layer(x)
        unique_indices = numpy.unique(self.indices)
        for idx in unique_indices:
            if idx != 0:
                pa_idx = self.embedding_tensors[idx].prompt_embedding
                mask = (self.indices == idx)
                try:
                    n_adapters = sum(mask) // pa_idx.shape[0]
                    hidden_states[mask] = pa_idx.repeat(n_adapters, 1)
                except Exception:
                    pass
        return hidden_states
