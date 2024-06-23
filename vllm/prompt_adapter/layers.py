from dataclasses import dataclass
from typing import Dict, List, Optional

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
        self.indices_gpu: torch.Tensor
        self.flag: bool = False

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
        self.indices_gpu = torch.tensor(base_indices, device="cuda")
        self.flag = torch.sum(self.indices_gpu) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.base_layer(x)
        if self.flag:
            unique_indices = torch.unique(self.indices_gpu)
            for idx in unique_indices:
                if idx != 0:
                    pa_idx = self.embedding_tensors[
                        idx.item()].prompt_embedding
                    mask = (self.indices_gpu == idx)
                    n_adapters = sum(mask) // pa_idx.shape[0]
                    hidden_states[mask] = pa_idx.repeat(n_adapters, 1)
        return hidden_states
