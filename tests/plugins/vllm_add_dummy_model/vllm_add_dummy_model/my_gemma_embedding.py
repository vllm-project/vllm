from typing import List, Optional, Union

import torch

from vllm.attention import AttentionMetadata
from vllm.model_executor.models.gemma2 import Gemma2EmbeddingModel
from vllm.sequence import IntermediateTensors


class MyGemma2Embedding(Gemma2EmbeddingModel):

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = super().forward(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        # Return all-zero embeddings
        return torch.zeros_like(hidden_states)
