from typing import List, Optional

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata
from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.llama import LlamaModel
from vllm.sequence import PoolerOutput


class LlamaEmbeddingModel(nn.Module):
    """A model that uses Llama with additional embedding functionalities.

   This class encapsulates the LlamaModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of LlamaModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.model = LlamaModel(config, linear_method, lora_config)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(input_ids, positions, kv_caches,
                                  attn_metadata, inputs_embeds)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        attention_metadata: AttentionMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, attention_metadata)

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        self.model.load_weights(model_name_or_path, cache_dir, load_format,
                                revision)
