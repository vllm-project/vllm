# Adapted from
# https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B/blob/main/modeling_qwen2_rm.py
# Copyright 2024 Kakao Corp. (Kanana-X Team)
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen2-Classification model compatible with HF weights."""
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsLoRA, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix


class Qwen2ForSequenceClassification(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # hidden_states from Qwen2Model has been reduced,
        # the input of score layer is not parallelized.
        self.score = RowParallelLinear(config.hidden_size,
                                       config.num_labels,
                                       quant_config=quant_config,
                                       input_is_parallel=False,
                                       bias=False,
                                       prefix=maybe_prefix(prefix, "score"))
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=False,
            softmax=True)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        logits, _ = self.score(hidden_states)
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)
