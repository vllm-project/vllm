# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B/blob/main/modeling_qwen2_rm.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen2-RM model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsPP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader


class ReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

    def forward(self, input):
        input, _ = input
        return self.activation(input)


class Qwen2ForRewardModel(nn.Module, SupportsPP):
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

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(config, cache_config, quant_config)

        self.score = nn.Sequential(
            ColumnParallelLinear(config.hidden_size,
                                 config.hidden_size,
                                 quant_config=quant_config),
            ReLU(),
            RowParallelLinear(config.hidden_size, 1,
                              quant_config=quant_config),
        )
        self._pooler = Pooler(pooling_type=PoolingType.ALL, normalize=False)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        logits, _ = self.score(hidden_states)
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
