# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_sequence_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (
    SequenceParallelLinearForBroastcast, SequenceParallelLinearForGather)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

from .interfaces import SupportsLoRA


class broastcastlayer:

    def __init__(self,
                 tp_size: int,
                 hidden_size: int,
                 dtype: Optional[torch.dtype] = None) -> None:
        self.tp_size = tp_size
        self.broastcast_streams = [torch.cuda.Stream() for _ in range(tp_size)]
        self.broatcasters = [
            SequenceParallelLinearForBroastcast(i) for i in range[self.tp_size]
        ]
        self.hidden_size = hidden_size
        self.q_size = self.hidden_size / self.tp_size
        self.dtype = dtype

    def forward(self, num_seqs: int) -> torch.Tensor:

        output = torch.empty(
            [self.tp_size, num_seqs, self.hidden_size // self.tp_size],
            dtype=self.dtype,
            device="cuda")
        for i in range(self.tp_size):
            with torch.cuda.stream(self.broastcast_streams[i]):
                output[i] = self.broatcasters[i]()
        return output


class gatherlayer:

    def __init__(self, tp_size: int, num_heads: int) -> None:
        self.tp_size = tp_size
        self.gather_streams = [torch.cuda.Stream() for _ in range(tp_size)]
        self.gathers = [
            SequenceParallelLinearForGather(i) for i in range[self.tp_size]
        ]
        self.num_heads = num_heads / self.tp_size

    def forward(self, attn_to_reduce: torch.Tensor,
                exp_sum_to_reduce: torch.Tensor,
                max_logits_to_reduce: torch.Tensor) -> None:
        for i in range(self.tp_size):
            with torch.cuda.stream(self.gather_streams[i]):
                start = i * self.num_heads
                end = (i + 1) * self.num_heads
                self.gather[i](attn_to_reduce[start:end],
                               exp_sum_to_reduce[start:end],
                               max_logits_to_reduce[start:end])


class OnlyAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_size = self.tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

        self.sp_rank = get_sequence_parallel_rank()
        self.broastcastlayer = broastcastlayer(self.tp_size, self.hidden_size,
                                               cache_config.cache_dtype)
        self.gatherlayer = gatherlayer(self.tp_size, self.total_num_heads)

    #     def __init__(
    #     self,
    #     num_heads: int,
    #     head_size: int,
    #     scale: float,
    #     num_kv_heads: Optional[int] = None,
    #     alibi_slopes: Optional[List[float]] = None,
    #     cache_config: Optional[CacheConfig] = None,
    #     quant_config: Optional[QuantizationConfig] = None,
    #     blocksparse_params: Optional[Dict[str, Any]] = None,
    # ) -> None:

    def forward(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        sp_rank: int,
    ) -> None:
        q = self.broastcastlayer(attn_metadata.num_long_decode_tokens)
        attn_to_reduce, exp_sum_to_reduce, max_logits_to_reduce = self.attn(
            q, kv_cache, attn_metadata, sp_rank)
        self.gatherlayer(attn_to_reduce, exp_sum_to_reduce,
                         max_logits_to_reduce)


class OnlyAttentionLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OnlyAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            quant_config=quant_config,
            cache_config=cache_config,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        # Self Attention
        self.self_attn(kv_cache, attn_metadata)


class AttentionModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.layer = OnlyAttentionLayer(config=config,
                                        cache_config=cache_config,
                                        quant_config=quant_config)
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i in range(len(self.num_hidden_layers)):
            layer = self.layer
            layer(
                kv_caches[i],
                attn_metadata,
            )


class OnlyAttentionModel(nn.Module, SupportsLoRA):
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
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = AttentionModel(config,
                                    cache_config,
                                    quant_config,
                                    lora_config=lora_config)

    def forward(
        self,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> None:
        self.model(kv_caches, attn_metadata)

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        pass

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        pass

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        pass

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        pass
