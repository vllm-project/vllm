# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2023 CTranslate2, and Michael Feil
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only GPTBigCode model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTBigCodeConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput

from .interfaces import SupportsLoRA


class GPTBigCodeAttention(nn.Module):

    def __init__(
        self,
        config: GPTBigCodeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        self.tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % self.tensor_model_parallel_world_size == 0
        self.num_heads = (total_num_heads //
                          self.tensor_model_parallel_world_size)
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.multi_query = config.multi_query
        if self.multi_query:
            total_num_kv_heads = 1
            self.num_kv_heads = 1
        else:
            total_num_kv_heads = total_num_heads
            self.num_kv_heads = self.num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )

        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scale,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.split(
            [
                self.hidden_size // self.tensor_model_parallel_world_size,
                self.kv_dim, self.kv_dim
            ],
            dim=-1,
        )
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPTBigMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPTBigCodeConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.act = get_act_fn(config.activation_function, quant_config,
                              intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPTBigCodeBlock(nn.Module):

    def __init__(
        self,
        config: GPTBigCodeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config, cache_config, quant_config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTBigMLP(inner_dim, config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPTBigCodeModel(nn.Module):

    def __init__(
        self,
        config: GPTBigCodeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention

        self.embed_dim = config.hidden_size
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.wte = VocabParallelEmbedding(self.vocab_size,
                                          self.embed_dim,
                                          org_num_embeddings=config.vocab_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList([
            GPTBigCodeBlock(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTBigCodeForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {"c_attn": ["c_attn"]}

    supported_lora_modules = ["c_fc", "c_proj", "wte", "c_attn"]

    embedding_modules = {
        "wte": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    embedding_padding_modules = []

    def __init__(
        self,
        config: GPTBigCodeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.transformer = GPTBigCodeModel(config, cache_config, quant_config,
                                           lora_config)
        self.lm_head = self.transformer.wte
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                continue
            if ".attn.bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # TODO (@robertgshaw2-neuralmagic): move to fp8 linear method
            if "c_attn.input_scale" in name or "c_attn.weight_scale" in name:
                weight_loader(param, loaded_weight, 'q')
                weight_loader(param, loaded_weight, 'k')
                weight_loader(param, loaded_weight, 'v')
            else:
                weight_loader(param, loaded_weight)
