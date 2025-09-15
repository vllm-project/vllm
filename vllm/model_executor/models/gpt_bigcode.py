# SPDX-License-Identifier: Apache-2.0

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
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import GPTBigCodeConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


class GPTBigCodeAttention(nn.Module):

    def __init__(
        self,
        config: GPTBigCodeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
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
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.split(
            [
                self.hidden_size // self.tensor_model_parallel_world_size,
                self.kv_dim, self.kv_dim
            ],
            dim=-1,
        )
        attn_output = self.attn(q, k, v)
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
        self.act = get_act_fn(config.activation_function)

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
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config,
                                        cache_config,
                                        quant_config,
                                        prefix=f"{prefix}.attn")
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTBigMLP(inner_dim, config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states=hidden_states, )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


@support_torch_compile
class GPTBigCodeModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

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
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPTBigCodeBlock(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.n_embd))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            hidden_states = inputs_embeds + self.wpe(position_ids)
        else:
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in self.h[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTBigCodeForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {"c_attn": ["c_attn"]}

    # LoRA specific attributes
    embedding_modules = {
        "wte": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.transformer = GPTBigCodeModel(vllm_config=vllm_config,
                                           prefix=prefix)
        if self.config.tie_word_embeddings:
            self.lm_head = self.transformer.wte
        else:
            self.lm_head = ParallelLMHead(
                self.transformer.vocab_size,
                self.transformer.embed_dim,
                org_num_embeddings=self.config.vocab_size)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.transformer(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                continue
            if ".attn.bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            if is_pp_missing_parameter(name, self):
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
            loaded_params.add(name)
        return loaded_params
