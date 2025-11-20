# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/bloom/modeling_bloom.py
# Copyright 2023 The vLLM team.
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""Inference-only BLOOM model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import BloomConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP, SupportsQuant
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class BloomAttention(nn.Module):
    def __init__(
        self,
        config: BloomConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.n_head
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scaling,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del position_ids  # Unused.
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.dense(attn_output)
        return output


class BloomMLP(nn.Module):
    def __init__(
        self,
        config: BloomConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            4 * hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_h_to_4h",
        )
        self.gelu_impl = get_act_fn("gelu")
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        x, _ = self.dense_4h_to_h(x)
        return x


class BloomBlock(nn.Module):
    def __init__(
        self,
        config: BloomConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.self_attention = BloomAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attention"
        )
        self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = BloomMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attention_output = self.self_attention(
            position_ids=position_ids,
            hidden_states=layernorm_output,
        )
        attention_output = attention_output + residual
        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output) + residual
        return output


@support_torch_compile
class BloomModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_dim = config.hidden_size

        # Embedding + LN Embedding
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
        )
        self.word_embeddings_layernorm = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )

        # Transformer blocks
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: BloomBlock(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.h",
        )

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = self.word_embeddings_layernorm(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states = layer(position_ids, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]

            if "query_key_value" in name:
                # NOTE: BLOOM's fused QKV's output_dim has the shape of
                # (num_heads * 3 * head_size), while the
                # required shape is (3 * num_heads * head_size).
                # Thus, we need weight conversion.
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads
                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim]
                        + (num_heads, 3, -1)
                        + loaded_weight_shape[output_dim + 1 :]
                    )
                    loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class BloomForCausalLM(nn.Module, SupportsPP, SupportsQuant):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.transformer = BloomModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.transformer.word_embeddings
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head.weight"])
        weights = _add_transformer_prefix(weights)
        return loader.load_weights(weights)


def _add_transformer_prefix(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if not name.startswith("transformer."):
            name = "transformer." + name
        yield name, tensor
