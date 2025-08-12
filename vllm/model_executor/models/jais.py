# coding=utf-8
# Adapted from
# https://huggingface.co/core42/jais-30b-chat-v3/blob/main/modeling_jais.py
# Copyright 2023 The vLLM team.
# Copyright 2023 the Jais authors and HuggingFace Inc. team.  All rights
# reserved.
# Copyright 2023 Cerebras Systems.
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
"""Inference-only Jais model compatible with HuggingFace weights."""

import math
from typing import List, Optional

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs import JAISConfig


class SwiGLUActivation(nn.Module):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)


def _get_alibi_slopes(n):

    def get_slopes_power_of_2(n):
        start = 2**(-(2**-(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return (get_slopes_power_of_2(closest_power_of_2) + _get_alibi_slopes(
            2 * closest_power_of_2)[0::2][:n - closest_power_of_2])


class JAISAttention(nn.Module):

    def __init__(
        self,
        config: JAISConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        if hasattr(config, "scale_qk_dot_by_d"):
            config.mup_scale_qk_dot_by_d = config.scale_qk_dot_by_d
        self.attn_scale_power = 1.0 if config.mup_scale_qk_dot_by_d else 0.5
        self.scale = self.head_dim**-self.attn_scale_power

        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            linear_method=linear_method,
        )

        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(total_num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end]
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            alibi_slopes=alibi_slopes,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class JAISMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: JAISConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.swiglu = config.activation_function == "swiglu"
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            linear_method=linear_method,
        )
        self.c_fc2 = (ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            linear_method=linear_method,
        ) if self.swiglu else None)
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            linear_method=linear_method,
        )

        self.act = SwiGLUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            hidden_states2, _ = self.c_fc2(hidden_states)
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = (self.act(hidden_states, hidden_states2)
                         if self.swiglu else self.act(hidden_states))
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class JAISBlock(nn.Module):

    def __init__(
        self,
        config: JAISConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = (config.n_inner if config.n_inner is not None else 4 *
                     hidden_size)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = JAISAttention(config, linear_method)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = JAISMLP(inner_dim, config, linear_method)

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


class JAISModel(nn.Module):

    def __init__(
        self,
        config: JAISConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = (nn.Embedding(config.max_position_embeddings,
                                 self.embed_dim)
                    if config.position_embedding_type != "alibi" else None)
        if hasattr(config, "embeddings_scale"):
            self.embeddings_scale = config.embeddings_scale
        else:
            self.embeddings_scale = config.mup_embeddings_scale
        self.h = nn.ModuleList([
            JAISBlock(config, linear_method)
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
        if self.wpe is not None:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        hidden_states *= torch.tensor(float(self.embeddings_scale),
                                      dtype=hidden_states.dtype)

        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class JAISLMHeadModel(nn.Module):

    def __init__(
        self,
        config: JAISConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.transformer = JAISModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        if hasattr(config, "width_scale"):
            self.output_logits_scale = config.width_scale
        else:
            self.output_logits_scale = (config.mup_output_alpha *
                                        config.mup_width_scale)
        self.logits_processor = LogitsProcessor(vocab_size=config.vocab_size,
                                                scale=self.output_logits_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "lm_head.weight" in name:
                # GPT-2 ties the weights of the embedding layer and the final
                # linear layer.
                continue
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            if "relative_pe" in name:
                continue
            if not name.startswith("transformer."):
                name = "transformer." + name
            param = params_dict[name]
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            # Note(zhuohan): the logic below might break quantized models.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
