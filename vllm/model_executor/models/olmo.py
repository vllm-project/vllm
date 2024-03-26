# coding=utf-8
# Adapted from
# https://github.com/allenai/OLMo/blob/v0.2.4/olmo/model.py and
# https://github.com/allenai/OLMo/blob/v0.2.4/hf_olmo/modeling_olmo.py
# Copyright 2023 The vLLM team.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# BSD 3-Clause License
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Inference-only OLMo model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
# this model must need this dependency
from hf_olmo import OLMoConfig
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput


class SwiGLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: OLMoConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        assert config.d_model % config.n_heads == 0
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = self.config.n_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = self.hidden_size // self.total_num_heads

        # Layer norms.
        self.attn_norm = nn.LayerNorm(config.d_model,
                                      elementwise_affine=False,
                                      bias=False)
        # Attention input projection. Projects x -> (q, k, v)
        self.att_proj = QKVParallelLinear(
            config.d_model,
            self.head_dim,
            self.total_num_heads,
            bias=config.include_bias,
            linear_method=linear_method,
        )

        # Rotary embeddings.
        if self.config.rope:
            rope_theta = getattr(config, "rope_theta", 10000)
            max_position_embeddings = getattr(config,
                                              "max_position_embeddings", 8192)
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
            )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling)

        # Attention output projection.
        self.attn_out = RowParallelLinear(
            config.d_model,
            config.d_model,
            bias=config.include_bias,
            linear_method=linear_method,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.attn_norm(hidden_states)
        qkv, _ = self.att_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.config.rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.attn_out(attn_output)
        return output


class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: OLMoConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = (config.mlp_hidden_size if config.mlp_hidden_size
                            is not None else config.mlp_ratio * config.d_model)

        # Layer norms.
        self.ff_norm = nn.LayerNorm(config.d_model,
                                    elementwise_affine=False,
                                    bias=False)

        # Feed-forward input projection.
        self.ff_proj = ColumnParallelLinear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            linear_method=linear_method,
        )

        # Activation function.
        # self.act = SiluAndMul()
        # self.act.output_multiplier = 0.5
        self.act = SwiGLU()
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Feed-forward output projection.
        self.ff_out = RowParallelLinear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            linear_method=linear_method,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        x = self.ff_norm(x)
        x, _ = self.ff_proj(x)
        x = self.act(x)
        x, _ = self.ff_out(x)
        x = og_x + x

        return x


class OlmoBlock(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self,
                 config: OLMoConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        # Attention block.
        self.attn = OlmoAttention(config, linear_method)

        # MLP block.
        self.mlp = OlmoMLP(config, linear_method)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block.
        og_x = hidden_states
        x = self.attn(positions, hidden_states, kv_cache, attn_metadata)
        x = x + og_x

        # MLP block.
        hidden_states = self.mlp(x)
        return hidden_states


class OlmoModel(nn.Module):

    def __init__(self,
                 config: OLMoConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=VocabParallelEmbedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                ),
                ln_f=nn.LayerNorm(config.d_model,
                                  elementwise_affine=False,
                                  bias=False),
            ))

        blocks = [
            OlmoBlock(config, linear_method) for i in range(config.n_layers)
        ]
        if self.config.block_group_size > 1:
            raise NotImplementedError("Block group size > 1 not supported yet")
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not config.weight_tying:
            self.transformer.update({
                "ff_out":
                ColumnParallelLinear(
                    config.d_model,
                    config.embedding_size or config.vocab_size,
                    bias=config.include_bias,
                    linear_method=linear_method,
                )
            })

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        # Apply blocks one-by-one.
        for block_idx, block in enumerate(self.transformer.blocks):
            # shape: (batch_size, seq_len, d_model)
            x = block(
                positions,
                x,
                kv_caches[block_idx],
                attn_metadata,
            )

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        return x


class OLMoForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """

    def __init__(self,
                 config: OLMoConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = OlmoModel(config, linear_method)
        self.lm_head_weight = (self.model.transformer.wte.weight
                               if config.weight_tying else
                               self.model.transformer.ff_out.weight)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
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
            # attention
            if ".att" in name:
                name = name.replace(".att", ".attn.att")
            # mlp
            if ".ff" in name and "transformer.ff_out" not in name:
                name = name.replace(".ff", ".mlp.ff")
            # there is no bias in olmo
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
