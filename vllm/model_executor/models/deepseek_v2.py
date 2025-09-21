# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only DeepseekV2/DeepseekV3 model."""
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv, direct_register_custom_op

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           disable_tp=is_sequence_parallel,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        x = nn.functional.pad(x, (0, 0, 0, pad_len))

    chunk = x.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(x, 0, start, chunk)


def sequence_parallel_chunk_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    seq_len = cdiv(x.size(0), tp_size)
    shape = list(x.shape)
    shape[0] = seq_len
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op(
    op_name="sequence_parallel_chunk",
    op_func=sequence_parallel_chunk,
    mutates_args=[],
    fake_impl=sequence_parallel_chunk_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: Union[DeepseekV2Config, DeepseekV3Config],
        parallel_config: ParallelConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        # The all_reduce at the end of attention (during o_proj) means that
        # inputs are replicated across each rank of the tensor parallel group.
        # If using expert-parallelism with DeepEP All2All ops, replicated
        # tokens results in useless duplicate computation and communication.
        #
        # In this case, ensure the input to the experts is sequence parallel
        # to avoid the excess work.
        #
        # Not needed for pplx-kernels as it can handle duplicate input tokens.
        self.is_sequence_parallel = (envs.VLLM_ALL2ALL_BACKEND
                                     in ("deepep_high_throughput",
                                         "deepep_low_latency")
                                     and parallel_config.enable_expert_parallel
                                     and self.tp_size > 1)

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32))
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        if config.n_shared_experts is None:
            self.experts = FusedMoE(
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.scoring_func,
                # we do scaling outside, set factor to 1.0 to avoid double mul
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )
            self.shared_experts = None
        else:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

            self.experts = SharedFusedMoE(
                shared_experts=self.shared_experts,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.scoring_func,
                # we do scaling outside, set factor to 1.0 to avoid double mul
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        if self.is_sequence_parallel:
            hidden_states = torch.ops.vllm.sequence_parallel_chunk(
                hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        fused_moe_out = self.experts(hidden_states=hidden_states,
                                     router_logits=router_logits)

        if self.shared_experts is not None:
            shared_output, final_hidden_states = fused_moe_out
        else:
            shared_output = None
            final_hidden_states = fused_moe_out

        # Fix FP16 overflow
        # See DeepseekV2DecoderLayer for more details.
        if hidden_states.dtype != torch.float16:
            final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            shared_output *= (1. / self.routed_scaling_factor)

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = (
                self.experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states))

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: Union[DeepseekV2Config, DeepseekV3Config],
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'

        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
                                         self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads,
                                                   self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim],
            value=0).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads,
            self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).
    
        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: Union[DeepseekV2Config, DeepseekV3Config],
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True)
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None else None,
            q_a_layernorm=self.q_a_layernorm
            if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
        )
        self.mla_attn = MultiHeadLatentAttention(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        if model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp,
                      DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile
class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts,
                            SupportsLoRA):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # `packed_modules_mapping` needs to be modified before
        # initializing DeepseekV2Model, as it is passed inplace to
        # quantization config init and may be used to select the
        # quant_method for relevant layers during initialization.
        self.fuse_qkv_a_proj = hasattr(
            config, "q_lora_rank") and config.q_lora_rank is not None
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = DeepseekV2Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = (config.num_hidden_layers -
                               config.first_k_dense_replace)
        self.num_expert_groups = config.n_group

        self.moe_layers: list[FusedMoE] = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No DeepseekV2MoE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.model.layers:
            if isinstance(layer.mlp, DeepseekV2MoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if ((param_name == "fused_qkv_a_proj")
                        and name_mapped not in params_dict):
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    success = weight_loader(param,
                                            loaded_weight,
                                            name_mapped,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                            return_success=True)
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


# Compatibility with
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/configuration_deepseek.py
def get_spec_layer_idx_from_weight_name(config: Union[DeepseekV2Config,
                                                      DeepseekV3Config],
                                        weight_name: str) -> Optional[int]:
    if (hasattr(config, "num_nextn_predict_layers")
            and config.num_nextn_predict_layers > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None
