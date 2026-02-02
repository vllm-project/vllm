# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo3_5_hybrid/modeling_olmo3_5_hybrid.py
# Copyright 2024 The vLLM team.
# Copyright 2026 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code combines OLMo2/OLMo3 attention with Gated DeltaNet linear attention
# for the OLMo 3.5 Hybrid architecture.
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
"""Inference-only OLMo 3.5 Hybrid model compatible with HuggingFace weights."""

import contextlib
from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import HasInnerState, IsHybrid, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _triton_pytorch_allocator(size: int, alignment: int, stream: int) -> int:
    """Allocator that uses PyTorch to allocate memory for Triton kernels."""
    return torch.empty(size, dtype=torch.int8, device="cuda").data_ptr()


with contextlib.suppress(Exception):
    triton.set_allocator(_triton_pytorch_allocator)

logger = init_logger(__name__)


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _make_conv1d_weight_loader(dim: int, tp_size: int, tp_rank: int):
    """Create a weight loader for conv1d that handles sharding."""

    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
        # loaded_weight shape: (dim, 1, kernel_size) or (dim, kernel_size)
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        # Shard along the first dimension
        shard_size = dim // tp_size
        start = tp_rank * shard_size
        end = start + shard_size
        param.data.copy_(loaded_weight[start:end])

    return weight_loader


class Olmo3_5HybridGatedDeltaNet(nn.Module, MambaBase):
    """
    Gated DeltaNet linear attention layer for OLMo 3.5 Hybrid.

    This implements the linear attention mechanism that replaces sliding window
    attention in the hybrid architecture.
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_kv_heads,
            self.num_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_value_heads
        self.num_kv_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_kv_heads
        self.value_dim = self.head_v_dim * self.num_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.use_gate = getattr(config, "linear_use_gate", True)
        self.allow_neg_eigval = getattr(config, "linear_allow_neg_eigval", False)
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        self.q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.a_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.a_proj",
        )
        self.b_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )

        if self.use_gate:
            self.g_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.value_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.q_conv1d_weight = nn.Parameter(
            torch.empty(self.key_dim // self.tp_size, 1, self.conv_kernel_size)
        )
        self.k_conv1d_weight = nn.Parameter(
            torch.empty(self.key_dim // self.tp_size, 1, self.conv_kernel_size)
        )
        self.v_conv1d_weight = nn.Parameter(
            torch.empty(self.value_dim // self.tp_size, 1, self.conv_kernel_size)
        )

        set_weight_attrs(
            self.q_conv1d_weight,
            {
                "weight_loader": _make_conv1d_weight_loader(
                    self.key_dim, self.tp_size, self.tp_rank
                )
            },
        )
        set_weight_attrs(
            self.k_conv1d_weight,
            {
                "weight_loader": _make_conv1d_weight_loader(
                    self.key_dim, self.tp_size, self.tp_rank
                )
            },
        )
        set_weight_attrs(
            self.v_conv1d_weight,
            {
                "weight_loader": _make_conv1d_weight_loader(
                    self.value_dim, self.tp_size, self.tp_rank
                )
            },
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.num_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_heads, self.tp_size),
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # use eps=1e-5 to match FLA's FusedRMSNormGated
        o_norm_eps = 1e-5
        if self.use_gate:
            self.o_norm = RMSNormGated(
                self.head_v_dim,
                eps=o_norm_eps,
                group_size=None,
                norm_before_gate=True,
                device=current_platform.current_device(),
                dtype=config.torch_dtype if hasattr(config, "torch_dtype") else None,
            )
        else:
            self.o_norm = RMSNorm(
                self.head_v_dim,
                eps=o_norm_eps,
            )

        self.o_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _get_combined_conv_weight(self):
        """Combine q, k, v conv weights for efficient processing."""
        return torch.cat(
            [
                self.q_conv1d_weight.squeeze(1),
                self.k_conv1d_weight.squeeze(1),
                self.v_conv1d_weight.squeeze(1),
            ],
            dim=0,
        )

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )

        num_kv_heads = self.num_kv_heads // self.tp_size
        num_heads = self.num_heads // self.tp_size

        query = rearrange(
            query, "l (h d) -> 1 l h d", h=num_kv_heads, d=self.head_k_dim
        )
        key = rearrange(key, "l (h d) -> 1 l h d", h=num_kv_heads, d=self.head_k_dim)
        value = rearrange(value, "l (h d) -> 1 l h d", h=num_heads, d=self.head_v_dim)

        # GQA expansion if needed
        if num_heads > num_kv_heads:
            expand_ratio = num_heads // num_kv_heads
            query = query.unsqueeze(3).expand(-1, -1, -1, expand_ratio, -1)
            query = query.reshape(1, query.shape[1], num_heads, self.head_k_dim)
            key = key.unsqueeze(3).expand(-1, -1, -1, expand_ratio, -1)
            key = key.reshape(1, key.shape[1], num_heads, self.head_k_dim)

        query = _l2_normalize(query, dim=-1)
        key = _l2_normalize(key, dim=-1)

        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        mixed_qkv = torch.cat([q, k, v], dim=-1)

        a, _ = self.a_proj(hidden_states)
        b, _ = self.b_proj(hidden_states)

        if self.use_gate:
            gate, _ = self.g_proj(hidden_states)
        else:
            gate = None

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.olmo3_5_hybrid_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        if self.use_gate:
            gate = gate.view(
                num_tokens, self.num_heads // self.tp_size, self.head_v_dim
            )
            core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            gate_flat = gate.reshape(-1, gate.shape[-1])
            core_attn_out_normed = self.o_norm(core_attn_out_flat, gate_flat)
            core_attn_out = core_attn_out_normed.view(
                num_tokens, self.num_heads // self.tp_size, self.head_v_dim
            )
        else:
            core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            core_attn_out_normed = self.o_norm(core_attn_out_flat)
            core_attn_out = core_attn_out_normed.view(
                num_tokens, self.num_heads // self.tp_size, self.head_v_dim
            )

        core_attn_out = rearrange(core_attn_out, "l h d -> l (h d)")
        output[:num_tokens], _ = self.o_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self._get_combined_conv_weight()

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                None,  # no bias
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                None,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        g, beta = fused_olmo3_5_hybrid_gdn_gating(
            self.A_log, a, b, self.dt_bias, self.allow_neg_eigval
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=False,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class Olmo3_5HybridAttention(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.tp_rank = get_tensor_model_parallel_rank()

        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=self.config.rms_norm_eps,
        )
        self.q_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        self.scaling = self.head_dim**-0.5

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

        rope_parameters = getattr(self.config, "rope_parameters", None)
        self._use_rope = (rope_parameters is not None) and (
            rope_parameters["rope_theta"] is not None
        )

        if self._use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=self.max_position_embeddings,
                rope_parameters=rope_parameters,
            )
        else:
            self.rotary_emb = None

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        if self._use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Olmo3_5HybridMLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.act_fn = SiluAndMul()

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Olmo3_5HybridDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        layer_idx = extract_layer_index(prefix)
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "linear_attention":
            self.linear_attn = Olmo3_5HybridGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
            # FLA layers use these norm names
            self.attention_layer_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.feedforward_layer_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = Olmo3_5HybridAttention(
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
            # Attention layers use these norm names
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_feedforward_layernorm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

        self.mlp = Olmo3_5HybridMLP(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            residual = hidden_states
            hidden_states = self.attention_layer_norm(hidden_states)

            attn_output = torch.empty_like(hidden_states)
            self.linear_attn(
                hidden_states=hidden_states,
                output=attn_output,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = self.feedforward_layer_norm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


@support_torch_compile
class Olmo3_5HybridModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: Olmo3_5HybridDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        conv_weight_mapping = {
            "q_conv1d.weight": "q_conv1d_weight",
            "k_conv1d.weight": "k_conv1d_weight",
            "v_conv1d.weight": "v_conv1d_weight",
        }

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue

            for hf_name, vllm_name in conv_weight_mapping.items():
                if hf_name in name:
                    name = name.replace(hf_name, vllm_name)
                    break

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "linear_attn" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                handled = True
                break

            if not handled:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Olmo3_5HybridForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsLoRA, IsHybrid
):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        self.model = Olmo3_5HybridModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.weight"] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)


def olmo3_5_hybrid_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def olmo3_5_hybrid_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="olmo3_5_hybrid_gdn_attention_core",
    op_func=olmo3_5_hybrid_gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=olmo3_5_hybrid_gdn_attention_core_fake,
)


@triton.jit
def fused_olmo3_5_hybrid_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    allow_neg_eigval: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)

    # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)

    # beta = self.b_proj(hidden_states).sigmoid()
    # if self.allow_neg_eigval: beta = beta * 2.0
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    if allow_neg_eigval:
        blk_beta_output = blk_beta_output * 2.0
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_olmo3_5_hybrid_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    allow_neg_eigval: bool = False,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_olmo3_5_hybrid_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        allow_neg_eigval,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
