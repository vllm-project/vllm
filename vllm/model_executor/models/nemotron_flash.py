# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/vllm-project/vllm/blob/e50c45467215f96068d95736b08d8a25f624e67d/vllm/model_executor/models/nemotron_h.py
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
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
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_flash import NemotronFlashConfig
from vllm.utils.torch_utils import direct_register_custom_op

# FLA kernels for DeltaNet/GLA
try:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_delta_rule = None
    fused_recurrent_delta_rule = None


class NemotronFlashMLP(nn.Module):
    """MLP for Nemotron-Flash."""

    def __init__(
        self,
        config: NemotronFlashConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NemotronFlashMoE(nn.Module):
    """Mixture of Experts layer for Nemotron-Flash."""

    def __init__(
        self,
        config: NemotronFlashConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.num_experts = getattr(config, "num_experts", 1)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 1)

        # Use ffn_expand_ratio if available, otherwise use intermediate_size
        if hasattr(config, "ffn_expand_ratio") and config.ffn_expand_ratio > 0:
            self.intermediate_size = config.hidden_size * config.ffn_expand_ratio
        elif hasattr(config, "moe_intermediate_size") and config.moe_intermediate_size:
            self.intermediate_size = config.moe_intermediate_size
        else:
            self.intermediate_size = config.intermediate_size

        tp_size = get_tensor_model_parallel_world_size()

        # Gate/router (only needed when num_experts > 1)
        if self.num_experts > 1:
            self.gate = ReplicatedLinear(
                self.hidden_size,
                self.num_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
        else:
            self.gate = None

        # Experts
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_experts,
        )
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            tp_size=tp_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",  # Add .experts to prefix
            expert_mapping=expert_mapping,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)
        else:
            # Single expert, no routing - create dummy logits pointing to expert 0
            router_logits = torch.ones(
                (num_tokens, 1), dtype=hidden_states.dtype, device=hidden_states.device
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        return final_hidden_states.view(num_tokens, hidden_size)


@CustomOp.register("nemotron_deltanet")
class NemotronDeltaNet(MambaBase, CustomOp):
    """
    DeltaNet/GLA layer for Nemotron-Flash.

    Reference implementation of DeltaNet (Gated Linear Attention) using PyTorch.
    Inherits from MambaBase to leverage V1 infrastructure for state management,
    caching, and speculative decoding.
    """

    def __init__(
        self,
        config: NemotronFlashConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_idx: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        self.num_meta_tokens = config.num_meta_tokens

        # DeltaNet dimensions
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size

        assert self.num_k_heads % tp_size == 0
        assert self.num_v_heads % tp_size == 0

        self.num_k_heads_per_rank = self.num_k_heads // tp_size
        self.num_v_heads_per_rank = self.num_v_heads // tp_size

        # Total dimensions
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim

        # Per-rank dimensions
        self.key_dim_per_rank = self.num_k_heads_per_rank * self.key_head_dim
        self.value_dim_per_rank = self.num_v_heads_per_rank * self.value_head_dim

        # Input projections: Q, K, V, beta
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.key_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_k_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )

        # Separate 1D convolutions for Q, K, V
        self.q_conv1d = ColumnParallelLinear(
            self.conv_kernel,
            self.key_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            self.conv_kernel,
            self.key_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            self.conv_kernel,
            self.value_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.v_conv1d",
        )

        # normalizes per head
        self.o_norm = RMSNorm(self.value_head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.use_gate = config.use_gate if hasattr(config, "use_gate") else False
        if self.use_gate:
            self.g_norm = nn.Parameter(
                torch.ones(self.num_v_heads_per_rank, dtype=torch.float32)
            )
        else:
            self.g_norm = None

        # Register this layer for custom op access via forward_context
        compilation_config = get_current_vllm_config().compilation_config
        # Only register if not already present (avoids issues with tracing)
        if prefix not in compilation_config.static_forward_context:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """Native (fallback) implementation."""
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """Forward pass using custom op."""
        torch.ops.vllm.nemotron_deltanet(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        DeltaNet forward pass.
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None or not isinstance(attn_metadata, dict):
            # Profile/tracing run - run through projections without state updates.
            # This ensures the compiled subgraphs see correct tensor shapes.
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            b, _ = self.b_proj(hidden_states)

            # Use v as placeholder for SSM output
            v_reshaped = rearrange(v, "t (h d) -> t h d", h=self.num_v_heads_per_rank)
            attn_output = self.o_norm(v_reshaped.to(hidden_states.dtype))
            attn_output = rearrange(attn_output, "t h d -> t (h d)")
            output_proj, _ = self.o_proj(attn_output)
            output.copy_(output_proj)
            return

        from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, LinearAttentionMetadata)

        num_actual_tokens = (
            attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
        )
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens

        # Project all tokens
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        b, _ = self.b_proj(hidden_states)

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        combined_conv_state = self_kv_cache[0]  # [batch, total_conv_dim, conv_kernel-1]
        # Get reference to SSM state (note: not a copy)
        ssm_state = self_kv_cache[
            1
        ]  # [batch, num_v_heads, key_head_dim, value_head_dim]

        state_indices_tensor = attn_metadata.state_indices_tensor

        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0

        key_dim = self.key_head_dim * self.num_k_heads
        q_slice = combined_conv_state[:, :key_dim, :]
        k_slice = combined_conv_state[:, key_dim : key_dim * 2, :]
        v_slice = combined_conv_state[:, key_dim * 2 :, :]

        # Create contiguous copies for conv operations
        q_conv_state = q_slice.transpose(1, 2).contiguous().transpose(1, 2)
        k_conv_state = k_slice.transpose(1, 2).contiguous().transpose(1, 2)
        v_conv_state = v_slice.transpose(1, 2).contiguous().transpose(1, 2)

        q_out_list = []
        k_out_list = []
        v_out_list = []

        q_conv_weights = self.q_conv1d.weight
        k_conv_weights = self.k_conv1d.weight
        v_conv_weights = self.v_conv1d.weight

        if has_decode:
            q_d, k_d, v_d = (
                q[:num_decodes],
                k[:num_decodes],
                v[:num_decodes],
            )
            state_indices_d = state_indices_tensor[:num_decodes]

            q_conv_out = causal_conv1d_update(
                q_d,
                q_conv_state,
                q_conv_weights,
                None,
                activation="silu",
                conv_state_indices=state_indices_d,
            )
            k_conv_out = causal_conv1d_update(
                k_d,
                k_conv_state,
                k_conv_weights,
                None,
                activation="silu",
                conv_state_indices=state_indices_d,
            )
            v_conv_out = causal_conv1d_update(
                v_d,
                v_conv_state,
                v_conv_weights,
                None,
                activation="silu",
                conv_state_indices=state_indices_d,
            )

            # Write updated conv states back to the KV cache for only the
            # decoded sequences.
            combined_conv_state[state_indices_d, :key_dim, :] = q_conv_state[
                state_indices_d
            ]
            combined_conv_state[state_indices_d, key_dim : key_dim * 2, :] = (
                k_conv_state[state_indices_d]
            )
            combined_conv_state[state_indices_d, key_dim * 2 :, :] = v_conv_state[
                state_indices_d
            ]

            q_out_list.append(q_conv_out)
            k_out_list.append(k_conv_out)
            v_out_list.append(v_conv_out)

        if has_prefill:
            q_p, k_p, v_p = (
                q[num_decodes:num_actual_tokens],
                k[num_decodes:num_actual_tokens],
                v[num_decodes:num_actual_tokens],
            )
            state_indices_p = state_indices_tensor[
                num_decodes : num_prefills + num_decodes
            ]
            # Get query_start_loc for prefill requests and adjust to start from 0
            query_start_loc_p = attn_metadata.query_start_loc[num_decodes:]
            query_start_loc_p_conv = query_start_loc_p - query_start_loc_p[0]

            has_initial_states_p = torch.zeros(
                num_prefills, dtype=torch.bool, device=query_start_loc_p.device
            )

            q_conv_out = causal_conv1d_fn(
                q_p.transpose(0, 1),
                q_conv_weights,
                None,
                activation="silu",
                conv_states=q_conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                metadata=None,
                query_start_loc=query_start_loc_p_conv,
            ).transpose(0, 1)

            k_conv_out = causal_conv1d_fn(
                k_p.transpose(0, 1),
                k_conv_weights,
                None,
                activation="silu",
                conv_states=k_conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                metadata=None,
                query_start_loc=query_start_loc_p_conv,
            ).transpose(0, 1)

            v_conv_out = causal_conv1d_fn(
                v_p.transpose(0, 1),
                v_conv_weights,
                None,
                activation="silu",
                conv_states=v_conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                metadata=None,
                query_start_loc=query_start_loc_p_conv,
            ).transpose(0, 1)

            # Write updated conv states back to the KV cache for only the
            # prefilled sequences.
            combined_conv_state[state_indices_p, :key_dim, :] = q_conv_state[
                state_indices_p
            ]
            combined_conv_state[state_indices_p, key_dim : key_dim * 2, :] = (
                k_conv_state[state_indices_p]
            )
            combined_conv_state[state_indices_p, key_dim * 2 :, :] = v_conv_state[
                state_indices_p
            ]

            q_out_list.append(q_conv_out)
            k_out_list.append(k_conv_out)
            v_out_list.append(v_conv_out)

        q = torch.cat(q_out_list, dim=0) if q_out_list else q[:0]
        k = torch.cat(k_out_list, dim=0) if k_out_list else k[:0]
        v = torch.cat(v_out_list, dim=0) if v_out_list else v[:0]

        # Reshape to [tokens, num_heads, head_dim]
        q = rearrange(q, "t (h d) -> t h d", h=self.num_k_heads_per_rank)
        k = rearrange(k, "t (h d) -> t h d", h=self.num_k_heads_per_rank)
        v = rearrange(v, "t (h d) -> t h d", h=self.num_v_heads_per_rank)

        # Process beta (input gate)
        beta = rearrange(torch.sigmoid(b), "t h -> t h")  # [tokens, num_heads]

        if not HAS_FLA:
            raise RuntimeError(
                "FLA library required for DeltaNet. Install with: pip install fla"
            )

        # Process decode tokens if present
        decode_output = None
        if has_decode:
            num_decode_tokens = num_decodes

            # For decode-only batches (no prefills), use the full q tensor size
            # which may be padded for CUDA graph compatibility.
            # For mixed batches, use num_decode_tokens which is the actual count.
            batch_size = num_decode_tokens if has_prefill else q.shape[0]

            q_d = q[:batch_size].unsqueeze(1)
            k_d = k[:batch_size].unsqueeze(1)
            v_d = v[:batch_size].unsqueeze(1)
            beta_d = beta[:batch_size].unsqueeze(1)

            # Get initial states for decode tokens
            # Use state_indices_tensor which matches the batch size
            initial_state_d = ssm_state.index_select(
                0, state_indices_tensor[:batch_size]
            )

            # Run FLA kernel on decode tokens
            decode_output_batch, final_state_d = fused_recurrent_delta_rule(
                q=q_d,
                k=k_d,
                v=v_d,
                beta=beta_d,
                initial_state=initial_state_d,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

            decode_output = decode_output_batch.squeeze(1)

            # Update states for decode tokens - only update actual tokens
            # to avoid corrupting state slot 0 with garbage from padded tokens
            if final_state_d is not None:
                batch_indices_d = (
                    state_indices_tensor[:num_decode_tokens]
                    .view(num_decode_tokens, 1, 1, 1)
                    .expand_as(final_state_d[:num_decode_tokens])
                )
                ssm_state.scatter_(
                    0, batch_indices_d, final_state_d[:num_decode_tokens]
                )

        # Process prefill tokens if present
        prefill_output = None
        if has_prefill:
            # Prefill tokens are after decode tokens
            q_p = q[num_decodes:num_actual_tokens]
            k_p = k[num_decodes:num_actual_tokens]
            v_p = v[num_decodes:num_actual_tokens]
            beta_p = beta[num_decodes:num_actual_tokens]

            # Get state indices for prefill requests
            state_indices_p = state_indices_tensor[
                num_decodes : num_decodes + num_prefills
            ]

            # Get cu_seqlens for prefill sequences only (offset by num_decodes)
            query_start_loc_p = attn_metadata.query_start_loc[num_decodes:]
            # Adjust to be relative to prefill start
            cu_seqlens_p = query_start_loc_p - query_start_loc_p[0]

            # Shape for chunk_delta_rule: [1, total_prefill_tokens, heads, dim]
            q_p = q_p.unsqueeze(0)
            k_p = k_p.unsqueeze(0)
            v_p = v_p.unsqueeze(0)
            beta_p = beta_p.unsqueeze(0)

            # start with zero initial state to avoid garbage from CUDA graph warmup
            state_shape = (num_prefills,) + ssm_state.shape[1:]
            initial_state_p = torch.zeros(
                state_shape, dtype=ssm_state.dtype, device=ssm_state.device
            )

            prefill_output_batch, final_state_p = chunk_delta_rule(
                q=q_p,
                k=k_p,
                v=v_p,
                beta=beta_p,
                initial_state=initial_state_p,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens_p,
            )

            prefill_output = prefill_output_batch.squeeze(0)

            # Update states for prefill sequences
            if final_state_p is not None:
                batch_indices_p = state_indices_p.view(num_prefills, 1, 1, 1).expand_as(
                    final_state_p
                )
                ssm_state.scatter_(0, batch_indices_p, final_state_p)

        # Combine outputs
        if has_decode and has_prefill:
            attn_output = torch.cat([decode_output, prefill_output], dim=0)
        elif has_decode:
            attn_output = decode_output
        else:
            attn_output = prefill_output

        attn_output = attn_output.to(hidden_states.dtype)

        attn_output = self.o_norm(attn_output)
        attn_output = rearrange(attn_output, "t h d -> t (h d)")

        output_proj, _ = self.o_proj(attn_output)

        # Only copy to the actual token range (handles padded inputs)
        num_output_tokens = output_proj.size(0)
        output[:num_output_tokens].copy_(output_proj)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return dtypes for (conv_state, temporal_state)."""
        assert self.model_config is not None
        assert self.cache_config is not None

        # Conv state uses the same dtype as model
        conv_dtype_tuple = MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )
        conv_dtype = (
            conv_dtype_tuple[0]
            if isinstance(conv_dtype_tuple, tuple)
            else conv_dtype_tuple
        )

        temporal_dtype = torch.float32

        return (conv_dtype, temporal_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return shapes for (conv_state, temporal_state)."""
        tp_size = get_tensor_model_parallel_world_size()

        total_conv_dim = (
            self.key_head_dim * self.num_k_heads * 2  # q and k
            + self.value_head_dim * self.num_v_heads  # v
        )
        conv_state_shape = (total_conv_dim // tp_size, self.conv_kernel - 1)

        # Temporal state shape: (num_v_heads/tp, key_head_dim, value_head_dim)
        temporal_state_shape = (
            self.num_v_heads // tp_size,
            self.key_head_dim,
            self.value_head_dim,
        )

        return (conv_state_shape, temporal_state_shape)

    @property
    def mamba_type(self) -> str:
        """Return the mamba type identifier for this layer."""
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        """Return the attention backend class for this layer."""
        from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend

        return LinearAttentionBackend

    def get_kv_cache_spec(self, vllm_config: "VllmConfig") -> "KVCacheSpec | None":
        """
        Set model_config and cache_config from vllm_config pre-state dtype/shape.
        """
        # Set configs from vllm_config if not already set
        if self.model_config is None:
            self.model_config = vllm_config.model_config
        if self.cache_config is None:
            self.cache_config = vllm_config.cache_config
        return super().get_kv_cache_spec(vllm_config)


def nemotron_deltanet(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op function for NemotronDeltaNet."""
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def nemotron_deltanet_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for tracing."""
    return


# Register custom op
direct_register_custom_op(
    op_name="nemotron_deltanet",
    op_func=nemotron_deltanet,
    mutates_args=["output"],
    fake_impl=nemotron_deltanet_fake,
)


class NemotronFlashSequentialAttention(nn.Module):
    """Sequential attention layer for Nemotron-Flash."""

    def __init__(
        self,
        config: NemotronFlashConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_meta_tokens = config.num_meta_tokens

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.attn = Attention(
            num_heads=self.num_attention_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Apply RoPE to queries and keys
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        output_proj, _ = self.o_proj(attn_output)
        output.copy_(output_proj)


class NemotronFlashDecoderLayer(nn.Module):
    """
    A single decoder layer for Nemotron-Flash
    Supports different layer types:mamba2, deltanet, attention, mlp
    """

    def __init__(
        self,
        config: NemotronFlashConfig,
        layer_idx: int,
        layer_type: str,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size

        # Input normalization
        if layer_type != "f":
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer-specific mixer
        if layer_type in ("mamba", "m2"):
            # Mamba2 layer
            self.mixer = MambaMixer2(
                hidden_size=config.hidden_size,
                ssm_state_size=config.ssm_state_size,
                conv_kernel_size=config.conv_kernel,
                intermediate_size=int(config.hidden_size * config.mamba_expand),
                use_conv_bias=config.use_conv_bias,
                use_bias=config.use_bias,
                n_groups=config.n_groups,
                num_heads=config.mamba_num_heads,
                head_dim=config.mamba_head_dim,
                rms_norm_eps=config.rms_norm_eps,
                activation=config.mamba_hidden_act,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mixer",
            )
        elif layer_type == "deltanet":
            self.mixer = NemotronDeltaNet(
                config=config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mixer",
            )
        elif layer_type in ("attention", "a"):
            # Sequential attention
            self.self_attn = NemotronFlashSequentialAttention(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
            self.mixer = self.self_attn
        elif layer_type == "mlp" or layer_type == "f":
            # MLP-only layer (no mixer, just MLP/MoE)
            self.mixer = None
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.has_moe = layer_type in ["f"]

        if self.has_moe:
            self.pre_moe_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.moe = NemotronFlashMoE(
                config=config, quant_config=quant_config, prefix=f"{prefix}.moe"
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.layer_type == "f":
            # MoE-only layer: pre_moe_layernorm + moe
            residual = hidden_states
            hidden_states = self.pre_moe_layernorm(hidden_states)
            hidden_states = self.moe(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Mixer layer (mamba/deltanet/attention)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if self.mixer is not None:
                if self.layer_type in ("mamba", "m2"):
                    hidden_states = self.mixer(hidden_states)
                elif self.layer_type == "deltanet":
                    self.mixer(hidden_states, output)
                    hidden_states = output
                elif self.layer_type in ("attention", "a"):
                    # Attention layer
                    self.self_attn(positions, hidden_states, output)
                    hidden_states = output

            hidden_states = residual + hidden_states

            # Attention layers also have MoE after the attention
            if self.has_moe:
                residual = hidden_states
                hidden_states = self.pre_moe_layernorm(hidden_states)
                hidden_states = self.moe(hidden_states)
                hidden_states = residual + hidden_states

        return hidden_states


@support_torch_compile
class NemotronFlashModel(nn.Module):
    """
    Nemotron-Flash model with hybrid architecture and meta tokens.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Normalize config
        if isinstance(hf_config, NemotronFlashConfig):
            config = hf_config
        else:
            config = NemotronFlashConfig.from_pretrained_hf_config(hf_config)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.num_meta_tokens = config.num_meta_tokens
        if self.num_meta_tokens > 0:
            # Initialize with zeros - will be overwritten by load_weights
            self.memory_tokens = nn.Parameter(
                torch.zeros(self.num_meta_tokens, config.hidden_size)
            )

        layers_block_type = config.layers_block_type

        def get_layer(prefix: str):
            # Extract layer index from prefix (model.layers.0 -> 0)
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return NemotronFlashDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                layer_type=layers_block_type[layer_idx],
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # inputs_embeds is provided by the outer forward w/ memory tokens applied
        assert inputs_embeds is not None, "inputs_embeds must be provided"
        hidden_states = inputs_embeds

        output = torch.empty_like(hidden_states)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, output)

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _apply_memory_tokens(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Replace first num_meta embeddings of each prefill sequence with meta tokens.

        Uses forward_context to get query_start_loc for proper sequence boundaries.
        """
        num_meta = self.num_meta_tokens

        # Get attention metadata for sequence boundaries
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # attn_metadata might be a dict keyed by prefix for hybrid models
        if isinstance(attn_metadata, dict):
            # Use the first available metadata for sequence info
            # (all should have same query_start_loc for embedding phase)
            attn_metadata = next(iter(attn_metadata.values()))

        if attn_metadata is None:
            return hidden_states

        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if query_start_loc is None or len(query_start_loc) <= 1:
            return hidden_states

        # Clone to avoid in-place modification
        result = hidden_states.clone()

        # For each sequence, check if it starts at position 0 (new prefill)
        # and replace its first num_meta embeddings
        num_seqs = len(query_start_loc) - 1

        for i in range(num_seqs):
            seq_start = query_start_loc[i].item()
            seq_end = query_start_loc[i + 1].item()
            seq_len = seq_end - seq_start

            if seq_start < len(positions) and positions[seq_start].item() == 0:
                # Replace first num_meta embeddings with memory_tokens
                tokens_to_replace = min(num_meta, seq_len)
                if tokens_to_replace > 0:
                    result[seq_start : seq_start + tokens_to_replace] = (
                        self.memory_tokens[:tokens_to_replace]
                    )
        return result


class NemotronFlashForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "A_log": "A",
            ".gla.": ".mixer.",
            ".mamba.": ".mixer.",
            ".final_layernorm.": ".norm.",
            ".ffn.": ".moe.experts.0.",
            ".pre_ffn_layernorm.": ".pre_moe_layernorm.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        """Return dtypes for state cache."""
        conv_dtype = MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )[0]
        return (conv_dtype, torch.float32)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        """Calculate state shapes for KV cache allocation.

        Note: This model has Mamba2 and DeltaNet layers. We need to return
        the shapes that result in the larger page size for padding.

        Args:
            vllm_config: vLLM config

        Returns:
            State shapes with the larger page size
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        if not isinstance(hf_config, NemotronFlashConfig):
            hf_config = NemotronFlashConfig.from_pretrained_hf_config(hf_config)

        intermediate_size = hf_config.hidden_size * hf_config.mamba_expand
        tp_size = parallel_config.tensor_parallel_size

        mamba2_shapes = MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=tp_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
        )

        key_dim = hf_config.num_attention_heads * (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        value_dim = hf_config.num_key_value_heads * (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        total_conv_dim = key_dim * 2 + value_dim
        deltanet_conv_shape = (total_conv_dim // tp_size, hf_config.conv_kernel - 1)
        deltanet_temporal_shape = (
            hf_config.num_key_value_heads // tp_size,
            hf_config.hidden_size // hf_config.num_attention_heads,  # key_head_dim
            hf_config.hidden_size // hf_config.num_attention_heads,  # value_head_dim
        )
        deltanet_shapes = (deltanet_conv_shape, deltanet_temporal_shape)

        from math import prod

        from vllm.utils.torch_utils import get_dtype_size

        conv_dtype = MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            cache_config.mamba_cache_dtype,
        )[0]

        mamba2_page_size = sum(
            prod(shape) * get_dtype_size(torch.bfloat16 if i == 0 else torch.float32)
            for i, shape in enumerate(mamba2_shapes)
        )

        deltanet_page_size = prod(deltanet_shapes[0]) * get_dtype_size(
            conv_dtype
        ) + prod(deltanet_shapes[1]) * get_dtype_size(torch.float32)

        if deltanet_page_size > mamba2_page_size:
            return deltanet_shapes
        return mamba2_shapes

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config

        if isinstance(hf_config, NemotronFlashConfig):
            config = hf_config
        else:
            config = NemotronFlashConfig.from_pretrained_hf_config(hf_config)

        self.config = config
        self.vllm_config = vllm_config

        self.model = NemotronFlashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is None:
            raise ValueError("input_ids is required for this model")

        # Check if we're in CUDA graph capture mode
        is_capturing = torch.cuda.is_current_stream_capturing()

        # compute embeddings in the outer forward
        hidden_states = self.model.embed_input_ids(input_ids)

        # Apply memory tokens only during real prefill inference, not capture/warmup
        if not is_capturing and self.model.num_meta_tokens > 0:
            # Check if these are dummy positions (all zeros) used during warmup
            # For real inference: positions go 0,1,2,... or have various values
            # For warmup/capture: all positions are 0
            #
            # We compute max - min: if > 0, we have real positions
            pos_range = (positions.max() - positions.min()).item()
            first_pos = positions[0].item()

            # Only apply memory tokens if positions span a range (real prefill)
            # or if we're doing decode (any position > 0)
            if pos_range > 0 or first_pos > 0:
                hidden_states = self.model._apply_memory_tokens(
                    hidden_states, positions
                )

        # Always pass embeddings to the inner model
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, hidden_states
        )

        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata=None
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return expert parameter mapping for FusedMoE weight loading."""
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=1,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def preprocess_weights():
            for name, weight in weights:
                if (
                    ".gla." in name
                    and "conv1d.weight" in name
                    and weight.dim() == 3
                    and weight.size(1) == 1
                ):
                    weight = weight.squeeze(1)

                yield (name, weight)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None,
        )
        return loader.load_weights(preprocess_weights(), mapper=self.hf_to_vllm_mapper)
