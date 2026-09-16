# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 MBZUAI-IFM.
# Copyright 2024 The Qwen team.
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
"""Inference-only K2Horizon model compatible with HuggingFace weights."""

import math
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.config import (
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_quant_dtype,
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl

logger = init_logger(__name__)


def split_to_interleaved(x):
    # Split halves:  x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    # Interleaved:   x0 y0 x1 y1 x2 y2 x3 y3 ...
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def interleaved_to_split(x):
    # Interleaved:   x0 y0 x1 y1 x2 y2 x3 y3 ...
    # Split halves:  x0 x1 x2 x3 ... y0 y1 y2 y3 ...
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def calc_router_weights(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None,
    score_func: str,
    top_k: int,
    scaling_factor: float | None,
    renormalize: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_func == "softmax":
        routing_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    else:
        assert score_func == "sigmoid", (
            f"Unsupported router score function: {score_func}"
        )
        routing_scores = torch.sigmoid(router_logits.to(torch.float32))

    selection_scores = routing_scores
    if e_score_correction_bias is not None:
        selection_scores = selection_scores + e_score_correction_bias.to(
            selection_scores
        )

    selected_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    routing_weights = torch.gather(routing_scores, dim=-1, index=selected_indices)
    if renormalize is None:
        renormalize = top_k > 1
    if renormalize:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    if scaling_factor is not None:
        routing_weights = routing_weights * scaling_factor
    return routing_weights.to(router_logits.dtype), selected_indices


def fused_mova_impl(
    config,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if ocp_mx_scheme is not None:
        raise NotImplementedError(
            f"Using ocp_mx_scheme={ocp_mx_scheme} in functional fused_experts call is "
            "deprecated. Please use OCP_MXQuantizationEmulationTritonExperts."
        )

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, K = w1.size()

    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    M = num_tokens

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    intermediate_cache1 = torch.empty(
        (M, top_k_num, N), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = torch.empty(
        (M, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    qhidden_states, a1q_scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=a1_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k_num,
        global_num_experts,
        expert_map,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )

    dispatch_fused_moe_kernel(
        qhidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # apply_router_weight_on_input,
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=w1_bias,
    )

    intermediate_cache1 = F.silu(intermediate_cache1)
    intermediate_cache1 *= topk_weights[:, :, None].to(intermediate_cache1.dtype)

    ops.moe_sum(intermediate_cache1, out_hidden_states)

    return out_hidden_states


class K2HorizonRMSNorm(RMSNorm):
    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        assert hidden_size % n_groups == 0
        super().__init__(hidden_size, eps=eps)
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.group_size = hidden_size // n_groups

    def forward(self, x, residual=None):
        if self.n_groups == 1:
            return super().forward(x, residual)

        if residual is not None:
            x = x + residual
            residual = x

        orig_shape = x.shape
        x_grouped = x.reshape(-1, self.n_groups, self.group_size).reshape(
            -1, self.group_size
        )
        y_grouped = torch.empty_like(x_grouped)

        # Reuse vLLM's RMSNorm CUDA kernel on each group.
        ops.rms_norm(
            y_grouped,
            x_grouped,
            torch.ones(self.group_size, dtype=x.dtype, device=x.device),
            self.variance_epsilon,
        )

        y = y_grouped.reshape(orig_shape) * self.weight
        return (y, residual) if residual is not None else y


class K2HorizonMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        return out


class K2HorizonSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=config.moe_gate_bias,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        if self.gate.bias is not None:
            self.gate.bias = nn.Parameter(self.gate.bias.float(), requires_grad=False)

        self.num_shared_experts = config.num_shared_experts
        if config.num_shared_experts > 0:
            self.shared_experts = K2HorizonMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=None,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            scoring_func=config.router_score_func,
            e_score_correction_bias=self.gate.bias,
            routed_scaling_factor=config.router_scaling_factor,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "K2HorizonSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class K2HorizonAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        rope_head_dim: int,
        query_key_norm: bool,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        gate_func: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
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
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rope_head_dim = rope_head_dim or self.head_dim
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.query_key_norm = query_key_norm
        if self.query_key_norm:
            self.q_norm = K2HorizonRMSNorm(
                hidden_size=self.q_size, n_groups=self.num_heads, eps=rms_norm_eps
            )
            self.k_norm = K2HorizonRMSNorm(
                hidden_size=self.kv_size, n_groups=self.num_kv_heads, eps=rms_norm_eps
            )

        self.gate_func = gate_func
        if self.gate_func is not None:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.query_key_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope_head_dim == self.head_dim:
            q, k = self.rotary_emb(positions, q, k)
        else:
            q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
            k = k.reshape(*k.shape[:-1], self.num_kv_heads, self.head_dim)

            q_rope, q_nope = torch.split(
                split_to_interleaved(q),
                split_size_or_sections=[
                    self.rope_head_dim,
                    self.head_dim - self.rope_head_dim,
                ],
                dim=-1,
            )
            k_rope, k_nope = torch.split(
                split_to_interleaved(k),
                split_size_or_sections=[
                    self.rope_head_dim,
                    self.head_dim - self.rope_head_dim,
                ],
                dim=-1,
            )

            q_rope, k_rope = self.rotary_emb(
                positions, interleaved_to_split(q_rope), interleaved_to_split(k_rope)
            )

            q = interleaved_to_split(
                torch.cat([split_to_interleaved(q_rope), q_nope], dim=-1)
            ).reshape(*q.shape[:-2], -1)

            k = interleaved_to_split(
                torch.cat([split_to_interleaved(k_rope), k_nope], dim=-1)
            ).reshape(*k.shape[:-2], -1)

        attn_output = self.attn(q, k, v)

        if self.gate_func is not None:
            gate, _ = self.gate_proj(hidden_states)
            if self.gate_func == "silu":
                gate = F.silu(gate)
            else:
                assert self.gate_func == "softplus"
                gate = F.softplus(gate, beta=math.log(2))

            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class K2HorizonMoVAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        rope_head_dim: int,
        query_key_norm: bool,
        num_experts: int,
        num_experts_per_tok: int,
        moe_gate_bias: bool,
        router_score_func: str,
        router_scaling_factor: float,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        gate_func: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
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
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_score_func = router_score_func
        self.router_scaling_factor = router_scaling_factor

        self.qk_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.total_num_heads * self.head_dim, self.kv_size * tp_size],
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qk_proj",
        )

        self.v_router = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=moe_gate_bias,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.v_router",
        )
        if self.v_router.bias is not None:
            self.v_router.bias = nn.Parameter(
                self.v_router.bias.float(), requires_grad=False
            )

        self.v_experts = nn.ModuleList(
            [
                ColumnParallelLinear(
                    hidden_size,
                    self.kv_size * tp_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.v_experts.{value}",
                )
                for value in range(num_experts)
            ]
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rope_head_dim = rope_head_dim or self.head_dim
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.query_key_norm = query_key_norm
        if self.query_key_norm:
            self.q_norm = K2HorizonRMSNorm(
                hidden_size=self.q_size, n_groups=self.num_heads, eps=rms_norm_eps
            )
            self.k_norm = K2HorizonRMSNorm(
                hidden_size=self.kv_size, n_groups=self.num_kv_heads, eps=rms_norm_eps
            )

        self.gate_func = gate_func
        if self.gate_func is not None:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        w1_shape = torch.Size(
            [
                self.num_experts,
                self.total_num_kv_heads * self.head_dim,
                hidden_size,
            ]
        )
        config_dtype = _get_config_dtype_str(
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            dtype=vllm_config.model_config.dtype,
        )
        self.fused_mova_config = try_get_optimal_moe_config(
            w1_shape=w1_shape,
            w2_shape=torch.Size([w1_shape[0], w1_shape[2], w1_shape[1]]),
            top_k=self.num_experts_per_tok,
            dtype=config_dtype,
            M=vllm_config.scheduler_config.max_num_batched_tokens,
            block_shape=None,
        )

    def compute_mova_v_sparse(self, hidden_states):
        router_logits, _ = self.v_router(hidden_states)

        routing_weights, selected_values = calc_router_weights(
            router_logits=router_logits,
            e_score_correction_bias=self.v_router.bias,
            score_func=self.router_score_func,
            top_k=self.num_experts_per_tok,
            scaling_factor=self.router_scaling_factor,
        )

        w1 = torch.stack(
            [expert.weight for expert in self.v_experts], dim=0
        ).contiguous()

        v = fused_mova_impl(
            config=self.fused_mova_config,
            hidden_states=hidden_states.contiguous(),
            w1=w1,
            topk_weights=routing_weights.contiguous(),
            topk_ids=selected_values.contiguous(),
            global_num_experts=self.num_experts,
            expert_map=None,
        )

        return v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qk, _ = self.qk_proj(hidden_states)
        q, k = qk.split([self.q_size, self.kv_size], dim=-1)
        v = self.compute_mova_v_sparse(hidden_states)

        if self.query_key_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope_head_dim == self.head_dim:
            q, k = self.rotary_emb(positions, q, k)
        else:
            q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
            k = k.reshape(*k.shape[:-1], self.num_kv_heads, self.head_dim)

            q_rope, q_nope = torch.split(
                split_to_interleaved(q),
                split_size_or_sections=[
                    self.rope_head_dim,
                    self.head_dim - self.rope_head_dim,
                ],
                dim=-1,
            )
            k_rope, k_nope = torch.split(
                split_to_interleaved(k),
                split_size_or_sections=[
                    self.rope_head_dim,
                    self.head_dim - self.rope_head_dim,
                ],
                dim=-1,
            )

            q_rope, k_rope = self.rotary_emb(
                positions, interleaved_to_split(q_rope), interleaved_to_split(k_rope)
            )

            q = interleaved_to_split(
                torch.cat([split_to_interleaved(q_rope), q_nope], dim=-1)
            ).reshape(*q.shape[:-2], -1)

            k = interleaved_to_split(
                torch.cat([split_to_interleaved(k_rope), k_nope], dim=-1)
            ).reshape(*k.shape[:-2], -1)

        attn_output = self.attn(q, k, v)

        if self.gate_func is not None:
            gate, _ = self.gate_proj(hidden_states)
            if self.gate_func == "silu":
                gate = F.silu(gate)
            else:
                assert self.gate_func == "softplus"
                gate = F.softplus(gate, beta=math.log(2))

            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class K2HorizonDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        is_sparse_layer = (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if is_sparse_layer and config.mova_num_experts > 0:
            self.self_attn = K2HorizonMoVAAttention(
                vllm_config=vllm_config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_parameters=config.rope_parameters,
                rope_head_dim=config.rope_head_dim,
                query_key_norm=config.query_key_norm,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=config.attention_bias,
                head_dim=config.head_dim,
                gate_func=config.attention_gate_func,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                dual_chunk_attention_config=dual_chunk_attention_config,
                num_experts=config.mova_num_experts,
                num_experts_per_tok=config.mova_num_experts_per_tok,
                moe_gate_bias=config.moe_gate_bias,
                router_score_func=config.router_score_func,
                router_scaling_factor=config.router_scaling_factor,
            )
        else:
            self.self_attn = K2HorizonAttention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                rope_parameters=config.rope_parameters,
                rope_head_dim=config.rope_head_dim,
                query_key_norm=config.query_key_norm,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=config.attention_bias,
                head_dim=config.head_dim,
                gate_func=config.attention_gate_func,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                dual_chunk_attention_config=dual_chunk_attention_config,
            )

        if is_sparse_layer:
            self.mlp = K2HorizonSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = K2HorizonMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps,
        )

        self.post_attention_layernorm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class K2HorizonModel(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[torch.nn.Module] = K2HorizonDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.norm = K2HorizonRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            eps=config.rms_norm_eps,
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        # ignore_suffixes = (
        #     ".bias",
        #     "_bias",
        #     ".weight_scale",
        #     "_weight_scale",
        #     ".input_scale",
        #     "_input_scale",
        # )
        ignore_suffixes = ()

        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # QK norm weights
            if name.endswith(".self_attn.q_norm.weight") or name.endswith(
                ".self_attn.k_norm.weight"
            ):
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()

                if loaded_weight.shape != param.shape:
                    if name.endswith(".self_attn.q_norm.weight"):
                        loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    else:
                        num_kv_heads = self.config.num_key_value_heads
                        if num_kv_heads >= tp_size:
                            loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                        else:
                            num_kv_head_replicas = tp_size // num_kv_heads
                            kv_rank = tp_rank // num_kv_head_replicas
                            loaded_weight = loaded_weight.chunk(num_kv_heads, dim=0)[
                                kv_rank
                            ]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            # MoVA v experts
            if ".self_attn.v_experts." in name:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                num_kv_heads = self.config.num_key_value_heads

                if loaded_weight.shape != param.shape:
                    if num_kv_heads >= tp_size:
                        loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    else:
                        num_kv_head_replicas = tp_size // num_kv_heads
                        kv_rank = tp_rank // num_kv_head_replicas
                        loaded_weight = loaded_weight.chunk(num_kv_heads, dim=0)[
                            kv_rank
                        ]

                default_weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                if ".self_attn." in name and param_name == "gate_up_proj":
                    assert weight_name == "gate_proj"
                    continue
                if (
                    ".self_attn." in name
                    and param_name == "qkv_proj"
                    and name.replace(weight_name, param_name) not in params_dict
                ):
                    qk_name = name.replace(weight_name, "qk_proj")
                    if is_pp_missing_parameter(qk_name, self):
                        continue
                    if qk_name not in params_dict:
                        continue

                    assert weight_name in ["q_proj", "k_proj"]
                    if weight_name == "k_proj":
                        tp_size = get_tensor_model_parallel_world_size()
                        num_kv_heads = self.config.num_key_value_heads
                        if num_kv_heads < tp_size:
                            num_kv_head_replicas = tp_size // num_kv_heads
                            loaded_weight = (
                                loaded_weight.reshape(
                                    num_kv_heads, -1, *loaded_weight.shape[1:]
                                )
                                .repeat_interleave(num_kv_head_replicas, dim=0)
                                .reshape(-1, *loaded_weight.shape[1:])
                            )

                    param = params_dict[qk_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(
                        param, loaded_weight, 0 if weight_name == "q_proj" else 1
                    )
                    loaded_params.add(qk_name)
                    break

                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
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

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class K2HorizonForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "qk_proj": [
            "q_proj",
            "k_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when K2HorizonMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = K2HorizonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, K2HorizonDecoderLayer)
            if isinstance(layer.mlp, K2HorizonSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_layer is not None:
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, K2HorizonSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
