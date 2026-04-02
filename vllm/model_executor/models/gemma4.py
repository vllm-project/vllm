# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
"""Gemma 4 model implementation for vLLM."""

from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
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
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _get_text_config(config):
    """Dereference text_config if config is a nested Gemma4Config.

    Gemma4 checkpoints use architectures=["Gemma4ForConditionalGeneration"]
    which yields a Gemma4Config with nested text_config. This function
    transparently returns the text config regardless of nesting.
    """
    if hasattr(config, "text_config"):
        return config.text_config
    return config


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
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
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4Router(nn.Module):
    """Router for Gemma4 MoE that preprocesses input before projection.

    Applies RMSNorm (no learned weight), root_size scaling
    (hidden_size^{-0.5}), then a learned per-dimension scale before
    projecting to expert logits.

    This preprocessing is applied ONLY to the router's input, not to
    the expert MLPs' input.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # RMSNorm without learned weight — pure normalization only
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, has_weight=False)
        # Per-dimension learned scale, applied after norm + root_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        # Constant 1/sqrt(hidden_size) scaling factor
        self.register_buffer(
            "root_size",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        # Project to expert logits; replicated across TP for consistent routing
        # GateLinear supports bf16 W/A → fp32 output, which is important
        # because the topk kernel often needs fp32 for stable routing.
        self.proj = GateLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw router logits [T, E]."""
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        router_logits, _ = self.proj(x)
        return router_logits


class Gemma4MoE(nn.Module):
    """Mixture of Experts for Gemma4 using vLLM's FusedMoE.

    Wraps FusedMoE with custom routing. The router projection is
    external (Gemma4Router) — this class only handles expert dispatch.

    Gemma4 routing: softmax over ALL experts → top-k → renormalize.
    per_expert_scale is folded into routing weights for mathematical
    correctness with FusedMoE's fused kernel.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        # Per-expert output scale folded into routing weights so that
        # FusedMoE's fused kernel computes: Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        # Gemma4 routing: softmax over ALL experts → top-k → renormalize.
        # FusedMoE's built-in fused_topk scopes softmax differently, so
        # a custom routing function is needed for numerical correctness.
        per_expert_scale = self.per_expert_scale

        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
            router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
            indicator = torch.nn.functional.one_hot(
                topk_ids, num_classes=gating_output.size(-1)
            ).sum(dim=-2)
            gate_weights = indicator * router_probabilities
            renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
            renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
            dispatch_weights = gate_weights / renorm_factor

            topk_weights = dispatch_weights.gather(1, topk_ids)

            # Fold per_expert_scale into routing weights
            expert_scales = per_expert_scale[topk_ids].to(topk_weights.dtype)
            topk_weights = topk_weights * expert_scales
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        # FusedMoE experts with custom Gemma4 routing
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            hidden_size=config.hidden_size,
            intermediate_size=getattr(
                config,
                "moe_intermediate_size",
                getattr(config, "expert_intermediate_size", None),
            ),
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=routing_function,
            activation="gelu",
        )

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return self.experts(x, router_logits)


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        use_k_eq_v: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.use_k_eq_v = use_k_eq_v

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma4 uses scaling=1.0.
        # Unlike Gemma2/3, query_pre_attn_scalar is NOT used here;
        # Q/K norms with learnable weights handle scaling implicitly.
        self.scaling = 1.0

        # QKVParallelLinear handles GQA correctly for all layer types.
        # k_eq_v layers load K weights into both K and V slots via
        # _weight_iterator remapping — no structural difference needed.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Q/K norms: output = norm(x) * weight (learnable per-head scale)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # V norm: no learnable scale (pure normalization only)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)

        # Determine layer type and sliding window
        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        # Initialize RoPE based on layer type.
        # Gemma4 uses different RoPE parameters for sliding vs full attention.
        if layer_type in config.rope_parameters:
            # Per-layer-type rope config (dict format).
            # rope_parameters already contains the correct
            # partial_rotary_factor per layer type (1.0 for full
            # attention, 1.0 for sliding). Do NOT override with
            # global_partial_rotary_factor — that config key is
            # not needed for Gemma4 — config uses per-layer rope_parameters.
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            # Legacy config format fallback.
            rope_parameters = dict(config.rope_parameters.copy())
            if self.is_sliding:
                rope_parameters["rope_theta"] = getattr(
                    config, "rope_local_base_freq", 10000.0
                )

        # KV sharing: layers in the last `num_kv_shared_layers` share KV
        # cache with earlier layers of the same type.
        kv_sharing_target_layer_name = None
        self.is_kv_shared_layer = False
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers > 0:
            first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers
            if layer_idx >= first_kv_shared_layer_idx:
                self.is_kv_shared_layer = True
                # Find the last non-shared layer of the same attention type
                prev_layers = config.layer_types[:first_kv_shared_layer_idx]
                current_layer_type = config.layer_types[layer_idx]
                kv_shared_layer_index = (
                    len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
                )
                if kv_shared_layer_index >= 0:
                    if ".layers." in prefix:
                        param_name_before_layers = prefix.split(".layers.")[0]
                    else:
                        raise ValueError(
                            "Unexpected prefix format for Gemma4Attention: "
                            f"'{prefix}'. Expected to contain '.layers.'."
                        )
                    kv_sharing_target_layer_name = (
                        f"{param_name_before_layers}.layers."
                        f"{kv_shared_layer_index}.self_attn.attn"
                    )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Unified QKV path (works for both k_eq_v and standard layers).
        # For k_eq_v, K weights are loaded into both K and V slots of
        # qkv_proj, so V == K automatically.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q norm (always applied)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            # Non-shared: apply K norm + RoPE, V norm
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            # Shared: only apply RoPE to Q
            q = self.rotary_emb(positions, q, k)[0]

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)

        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )

        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx

        # Gemma4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        # Determine if this full-attention layer uses k_eq_v
        # (laptop variant: no v_proj, K reused as V on full attention layers)
        use_k_eq_v = self.is_full_attention and getattr(
            config, "attention_k_eq_v", False
        )

        # For k_eq_v full-attention layers, use num_global_key_value_heads
        # as the KV head count when k_eq_v is enabled.
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        self.self_attn = Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            use_k_eq_v=use_k_eq_v,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
        )

        # Compute per-layer intermediate_size from config.
        # When use_double_wide_mlp is set, intermediate_size doubles for
        # KV-shared layers (layers >= first_kv_shared_layer_idx).
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        layer_intermediate_size = config.intermediate_size * (
            2 if use_double_wide_mlp else 1
        )

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Layer norms: output = norm(x) * weight
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE (Mixture of Experts) — router + expert block parallel to MLP
        self.enable_moe_block = getattr(config, "enable_moe_block", False) or getattr(
            config, "use_second_mlp_block", False
        )
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.router",
            )
            self.moe = Gemma4MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.router = None
            self.moe = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

        # Per-Layer Embedding (PLE) components — present in each decoder layer
        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            # Gate: projects hidden_states → per-layer dim for gating
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_input_gate",
                return_bias=False,
            )
            # Projection: projects gated per-layer input back → hidden size
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_projection",
                return_bias=False,
            )
            # Post-PLE norm: output = norm(x) * weight
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar (loaded from checkpoint) — applies to ALL text layers
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gemma4 residual pattern:
        # 1. input_norm(x) → attn → post_attn_norm → ADD residual
        # 2. pre_ff_norm → mlp → post_ff_norm → ADD residual
        residual = hidden_states

        hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        # MLP runs unconditionally (same inputs for MoE and non-MoE)
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # Router and MoE experts see the residual (pre-MLP state),
            # matching the HF transformers forward path
            router_logits = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine MLP and MoE outputs
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # Apply PLE (Per-Layer Embedding) if configured
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution

        # Apply layer scalar for full-attention layers
        # Apply per-layer scalar (all text layers)
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, None


@support_torch_compile
class Gemma4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # PLE config values (default to 0 if not present — disables PLE)
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        self.vocab_size_per_layer_input = getattr(
            config, "vocab_size_per_layer_input", config.vocab_size
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        # Per-Layer Embedding (PLE) components
        if (
            self.hidden_size_per_layer_input is not None
            and self.hidden_size_per_layer_input > 0
        ):
            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input,
                total_ple_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
            # Scaled embedding factor (from config, not hardcoded)
            # Register as buffer so it moves to GPU with the model
            # and interacts correctly with torch.compile AOT caching.
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            # Projection: hidden_size → total_ple_dim
            # ColumnParallelLinear with gather_output=True
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            # PLE projection norm: output = norm(x) * weight
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
            # Scale factor for combining projection + per_layer_inputs
            # Register as buffer so it moves to GPU with the model
            # and interacts correctly with torch.compile AOT caching.
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            # Scaled projection: multiply output by hidden_size**-0.5.
            # Register as buffer for GPU placement and torch.compile.
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(config.hidden_size**-0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.embed_scale_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        # Final norm: output = norm(x) * weight
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Embedding scale = sqrt(hidden_size)
        # Downcast to model dtype (bfloat16 etc.) for numerical parity
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )
        # Custom factory that includes per_layer_inputs for PLE-enabled PP.
        # per_layer_inputs has shape (batch, num_layers, per_layer_dim),
        # which differs from the standard (batch, hidden_size) shape,
        # so we can't use the default factory.
        ple_dim = self.hidden_size_per_layer_input
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        def _make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            tensors: dict[str, torch.Tensor] = {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "residual": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
            if ple_dim and ple_dim > 0:
                tensors["per_layer_inputs"] = torch.zeros(
                    (batch_size, num_layers, ple_dim),
                    dtype=dtype,
                    device=device,
                )
            return IntermediateTensors(tensors)

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get per-layer embeddings from embed_tokens_per_layer.

        Returns:
            Per-layer embeddings (num_tokens, num_layers,
            hidden_size_per_layer_input)
        """
        if self.embed_tokens_per_layer is None:
            return None

        # Handle out-of-vocab tokens for PLE (vocab_size_per_layer_input may
        # be smaller than the main vocab_size).
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )

        # Get packed per-layer embeddings: (num_tokens, total_ple_dim)
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)

        # Apply embed_scale (sqrt of per-layer hidden dim)
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return per_layer_embeds

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project inputs_embeds and combine with per_layer_inputs.

        Steps:
        1. Project inputs_embeds: hidden_size → total_ple_dim
        2. Scale by hidden_size^{-0.5}
        3. Reshape to (num_tokens, num_layers, per_layer_dim)
        4. Normalize with per_layer_projection_norm
        5. Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        """
        if self.per_layer_model_projection is None:
            return None

        # Project from hidden_size to total_ple_dim
        # Scaled projection: output = linear(input, weight) * scale
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        # Normalize
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        # Combine: (projection + per_layer_inputs) * scale
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                # When called from the multimodal wrapper, raw PLE
                # embeddings are pre-computed and passed explicitly.
                # Project them through per_layer_model_projection.
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                # Compute per-layer inputs for PLE
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")

        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            # Extract the per-layer embedding for this specific layer
            if per_layer_inputs is not None:
                actual_layer_idx = self.start_layer + layer_idx
                layer_per_input = per_layer_inputs[
                    :, actual_layer_idx, :
                ]  # (num_tokens, per_layer_dim)
            else:
                layer_per_input = None
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "per_layer_inputs": per_layer_inputs,
                }
            )
        # Gemma4 incorporates residual into hidden_states directly
        # Apply norm without residual fusion when possible.
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MoE expert weight mapping: checkpoint 3D packed tensors are
        # exploded in _weight_iterator to per-expert 2D weights like:
        #   moe.experts.{id}.gate_proj → FusedMoE w1 (shard of w13)
        #   moe.experts.{id}.up_proj   → FusedMoE w3 (shard of w13)
        #   moe.experts.{id}.down_proj → FusedMoE w2
        # We build the mapping directly since Gemma4 uses bare param
        # names (no .weight suffix) unlike standard MoE checkpoints.
        num_experts = getattr(self.config, "num_experts", None) or 0
        expert_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_weight"
                if proj_name in ["gate_proj", "up_proj"]
                else "experts.w2_weight",
                f"experts.{expert_id}.{proj_name}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, proj_name in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
        ]
        params_dict = dict(self.named_parameters())
        # Include buffers (e.g. layer_scalar) so they can be loaded too
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                stacked_name = name.replace(shard_name, param_name)
                # k_eq_v layers use separate q_proj/k_proj instead of
                # packed qkv_proj. If the stacked param doesn't exist,
                # skip this mapping and fall through to direct load.
                if stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                param = params_dict[stacked_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    moe_name = name.replace(weight_name, param_name)
                    if moe_name not in params_dict:
                        continue
                    if is_pp_missing_parameter(moe_name, self):
                        continue
                    param = params_dict[moe_name]
                    # Expert weights are already in the correct
                    # orientation for FusedMoE after _weight_iterator:
                    #   gate/up: [I, H] → w1/w3 expects [I, H]
                    #   down:    [H, I] → w2 expects [H, I]
                    assert loaded_weight.dim() == 2, (
                        f"Expected 2D expert weight for {weight_name}, "
                        f"got shape {loaded_weight.shape}"
                    )
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name + ".weight",
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(moe_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

class Gemma4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,
                        MixtureOfExperts):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Gemma4ForConditionalGeneration already loads the text stack
            # from `model.language_model.*`. We reuse that same checkpoint
            # and adapter naming for the text-only Gemma4ForCausalLM path,
            # so LoRA keys from the conditional wrapper map onto `model.*`.
            "model.language_model.": "model.",
        }
    )
    # Note: qkv_proj packing applies to non-k_eq_v layers (sliding
    # attention and full attention without k_eq_v). k_eq_v layers use
    # separate q_proj + k_proj without packing.
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
        config = _get_text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # --- MixtureOfExperts protocol ---
        self.expert_weights: list[list[torch.Tensor]] = []
        self.moe_layers: list[nn.Module] = []
        example_moe: Gemma4MoE | None = None

        for layer in self.model.layers:
            if hasattr(layer, "moe") and isinstance(layer.moe, Gemma4MoE):
                example_moe = layer.moe
                self.moe_layers.append(layer.moe.experts)

        self.num_moe_layers = len(self.moe_layers)

        if example_moe is not None:
            self.num_logical_experts = example_moe.num_experts
            self.num_physical_experts = example_moe.num_experts
            self.num_local_physical_experts = example_moe.num_experts
            self.num_routed_experts = example_moe.num_experts
        else:
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Checkpoint weight names use "language_model." prefix (from the
        # Gemma4ForConditionalGeneration wrapper). Strip it to map to our
        # model tree which is just "model.*".
        def _weight_iterator():
            use_k_eq_v = getattr(self.config, "attention_k_eq_v", False)
            # Build set of k_eq_v layer indices (full_attention layers
            # when attention_k_eq_v is enabled). These layers have k_proj
            # but no v_proj in checkpoint — we duplicate k_proj as v_proj.
            k_eq_v_layer_indices: set[int] = set()
            if use_k_eq_v:
                for idx, lt in enumerate(self.config.layer_types):
                    if lt == "full_attention":
                        k_eq_v_layer_indices.add(idx)

            for name, weight in weights:
                # Remap "language_model." → "" to match our model tree.
                # Checkpoint: model.language_model.layers.X.*
                # Our model:  model.layers.X.*
                name = name.replace("language_model.", "")

                # Remap new HF checkpoint naming to internal vLLM
                # naming: HF moved per_expert_scale to router and
                # renamed moe → experts in the MoE block.
                name = name.replace(
                    ".router.per_expert_scale",
                    ".moe.per_expert_scale",
                )
                if ".experts.gate_up_proj" in name:
                    name = name.replace(
                        ".experts.gate_up_proj",
                        ".moe.gate_up_proj",
                    )
                elif ".experts.down_proj" in name:
                    name = name.replace(
                        ".experts.down_proj",
                        ".moe.down_proj",
                    )

                # MoE expert weights: checkpoint stores as 3D packed
                # tensors.  Explode into per-expert 2D weights for
                # FusedMoE weight_loader.
                #
                # Checkpoint format:
                #   moe.gate_up_proj: [E, 2*I, H]  (fused gate + up)
                #   moe.down_proj:    [E, H, I]
                #
                # FusedMoE expects per-expert:
                #   w1 (gate): [I, H]   — first half of gate_up
                #   w3 (up):   [I, H]   — second half of gate_up
                #   w2 (down): [H, I]   — as-is from checkpoint
                #
                # No transpose needed: checkpoint orientation already
                # matches FusedMoE's expected layout.
                if "moe.gate_up_proj" in name and weight.dim() == 3:
                    num_experts = weight.size(0)
                    intermediate_size = weight.size(1) // 2
                    for expert_id in range(num_experts):
                        gate_weight = weight[expert_id, :intermediate_size, :]
                        up_weight = weight[expert_id, intermediate_size:, :]
                        base = name.replace("moe.", f"moe.experts.{expert_id}.")
                        yield base.replace("gate_up_proj", "gate_proj"), gate_weight
                        yield base.replace("gate_up_proj", "up_proj"), up_weight
                    continue

                if "moe.down_proj" in name and weight.dim() == 3:
                    num_experts = weight.size(0)
                    for expert_id in range(num_experts):
                        expert_name = name.replace("moe.", f"moe.experts.{expert_id}.")
                        yield expert_name, weight[expert_id]
                    continue

                # k_eq_v layers: checkpoint has k_proj but no v_proj.
                # QKVParallelLinear expects both, so duplicate k_proj
                # as v_proj so V gets identical weights to K.
                # ONLY for full_attention layers — sliding layers have
                # their own real v_proj weights.
                if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                    m = re.search(r"layers\.(\d+)\.", name)
                    if m and int(m.group(1)) in k_eq_v_layer_indices:
                        yield name, weight
                        yield name.replace("k_proj", "v_proj"), weight.clone()
                        continue

                yield name, weight

        # Skip multimodal weights — handled by the multimodal wrapper.
        # Also skip lm_head when weights are tied.
        skip = [
            "audio_tower.",
            "vision_tower.",
            "embed_audio.",
            "embed_vision.",
        ]
        if self.config.tie_word_embeddings:
            skip.append("lm_head.")

        loader = AutoWeightsLoader(self, skip_substrs=skip)
        return loader.load_weights(_weight_iterator())
