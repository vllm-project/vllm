# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    RMSNormGated,
    layernorm_fn,
)
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01LinearAttention,
    MiniMaxText01LinearKernel,
    MiniMaxText01RMSNormTP,
    clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode,
    linear_attention_prefill_and_mix,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.bailing_moe import BailingMLP
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

from .interfaces import HasInnerState, IsHybrid, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def is_linear_layer(layer_idx, layer_group_size):
    if layer_idx is None:
        return False
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


def _build_rope_parameters(config: PretrainedConfig) -> dict | None:
    rope_parameters = copy.deepcopy(getattr(config, "rope_parameters", None)) or {}
    if "rope_theta" not in rope_parameters and hasattr(config, "rope_theta"):
        rope_parameters["rope_theta"] = config.rope_theta
    if "partial_rotary_factor" not in rope_parameters and hasattr(
        config, "partial_rotary_factor"
    ):
        rope_parameters["partial_rotary_factor"] = config.partial_rotary_factor

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_scaling = copy.deepcopy(rope_scaling)
        if "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type")
        rope_parameters.update(rope_scaling)

    return rope_parameters or None


class BailingMoeV25MLAAttention(nn.Module):
    """
    MLA Attention for BailingMoeV2.5 full attention layers.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.layer_id = layer_id
        self.prefix = prefix

        # MLA dimensions
        self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = getattr(config, "v_head_dim", 128)

        # LoRA ranks
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5

        # KV projections
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if self.q_lora_rank is not None:
            # Use fused_qkv_a_proj when q_lora_rank is set
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None
        else:
            # Direct projections when no q_lora_rank
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None

        rope_parameters = _build_rope_parameters(config)
        max_position = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            head_size=self.qk_rope_head_dim,
            max_position=max_position,
            is_neox_style=False,
            rope_parameters=rope_parameters or None,
            dtype=torch.float32,
        )

        # Build MLAModules for MultiHeadLatentAttentionWrapper
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
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
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MLA attention."""
        return self.mla_attn(positions, hidden_states)


class BailingMoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class BailingMoeV25(nn.Module):
    """Bailing MoE v2.5 - standalone implementation for linear attention model."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "",
    ):
        super().__init__()

        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        norm_topk_prob = getattr(config, "norm_topk_prob", None)
        # Ring-2.5 reference implementations normalize routing weights by default.
        self.norm_expert_prob = True if norm_topk_prob is None else bool(norm_topk_prob)
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        self.num_shared_experts = config.num_shared_experts
        self.score_function = getattr(config, "score_function", None)
        self.n_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.use_grouped_topk = self.n_group is not None and self.topk_group is not None
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None or router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # Gate for routing
        self.gate = BailingMoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        correction_bias = (
            self.gate.expert_bias if self.gate.expert_bias is not None else None
        )
        if self.score_function is not None:
            assert (self.score_function == "softmax" and correction_bias is None) or (
                self.score_function == "sigmoid" and correction_bias is not None
            ), (
                "score_function and correction_bias should be "
                "(softmax, None) or (sigmoid, not None)"
            )

        # Shared experts (using BailingMLP)
        if self.num_shared_experts > 0:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            self.shared_experts = BailingMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        # Routed experts using SharedFusedMoE
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=self.norm_expert_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.score_function,
            e_score_correction_bias=correction_bias,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            use_grouped_topk=self.use_grouped_topk,
            router_logits_dtype=self.router_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        # Ensure contiguous token-major layout before router/projections.
        hidden_states = hidden_states.contiguous().view(-1, hidden_size)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states.to(self.router_dtype))
        router_logits = router_logits.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        # Handle tuple return from SharedFusedMoE
        if self.shared_experts is not None:
            shared_output, final_hidden_states = final_hidden_states
        else:
            shared_output = None

        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_size)


BailingRMSNormTP = MiniMaxText01RMSNormTP


class BailingGroupRMSNormGate(RMSNormGated):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            hidden_size,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            device=device,
            dtype=dtype,
            activation="sigmoid",
        )
        # Add custom weight loader for TP sharding
        self.weight.weight_loader = self._weight_loader

    @staticmethod
    def _weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load weight with TP sharding."""
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = loaded_weight.shape[0] // tp_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard].contiguous())


class BailingMoELinearAttention(nn.Module, MambaBase):
    """
    Bailing MoE Linear Attention implementation using minimax backend.

    This implements the linear attention mechanism from sglang, adapted for vLLM's
    v1 engine with MambaBase interface support.
    """

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return state shape for linear attention cache.

        Must match the calculation in get_mamba_state_shape_from_config.
        """
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=self.total_num_heads,
            tp_size=self.tp_size,
            head_dim=self.head_dim,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return state dtype for linear attention cache.

        Must match the calculation in get_mamba_state_dtype_from_config.
        """
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "linear_attn",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_attention_heads  # MHA
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // self.total_num_heads
        )

        self.hidden_inner_size = self.head_dim * self.total_num_heads
        self.scaling = self.head_dim**-0.5

        assert self.total_num_heads % self.tp_size == 0
        self.tp_heads = self.total_num_heads // self.tp_size

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 600000)

        self.tp_kv_heads = self.total_kv_heads // self.tp_size
        self.q_size_per_rank = self.head_dim * self.tp_heads
        self.kv_size_per_rank = self.head_dim * self.tp_kv_heads

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.linear_backend = "minimax"
        self.linear_scale = self.linear_backend == "minimax"
        self.linear_rope = getattr(config, "linear_rope", True)
        if hasattr(config, "use_linear_silu"):
            self.linear_silu = config.use_linear_silu
        elif hasattr(config, "linear_silu"):
            self.linear_silu = config.linear_silu
        else:
            self.linear_silu = False

        # Block size for lightning attention
        self.BLOCK = getattr(config, "block", 256)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_heads,  # MHA: kv_heads = num_heads
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.dense = RowParallelLinear(
            self.hidden_inner_size,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
            reduce_results=True,
        )

        self.group_norm_size = getattr(config, "group_norm_size", 1)
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        assert self.tp_size <= self.group_norm_size, (
            "tp_size must be <= group_norm_size for local rms norm"
        )
        assert self.group_norm_size % self.tp_size == 0, (
            "group_norm_size must be divisible by tp_size"
        )

        # When group_norm_size == 1, group_size equals hidden_size // tp_size
        self.g_norm = BailingGroupRMSNormGate(
            hidden_size=self.hidden_inner_size // self.tp_size,
            eps=self.rms_norm_eps,
            group_size=(
                self.hidden_inner_size // self.group_norm_size
                if self.group_norm_size > 1
                else self.hidden_inner_size // self.tp_size
            ),
        )

        # use fp32 rotary embedding
        rope_parameters = _build_rope_parameters(config)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            is_neox_style=True,
            dtype=torch.float32,
            rope_parameters=rope_parameters or None,
        )

        # Build slope tensor for linear attention decay
        num_hidden_layers = config.num_hidden_layers
        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(
            self.total_num_heads
        )
        if num_hidden_layers <= 1:
            self.slope_rate = slope_rate * (1 + 1e-5)
        else:
            self.slope_rate = slope_rate * (
                1 - layer_id / (num_hidden_layers - 1) + 1e-5
            )
        self.tp_slope = self.slope_rate[
            self.tp_rank * self.tp_heads : (self.tp_rank + 1) * self.tp_heads
        ].contiguous()

        # Register for compilation
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Load weight for linear attention layers.

        For FP8 quantized parameters, we need to use the weight_loader if available,
        as it handles special cases like tensor parallelism sharding.
        """
        # Check if param has a weight_loader (for vLLM ModelWeightParameter)
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            # Use the weight_loader which handles TP sharding and quantization
            weight_loader(param, loaded_weight)
        else:
            # Fall back to direct copy for standard tensors
            assert param.size() == loaded_weight.size(), (
                f"Shape mismatch: {param.shape} vs {loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Forward method called by torch.ops.vllm.linear_attention"""
        torch.ops.vllm.linear_attention(
            hidden_states,
            output,
            positions,
            self.prefix,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Actual forward implementation."""
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, LinearAttentionMetadata)
            num_actual_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        else:
            num_actual_tokens = hidden_states.shape[0]

        # QKV projection
        qkv, _ = self.query_key_value(hidden_states[:num_actual_tokens])

        # use rotary_emb support fp32
        qkv = qkv.to(torch.float32)
        if self.linear_silu:
            qkv = F.silu(qkv)

        # Split q, k, v
        q, k, v = torch.split(
            qkv,
            [self.q_size_per_rank, self.kv_size_per_rank, self.kv_size_per_rank],
            dim=-1,
        )

        # Apply QK norm if needed
        if self.use_qk_norm:
            q = q.reshape(-1, self.tp_heads, self.head_dim)
            k = k.reshape(-1, self.tp_kv_heads, self.head_dim)
            q = layernorm_fn(
                q,
                self.query_layernorm.weight.data,
                bias=None,
                eps=self.rms_norm_eps,
                is_rms_norm=True,
            )
            k = layernorm_fn(
                k,
                self.key_layernorm.weight.data,
                bias=None,
                eps=self.rms_norm_eps,
                is_rms_norm=True,
            )
            q = q.reshape(-1, self.q_size_per_rank)
            k = k.reshape(-1, self.kv_size_per_rank)

        # Apply rotary embeddings
        if self.linear_rope:
            q, k = self.rotary_emb(positions[:num_actual_tokens], q, k)

        # Reshape to [batch, heads, seq_len, head_dim]
        q = q.view((qkv.shape[0], self.tp_heads, self.head_dim))
        k = k.view((qkv.shape[0], self.tp_kv_heads, self.head_dim))
        v = v.view((qkv.shape[0], self.tp_kv_heads, self.head_dim))

        # Apply scaling if using minimax backend
        if self.linear_scale:
            q = q * self.scaling

        # Get KV cache and state indices
        if attn_metadata is not None:
            kv_cache = self.kv_cache[forward_context.virtual_engine][0]
            state_indices_tensor = attn_metadata.state_indices_tensor
            clear_linear_attention_cache_for_new_sequences(
                kv_cache, state_indices_tensor, attn_metadata
            )

        # Compute attention
        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if attn_metadata is None:
            hidden = torch.empty(
                (q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype
            )
        else:
            if not decode_only:
                hidden = self._prefill_and_mix_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
            else:
                hidden = self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )

        # Apply group norm and gate (matching SGLang behavior)
        gate, _ = self.g_proj(hidden_states[:num_actual_tokens])

        if self.group_norm_size > 1:
            hidden = self.g_norm(hidden, gate)
        else:
            hidden = self.g_norm(hidden)
            hidden = F.sigmoid(gate) * hidden

        hidden = hidden.to(hidden_states.dtype)

        # Output projection
        dense_out, _ = self.dense(hidden)
        output[:num_actual_tokens] = dense_out

    def _prefill_and_mix_infer(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata
    ):
        """Handle prefill (mixed with decode if any)."""
        return linear_attention_prefill_and_mix(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            state_indices_tensor=state_indices_tensor,
            attn_metadata=attn_metadata,
            slope_rate=self.tp_slope,
            block_size=self.BLOCK,
            decode_fn=self._decode_infer,
            prefix_fn=MiniMaxText01LinearKernel.jit_linear_forward_prefix,
            layer_idx=self.layer_id,
        )

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        """Handle decode (single token per sequence)."""
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_prefills = attn_metadata.num_prefills
        hidden = linear_attention_decode(
            q,
            k,
            v,
            kv_cache,
            self.tp_slope,
            state_indices_tensor,
            q_start=num_prefill_tokens,
            q_end=None,
            slot_start=num_prefills,
            slot_end=None,
            block_size=32,
        )
        return hidden


class BailingMoeV25DecoderLayer(nn.Module):
    """Decoder layer supporting both linear and full attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "layer",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # Determine attention type (0 = linear, 1 = full)
        self.attention_type = getattr(config, "attention_type", 1)

        if self.attention_type == 0:  # Linear attention
            self.self_attn = BailingMoELinearAttention(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
                model_config=model_config,
                cache_config=cache_config,
            )
        else:  # Full attention
            self.self_attn = BailingMoeV25MLAAttention(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        # MLP/MoE
        is_moe_layer = config.num_experts > 1 and layer_id >= getattr(
            config, "first_k_dense_replace", 0
        )

        if is_moe_layer:
            self.mlp = BailingMoeV25(
                config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = BailingMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        # Layer norms
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Input layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        if self.attention_type == 0:
            # Linear attention uses output tensor
            self_attention_output = torch.zeros_like(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            # Full attention
            self_attention_output = self.self_attn(hidden_states, positions)

        hidden_states, residual = self.post_attention_layernorm(
            self_attention_output, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class BailingMoeV25Model(nn.Module):
    """Bailing MoE v2.5 Model with hybrid attention support."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        # Determine layer types based on layer_group_size
        self.layer_group_size = getattr(config, "layer_group_size", 1)
        self.num_layers = config.num_hidden_layers

        # decoder_attention_types: 0 = linear, 1 = full
        self.decoder_attention_types = [
            0 if is_linear_layer(i, self.layer_group_size) else 1
            for i in range(self.num_layers)
        ]

        # Embeddings
        if get_pp_group().is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                org_num_embeddings=self.vocab_size,
            )
        else:
            from vllm.model_executor.models.utils import PPMissingLayer

            self.word_embeddings = PPMissingLayer()

        # Layers
        def layer_fn(prefix):
            layer_idx = int(prefix.split(".")[-1])
            layer_config = copy.deepcopy(config)
            layer_config.attention_type = self.decoder_attention_types[layer_idx]

            return BailingMoeV25DecoderLayer(
                config=layer_config,
                quant_config=quant_config,
                layer_id=layer_idx,
                prefix=prefix,
                model_config=model_config,
                cache_config=cache_config,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.num_layers, layer_fn, prefix=f"{prefix}.layers"
        )

        # Final norm
        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            from vllm.model_executor.models.utils import PPMissingLayer

            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if residual is not None:
                hidden_states, _ = self.norm(hidden_states, residual)
            else:
                hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Get expert parameter mapping for MoE layers."""
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint weights with simplified mapping."""

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        # Stacked parameter mappings (fused projections)
        stacked_mappings = [
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Expert parameter mappings from FusedMoE
        expert_mappings = list(self.get_expert_mapping())

        def load_param(name: str, tensor: torch.Tensor, shard_id=None) -> bool:
            """Load a single parameter."""
            if name not in params_dict or is_pp_missing_parameter(name, self):
                return False
            if name.endswith(".bias") and name not in params_dict:
                return False

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)

            if shard_id is None:
                weight_loader(param, tensor)
            elif isinstance(shard_id, int):
                weight_loader(param, tensor, shard_id)
            else:
                # Expert param: (expert_id, shard_id)
                weight_loader(
                    param, tensor, name, expert_id=shard_id[0], shard_id=shard_id[1]
                )

            loaded_params.add(name)
            return True

        def normalize_name(name: str) -> str | None:
            """Normalize checkpoint name to model parameter name."""
            # Skip special weights
            if name.startswith("model.mtp"):
                return None
            # Remove 'model.' prefix if present
            # (e.g., 'model.layers.0...' -> 'layers.0...')
            name = name.removeprefix("model.")
            # Map attention.dense based on layer type
            if "attention.dense" in name:
                layer_idx = (
                    int(name.split("layers.")[1].split(".")[0])
                    if "layers." in name
                    else 0
                )
                attn_name = (
                    "self_attn.dense"
                    if is_linear_layer(layer_idx, self.config.layer_group_size)
                    else "self_attn.o_proj"
                )
                name = name.replace("attention.dense", attn_name)

            # Standard mappings
            name = name.replace("attention.", "self_attn.")
            name = name.replace(
                "mlp.gate.e_score_correction_bias", "mlp.gate.expert_bias"
            )

            return maybe_remap_kv_scale_name(name, params_dict)

        for orig_name, weight in weights:
            norm_name = normalize_name(orig_name)
            if norm_name is None:
                continue

            # Try stacked mappings
            loaded = False
            for param_suf, weight_suf, shard_id in stacked_mappings:
                if weight_suf not in norm_name:
                    continue
                mapped = norm_name.replace(weight_suf, param_suf).replace(
                    "attention.", "self_attn."
                )
                if load_param(mapped, weight, shard_id):
                    loaded = True
                    break
            if loaded:
                continue

            # Handle expert weights
            if "mlp.experts" in norm_name:
                # Expert bias
                if (
                    "mlp.experts.e_score_correction_bias" in norm_name
                    or "mlp.experts.expert_bias" in norm_name
                ):
                    alt = norm_name.replace(
                        "mlp.experts.e_score_correction_bias", "mlp.gate.expert_bias"
                    ).replace("mlp.experts.expert_bias", "mlp.gate.expert_bias")
                    if load_param(alt, weight) or load_param(norm_name, weight):
                        continue

                # Routed experts
                for param_name, weight_name, expert_id, shard_id in expert_mappings:
                    if weight_name not in norm_name:
                        continue
                    mapped = norm_name.replace(weight_name, param_name)
                    if load_param(mapped, weight, (expert_id, shard_id)):
                        break
                continue

            # General parameters
            load_param(norm_name, weight)

        return loaded_params


class BailingMoeV25ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    """Bailing MoE v2.5 For CausalLM."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = BailingMoeV25Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        """Calculate shape for linear attention cache."""
        config = vllm_config.model_config.hf_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Return base state shape from linear attention (no padding)
        return MambaStateShapeCalculator.linear_attention_state_shape(
            num_heads=config.num_attention_heads,
            tp_size=tp_size,
            head_dim=head_dim,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.linear_attention_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
