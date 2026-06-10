# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import torch
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    RMSNormGated,
    layernorm_fn,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.linear.bailing_spec_decode_ops import (
    bailing_linear_attention_decode_spec,
)
from vllm.model_executor.layers.mamba.linear.base import LinearAttention
from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
    MiniMaxText01LinearAttention,
    MiniMaxText01LinearKernel,
    clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode,
    linear_attention_prefill_and_mix,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata


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


# --8<-- [start:bailing_moe_linear_attention]
@PluggableLayer.register("bailing_moe_linear_attention")
class BailingMoELinearAttention(LinearAttention):
    """Pluggable Bailing MoE Linear Attention layer which allows OOT backends
    to add custom implementations.

    This implements the linear attention mechanism from sglang, adapted for
    vLLM's v1 engine with MambaBase interface support.
    """

    # --8<-- [end:bailing_moe_linear_attention]
    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "linear_attn",
    ):
        super().__init__(config, vllm_config, prefix)

        self.scaling = self.head_dim**-0.5

        self.tp_heads = self.num_heads // self.tp_size

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 600000)

        self.tp_kv_heads = self.num_heads // self.tp_size
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

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_heads,  # MHA: kv_heads = num_heads
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=self.quant_config,
            prefix=f"{prefix}.query_key_value",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.dense = RowParallelLinear(
            self.hidden_inner_size,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=self.quant_config,
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
            rope_parameters=rope_parameters or None,
        )

        # Build slope tensor for linear attention decay
        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        if self.num_hidden_layers <= 1:
            self.slope_rate = slope_rate * (1 + 1e-5)
        else:
            self.slope_rate = slope_rate * (
                1 - self.layer_idx / (self.num_hidden_layers - 1) + 1e-5
            )
        self.tp_slope = self.slope_rate[
            self.tp_rank * self.tp_heads : (self.tp_rank + 1) * self.tp_heads
        ].contiguous()

        # Register for compilation
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self):
        from vllm.v1.attention.backends.bailing_linear_attn import (
            BailingLinearAttentionBackend,
        )

        return BailingLinearAttentionBackend

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
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]  # type: ignore
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
            kv_cache = self.kv_cache[0]
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
            layer_idx=self.layer_idx,
        )

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        """Handle decode (single token per sequence)."""
        decode_state_indices = getattr(attn_metadata, "state_indices_tensor_d", None)
        num_accepted_tokens = getattr(attn_metadata, "num_accepted_tokens", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc_d", None)
        if (
            decode_state_indices is not None
            and decode_state_indices.dim() > 1
            and num_accepted_tokens is not None
            and query_start_loc is not None
        ):
            return bailing_linear_attention_decode_spec(
                q,
                k,
                v,
                kv_cache,
                self.tp_slope,
                decode_state_indices,
                query_start_loc,
                num_accepted_tokens,
                q_start=0,
                q_end=attn_metadata.num_decode_tokens,
                slot_start=0,
                slot_end=attn_metadata.num_decodes,
                block_size=32,
            )
        decode_state_indices = (
            state_indices_tensor
            if decode_state_indices is None
            else decode_state_indices
        )
        hidden = linear_attention_decode(
            q,
            k,
            v,
            kv_cache,
            self.tp_slope,
            decode_state_indices,
            q_start=0,
            q_end=attn_metadata.num_decode_tokens,
            slot_start=0,
            slot_end=attn_metadata.num_decodes,
            block_size=32,
        )
        return hidden
