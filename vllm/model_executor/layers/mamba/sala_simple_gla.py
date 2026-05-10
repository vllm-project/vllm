# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SALALightningAttention — Simple GLA layer for MiniCPM-SALA.

Structural reference: MiniMaxText01LinearAttention (linear_attn.py).
Kernel reference:     fla.ops.simple_gla (chunk_simple_gla / fused_recurrent_simple_gla).
HF reference:         openbmb/MiniCPM-SALA LightningAttention.forward().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01RMSNormTP,
    MiniMaxText01LinearAttention,
    clear_linear_attention_cache_for_new_sequences,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.minicpm_sala import MiniCPMSALAConfig

# Minimum fla for the Simple GLA API used here (``head_first`` removed; inputs
# are always ``[B, T, H, D]``). Revisit when bumping if upstream changes kwargs.
_MIN_FLASH_LINEAR_ATTENTION_VERSION = "0.5.0"


def _lazy_import_fla():
    """Lazy-import flash-linear-attention (GPU + Triton required).

    Returns (chunk_simple_gla, fused_recurrent_simple_gla).
    Raises ImportError with a clear message if the package is unavailable or too
    old for the kernel contract in this module.
    """
    from packaging.version import Version, parse as parse_version

    suffix = (
        f"`flash-linear-attention>={_MIN_FLASH_LINEAR_ATTENTION_VERSION}` "
        "(Simple GLA kernel contract: tensors shaped `[B, T, H, D]`).\n"
        "Install or upgrade:\n"
        f"  pip install -U 'flash-linear-attention>="
        f"{_MIN_FLASH_LINEAR_ATTENTION_VERSION}'\n"
        "Then verify:\n"
        "  python -c 'from fla.ops.simple_gla import chunk_simple_gla'"
    )
    base_msg = "MiniCPM-SALA Lightning layers require " + suffix

    try:
        import fla
    except ImportError as exc:
        raise ImportError(base_msg) from exc

    ver_str = getattr(fla, "__version__", "") or "0"
    if parse_version(ver_str) < Version(_MIN_FLASH_LINEAR_ATTENTION_VERSION):
        raise ImportError(f"{base_msg}\n(found flash-linear-attention {ver_str!r}).")

    try:
        from fla.ops.simple_gla import (
            chunk_simple_gla,
            fused_recurrent_simple_gla,
        )
    except ImportError as exc:
        raise ImportError(base_msg) from exc

    return chunk_simple_gla, fused_recurrent_simple_gla


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA key/value heads: (tokens, kv_heads, d) -> (tokens, q_heads, d)."""
    if n_rep == 1:
        return x
    tokens, kv_h, d = x.shape
    return (
        x[:, :, None, :].expand(tokens, kv_h, n_rep, d).reshape(tokens, kv_h * n_rep, d)
    )


class SALALightningAttention(nn.Module, MambaBase):
    """Simple GLA recurrent attention layer for MiniCPM-SALA Lightning blocks.

    Reuses the existing ``"linear_attention"`` backend (LinearAttentionMetadata)
    so no new backend registry entry is required.

    Key contracts
    -------------
    * ``mamba_type = "linear_attention"`` — reuses LinearAttentionBackend.
    * One state tensor per layer: shape ``(num_heads // tp, head_dim, head_dim)``,
      dtype ``float32``.
    * New-sequence detection: ``context_len == 0`` from
      ``clear_linear_attention_cache_for_new_sequences()``.
    * Slopes: built from the same ``_build_slope_tensor`` as MiniMax, then
      negated (``* -1``) and TP-sharded before use.
    * GQA repeat: k/v are repeated from ``nkv`` to ``nh`` heads *after* TP
      sharding and qk-norm; the state is indexed on the full ``nh // tp``
      head count.
    * Fused residual/RMSNorm path is **not** used — MiniCPM-SALA uses a
      layer-local scaled residual in the decoder, handled by the parent layer.
    """

    # -------------------------------------------------------------------------
    # MambaBase interface
    # -------------------------------------------------------------------------

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None and self.cache_config is not None
        return MambaStateDtypeCalculator.simple_gla_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int, int], ...]:
        return MambaStateShapeCalculator.simple_gla_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(
        self,
        config: MiniCPMSALAConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.lightning_nh  # total q heads
        self.num_kv_heads: int = config.lightning_nkv  # total k/v heads
        self.head_dim: int = config.lightning_head_dim
        self.num_kv_groups: int = self.num_heads // self.num_kv_heads

        # scale applied inside the fla kernel (1/sqrt(d) or 1/d)
        if config.lightning_scale == "1/sqrt(d)":
            self.scale: float = self.head_dim**-0.5
        elif config.lightning_scale == "1/d":
            self.scale = self.head_dim**-1.0
        else:
            self.scale = 1.0

        self.use_output_gate: bool = config.use_output_gate
        self.use_output_norm: bool = config.use_output_norm
        self.use_qk_norm: bool = config.qk_norm
        self.use_rope: bool = config.lightning_use_rope

        self.tp_size: int = get_tensor_model_parallel_world_size()
        self.tp_rank: int = get_tensor_model_parallel_rank()

        assert (
            self.num_heads % self.tp_size == 0
        ), f"lightning_nh ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        assert (
            self.num_kv_heads % self.tp_size == 0
        ), f"lightning_nkv ({self.num_kv_heads}) must be divisible by tp_size ({self.tp_size})"

        self.local_heads: int = self.num_heads // self.tp_size
        self.local_kv_heads: int = self.num_kv_heads // self.tp_size

        # ---- Projections (match HF checkpoint key names) ----
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # ---- Optional: QK norm (per head_dim, weight replicated across TP) ----
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ---- Optional: output norm ----
        if self.use_output_norm:
            # HF defines o_norm over the full (num_heads * head_dim) vector.
            # At TP>1 the activation is sharded, so use the MiniMax TP-aware
            # RMSNorm: it shards the checkpoint weight and all-reduces variance.
            self.o_norm = MiniMaxText01RMSNormTP(
                self.num_heads * self.head_dim,
                eps=config.rms_norm_eps,
            )

        # ---- Optional: output gate (z_proj) ----
        if self.use_output_gate:
            self.z_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.z_proj",
            )

        # ---- RoPE ----
        if self.use_rope:
            rope_params = None
            if config.rope_scaling:
                rope_params = {"rope_theta": config.rope_theta, **config.rope_scaling}
            else:
                rope_params = {"rope_theta": config.rope_theta}
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_params,
            )

        # ---- Decay slopes (same ALiBi-style as MiniMax, then negated) ----
        # Build total-head slopes, then shard by TP rank.
        # Negation matches HF: s = _build_slope_tensor(nh) * (-1.0)
        all_slopes = MiniMaxText01LinearAttention._build_slope_tensor(self.num_heads)
        all_slopes = all_slopes * -1.0  # shape (nh, 1, 1), values negative
        # Shard: each TP rank owns [rank*local_heads : (rank+1)*local_heads]
        tp_slope = all_slopes[
            self.tp_rank * self.local_heads : (self.tp_rank + 1) * self.local_heads
        ].contiguous()  # (local_heads, 1, 1)
        self.register_buffer("tp_slope", tp_slope, persistent=False)

        # ---- Register in static_forward_context (no-compile pattern) ----
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # ---- Placeholder for block allocator (set by MambaBase.get_kv_cache_spec) ----
        self.kv_cache: tuple[torch.Tensor, ...] = (torch.tensor([]),)

    # -------------------------------------------------------------------------
    # Kernel helpers
    # -------------------------------------------------------------------------

    def _prefill_batch(
        self,
        q: torch.Tensor,  # (tokens, local_heads, head_dim)
        k: torch.Tensor,  # (tokens, local_kv_heads, head_dim)
        v: torch.Tensor,  # (tokens, local_kv_heads, head_dim)
        kv_cache: torch.Tensor,  # (num_blocks, local_heads, head_dim, head_dim)
        state_indices_tensor: torch.Tensor,
        attn_metadata: LinearAttentionMetadata,
    ) -> torch.Tensor:  # (prefill_tokens, local_heads * head_dim)
        chunk_simple_gla, _ = _lazy_import_fla()

        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        if num_prefills <= 0:
            return torch.empty(
                (0, self.local_heads * self.head_dim),
                device=q.device,
                dtype=q.dtype,
            )

        offset = attn_metadata.num_decode_tokens
        start = attn_metadata.query_start_loc[offset].item()
        end = attn_metadata.query_start_loc[offset + num_prefills].item()
        prefill_state_indices = state_indices_tensor[offset : offset + num_prefills]

        # GQA repeat k/v to match q head count
        q_prefill = q[start:end]
        k_prefill = _repeat_kv(k[start:end], self.num_kv_groups)
        v_prefill = _repeat_kv(v[start:end], self.num_kv_groups)

        # Varlen prefill: packed tokens in batch dimension 1 plus cu_seqlens.
        # MiniCPM-SALA HF uses the same FLA contract when attention_mask is set.
        q_4d = q_prefill.unsqueeze(0).to(torch.float32)  # (1, total_T, h, d)
        k_4d = k_prefill.unsqueeze(0).to(torch.float32)
        v_4d = v_prefill.unsqueeze(0).to(torch.float32)

        init_state = kv_cache[prefill_state_indices].to(torch.float32)
        cu_seqlens = (
            attn_metadata.query_start_loc[offset : offset + num_prefills + 1] - start
        ).to(torch.int32)

        out, final_state = chunk_simple_gla(
            q=q_4d,
            k=k_4d,
            v=v_4d,
            g_gamma=self.tp_slope,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        kv_cache[prefill_state_indices] = final_state.to(kv_cache.dtype)

        # out: (1, total_T, h, d) -> (total_T, h*d)
        return out.squeeze(0).reshape(
            q_prefill.shape[0], self.local_heads * self.head_dim
        )

    def _decode_batch(
        self,
        q: torch.Tensor,  # (tokens, local_heads, head_dim)
        k: torch.Tensor,  # (tokens, local_kv_heads, head_dim)
        v: torch.Tensor,  # (tokens, local_kv_heads, head_dim)
        kv_cache: torch.Tensor,
        state_indices_tensor: torch.Tensor,
        attn_metadata: LinearAttentionMetadata,
    ) -> torch.Tensor:  # (decode_tokens, local_heads * head_dim)
        _, fused_recurrent_simple_gla = _lazy_import_fla()

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        if num_decode_tokens <= 0:
            return torch.empty(
                (0, self.local_heads * self.head_dim),
                device=q.device,
                dtype=q.dtype,
            )
        # LinearAttentionMetadata represents normal decode as one token per
        # sequence. Keep this explicit so speculative/packed variants fail
        # loudly instead of silently corrupting recurrent state.
        assert num_decode_tokens == num_decodes, (
            "Simple GLA batched decode expects one token per decode sequence: "
            f"{num_decode_tokens=} vs {num_decodes=}"
        )

        decode_state_indices = state_indices_tensor[:num_decodes]
        q_decode = q[:num_decode_tokens]
        k_decode = _repeat_kv(k[:num_decode_tokens], self.num_kv_groups)
        v_decode = _repeat_kv(v[:num_decode_tokens], self.num_kv_groups)

        # Batch decode: (batch=num_decodes, seqlen=1, heads, head_dim)
        q_4d = q_decode.unsqueeze(1).to(torch.float32)
        k_4d = k_decode.unsqueeze(1).to(torch.float32)
        v_4d = v_decode.unsqueeze(1).to(torch.float32)

        init_state = kv_cache[decode_state_indices].to(torch.float32)

        out, final_state = fused_recurrent_simple_gla(
            q=q_4d,
            k=k_4d,
            v=v_4d,
            g_gamma=self.tp_slope,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
        )
        kv_cache[decode_state_indices] = final_state.to(kv_cache.dtype)

        return out.squeeze(1).reshape(
            num_decode_tokens, self.local_heads * self.head_dim
        )

    # -------------------------------------------------------------------------
    # Main forward (no-compile path called by custom op)
    # -------------------------------------------------------------------------

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: LinearAttentionMetadata | None = None

        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, LinearAttentionMetadata)
            num_actual_tokens = (
                attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
            )
        else:
            num_actual_tokens = hidden_states.shape[0]

        hs = hidden_states[:num_actual_tokens]

        # ---- Projections ----
        q, _ = self.q_proj(hs)  # (T, local_heads * head_dim)
        k, _ = self.k_proj(hs)  # (T, local_kv_heads * head_dim)
        v, _ = self.v_proj(hs)

        # Reshape to (T, heads, head_dim)
        q = q.view(num_actual_tokens, self.local_heads, self.head_dim)
        k = k.view(num_actual_tokens, self.local_kv_heads, self.head_dim)
        v = v.view(num_actual_tokens, self.local_kv_heads, self.head_dim)

        # ---- QK norm (per head, replicated weight) ----
        if self.use_qk_norm:
            # Apply RMSNorm on the head_dim dimension for each token and head
            orig_shape = q.shape
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(orig_shape)
            k_shape = k.shape
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k_shape)

        # ---- RoPE ----
        if self.use_rope:
            # positions: (T,); rotary_emb expects (T, local_heads, head_dim)
            # We apply to q (local_heads) and k (local_kv_heads) separately.
            # vLLM rotary_emb.forward takes (positions, query, key) as flat tensors.
            q_flat = q.reshape(num_actual_tokens, self.local_heads * self.head_dim)
            k_flat = k.reshape(num_actual_tokens, self.local_kv_heads * self.head_dim)
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(num_actual_tokens, self.local_heads, self.head_dim)
            k = k_flat.view(num_actual_tokens, self.local_kv_heads, self.head_dim)

        # ---- z_proj for output gate (project before attention compute) ----
        if self.use_output_gate:
            z, _ = self.z_proj(hs)  # (T, local_heads * head_dim)

        # ---- State routing ----
        if attn_metadata is None:
            # Profile run (no metadata): return zero output
            hidden = torch.zeros(
                num_actual_tokens,
                self.local_heads * self.head_dim,
                device=q.device,
                dtype=q.dtype,
            )
        else:
            kv_cache = self.kv_cache[0]
            state_indices_tensor = attn_metadata.state_indices_tensor
            clear_linear_attention_cache_for_new_sequences(
                kv_cache, state_indices_tensor, attn_metadata
            )

            num_decode = attn_metadata.num_decode_tokens
            num_prefills = getattr(attn_metadata, "num_prefills", 0)
            decode_only = num_prefills == 0

            hidden_parts: list[torch.Tensor] = []

            # ---- Decode path ----
            if num_decode > 0:
                hidden_parts.append(
                    self._decode_batch(
                        q, k, v, kv_cache, state_indices_tensor, attn_metadata
                    )
                )

            # ---- Prefill path (and mixed) ----
            if not decode_only:
                hidden_parts.append(
                    self._prefill_batch(
                        q, k, v, kv_cache, state_indices_tensor, attn_metadata
                    )
                )

            if not hidden_parts:
                hidden = torch.empty(
                    (0, self.local_heads * self.head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
            else:
                hidden = torch.cat(hidden_parts, dim=0).to(hidden_states.dtype)

        # ---- Optional output norm ----
        if self.use_output_norm:
            hidden = self.o_norm(hidden)

        # ---- Optional output gate ----
        if self.use_output_gate:
            hidden = F.sigmoid(z) * hidden

        # ---- Output projection ----
        output[:num_actual_tokens], _ = self.o_proj(hidden)

    # -------------------------------------------------------------------------
    # forward() — dispatches through the no-compile custom op
    # -------------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # Reuse the existing registered custom op so graph compilation skips
        # this layer (fla Triton kernels cannot be captured by torch.compile).
        torch.ops.vllm.linear_attention(
            hidden_states,
            output,
            positions,
            self.prefix,
        )
