# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Shared vision attention stack for Dots dense / MoE ViT encoders.

Exports every attention backend (eager, eager_v2, sdpa, flash_attention_2,
flash_attention_3), RoPE helpers, and block wiring utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

_flash_attn_available = is_fa_version_supported(2)
_flash_attn3_available = is_fa_version_supported(3)
_flash_attn3_varlen_func = flash_attn_varlen_func
_flash_attn3_use_positional_args = False


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    """2D vision RoPE frequency table.

    When ``cache_seq_len`` is set, precomputes and reuses frequencies up to that
    length (MoE default). Otherwise frequencies are computed on each forward
    (dense default).
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        cache_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_seq_len = cache_seq_len
        self.freqs_cache: torch.Tensor
        if cache_seq_len is not None:
            self.register_buffer(
                "freqs_cache",
                self._compute_freqs(cache_seq_len),
                persistent=False,
            )

    def _compute_freqs(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)

    def forward(self, seqlen: int) -> torch.Tensor:
        if self._cache_seq_len is None:
            return self._compute_freqs(seqlen)
        if seqlen > self.freqs_cache.shape[0]:
            self.freqs_cache = self._compute_freqs(seqlen)
        return self.freqs_cache[:seqlen]


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Q/K norm inside attention; matches Dots ViT RMSNorm (fp32 reduce, then cast back)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


@dataclass(frozen=True)
class VisionAttentionParams:
    dim: int
    num_heads: int
    bias: bool = True
    is_causal: bool = False
    rms_norm_eps: float = 1e-5
    use_qk_norm: bool = False


def vision_attention_params_from_config(config: Any) -> VisionAttentionParams:
    return VisionAttentionParams(
        dim=config.embed_dim,
        num_heads=config.num_attention_heads,
        bias=getattr(config, "use_bias", True),
        is_causal=config.is_causal,
        rms_norm_eps=config.rms_norm_eps,
        use_qk_norm=getattr(config, "use_qk_norm", False),
    )


class _VisionAttentionBase(nn.Module):
    """QKV projection, optional Q/K norm, and output projection."""

    def __init__(self, params: VisionAttentionParams) -> None:
        super().__init__()
        self.num_heads = params.num_heads
        self.head_dim = params.dim // params.num_heads
        self.is_causal = params.is_causal
        self.use_qk_norm = params.use_qk_norm
        self.qkv = nn.Linear(params.dim, params.dim * 3, bias=params.bias)
        self.proj = nn.Linear(params.dim, params.dim, bias=params.bias)
        if self.use_qk_norm:
            self.q_norm = _RMSNorm(self.head_dim, eps=params.rms_norm_eps)
            self.k_norm = _RMSNorm(self.head_dim, eps=params.rms_norm_eps)

    def _qkv_with_rope(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        return q, k, v


class VisionAttention(_VisionAttentionBase):
    """Eager attention with a dense block-diagonal mask (cu_seqlens boundaries)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self._qkv_with_rope(hidden_states, rotary_pos_emb)

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionAttentionV2(_VisionAttentionBase):
    """Eager attention per varlen segment (lower peak memory than :class:`VisionAttention`)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self._qkv_with_rope(hidden_states, rotary_pos_emb)

        if seqlens is None:
            seqlens = torch.diff(cu_seqlens).tolist()

        outputs = []
        for q_i, k_i, v_i in zip(
            torch.split(q, seqlens, 0),
            torch.split(k, seqlens, 0),
            torch.split(v, seqlens, 0),
        ):
            q_i = q_i.transpose(0, 1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)
            out = torch.matmul(q_i, k_i.transpose(1, 2)) / math.sqrt(self.head_dim)
            out = F.softmax(out, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(out, v_i).transpose(0, 1)
            outputs.append(out)

        attn_output = torch.concat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionFlashAttention2(_VisionAttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self._qkv_with_rope(hidden_states, rotary_pos_emb)
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=self.is_causal,
            fa_version=2,
        ).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionFlashAttention3(_VisionAttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self._qkv_with_rope(hidden_states, rotary_pos_emb)
        if _flash_attn3_use_positional_args:
            attn_output = _flash_attn3_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=self.is_causal,
            )
        else:
            result = _flash_attn3_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=self.is_causal,
                fa_version=3,
            )
            attn_output = result[0] if isinstance(result, tuple) else result
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionSdpaAttention(_VisionAttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_pos_emb: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self._qkv_with_rope(hidden_states, rotary_pos_emb)

        attention_mask = torch.zeros(
            [1, seq_length, seq_length], device=q.device, dtype=torch.bool
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = True

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if attention_mask.stride(-1) != 1:
            attention_mask = torch.empty_like(
                attention_mask, memory_format=torch.contiguous_format
            ).copy_(attention_mask)

        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attention_mask, dropout_p=0.0
            )

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


DOTS_VISION_ATTENTION_CLASSES: dict[str, type[nn.Module]] = {
    "eager": VisionAttention,
    "eager_v2": VisionAttentionV2,
    "flash_attention_2": VisionFlashAttention2,
    "flash_attention_3": VisionFlashAttention3,
    "sdpa": VisionSdpaAttention,
}


def resolve_attn_implementation(
    attn_implementation: str,
    *,
    eager_fallback: str = "eager",
) -> str:
    """Apply FA3 → FA2 → eager fallback when backends are missing.

    ``eager_fallback`` selects which eager backend is used when flash-attn is
    unavailable. MoE historically used per-segment eager (``eager_v2``); dense
    uses full-mask ``eager``.
    """
    if attn_implementation == "flash_attention_3" and not _flash_attn3_available:
        print(
            "flash attention 3 not available! fallback to flash attention 2 implementation"
        )
        attn_implementation = "flash_attention_2"
    if attn_implementation == "flash_attention_2" and not _flash_attn_available:
        print("flash attention 2 not available! fallback to eager implementation")
        attn_implementation = eager_fallback
    return attn_implementation


def build_vision_attention(
    attn_implementation: str,
    config: Any,
    *,
    dim: int | None = None,
    num_heads: int | None = None,
    bias: bool | None = None,
    eager_fallback: str = "eager",
) -> nn.Module:
    """Instantiate a vision attention module from ``attn_implementation`` and config."""
    attn_implementation = resolve_attn_implementation(
        attn_implementation, eager_fallback=eager_fallback
    )
    if attn_implementation not in DOTS_VISION_ATTENTION_CLASSES:
        raise ValueError(
            f"Unknown attn_implementation {attn_implementation!r}; "
            f"expected one of {sorted(DOTS_VISION_ATTENTION_CLASSES)}"
        )
    if dim is None:
        params = vision_attention_params_from_config(config)
    else:
        params = VisionAttentionParams(
            dim=dim,
            num_heads=num_heads
            if num_heads is not None
            else config.num_attention_heads,
            bias=bias if bias is not None else getattr(config, "use_bias", True),
            is_causal=config.is_causal,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=getattr(config, "use_qk_norm", False),
        )
    return DOTS_VISION_ATTENTION_CLASSES[attn_implementation](params)


def attn_uses_seqlens(attn_implementation: str) -> bool:
    return attn_implementation == "eager_v2"


def prepare_seqlens_for_attention(
    attn_implementation: str,
    cu_seqlens: torch.Tensor,
    *,
    eager_fallback: str = "eager",
) -> list[int] | None:
    """Return per-segment lengths when the resolved backend needs ``seqlens``."""
    resolved = resolve_attn_implementation(
        attn_implementation, eager_fallback=eager_fallback
    )
    if attn_uses_seqlens(resolved):
        return torch.diff(cu_seqlens).tolist()
    return None


def apply_vision_attention_residual(
    attn: nn.Module,
    norm: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: torch.Tensor,
    *,
    seqlens: list[int] | None = None,
    uses_seqlens: bool = False,
) -> torch.Tensor:
    """Pre-norm residual attention used by dense / MoE vision blocks."""
    attn_kwargs: dict[str, Any] = {"rotary_pos_emb": rotary_pos_emb}
    if uses_seqlens:
        attn_kwargs["seqlens"] = seqlens
    return hidden_states + attn(
        norm(hidden_states),
        cu_seqlens,
        max_seqlen,
        **attn_kwargs,
    )


__all__ = [
    "DOTS_VISION_ATTENTION_CLASSES",
    "VisionAttention",
    "VisionAttentionParams",
    "VisionAttentionV2",
    "VisionFlashAttention2",
    "VisionFlashAttention3",
    "VisionRotaryEmbedding",
    "VisionSdpaAttention",
    "apply_rotary_pos_emb_vision",
    "apply_vision_attention_residual",
    "attn_uses_seqlens",
    "build_vision_attention",
    "prepare_seqlens_for_attention",
    "resolve_attn_implementation",
    "rotate_half",
    "vision_attention_params_from_config",
]
