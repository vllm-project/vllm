# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 and FP8 per-token-head KV cache quantization backends.

The attention read path for these modes lives in the core kernel
(:mod:`vllm.v1.attention.ops.triton_unified_attention`) — the only thing
that differs from the unquantized path is loading per-(token, head)
scales and fusing them into S/P, both gated by a constexpr branch.

This module owns the write side: a small reshape kernel that does
per-(token, head) absmax + symmetric quantization and writes both the
quantized cache values and the float32 scale buffer.  INT8 and FP8 share
the same kernel and only differ in (QUANT_MAX, QUANT_MIN) and the cache
storage dtype (which Triton infers from the cache pointer).

Symmetric quantization::

    scale = absmax / QUANT_MAX
    q     = clamp(round(x / scale), QUANT_MIN, QUANT_MAX)
    x_hat = q * scale
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVBackend
from vllm.v1.kv_cache_interface import KVQuantMode

# ---------------------------------------------------------------------------
# Per-mode quantization range
# ---------------------------------------------------------------------------
_INT8_QUANT_MAX = 127.0
_INT8_QUANT_MIN = -128.0
_FP8_QUANT_MIN, _FP8_QUANT_MAX = get_fp8_min_max()


# ---------------------------------------------------------------------------
# Reshape kernel: per-(token, head) absmax → scale, store quantized + scale
# ---------------------------------------------------------------------------
@triton.jit
def _reshape_cache_per_token_head_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    QUANT_MIN: tl.constexpr,
):
    """Per-(token, head) dynamic quantization for INT8 / FP8.

    One scale = absmax / QUANT_MAX per (token, head).  Cache storage
    dtype (int8 / fp8_e4m3 / fp8_e4m3fnuz / fp8_e5m2) is inferred from
    the cache pointer.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    dim_offs = tl.arange(0, HEAD_SIZE_PADDED)

    # ---- Key ---------------------------------------------------------------
    k_mask = dim_offs < head_size
    k_h = tl.load(
        key_ptr + tok * stride_key_tok + head * stride_key_head + dim_offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    k_scale = tl.maximum(tl.max(tl.abs(k_h)) / QUANT_MAX, 1e-6)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    k_q = tl.clamp(k_h * (1.0 / k_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + dim_offs,
        k_q,
        mask=k_mask,
    )

    # ---- Value -------------------------------------------------------------
    v_mask = dim_offs < head_size_v
    v_h = tl.load(
        value_ptr + tok * stride_val_tok + head * stride_val_head + dim_offs,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    v_scale = tl.maximum(tl.max(tl.abs(v_h)) / QUANT_MAX, 1e-6)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )

    v_q = tl.clamp(v_h * (1.0 / v_scale), QUANT_MIN, QUANT_MAX)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + dim_offs,
        v_q,
        mask=v_mask,
    )


def _run_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_max: float,
    quant_min: float,
) -> None:
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    block_size = key_cache.shape[1]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))

    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, head_size_padded // 32))

    _reshape_cache_per_token_head_kernel[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=block_size,
        head_size=head_size,
        head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded,
        QUANT_MAX=quant_max,
        QUANT_MIN=quant_min,
        num_warps=num_warps,
    )


class _PerTokenHeadBackend(QuantKVBackend):
    """Common implementation for INT8 / FP8 per-token-head modes.

    Subclasses set ``mode``, ``_quant_max`` and ``_quant_min``.  The
    attention read path lives in the core kernel; this backend only
    owns the write path and the scale-cache allocation.
    """

    packing_factor = 1
    needs_scale_caches = True

    _quant_max: float
    _quant_min: float

    def allocate_scale_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (num_blocks, block_size, num_kv_heads)
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
        )

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{self.mode.name} requires k_scale_cache / v_scale_cache"
        )
        _run_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            quant_max=self._quant_max,
            quant_min=self._quant_min,
        )


class Int8PerTokenHeadBackend(_PerTokenHeadBackend):
    """KV cache backend for ``KVQuantMode.INT8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT8_PER_TOKEN_HEAD
    _quant_max = _INT8_QUANT_MAX
    _quant_min = _INT8_QUANT_MIN


class Fp8PerTokenHeadBackend(_PerTokenHeadBackend):
    """KV cache backend for ``KVQuantMode.FP8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.FP8_PER_TOKEN_HEAD
    _quant_max = _FP8_QUANT_MAX
    _quant_min = _FP8_QUANT_MIN


register(Int8PerTokenHeadBackend())
register(Fp8PerTokenHeadBackend())
