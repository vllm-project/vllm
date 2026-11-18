# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash-family draft's context K/V store as one launch (ROCm).

``DFlashQwen3Model.precompute_and_store_context_kv`` projects the target's
context hidden states to every draft layer's K / V with one GEMM, then runs,
for the release Kimi-K3.1 DSpark draft (5 GQA layers, 2 KV heads a TP rank,
head dim 64): the layer-major K / V copy, the grouped K RMSNorm, the position
repeat, the in-place K RoPE and one ``reshape_and_cache`` per layer: nine
launches, ~41 us a step on MI355X at c1, for a few context rows.

Here a program takes one context row of the GEMM output (token-major,
``[L, 2, nkv, hd]``) and, for each layer, normalizes K (FP32, rounded to bf16
after the weight as ``rms_norm``), rotates it (FP32 from the bf16 normed K and
the bf16 cos / sin, rounded once as ``rotary_embedding``) and writes K and V
into the layer's ``[K | V]`` cache row at the row's slot (negative slots
skipped, as ``reshape_and_cache``). Only the variance's summation order
differs from ``rms_norm``'s block reduction. Graph capturable: one launch,
no host data.

Serves bf16 caches without KV quantization (``kv_cache_dtype=auto``), the
TRITON_ATTN cache layout (``[blocks, kv heads, block size, 2 * hd]`` up to
strides; also KIMI_K3_FLYDSL_SWA_GQA's), full-head RoPE and up to ``MAX_LAYERS``
layers. ``levers/ctxkv.py`` checks those and keeps the stock path otherwise.
"""

import torch

from vllm.triton_utils import tl, triton

MAX_LAYERS = 8


@triton.jit
def _store_layer(
    kv_row,
    cache_ptr,
    slot_ptr,
    knw_ptr,
    t,
    cos,
    sin,
    sign,
    h,
    d,
    pd,
    eps,
    v_off,
    c_blk,
    c_tok,
    c_head,
    block_size,
    LAYER: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
):
    base = kv_row + LAYER * 2 * NKV * HD
    k = tl.load(base + h * HD + d).to(tl.float32)
    kp = tl.load(base + h * HD + pd).to(tl.float32)
    v = tl.load(base + NKV * HD + h * HD + d)
    r = tl.rsqrt(tl.sum(k * k, axis=1) / HD + eps)[:, None]
    w = tl.load(knw_ptr + LAYER * HD + d).to(tl.float32)
    wp = tl.load(knw_ptr + LAYER * HD + pd).to(tl.float32)
    kn = (k * r * w).to(v.dtype).to(tl.float32)
    knp = (kp * r * wp).to(v.dtype).to(tl.float32)
    out = (kn * cos + sign * knp * sin).to(v.dtype)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    ok = (slot >= 0) & (h >= 0)
    slot = tl.maximum(slot, 0)
    dst = (
        cache_ptr
        + (slot // block_size) * c_blk
        + (slot % block_size) * c_tok
        + h * c_head
        + d
    )
    tl.store(dst, out, mask=ok)
    tl.store(dst + v_off, v, mask=ok)


@triton.jit
def _ctx_kv_store_kernel(
    kv_ptr,
    kv_tok_stride,
    pos_ptr,
    knw_ptr,
    cs_ptr,
    cs_stride,
    eps,
    c0, c1, c2, c3, c4, c5, c6, c7,
    s0, s1, s2, s3, s4, s5, s6, s7,
    v_off,
    c_blk,
    c_tok,
    c_head,
    block_size,
    L: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    NEOX: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    h = tl.arange(0, NKV)[:, None]
    d = tl.arange(0, HD)[None, :]
    HALF: tl.constexpr = HD // 2
    if NEOX:
        pd = (d + HALF) % HD
        ci = d % HALF
        first = d < HALF
    else:
        pd = d ^ 1
        ci = d // 2
        first = (d % 2) == 0
    sign = tl.where(first, -1.0, 1.0)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    kv_row = kv_ptr + t * kv_tok_stride
    # The stock path casts the cos / sin cache to the activation dtype.
    act = kv_ptr.dtype.element_ty
    cos = tl.load(cs_ptr + pos * cs_stride + ci).to(act).to(tl.float32)
    sin = tl.load(cs_ptr + pos * cs_stride + HALF + ci).to(act).to(tl.float32)
    if L > 0:
        _store_layer(
            kv_row, c0, s0, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 0, NKV, HD,
        )  # fmt: skip
    if L > 1:
        _store_layer(
            kv_row, c1, s1, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 1, NKV, HD,
        )  # fmt: skip
    if L > 2:
        _store_layer(
            kv_row, c2, s2, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 2, NKV, HD,
        )  # fmt: skip
    if L > 3:
        _store_layer(
            kv_row, c3, s3, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 3, NKV, HD,
        )  # fmt: skip
    if L > 4:
        _store_layer(
            kv_row, c4, s4, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 4, NKV, HD,
        )  # fmt: skip
    if L > 5:
        _store_layer(
            kv_row, c5, s5, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 5, NKV, HD,
        )  # fmt: skip
    if L > 6:
        _store_layer(
            kv_row, c6, s6, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 6, NKV, HD,
        )  # fmt: skip
    if L > 7:
        _store_layer(
            kv_row, c7, s7, knw_ptr, t, cos, sin, sign, h, d, pd, eps, v_off,
            c_blk, c_tok, c_head, block_size, 7, NKV, HD,
        )  # fmt: skip


def cache_views(kv_cache: torch.Tensor, head_dim: int):
    """TRITON_ATTN's K / V views of a ``[blocks, kv heads, block size, 2 * hd]``
    cache: ``[blocks, block size, kv heads, hd]`` each."""
    return kv_cache.transpose(1, 2).split(head_dim, dim=-1)


def layout_why_not(
    kv_caches: list[torch.Tensor], num_kv_heads: int, head_dim: int
) -> str | None:
    """Why the caches are outside the launch (None: inside)."""
    if not 0 < len(kv_caches) <= MAX_LAYERS:
        return f"{len(kv_caches)} layers"
    ref = None
    for c in kv_caches:
        if not isinstance(c, torch.Tensor) or c.dim() != 4:
            return "KV cache is not one 4-D tensor a layer"
        if c.dtype not in (torch.bfloat16, torch.float16):
            return f"KV cache dtype {c.dtype}"
        if c.shape[1] != num_kv_heads or c.shape[3] != 2 * head_dim:
            return f"KV cache shape {tuple(c.shape)}"
        k, v = cache_views(c, head_dim)
        if k.stride(3) != 1:
            return "KV cache head dim is not contiguous"
        key = (
            tuple(c.shape),
            k.stride(),
            v.data_ptr() - k.data_ptr(),
            c.dtype,
        )
        if ref is None:
            ref = key
        elif key != ref:
            return "KV caches differ across layers"
    return None


def store_context_kv(
    all_kv: torch.Tensor,
    positions: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    is_neox: bool,
    kv_caches: list[torch.Tensor],
    slot_mappings: list[torch.Tensor],
) -> None:
    """``all_kv``: ``[num_ctx, L, 2, nkv, hd]`` with contiguous rows (token-major
    GEMM output); ``k_norm_weights``: ``[L, hd]``; ``cos_sin_cache``:
    ``[max_pos, hd]`` (cos | sin halves); one cache and slot mapping a layer.
    The caller checks ``layout_why_not``."""
    num_ctx, L, _, nkv, hd = all_kv.shape
    if num_ctx == 0:
        return
    k, v = cache_views(kv_caches[0], hd)
    caches = list(kv_caches) + [kv_caches[0]] * (MAX_LAYERS - L)
    slots = list(slot_mappings) + [slot_mappings[0]] * (MAX_LAYERS - L)
    _ctx_kv_store_kernel[(num_ctx,)](
        all_kv,
        all_kv.stride(0),
        positions,
        k_norm_weights,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        eps,
        *caches,
        *slots,
        (v.data_ptr() - k.data_ptr()) // k.element_size(),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.shape[1],
        L=L,
        NKV=nkv,
        HD=hd,
        NEOX=is_neox,
        num_warps=1,
    )
