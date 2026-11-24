# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared per-forward M-RoPE rows for HY4 fused rotary kernels."""

import math

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope
from vllm.triton_utils import tl, triton


@eager_break_during_capture
def _bind_token_rope(
    rotary_emb: nn.Module, rows: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Keep this graph's rows alive and restore them before eager indexers."""
    rotary_emb._hy4_token_rope = rows


def set_token_rope(
    rotary_emb: nn.Module, positions: torch.Tensor, x: torch.Tensor
) -> None:
    """Build this forward's per-token M-RoPE cos/sin rows once, before the
    layers run; `token_rope` reads them. `x` sets the cache dtype."""
    if positions.ndim == 1:
        rotary_emb._hy4_token_rope = None
        return
    sel = getattr(rotary_emb, "_hy4_mrope_sel", None)
    if sel is None:
        # T/H/W position row of each cos/sin channel, via vLLM's M-RoPE layout.
        sec = rotary_emb.mrope_section
        ids = torch.arange(3, device=positions.device)[:, None].expand(3, sum(sec))
        if rotary_emb.mrope_interleaved:
            sel = apply_interleaved_rope(ids, sec)
        else:
            sel = torch.cat([m[i] for i, m in enumerate(ids.split(sec, dim=-1))])
        sel = rotary_emb._hy4_mrope_sel = sel.repeat(2)
    num_tokens = positions.shape[1]
    cache = rotary_emb._match_cos_sin_cache_dtype(x)
    rows = cache[positions].gather(0, sel.expand(1, num_tokens, -1))[0]
    arange = torch.arange(num_tokens, device=positions.device)
    _bind_token_rope(rotary_emb, (arange, rows))


def token_rope(
    rotary_emb: nn.Module, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(positions, cos_sin_cache)` for the 1-D fused RoPE kernels.

    [3, T] M-RoPE positions become `arange(T)` into the per-token cos/sin
    rows built by `set_token_rope` for this forward.
    """
    if positions.ndim == 1:
        return positions, cos_sin_cache
    rows = getattr(rotary_emb, "_hy4_token_rope", None)
    assert rows is not None and rows[0].shape[0] == positions.shape[1], (
        "set_token_rope must run before the layers"
    )
    return rows


def apply_mrope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate strided Q/K in place using this forward's shared token rows."""
    rope_pos, cos_sin = token_rope(
        rotary_emb, positions, rotary_emb._match_cos_sin_cache_dtype(query)
    )
    torch.ops._C.rotary_embedding(
        rope_pos,
        query,
        key,
        rotary_emb.head_size,
        cos_sin,
        rotary_emb.is_neox_style,
    )
    return query, key


@triton.jit
def _rope_into(
    POS,
    SRC,
    SRC_S0,
    SRC_S1,
    CS,
    CS_S0,
    DST,
    DST_S0,
    DST_S1,
    HALF: tl.constexpr,
    HB: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HB + tl.arange(0, HB)
    pairs = tl.arange(0, HALF)
    pos = tl.load(POS + t)
    cos = tl.load(CS + pos * CS_S0 + pairs).to(tl.float32)[None, :]
    sin = tl.load(CS + pos * CS_S0 + HALF + pairs).to(tl.float32)[None, :]
    src = SRC + t * SRC_S0 + heads[:, None] * SRC_S1 + 2 * pairs[None, :]
    x1 = tl.load(src).to(tl.float32)
    x2 = tl.load(src + 1).to(tl.float32)
    dst = DST + t * DST_S0 + heads[:, None] * DST_S1 + 2 * pairs[None, :]
    tl.store(dst, (x1 * cos - x2 * sin).to(DST.dtype.element_ty))
    tl.store(dst + 1, (x2 * cos + x1 * sin).to(DST.dtype.element_ty))


def rope_into(
    positions: torch.Tensor,
    src: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    """Write interleaved-RoPE'd `src` (tokens, heads, rot_dim) into `dst`.

    `cos_sin_cache` is `(positions, rot_dim)`: cos then sin halves.
    """
    num_tokens, num_heads, rot_dim = src.shape
    assert src.stride(-1) == 1 and dst.stride(-1) == 1 and dst.shape == src.shape
    block_heads = math.gcd(num_heads, 16)
    if num_tokens:
        _rope_into[(num_tokens, num_heads // block_heads)](
            positions,
            src,
            src.stride(0),
            src.stride(1),
            cos_sin_cache,
            cos_sin_cache.stride(0),
            dst,
            dst.stride(0),
            dst.stride(1),
            rot_dim // 2,
            block_heads,
        )


def fused_mrope_norm(rotary_emb, positions, q_c, kv_lora, q_norm, kv_norm):
    """Normalize Q/KV and rotate K using shared per-token rows."""
    rope_pos, cache = token_rope(
        rotary_emb, positions, rotary_emb._match_cos_sin_cache_dtype(kv_lora)
    )
    kv_c, k_pe = kv_lora.split([512, 64], dim=-1)
    kv_out = torch.empty_like(kv_c)
    k_out = torch.empty_like(k_pe)
    unused_topk = torch.empty((1, 1), device=q_c.device, dtype=torch.int32)
    q_out = fused_norm_rope(
        rope_pos,
        q_c,
        q_norm.weight,
        q_norm.variance_epsilon,
        kv_c,
        kv_norm.weight,
        kv_norm.variance_epsilon,
        k_pe,
        cache,
        None,
        None,
        None,
        0.0,
        None,
        unused_topk,
        has_indexer=False,
        kv_c_out=kv_out,
        k_pe_out=k_out,
    )
    return q_out, kv_out, k_out.unsqueeze(1), rope_pos, cache
