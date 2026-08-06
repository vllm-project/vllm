# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling short-convolution kernels backed by a paged sliding-window state cache.

Each layer's 4 conv streams (K, V, attn-output, mlp-output) share one paged KV
cache ``[num_blocks, H, N, D]`` (head-major; see ``sconv_swa_attn.py``). A stream
occupies the contiguous D-sub-range ``[off_s, off_s + ws)`` across all ``H``
heads, so its flat per-token width is ``H * ws`` and that is the conv channel
dim. The cache stores the conv *input* at every absolute position.

``fused_sconv`` is the single-launch path used by the model: per token it writes
the current input to its slot and convolves the ``W`` taps ending at its
absolute position. A tap landing inside the current forward is read from the
immutable input ``x`` (row ``src - pos + pid_t``); only pre-forward taps are read
from the paged cache (window position ``src`` -> physical block via
``block_table[req, src // N]``). The just-written slot is never read back this
step, so there is no write/read hazard within or across programs -- which is
why this needs no decode-vs-prefill split and is valid for prefill / decode /
spec alike.

All kernels address the cache purely by ``(slot, absolute_position)`` and
allocate nothing inside the captured region; their grids depend only on the
token count (``fused_sconv`` on a fixed token/channel tiling), so the same
path replays correctly under eager, breakable PIECEWISE, and FULL cudagraphs
without any data-dependent shape or branch.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_sconv_kernel(
    x_ptr,  # [T, H*WS] head-major current-token inputs (also residual)
    cache_ptr,  # [num_blocks, H, N, D] paged (page-strided view)
    weight_ptr,  # [H*WS, W]
    out_ptr,  # [T, H*WS]
    pos_ptr,  # [T] int64 absolute position per token
    seq_idx_ptr,  # [T] int32 token -> batch request
    slot_ptr,  # [T] int64 flat slot (block*N + blk_off); < 0 => PAD
    block_table_ptr,  # [num_reqs, max_blocks] int32 block_table
    qstart_ptr,  # [T] int32 first x-row of the token's request
    T,  # num tokens
    stride_x_t,
    stride_c_blk,
    stride_c_h,
    stride_c_n,
    stride_c_d,
    stride_w_d,
    stride_w_w,
    stride_bt_r,
    MAX_BLOCKS,
    N,  # block_size
    W: tl.constexpr,
    USE_SILU: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    OFF_S: tl.constexpr,
    WS: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program owns a [BT tokens, BLOCK_C channels] tile. The flat channel
    # index packs all H heads head-major (head = c // WS, in-stream offset =
    # c % WS), so one program spans heads -- no per-head launch and no
    # next_power_of_2(WS) lane waste.
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    toff = pid_t * BT + tl.arange(0, BT)  # [BT] token rows
    coff = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C] flat channels
    C = H * WS
    t_mask = toff < T
    c_mask = coff < C
    head = tl.minimum(coff // WS, H - 1)  # clamp keeps masked lanes in-buffer
    cd = OFF_S + coff % WS  # cache D-index of the channel's stream slot

    slot = tl.load(slot_ptr + toff, mask=t_mask, other=-1)  # [BT]
    valid = slot >= 0
    pos = tl.load(pos_ptr + toff, mask=t_mask, other=0)
    req = tl.load(seq_idx_ptr + toff, mask=t_mask, other=0)
    qstart = tl.load(qstart_ptr + toff, mask=t_mask, other=0)

    tc_mask = t_mask[:, None] & c_mask[None, :]

    # 1) Insert each token's input into its paged slot (skip PAD rows).
    xv = tl.load(x_ptr + toff[:, None] * stride_x_t + coff[None, :], mask=tc_mask)
    safe_slot = tl.maximum(slot, 0)
    dst = (
        cache_ptr
        + (safe_slot // N)[:, None] * stride_c_blk
        + head[None, :] * stride_c_h
        + (safe_slot % N)[:, None] * stride_c_n
        + cd[None, :] * stride_c_d
    )
    tl.store(dst, xv, mask=tc_mask & valid[:, None])

    # 2) Convolve the W taps ending at each token's `pos`. Each tap is read from
    # `x` if it falls inside this forward (row >= the request's first row), else
    # from the paged cache. Exactly one source is unmasked per tap, so we sum both.
    acc = tl.zeros([BT, BLOCK_C], dtype=tl.float32)
    for iw in tl.static_range(W):
        src = pos - (W - 1) + iw  # [BT] absolute window position
        row = toff - (W - 1) + iw  # [BT] x-row of `src` (== src - pos + token)
        in_win = valid & (src >= 0)
        intra = in_win & (row >= qstart)
        cached = in_win & (row < qstart)
        # intra-forward tap: read the immutable input x (never the slot we just
        # wrote), so there is no write/read hazard.
        safe_row = tl.maximum(row, 0)
        xt = tl.load(
            x_ptr + safe_row[:, None] * stride_x_t + coff[None, :],
            mask=c_mask[None, :] & intra[:, None],
            other=0.0,
        ).to(tl.float32)
        # pre-forward tap: read from the paged cache via the block table. Clamp
        # addressing terms; the load is masked off when out of window.
        safe_src = tl.maximum(src, 0)
        safe_lblk = tl.minimum(safe_src // N, MAX_BLOCKS - 1)
        blk = tl.load(
            block_table_ptr + req * stride_bt_r + safe_lblk, mask=cached, other=0
        ).to(tl.int64)
        cbase = (
            cache_ptr
            + blk[:, None] * stride_c_blk
            + head[None, :] * stride_c_h
            + (safe_src % N)[:, None] * stride_c_n
            + cd[None, :] * stride_c_d
        )
        cv = tl.load(cbase, mask=c_mask[None, :] & cached[:, None], other=0.0).to(
            tl.float32
        )
        wv = tl.load(
            weight_ptr + coff * stride_w_d + iw * stride_w_w, mask=c_mask, other=0.0
        ).to(tl.float32)
        acc += (xt + cv) * wv[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if USE_RESIDUAL:
        acc += xv.to(tl.float32)

    tl.store(
        out_ptr + toff[:, None] * stride_x_t + coff[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=tc_mask,
    )


def fused_sconv(
    x: torch.Tensor,  # [T, H*ws] head-major current-token inputs
    weight: torch.Tensor,  # [H*ws, W]
    cache: torch.Tensor,  # [num_blocks, H, N, D] paged
    positions: torch.Tensor,  # [T] int64 absolute position per token
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
    seq_idx: torch.Tensor,  # [T] int32 token -> batch request
    slot_mapping: torch.Tensor,  # [T] int64 flat slot (PAD = -1 => skip)
    query_start: torch.Tensor,  # [T] int32 first x-row of the token's request
    off_s: int,
    ws: int,
    block_size: int,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Single-launch insert + depthwise causal conv1d over the paged cache.

    Reads same-forward taps from ``x`` and pre-forward taps from the cache, so
    it is race-free in one launch for prefill / decode / spec and cudagraph-safe
    under eager / piecewise / full capture.
    """
    T = x.shape[0]
    out = torch.empty_like(x)
    if T == 0:
        return out
    assert x.is_contiguous()
    assert cache.stride(3) == 1, "cache D-dim must be contiguous"
    H = cache.shape[1]
    W = weight.shape[1]
    C = H * ws  # flat conv channel dim (all heads, head-major)
    # Tile BT tokens x BLOCK_C channels per program: enough work per CTA to
    # amortize the per-token addressing, while keeping the grid large for
    # prefill. BLOCK_C spans heads so there is no per-head launch. A ~2K-element
    # tile at 4 warps measured best on Blackwell; larger tiles spill registers.
    BLOCK_C = min(triton.next_power_of_2(C), 256)
    BT = 8
    grid = (triton.cdiv(T, BT), triton.cdiv(C, BLOCK_C))
    _fused_sconv_kernel[grid](
        x,
        cache,
        weight,
        out,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        T,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        weight.stride(0),
        weight.stride(1),
        block_table.stride(0),
        block_table.shape[1],
        block_size,
        W=W,
        USE_SILU=activation in ("silu", "swish"),
        USE_RESIDUAL=use_residual,
        OFF_S=off_s,
        WS=ws,
        H=H,
        BT=BT,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


@triton.jit
def _seq_metadata_kernel(
    qsl_ptr,  # [num_reqs + 1] int32 cumulative query start rows
    seq_idx_ptr,  # [T] int32 out: token -> owning request
    query_start_ptr,  # [T] int32 out: first x-row of the token's request
    num_reqs,
    num_actual_tokens,
    num_padded_tokens,
    n_iters,  # ceil(log2(num_reqs)): binary-search depth
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tok = offs.to(tl.int32)
    # Largest j in [0, num_reqs) with qsl[j] <= tok.
    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], num_reqs - 1, tl.int32)
    for _ in range(n_iters):
        mid = (lo + hi + 1) // 2
        below = tl.load(qsl_ptr + mid) <= tok
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    actual = offs < num_actual_tokens
    padded = offs < num_padded_tokens
    query_start = tl.load(qsl_ptr + lo)
    tl.store(seq_idx_ptr + offs, tl.where(actual, lo, 0), mask=padded)
    tl.store(
        query_start_ptr + offs,
        tl.where(actual, query_start, 0),
        mask=padded,
    )


def sconv_seq_metadata(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    seq_idx_out: torch.Tensor,
    query_start_out: torch.Tensor,
    num_padded_tokens: int | None = None,
) -> None:
    """Fill static per-token seq_idx / query_start buffers in one launch.

    Replaces the arange + searchsorted + clamp + gather + 2x copy chain of the
    sconv metadata build with a single kernel writing both persistent buffers.
    Padded rows are filled with zero and must have ``slot_mapping == -1``.
    """
    if num_padded_tokens is None:
        num_padded_tokens = num_actual_tokens
    if num_padded_tokens < num_actual_tokens:
        raise ValueError("num_padded_tokens must cover all actual tokens")
    if num_padded_tokens > seq_idx_out.shape[0]:
        raise ValueError("seq_idx_out is too small for the padded token count")
    if num_padded_tokens > query_start_out.shape[0]:
        raise ValueError("query_start_out is too small for the padded token count")

    BLOCK = 256
    n_iters = (num_reqs - 1).bit_length()
    grid = (triton.cdiv(num_padded_tokens, BLOCK),)
    _seq_metadata_kernel[grid](
        query_start_loc,
        seq_idx_out,
        query_start_out,
        num_reqs,
        num_actual_tokens,
        num_padded_tokens,
        n_iters,
        BLOCK=BLOCK,
    )
