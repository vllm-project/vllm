# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Varlen causal short conv + SiLU as a tile stencil, planar output.

Written for the GLM-5.3-Flash KDA prefill path (`kda_prefill_conv`: the
default `auto` takes it on SM10x); the kernels are width-4 `causal_conv1d_fn`
replacements for any caller.

Same math and conv-state contract as `causal_conv1d_fn` (vLLM Triton, width
`W`; state positions `0..W-2` of `conv_states[cache_indices[s]]` hold the
`W-1` tokens before sequence `s` when `has_initial_state[s]`, and receive
its last `W-1` tokens), with two differences:

- The output is planar, `[num_planes, T, dim // num_planes]`: for the merged
  q|k|v channels each of q, k and v is a dense `[T, H * D]` tensor, so
  consumers that need contiguous q/k/v do not copy them.
- The main kernel is a sequence-agnostic width-4 stencil over tiles of 16
  tokens x BLOCK_N channels (no per-token metadata): each tile is one 2D
  load (registers, the next tile prefetched; or TMA through shared memory,
  pipelined over TILES tiles per program), rows are unpacked in registers
  and slid through the window, and a tile's last 3 rows are the next one's
  halo in both variants. A second launch recomputes the first 3 tokens of
  every sequence with the initial conv state (or zeros) as history; a third
  updates the state. causal_conv1d_fn instead loops token by token with
  4-byte loads.

Conv states are bit-identical to `causal_conv1d_fn`'s for conv states in the
dtype of `x` (bf16 with bf16, the layer's default pairing) and for fp32 weights
with fp32 states, and so are the outputs where Triton contracts that kernel's
fp32 multiply-adds into fused multiply-adds in tap order (the planar kernel
writes them explicitly, see `_conv_silu`), as it does on the GPUs this was
tested on; otherwise outputs differ by 1 bf16 ulp on a few elements per million.
Other dtype pairings are refused: `causal_conv1d_fn` casts `x` to the
conv-state dtype before multiplying while this kernel multiplies `x` as is, so
the two round differently unless that cast is exact and both form their
products in fp32.
The one intended difference: rows of sequences on a null cache slot are
written (as if the sequence had no initial state) where `causal_conv1d_fn`
leaves them unset.
"""

import math
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

logger = init_logger(__name__)

# Geometry the kernels are written for: the stencil kernels unpack tiles of
# TILE_TOKENS tokens in registers (_conv_tile spells out its 16 rows), the conv
# has CONV_WIDTH taps (so a sequence's history is CONV_WIDTH - 1 tokens) and the
# per-sequence kernels take up to _SEQ_BLOCK_N channels per program (the state
# update exactly that; the fixup the largest power of two not above it that
# divides the plane width, so a program stays inside one plane).
# Globals read inside a kernel must be tl.constexpr instances; host code uses
# the plain ints (without Triton, tl.constexpr is a placeholder that returns
# its argument, and the module must still import where the GLM-5.3-Flash model
# is imported without Triton: CPU builds, see tests/test_triton_utils.py).
_TILE_TOKENS = 16
_CONV_WIDTH = 4
TILE_TOKENS = tl.constexpr(_TILE_TOKENS)
CONV_WIDTH = tl.constexpr(_CONV_WIDTH)
_SEQ_BLOCK_N = 1024

# Launch defaults of the stencil kernels, chosen with the conv-only benchmark
# (benchmarks/kernels/benchmark_glm5next_kda_prefill_conv.py) on one SM10x GPU
# at 64 to 8 local heads: 512 channels x 8 tiles of 16 tokens per program, 4
# warps, a 3-stage TMA pipeline. Nothing is autotuned; the public keyword
# arguments exist to re-measure on other GPUs.
_DEFAULT_BLOCK_N = 512
_DEFAULT_NUM_WARPS = 4
_DEFAULT_TILES = 8
_DEFAULT_NUM_STAGES = 3


@triton.jit
def _split_rows(v):
    """[BN, 2k] -> (rows 0, 2, .., rows 1, 3, ..), each [BN, k]; in registers."""
    return tl.split(tl.reshape(v, [v.shape[0], v.shape[1] // 2, 2]))


@triton.jit
def _join_rows(a, b):
    """Inverse of _split_rows."""
    return tl.reshape(tl.join(a, b), [a.shape[0], a.shape[1] * 2])


@triton.jit
def _rows4(v):
    """[BN, 4] -> its 4 rows as [BN] tensors."""
    e, o = _split_rows(v)
    r0, r2 = _split_rows(e)
    r1, r3 = _split_rows(o)
    n: tl.constexpr = v.shape[0]
    return (
        tl.reshape(r0, [n]),
        tl.reshape(r1, [n]),
        tl.reshape(r2, [n]),
        tl.reshape(r3, [n]),
    )


@triton.jit
def _unrows4(r0, r1, r2, r3):
    """Inverse of _rows4."""
    n: tl.constexpr = r0.shape[0]
    e = _join_rows(tl.reshape(r0, [n, 1]), tl.reshape(r2, [n, 1]))
    o = _join_rows(tl.reshape(r1, [n, 1]), tl.reshape(r3, [n, 1]))
    return _join_rows(e, o)


@triton.jit
def _conv_silu(a, b, c, d, w0, w1, w2, w3, acc0, SILU_ACTIVATION: tl.constexpr):
    # Same accumulation order as causal_conv1d_fn (oldest tap first). With fp32
    # weights the taps are explicit fused multiply-adds (one rounding per tap):
    # that is what causal_conv1d_fn's fp32 multiply-adds compile to, and
    # written as separate multiplies and adds the compiler does not always
    # contract them, which costs 1 bf16 ulp on a few outputs per million. With
    # bf16 weights the product is a bf16 multiply in both kernels, so
    # mul-then-add stays.
    if w0.dtype == tl.float32:
        acc = tl.fma(a.to(tl.float32), w0, acc0)
        acc = tl.fma(b.to(tl.float32), w1, acc)
        acc = tl.fma(c.to(tl.float32), w2, acc)
        acc = tl.fma(d.to(tl.float32), w3, acc)
    else:
        acc = acc0 + a * w0
        acc += b * w1
        acc += c * w2
        acc += d * w3
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    return acc


@triton.jit
def _conv_tile(
    xt,
    h1,
    h2,
    h3,
    w0,
    w1,
    w2,
    w3,
    acc0,
    dt: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
):
    """Conv + SiLU of the TILE_TOKENS (16) rows `xt` [BN, 16] given the 3
    rows before them; returns the outputs [BN, 16] and the tile's last 3 rows
    (the next tile's history)."""
    tl.static_assert(xt.shape[1] == TILE_TOKENS)
    tl.static_assert(CONV_WIDTH == 4)
    e, o = _split_rows(xt)
    s0, s2 = _split_rows(e)  # rows 0,4,8,12 / 2,6,10,14
    s1, s3 = _split_rows(o)  # rows 1,5,9,13 / 3,7,11,15
    x0, x4, x8, x12 = _rows4(s0)
    x1, x5, x9, x13 = _rows4(s1)
    x2, x6, x10, x14 = _rows4(s2)
    x3, x7, x11, x15 = _rows4(s3)
    y0 = _conv_silu(h1, h2, h3, x0, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y1 = _conv_silu(h2, h3, x0, x1, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y2 = _conv_silu(h3, x0, x1, x2, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y3 = _conv_silu(x0, x1, x2, x3, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y4 = _conv_silu(x1, x2, x3, x4, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y5 = _conv_silu(x2, x3, x4, x5, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y6 = _conv_silu(x3, x4, x5, x6, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y7 = _conv_silu(x4, x5, x6, x7, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y8 = _conv_silu(x5, x6, x7, x8, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y9 = _conv_silu(x6, x7, x8, x9, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y10 = _conv_silu(x7, x8, x9, x10, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y11 = _conv_silu(x8, x9, x10, x11, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y12 = _conv_silu(x9, x10, x11, x12, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y13 = _conv_silu(x10, x11, x12, x13, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y14 = _conv_silu(x11, x12, x13, x14, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    y15 = _conv_silu(x12, x13, x14, x15, w0, w1, w2, w3, acc0, SILU_ACTIVATION).to(dt)
    u0 = _unrows4(y0, y4, y8, y12)
    u1 = _unrows4(y1, y5, y9, y13)
    u2 = _unrows4(y2, y6, y10, y14)
    u3 = _unrows4(y3, y7, y11, y15)
    return _join_rows(_join_rows(u0, u2), _join_rows(u1, u3)), x13, x14, x15


@triton.jit
def _load_weights(
    weight_ptr,
    bias_ptr,
    offs_f,
    mask_f,
    weight_stride_0,
    weight_stride_1,
    HAS_BIAS: tl.constexpr,
):
    w_row = weight_ptr + offs_f * weight_stride_0
    w0 = tl.load(w_row, mask=mask_f, other=0.0)
    w1 = tl.load(w_row + weight_stride_1, mask=mask_f, other=0.0)
    w2 = tl.load(w_row + 2 * weight_stride_1, mask=mask_f, other=0.0)
    w3 = tl.load(w_row + 3 * weight_stride_1, mask=mask_f, other=0.0)
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + offs_f, mask=mask_f, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros(offs_f.shape, dtype=tl.float32)
    return w0, w1, w2, w3, acc0


@triton.jit(do_not_specialize=["num_tokens"])
def _planar_conv_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_tokens,
    dim,
    plane_dim,
    x_stride_0,
    weight_stride_0,
    weight_stride_1,
    out_stride_0,
    out_stride_1,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
):
    # Sequence-agnostic width-4 stencil over TILES consecutive TILE_TOKENS-token
    # blocks x BLOCK_N channels: every token takes its CONV_WIDTH - 1
    # predecessors from x. The first CONV_WIDTH - 1 tokens of each sequence are
    # redone by the fixup kernel. The next block's rows are loaded before the
    # current block is computed and stored; a block's last CONV_WIDTH - 1 rows
    # are the next block's halo.
    rows = tl.arange(0, TILE_TOKENS)
    t_first = tl.program_id(0).to(tl.int64) * (TILE_TOKENS * TILES)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    w0, w1, w2, w3, acc0 = _load_weights(
        weight_ptr, bias_ptr, offs_f, mask_f, weight_stride_0, weight_stride_1, HAS_BIAS
    )
    # BLOCK_N divides plane_dim, so a program writes inside one plane.
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    out_base = out_ptr + plane.to(tl.int64) * out_stride_0 + offs_f - plane * plane_dim
    x_base = x_ptr + offs_f
    dt: tl.constexpr = out_ptr.dtype.element_ty

    halo_t = t_first - CONV_WIDTH + tl.arange(0, CONV_WIDTH)
    hx = tl.load(
        x_base[:, None] + halo_t[None, :] * x_stride_0,
        mask=mask_f[:, None] & ((halo_t >= 0) & (halo_t < num_tokens))[None, :],
        other=0.0,
    )
    _, h1, h2, h3 = _rows4(hx)
    xt = tl.load(
        x_base[:, None] + (t_first + rows)[None, :] * x_stride_0,
        mask=mask_f[:, None] & ((t_first + rows) < num_tokens)[None, :],
        other=0.0,
    )
    for it in range(TILES):
        offs_t = t_first + it * TILE_TOKENS + rows
        nxt_t = offs_t + TILE_TOKENS
        xn = tl.load(
            x_base[:, None] + nxt_t[None, :] * x_stride_0,
            mask=mask_f[:, None] & ((nxt_t < num_tokens) & (it + 1 < TILES))[None, :],
            other=0.0,
        )
        yt, h1, h2, h3 = _conv_tile(
            xt, h1, h2, h3, w0, w1, w2, w3, acc0, dt, SILU_ACTIVATION
        )
        tl.store(
            out_base[:, None] + offs_t[None, :] * out_stride_1,
            yt,
            mask=mask_f[:, None] & (offs_t < num_tokens)[None, :],
        )
        xt = xn


@triton.jit
def _planar_conv_tma_kernel(
    x_desc,
    x_halo_desc,
    out_desc,
    weight_ptr,
    bias_ptr,
    dim,
    plane_dim,
    weight_stride_0,
    weight_stride_1,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # As _planar_conv_kernel, with TMA loads (pipelined over the tiles) and
    # TMA stores. The token bound is the descriptors' global shape: rows
    # outside [0, num_tokens) read as zero and stores there are dropped, so
    # the kernel takes no num_tokens argument. The halo is loaded once per
    # program; a tile's last rows are the next tile's history, as in the
    # register kernel.
    t_first = tl.program_id(0) * (TILE_TOKENS * TILES)
    f0 = tl.program_id(1) * BLOCK_N
    offs_f = f0 + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    w0, w1, w2, w3, acc0 = _load_weights(
        weight_ptr, bias_ptr, offs_f, mask_f, weight_stride_0, weight_stride_1, HAS_BIAS
    )
    plane = f0 // plane_dim
    col0 = f0 - plane * plane_dim
    dt: tl.constexpr = out_desc.dtype
    _, h1, h2, h3 = _rows4(tl.trans(x_halo_desc.load([t_first - CONV_WIDTH, f0])))
    for it in tl.range(0, TILES, num_stages=NUM_STAGES):
        t0 = t_first + it * TILE_TOKENS
        xt = tl.trans(x_desc.load([t0, f0]))  # [BLOCK_N, TILE_TOKENS]
        yt, h1, h2, h3 = _conv_tile(
            xt, h1, h2, h3, w0, w1, w2, w3, acc0, dt, SILU_ACTIVATION
        )
        out_desc.store(
            [plane, t0, col0], tl.reshape(tl.trans(yt), [1, TILE_TOKENS, BLOCK_N])
        )


@triton.jit
def _slot_flags(
    cache_indices_ptr,
    has_initial_state_ptr,
    seq,
    cache_indices_stride_0,
    has_initial_state_stride_0,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
):
    """The conv-state contract of sequence `seq`, shared by the fixup and
    the state update: `(cache_idx, null_or_pad, has_state)`. A sequence on a
    pad or null slot owns no state: nothing is read from or written to the
    pool for it and `has_state` is false, so its first tokens are computed
    with zero history."""
    cache_idx = tl.load(cache_indices_ptr + seq * cache_indices_stride_0).to(tl.int64)
    null_or_pad = (cache_idx == pad_slot_id) | (cache_idx == null_block_id)
    has_state = (
        tl.load(has_initial_state_ptr + seq * has_initial_state_stride_0).to(tl.int1)
        & (cache_idx != pad_slot_id)
        & (cache_idx != null_block_id)
    )
    return cache_idx, null_or_pad, has_state


# The per-sequence kernels do not specialize on the strides that vary between
# steps (the row stride of x: fused projection row or gathered dense rows; the
# output's plane stride, which follows the token count; the cache-index stride:
# the block table's width; the flag stride, for other callers). They touch three
# rows per sequence, so they give up the 16-byte vector width of those accesses
# and lose nothing (measured per kernel on one SM10x GPU: the fixup runs 2.8x
# faster on a step of two 4K prompts and 500 single-token rows without them, the
# state update and the single-sequence steps are unchanged), and one compile key
# per conv-state pool layout covers every step (see planar_conv_warmup_plan).
# The fixup leaves the conv-state strides unspecialized as well: in the SD pool
# layout (vLLM's default, channel stride 1) Triton otherwise specializes that
# stride, picks another register layout for the three state loads and converts
# it through 2 KB of shared memory per program, which made the fixup 2.6x
# slower than in the DS layout (140 against 53 us on one 4K prompt with 600
# single-token rows at 64 heads on one SM10x GPU); unspecialized it runs the
# same in both layouts, 10 percent slower in DS. The state update keeps the
# specialization: it stores rather than loads there and is faster with it.
# Neither kernel specializes on the alignment of its cache-index, flag and
# offset pointers, which it reads one scalar per sequence: an offset view of a
# block table would otherwise compile a second variant at runtime.
@triton.jit(
    do_not_specialize=[
        "x_stride_0",
        "out_stride_0",
        "conv_states_stride_0",
        "conv_states_stride_1",
        "conv_states_stride_2",
        "cache_indices_stride_0",
        "has_initial_state_stride_0",
    ],
    do_not_specialize_on_alignment=[
        "cache_indices_ptr",
        "has_initial_state_ptr",
        "query_start_loc_ptr",
    ],
)
def _planar_conv_fixup_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    out_ptr,
    dim,
    plane_dim,
    x_stride_0,
    weight_stride_0,
    weight_stride_1,
    conv_states_stride_0,
    conv_states_stride_1,
    conv_states_stride_2,
    out_stride_0,
    out_stride_1,
    cache_indices_stride_0,
    has_initial_state_stride_0,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Recompute the first CONV_WIDTH - 1 (3) tokens of sequence program_id(0):
    # their taps before the sequence start come from the initial conv state
    # (or zero).
    tl.static_assert(CONV_WIDTH == 4)
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - start).to(tl.int32)
    if seq_len <= 0:
        return
    cache_idx, _, has_state = _slot_flags(
        cache_indices_ptr,
        has_initial_state_ptr,
        seq,
        cache_indices_stride_0,
        has_initial_state_stride_0,
        pad_slot_id,
        null_block_id,
    )
    cache_idx = tl.where(has_state, cache_idx, 0)
    w0, w1, w2, w3, acc0 = _load_weights(
        weight_ptr, bias_ptr, offs_f, mask_f, weight_stride_0, weight_stride_1, HAS_BIAS
    )
    plane = (tl.program_id(1) * BLOCK_N) // plane_dim
    out_base = out_ptr + plane.to(tl.int64) * out_stride_0 + offs_f - plane * plane_dim
    x_base = x_ptr + offs_f
    st_base = (
        conv_states_ptr
        + cache_idx * conv_states_stride_0
        + offs_f * conv_states_stride_1
    )
    m_st = mask_f & has_state
    s0 = tl.load(st_base, mask=m_st, other=0.0)
    s1 = tl.load(st_base + conv_states_stride_2, mask=m_st, other=0.0)
    s2 = tl.load(st_base + 2 * conv_states_stride_2, mask=m_st, other=0.0)
    x0 = tl.load(x_base + start * x_stride_0, mask=mask_f, other=0.0)
    y = _conv_silu(s0, s1, s2, x0, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + start * out_stride_1, y.to(out_ptr.dtype.element_ty), mask=mask_f
    )
    m1 = mask_f & (seq_len > 1)
    x1 = tl.load(x_base + (start + 1) * x_stride_0, mask=m1, other=0.0)
    y = _conv_silu(s1, s2, x0, x1, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + (start + 1) * out_stride_1,
        y.to(out_ptr.dtype.element_ty),
        mask=m1,
    )
    m2 = mask_f & (seq_len > 2)
    x2 = tl.load(x_base + (start + 2) * x_stride_0, mask=m2, other=0.0)
    y = _conv_silu(s2, x0, x1, x2, w0, w1, w2, w3, acc0, SILU_ACTIVATION)
    tl.store(
        out_base + (start + 2) * out_stride_1,
        y.to(out_ptr.dtype.element_ty),
        mask=m2,
    )


@triton.jit
def _planar_conv_new_state(
    x_ptr,
    state_base,
    seq_start,
    seq_len,
    has_state,
    offs_f,
    mask_f,
    x_stride_0,
    conv_states_stride_2,
    k: tl.constexpr,
):
    # New state position k holds sequence position seq_len - (CONV_WIDTH - 1) + k:
    # from x, or (short sequences) the old state shifted left by seq_len.
    rel = seq_len - (CONV_WIDTH - 1) + k
    x_val = tl.load(
        x_ptr + (seq_start + tl.maximum(rel, 0)) * x_stride_0 + offs_f,
        mask=mask_f & (rel >= 0),
        other=0.0,
    )
    s_val = tl.load(
        state_base + tl.minimum(k + seq_len, CONV_WIDTH - 2) * conv_states_stride_2,
        mask=mask_f & (rel < 0) & has_state,
        other=0.0,
    )
    return tl.where(rel >= 0, x_val, s_val)


@triton.jit(
    do_not_specialize=[
        "x_stride_0",
        "cache_indices_stride_0",
        "has_initial_state_stride_0",
    ],
    do_not_specialize_on_alignment=[
        "cache_indices_ptr",
        "has_initial_state_ptr",
        "query_start_loc_ptr",
    ],
)
def _planar_conv_state_update_kernel(
    x_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    dim,
    x_stride_0,
    conv_states_stride_0,
    conv_states_stride_1,
    conv_states_stride_2,
    cache_indices_stride_0,
    has_initial_state_stride_0,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Write the CONV_WIDTH - 1 (3) state positions of sequence program_id(0).
    tl.static_assert(CONV_WIDTH == 4)
    seq = tl.program_id(0)
    offs_f = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = offs_f < dim
    cache_idx, null_or_pad, has_state = _slot_flags(
        cache_indices_ptr,
        has_initial_state_ptr,
        seq,
        cache_indices_stride_0,
        has_initial_state_stride_0,
        pad_slot_id,
        null_block_id,
    )
    seq_start = tl.load(query_start_loc_ptr + seq).to(tl.int64)
    seq_len = (tl.load(query_start_loc_ptr + seq + 1) - seq_start).to(tl.int32)
    if null_or_pad | (seq_len <= 0):
        return
    state_base = (
        conv_states_ptr
        + cache_idx * conv_states_stride_0
        + offs_f * conv_states_stride_1
    )
    # All loads happen before the first store.
    v0 = _planar_conv_new_state(
        x_ptr,
        state_base,
        seq_start,
        seq_len,
        has_state,
        offs_f,
        mask_f,
        x_stride_0,
        conv_states_stride_2,
        0,
    )
    v1 = _planar_conv_new_state(
        x_ptr,
        state_base,
        seq_start,
        seq_len,
        has_state,
        offs_f,
        mask_f,
        x_stride_0,
        conv_states_stride_2,
        1,
    )
    v2 = _planar_conv_new_state(
        x_ptr,
        state_base,
        seq_start,
        seq_len,
        has_state,
        offs_f,
        mask_f,
        x_stride_0,
        conv_states_stride_2,
        2,
    )
    tl.debug_barrier()
    tl.store(state_base, v0, mask=mask_f)
    tl.store(state_base + conv_states_stride_2, v1, mask=mask_f)
    tl.store(state_base + 2 * conv_states_stride_2, v2, mask=mask_f)


def planar_conv_impl(
    impl: str,
    x: Any,
    out: torch.Tensor | None = None,
    num_planes: int = 3,
    block_n: int = _DEFAULT_BLOCK_N,
) -> str:
    """The kernel `causal_conv1d_planar(x, ..., impl=impl)` runs: "tma"
    or "regs".

    TMA descriptors need SM90+, 16-byte aligned base addresses and row
    strides for `x` and `out` (`out=None` stands for the dense output
    the function allocates itself), and channel blocks of at least 16 bytes
    (`block_n` as the kernel runs it, `gcd(block_n, dim // num_planes)`).
    `impl="auto"` takes the TMA kernel when they can address the tensors
    and the register kernel otherwise; `impl="tma"` raises instead of
    falling back, so a caller that expects the TMA kernel is told rather than
    served the slower one; `impl="regs"` is the register kernel. `x` may be
    a `TritonWarmupTensor` standing for a layer's rows (the warmup plan asks
    which kernel they take).
    """
    if impl not in ("auto", "tma", "regs"):
        raise ValueError(f"Unsupported impl {impl!r}: 'auto', 'tma' or 'regs'")
    if impl == "regs":
        return "regs"
    e = x.dtype.itemsize
    plane_dim = x.shape[1] // num_planes
    block_n = math.gcd(block_n, plane_dim)
    x_ok = x.data_ptr() % 16 == 0 and (x.stride(0) * e) % 16 == 0
    if out is None:
        # A fresh allocation: 16-byte aligned base, both strides multiples of
        # the plane width.
        out_ok = (plane_dim * e) % 16 == 0
    else:
        out_ok = (
            out.data_ptr() % 16 == 0
            and (out.stride(0) * e) % 16 == 0
            and (out.stride(1) * e) % 16 == 0
        )
    block_ok = block_n * e >= 16
    if (
        current_platform.is_cuda()
        and current_platform.has_device_capability(90)
        and x_ok
        and out_ok
        and block_ok
    ):
        return "tma"
    if impl == "tma":
        missing = []
        if not current_platform.is_cuda():
            missing.append("a CUDA device")
        elif not current_platform.has_device_capability(90):
            missing.append("SM90 or newer (this device is older)")
        if not x_ok:
            missing.append(
                f"16-byte aligned x rows (row stride {x.stride(0) * e} bytes at "
                f"{x.data_ptr() % 16} mod 16)"
            )
        if not out_ok:
            missing.append("16-byte aligned output rows")
        if not block_ok:
            missing.append(
                f"channel blocks of at least 16 bytes (block_n {block_n} x {e} bytes)"
            )
        raise ValueError(
            f"impl='tma' needs {' and '.join(missing)}; use impl='auto' to "
            "fall back to the register kernel"
        )
    return "regs"


# Dispatch guard. The planar path (three launches, no per-token metadata) beats
# causal_conv1d_fn once the prefill tokens pay for the launches and for the
# single-token decode rows that share the step; per row the planar path costs
# about what the stock kernel costs since the fixup's SD-layout fix (it was
# twice), so the 6 prefill tokens charged per row below are conservative.
# Break-even in GPU time from a conv-only sweep on one SM10x GPU (bf16;
# local head counts of the TP1 to TP8 shards of GLM-5.3-Flash; the benchmark's
# --gpu-time): below 1K prefill tokens at 64 and 32 heads, about 1.6K at 16 and
# 3.3K at 8 for a single sequence, plus 5 to 6 tokens per decode row at every
# head count. The intercepts below add a margin; below 8 heads nothing was
# measured and the trend puts break-even higher still. Re-measure on other GPUs
# before changing them. Host time is not in these figures: the three launches
# cost about 40 us more per call than the stock kernel's one, which a prefill
# step hides unless it is launch-bound (the benchmark's wall-clock columns show
# that floor at small shapes).
PLANAR_MIN_PREFILL_TOKENS = {64: 1024, 32: 1536, 16: 2048, 8: 4096}
PLANAR_TOKENS_PER_DECODE_ROW = 6


def planar_conv_pays_off(
    num_prefill_tokens: int, num_decode_rows: int, local_num_heads: int
) -> bool:
    """Whether `causal_conv1d_planar` is expected to beat
    `causal_conv1d_fn` on a step with `num_prefill_tokens` prefill tokens
    and `num_decode_rows` single-token rows at `local_num_heads` heads:
    `prefill tokens > intercept(heads) + 6 x decode rows`, with the intercept
    of the largest tabulated head count not above `local_num_heads` (twice
    the largest intercept below the table).
    """
    intercept = 2 * max(PLANAR_MIN_PREFILL_TOKENS.values())
    for heads, tokens in sorted(PLANAR_MIN_PREFILL_TOKENS.items(), reverse=True):
        if local_num_heads >= heads:
            intercept = tokens
            break
    return (
        num_prefill_tokens > intercept + PLANAR_TOKENS_PER_DECODE_ROW * num_decode_rows
    )


# --- launches and JIT warmup -------------------------------------------------
# Every launch goes through a dispatcher (docs/contributing/jit_kernel_warmup.md)
# that maps a call's tensors to the grid and the derived launch arguments;
# vLLM's helper binds the kernel's pointer and stride arguments to the
# dispatcher's inputs by name (`x` -> `x_ptr` and `x_stride_0`, ...; a scalar
# input with a kernel argument's name, such as `pad_slot_id`, directly). A
# warmup-inputs function describes one layer configuration's tensors (dtype,
# shape, strides, alignment) as `TritonWarmupTensor` stand-ins, so the same
# dispatch yields the compile keys at startup without real tensors or launches.
# The TMA kernel builds its descriptors inside the dispatch; Triton accepts them
# on warmup tensors.


def _stencil_warmup_inputs(
    *,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    dim: int,
    num_planes: int,
    row_stride: int,
    block_n: int,
    tiles: int,
    num_warps: int,
    num_stages: int | None = None,
) -> dict[str, Any]:
    """One tile of `dim` channels in rows of `row_stride` elements (the
    register kernel without `num_stages`, the TMA kernel with it). The stencil
    kernels do not specialize on the token count, and with plane widths that
    are multiples of 16 elements (every KDA shard) the output strides' Triton
    classes do not depend on it either, so one case per row stride covers
    every step."""
    case = dict(
        x=TritonWarmupTensor(dtype, shape=(_TILE_TOKENS, dim), strides=(row_stride, 1)),
        weight=TritonWarmupTensor(weight_dtype, shape=(dim, _CONV_WIDTH)),
        bias=None
        if bias_dtype is None
        else TritonWarmupTensor(bias_dtype, shape=(dim,)),
        out=TritonWarmupTensor(
            dtype, shape=(num_planes, _TILE_TOKENS, dim // num_planes)
        ),
        block_n=block_n,
        tiles=tiles,
        num_warps=num_warps,
        activation="silu",
    )
    if num_stages is not None:
        case["num_stages"] = num_stages
    return case


def _per_seq_warmup_inputs(
    *,
    dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    state_len: int,
    dim: int,
    conv_state_dim_first: bool,
    weight_dtype: torch.dtype | None = None,
    bias_dtype: torch.dtype | None = None,
    num_planes: int | None = None,
) -> dict[str, Any]:
    """Two sequences over one slot of the conv-state pool as the kernels see
    it, `[slot, dim, state_len]` with `state_len >= width - 1` (plus the
    speculative tokens, which changes the stride classes): the DS layout
    directly, the SD layout as the transposed view kda.py passes. The
    per-sequence kernels do not specialize on the strides that vary between
    steps, so one case per pool covers every step. With `num_planes` (and the
    weight and bias dtypes) the case is the fixup's, else the state update's."""
    strides = (
        (state_len * dim, state_len, 1)
        if conv_state_dim_first
        else (state_len * dim, 1, dim)
    )
    case = dict(
        x=TritonWarmupTensor(dtype, shape=(_TILE_TOKENS, dim)),
        conv_states=TritonWarmupTensor(
            conv_state_dtype, shape=(1, dim, state_len), strides=strides
        ),
        cache_indices=TritonWarmupTensor(torch.int32, shape=(2,)),
        has_initial_state=TritonWarmupTensor(torch.bool, shape=(2,)),
        query_start_loc=TritonWarmupTensor(torch.int32, shape=(3,)),
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
    )
    if num_planes is not None:
        case.update(
            weight=TritonWarmupTensor(weight_dtype, shape=(dim, _CONV_WIDTH)),
            bias=None
            if bias_dtype is None
            else TritonWarmupTensor(bias_dtype, shape=(dim,)),
            out=TritonWarmupTensor(
                dtype, shape=(num_planes, _TILE_TOKENS, dim // num_planes)
            ),
            activation="silu",
        )
    return case


def _stencil_launch(x, out, bias, block_n, tiles, num_warps, activation):
    """Grid and derived launch arguments shared by the two stencil kernels."""
    num_tokens, dim = x.shape
    grid = (triton.cdiv(num_tokens, _TILE_TOKENS * tiles), triton.cdiv(dim, block_n))
    return grid, dict(
        dim=dim,
        plane_dim=out.shape[2],
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        TILES=tiles,
        num_warps=num_warps,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_planar_conv_kernel, warmup_inputs=_stencil_warmup_inputs
)
def _planar_stencil_regs(
    x, weight, bias, out, block_n, tiles, num_warps, activation
) -> DispatchSpec:
    grid, kwargs = _stencil_launch(x, out, bias, block_n, tiles, num_warps, activation)
    return grid, dict(kwargs, num_tokens=x.shape[0])


@triton_kernel_dispatcher_with_warmup(
    kernel=_planar_conv_tma_kernel, warmup_inputs=_stencil_warmup_inputs
)
def _planar_stencil_tma(
    x, weight, bias, out, block_n, tiles, num_warps, num_stages, activation
) -> DispatchSpec:
    from triton.tools.tensor_descriptor import TensorDescriptor

    num_tokens, dim = x.shape
    num_planes, _, plane_dim = out.shape
    grid, kwargs = _stencil_launch(x, out, bias, block_n, tiles, num_warps, activation)
    # The descriptors carry the token bound: rows outside [0, num_tokens) read
    # as zero and stores there are dropped.
    return grid, dict(
        kwargs,
        x_desc=TensorDescriptor(
            x, [num_tokens, dim], [x.stride(0), 1], [_TILE_TOKENS, block_n]
        ),
        x_halo_desc=TensorDescriptor(
            x, [num_tokens, dim], [x.stride(0), 1], [_CONV_WIDTH, block_n]
        ),
        out_desc=TensorDescriptor(
            out,
            [num_planes, num_tokens, plane_dim],
            [out.stride(0), out.stride(1), 1],
            [1, _TILE_TOKENS, block_n],
        ),
        NUM_STAGES=num_stages,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_planar_conv_fixup_kernel, warmup_inputs=_per_seq_warmup_inputs
)
def _planar_fixup(
    x,
    weight,
    bias,
    conv_states,
    cache_indices,
    has_initial_state,
    query_start_loc,
    out,
    activation,
    pad_slot_id,
    null_block_id,
) -> DispatchSpec:
    dim, plane_dim = x.shape[1], out.shape[2]
    # Programs must not straddle planes.
    block_n = math.gcd(_SEQ_BLOCK_N, plane_dim)
    return (query_start_loc.shape[0] - 1, triton.cdiv(dim, block_n)), dict(
        dim=dim,
        plane_dim=plane_dim,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_planar_conv_state_update_kernel, warmup_inputs=_per_seq_warmup_inputs
)
def _planar_state_update(
    x,
    conv_states,
    cache_indices,
    has_initial_state,
    query_start_loc,
    pad_slot_id,
    null_block_id,
) -> DispatchSpec:
    dim = x.shape[1]
    return (query_start_loc.shape[0] - 1, triton.cdiv(dim, _SEQ_BLOCK_N)), dict(
        dim=dim, BLOCK_N=_SEQ_BLOCK_N, num_warps=4
    )


def planar_conv_warmup_plan(
    *,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    dim: int,
    num_planes: int,
    row_strides: tuple[int, ...],
    conv_state_dim_first: bool,
    conv_state_dtype: torch.dtype | None = None,
    state_len: int = _CONV_WIDTH - 1,
    impl: str = "auto",
    block_n: int = _DEFAULT_BLOCK_N,
    tiles: int = _DEFAULT_TILES,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> list[tuple[Any, dict[str, Any]]]:
    """The (dispatcher, warmup kwargs) pairs that cover one layer's planar
    conv; `dispatcher.register_warmup(**kwargs)` on each registers them for
    startup JIT warmup.

    For each row stride of `x` (the fused projection's row; the dense row of
    the gathered non-spec rows of a spec-decode step) the stencil kernel that
    `planar_conv_impl` selects for it, then the fixup and the state update
    once for the conv-state pool (`conv_state_dtype`, default the activation
    dtype; `state_len` positions per channel, width - 1 plus the speculative
    tokens; the layout): three or four compile keys per layer.
    """
    plane_dim = dim // num_planes
    block_n = math.gcd(block_n, plane_dim)
    stencil = dict(
        dtype=dtype,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        dim=dim,
        num_planes=num_planes,
        block_n=block_n,
        tiles=tiles,
        num_warps=num_warps,
    )
    plan: list[tuple[Any, dict[str, Any]]] = []
    for row_stride in dict.fromkeys(row_strides):
        rows = TritonWarmupTensor(
            dtype, shape=(_TILE_TOKENS, dim), strides=(row_stride, 1)
        )
        if planar_conv_impl(impl, rows, None, num_planes, block_n) == "tma":
            plan.append(
                (
                    _planar_stencil_tma,
                    dict(stencil, row_stride=row_stride, num_stages=num_stages),
                )
            )
        else:
            plan.append((_planar_stencil_regs, dict(stencil, row_stride=row_stride)))
    per_seq = dict(
        dtype=dtype,
        conv_state_dtype=dtype if conv_state_dtype is None else conv_state_dtype,
        state_len=state_len,
        dim=dim,
        conv_state_dim_first=conv_state_dim_first,
    )
    plan.append(
        (
            _planar_fixup,
            dict(
                per_seq,
                weight_dtype=weight_dtype,
                bias_dtype=bias_dtype,
                num_planes=num_planes,
            ),
        )
    )
    plan.append((_planar_state_update, per_seq))
    return plan


def causal_conv1d_planar(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    num_planes: int = 3,
    activation: str | None = "silu",
    out: torch.Tensor | None = None,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    tiles: int = _DEFAULT_TILES,
    impl: str = "auto",
    num_stages: int = _DEFAULT_NUM_STAGES,
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """Causal conv1d (+ SiLU) over a varlen token batch, planar output.

    Args:
        x: `[T, dim]` token-major input; rows may be strided (e.g. the
            q|k|v slice of a wider projection), channels must be dense.
        weight: `[dim, width]` conv weight.
        bias: optional `[dim]` bias.
        conv_states: `[num_slots, dim, >= width - 1]` view of the conv
            state pool (any strides, so both SD and DS layouts work) in the
            dtype of `x`, or fp32 with fp32 weights; updated in place for
            every sequence with a real cache slot.
        query_start_loc: `[N + 1]` contiguous int32 cumulative token offsets.
        cache_indices: `[>= N]` cache slot per sequence, may be strided
            (e.g. column 0 of the block table). Sequences on a null slot read
            and write no state and their output rows are computed with zero
            history (`causal_conv1d_fn` skips them and leaves the rows
            unset); a pad slot is treated the same way here, while
            `causal_conv1d_fn` expects pad slots on empty sequences only.
        has_initial_state: `[>= N]` bool, read the old state as history.
        num_planes: output planes; `dim` must be divisible by it.
        activation: `"silu"`/`"swish"` or `None`.
        out: optional `[num_planes, >= T, dim // num_planes]` output with a
            dense last dim.
        block_n: channels per program (16 tokens each), a power of 2;
            replaced by `gcd(block_n, dim // num_planes)` (the largest
            power of 2 not above it that divides the plane width). Blocks
            under 16 bytes run on the register kernel (see
            `planar_conv_impl`).
        num_warps: warps per program.
        tiles: consecutive 16-token blocks per program. The defaults of these
            three and of `num_stages` are the module's `_DEFAULT_*` values
            (measured on one GPU, see there).
        impl: "regs" (16-row tiles loaded into registers, the next tile
            prefetched), "tma" (TMA loads/stores through shared memory,
            pipelined over the tiles; needs SM90+ and 16-byte aligned rows and
            raises otherwise) or "auto" ("tma" where possible, else "regs";
            see `planar_conv_impl`). The kernel in use is logged once.
        num_stages: TMA pipeline depth (impl="tma").
        pad_slot_id: the cache slot that marks a padded (empty) sequence
            (vLLM's constant by default).
        null_block_id: the cache slot that marks a sequence without a slot
            (vLLM's constant by default); its rows are computed with zero
            history and no state is read or written.

    Returns:
        `[num_planes, T, dim // num_planes]` output in `x.dtype`.

    """
    if x.dim() != 2 or x.stride(1) != 1:
        raise ValueError(
            "x must be a 2D [T, dim] tensor with dense channels, got shape "
            f"{tuple(x.shape)} with strides {x.stride()}"
        )
    num_tokens, dim = x.shape
    if weight.dim() != 2 or weight.shape[1] != _CONV_WIDTH:
        raise ValueError(
            f"causal_conv1d_planar implements conv width {_CONV_WIDTH} "
            f"only, got a weight of shape {tuple(weight.shape)}"
        )
    width = weight.shape[1]
    if weight.shape[0] != dim:
        raise ValueError(f"weight has {weight.shape[0]} channels, x has {dim}")
    if bias is not None and (
        bias.dim() != 1 or bias.numel() != dim or bias.device != x.device
    ):
        raise ValueError(
            f"bias must be a [{dim}] tensor on {x.device}, got shape "
            f"{tuple(bias.shape)} on {bias.device}"
        )
    if num_planes < 1 or dim % num_planes != 0:
        raise ValueError(f"dim {dim} is not divisible by num_planes {num_planes}")
    if (
        conv_states.dim() != 3
        or conv_states.shape[1] != dim
        or conv_states.shape[2] < width - 1
    ):
        raise ValueError(
            f"conv_states must be [num_slots, {dim}, >= {width - 1}], got shape "
            f"{tuple(conv_states.shape)}"
        )
    if conv_states.dtype != x.dtype and not (
        weight.dtype == torch.float32 and conv_states.dtype == torch.float32
    ):
        raise TypeError(
            f"conv_states dtype {conv_states.dtype} with x dtype {x.dtype} and "
            f"weight dtype {weight.dtype}: causal_conv1d_fn casts x to the "
            "conv-state dtype before multiplying and this kernel does not, so "
            "only conv states in x's dtype, or fp32 states with fp32 weights, "
            "give the same results"
        )
    if activation not in ("silu", "swish", None):
        raise ValueError(
            f"Unsupported activation {activation!r}: 'silu', 'swish' or None"
        )
    if block_n < 1 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a power of two, got {block_n}")
    if tiles < 1 or num_stages < 1 or num_warps < 1 or num_warps & (num_warps - 1):
        raise ValueError(
            f"tiles ({tiles}) and num_stages ({num_stages}) must be positive and "
            f"num_warps ({num_warps}) a positive power of two"
        )
    if (
        query_start_loc.dim() != 1
        or query_start_loc.stride(0) != 1
        or query_start_loc.dtype.is_floating_point
        or query_start_loc.dtype == torch.bool
    ):
        raise ValueError(
            "query_start_loc must be a contiguous 1D integer tensor, got shape "
            f"{tuple(query_start_loc.shape)} with strides {query_start_loc.stride()} "
            f"of dtype {query_start_loc.dtype}"
        )
    num_seqs = query_start_loc.shape[0] - 1
    if cache_indices.dim() != 1 or has_initial_state.dim() != 1:
        raise ValueError(
            "cache_indices and has_initial_state must be 1D, got shapes "
            f"{tuple(cache_indices.shape)} and {tuple(has_initial_state.shape)}"
        )
    if cache_indices.dtype.is_floating_point or cache_indices.dtype == torch.bool:
        raise ValueError(
            f"cache_indices must be an integer tensor, got {cache_indices.dtype}"
        )
    if cache_indices.numel() < num_seqs or has_initial_state.numel() < num_seqs:
        raise ValueError(
            f"cache_indices ({cache_indices.numel()}) and has_initial_state "
            f"({has_initial_state.numel()}) need an entry per sequence ({num_seqs})"
        )
    for name, t in (
        ("weight", weight),
        ("conv_states", conv_states),
        ("query_start_loc", query_start_loc),
        ("cache_indices", cache_indices),
        ("has_initial_state", has_initial_state),
    ):
        if t.device != x.device:
            raise TypeError(f"{name} must be on {x.device}, got {t.device}")
    plane_dim = dim // num_planes
    # Programs must not straddle planes.
    block_n = math.gcd(block_n, plane_dim)
    if out is None:
        out = torch.empty(
            (num_planes, num_tokens, plane_dim), dtype=x.dtype, device=x.device
        )
    elif (
        out.dim() != 3
        or out.shape[0] != num_planes
        or out.shape[1] < num_tokens
        or out.shape[2] != plane_dim
        or out.stride(2) != 1
    ):
        raise ValueError(
            f"out must be [{num_planes}, >= {num_tokens}, {plane_dim}] with a "
            f"dense last dim, got shape {tuple(out.shape)} with strides "
            f"{out.stride()}"
        )
    elif out.dtype != x.dtype or out.device != x.device:
        raise TypeError(
            f"out must match x in dtype and device ({x.dtype}, {x.device}), got "
            f"{out.dtype}, {out.device}"
        )
    impl_used = planar_conv_impl(impl, x, out, num_planes, block_n)
    logger.info_once(
        "causal_conv1d_planar: %s kernel for %d channels in %d planes (x row "
        "stride %d bytes, impl=%r)",
        impl_used,
        dim,
        num_planes,
        x.stride(0) * x.element_size(),
        impl,
    )
    if num_tokens == 0 or num_seqs <= 0:
        return out[:, :num_tokens]
    if impl_used == "tma":
        _planar_stencil_tma(
            x, weight, bias, out, block_n, tiles, num_warps, num_stages, activation
        )
    else:
        _planar_stencil_regs(
            x, weight, bias, out, block_n, tiles, num_warps, activation
        )
    _planar_fixup(
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        activation,
        pad_slot_id,
        null_block_id,
    )
    _planar_state_update(
        x,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        pad_slot_id,
        null_block_id,
    )
    return out[:, :num_tokens]
