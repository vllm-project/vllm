# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill short convolution over the packed Q|K|V projection in one launch.

The stock prefill runs `causal_conv1d_fn` once per Q, K and V. Each call first
casts the row-strided bf16 slice of the in_proj output to a dense FP32 copy (the
conv state dtype), convolves it into an FP32 output and casts that back to bf16:
per layer and rank at a 16K chunk that is 3 x (76 + 52 + 19) us, of which only
the convolution reads and writes the data the layer needs.

This kernel reads the bf16 Q|K|V columns of the in_proj output in place, does the
stock arithmetic in FP32 (bf16 -> FP32 is exact; same product order, FP32 weights,
FP32 conv state, SiLU), and stores bf16 Q, K and V as three dense `(tokens, dim)`
planes of one buffer, the layout the stock casts produce. FP32 -> bf16 rounds to
nearest even, as the stock `.to(bfloat16)` does, so the outputs and the updated
conv state are bit-identical to the stock path.

Programs map to (sequence, token chunk) on the device by a radix-32 search over
`query_start_loc`, so no host-side chunk table is needed and the launch is CUDA
graph safe. A program streams its tokens over one channel vector with the 3
history rows in registers and the next 8 input rows prefetched in a register
queue. As in the stock kernel, only a sequence's first chunk reads
its conv state, and that same program then writes the updated state.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _config(part_dim: int) -> tuple[int, int, int, int]:
    """(BLOCK_M unrolled tokens, NUM_BLOCKS blocks per program, BLOCK_N channels,
    warps). BLOCK_N divides part_dim so that a program writes one plane."""
    # Measured on GB300 at 2K-16K tokens x 3 x 1536 channels: one warp per
    # 128 channels and 64-token programs stream at ~4.8 TB/s of read + write.
    return 8, 8, 128 if part_dim % 128 == 0 else 64, 1


@triton.jit
def _load_row(x_row, r, end, stride_x_token):
    return tl.load(x_row + r * stride_x_token, mask=r < end, other=0.0)


@triton.jit
def _packed_causal_conv1d_fwd_kernel(
    x_ptr,  # (tokens, dim) channel-contiguous, any row stride
    w_ptr,  # (dim, 4), FP32
    bias_ptr,  # (dim,) or None
    state_ptr,  # (cache lines, dim, >= 3), FP32
    cache_indices_ptr,  # (seqs,)
    has_initial_state_ptr,  # (seqs,)
    query_start_loc_ptr,  # (seqs + 1,)
    out_ptr,  # (dim // part_dim, tokens, part_dim)
    num_seqs,
    search_steps,
    search_stride,
    num_col_blocks,
    part_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    stride_cache_indices,
    stride_out_part,
    stride_out_token,
    null_block_id,
    HAS_BIAS: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Width-4 taps: 3 history rows. Every tensor is one [BLOCK_N] vector in one
    # layout, so each thread owns its channels for the whole program.
    STATE_LEN: tl.constexpr = 3
    CHUNK: tl.constexpr = BLOCK_M * NUM_BLOCKS
    # The channel block varies fastest: CTAs running together cover whole rows.
    pid = tl.program_id(0) // num_col_blocks
    col_block = tl.program_id(0) % num_col_blocks

    # This program's (sequence, chunk). Sequence s owns the programs [g(s), g(s+1))
    # with g(s) = (query_start_loc[s] - query_start_loc[0]) // CHUNK + s: g strictly
    # increases, a sequence of n tokens gets at least cdiv(n, CHUNK) programs, and
    # g(num_seqs) <= cdiv(tokens, CHUNK) + num_seqs = the grid's rows. A radix-32
    # search finds the last s with g(s) <= pid: one 32-wide parallel load per level,
    # one level up to 32 sequences and two up to 1024 (a linear scan costs O(seqs)
    # and was slower than stock at ~500+ sequences; a binary search's dependent
    # loads cost ~4 us per layer at 4-32 sequences).
    qsl0 = tl.load(query_start_loc_ptr)
    lo = pid * 0
    stride = search_stride
    for _ in range(0, search_steps):
        idx = lo + tl.arange(0, 32) * stride
        in_range = idx < num_seqs
        g = (
            tl.load(query_start_loc_ptr + idx, mask=in_range, other=0) - qsl0
        ) // CHUNK + idx
        lo += (tl.sum((in_range & (g <= pid)).to(tl.int32)) - 1) * stride
        stride = stride // 32
    seq = lo
    seq_start_i = tl.load(query_start_loc_ptr + seq)
    chunk = pid - ((seq_start_i - qsl0) // CHUNK + seq)
    seqlen = tl.load(query_start_loc_ptr + seq + 1) - seq_start_i
    if chunk * CHUNK >= seqlen:  # spare program (also: empty sequences)
        return

    seq_start = seq_start_i.to(tl.int64)
    slot = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa: SIM102 (constexpr guard of a runtime branch)
        if slot == null_block_id:
            return
    load_init = tl.load(has_initial_state_ptr + seq).to(tl.int1)

    col0 = col_block * BLOCK_N
    feats = col0 + tl.arange(0, BLOCK_N)
    x_row = x_ptr + seq_start * stride_x_token + feats
    state_base = state_ptr + slot * stride_state_seq + feats * stride_state_dim
    row0 = chunk * CHUNK
    end = tl.minimum(seqlen, row0 + CHUNK)

    # Rows are prefetched 8 ahead in a register queue: a row's load is issued 8
    # rows before its use. (In program order a load follows the previous row's
    # store, and the compiler cannot hoist it there since x and out may alias;
    # loading each row just before its use leaves one row in flight per warp.)
    q0 = _load_row(x_row, row0 + 0, end, stride_x_token)
    q1 = _load_row(x_row, row0 + 1, end, stride_x_token)
    q2 = _load_row(x_row, row0 + 2, end, stride_x_token)
    q3 = _load_row(x_row, row0 + 3, end, stride_x_token)
    q4 = _load_row(x_row, row0 + 4, end, stride_x_token)
    q5 = _load_row(x_row, row0 + 5, end, stride_x_token)
    q6 = _load_row(x_row, row0 + 6, end, stride_x_token)
    q7 = _load_row(x_row, row0 + 7, end, stride_x_token)

    # History rows row0 - 3 .. row0 - 1: input rows, or before the sequence the
    # conv state (zero without an initial state).
    h0 = _history_row(
        x_row,
        state_base,
        row0 - 3,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )
    h1 = _history_row(
        x_row,
        state_base,
        row0 - 2,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )
    h2 = _history_row(
        x_row,
        state_base,
        row0 - 1,
        load_init,
        stride_x_token,
        stride_state_token,
        STATE_LEN,
    )

    w0 = tl.load(w_ptr + feats * stride_w_dim + 0 * stride_w_width)
    w1 = tl.load(w_ptr + feats * stride_w_dim + 1 * stride_w_width)
    w2 = tl.load(w_ptr + feats * stride_w_dim + 2 * stride_w_width)
    w3 = tl.load(w_ptr + feats * stride_w_dim + 3 * stride_w_width)
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + feats).to(tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    part = col0 // part_dim
    out_row = (
        out_ptr
        + part.to(tl.int64) * stride_out_part
        + seq_start * stride_out_token
        + (col0 - part * part_dim)
        + tl.arange(0, BLOCK_N)
    )
    num_blocks = tl.cdiv(end - row0, BLOCK_M)
    for b in range(0, num_blocks):
        for i in tl.static_range(BLOCK_M):
            r = row0 + b * BLOCK_M + i
            cur = q0.to(tl.float32)
            q0 = q1
            q1 = q2
            q2 = q3
            q3 = q4
            q4 = q5
            q5 = q6
            q6 = q7
            q7 = _load_row(x_row, r + 8, end, stride_x_token)
            # The stock chain, oldest tap first: fma(x_j, w_j, acc). Explicit
            # FMAs so the contraction does not depend on the compiler.
            acc = tl.fma(h0, w0, acc0)
            acc = tl.fma(h1, w1, acc)
            acc = tl.fma(h2, w2, acc)
            acc = tl.fma(cur, w3, acc)
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            tl.store(
                out_row + r * stride_out_token,
                acc.to(out_ptr.dtype.element_ty),
                mask=r < end,
            )
            h0 = h1
            h1 = h2
            h2 = cur

    if chunk == 0:
        # New state = the last 3 inputs, shifted in after the old state when the
        # sequence is shorter. All old-state reads precede the stores, and the
        # same thread owns a channel throughout.
        n0 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            0,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        n1 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            1,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        n2 = _new_state_row(
            x_row,
            state_base,
            seqlen,
            load_init,
            2,
            stride_x_token,
            stride_state_token,
            STATE_LEN,
        )
        tl.store(state_base + 0 * stride_state_token, n0)
        tl.store(state_base + 1 * stride_state_token, n1)
        tl.store(state_base + 2 * stride_state_token, n2)


@triton.jit
def _history_row(
    x_row,
    state_base,
    pos,
    load_init,
    stride_x_token,
    stride_state_token,
    STATE_LEN: tl.constexpr,
):
    return tl.where(
        pos >= 0,
        tl.load(x_row + pos * stride_x_token, mask=pos >= 0, other=0.0).to(tl.float32),
        tl.load(
            state_base + (STATE_LEN + pos) * stride_state_token,
            mask=(pos < 0) & load_init,
            other=0.0,
        ),
    )


@triton.jit
def _new_state_row(
    x_row,
    state_base,
    seqlen,
    load_init,
    t,
    stride_x_token,
    stride_state_token,
    STATE_LEN: tl.constexpr,
):
    src = seqlen - STATE_LEN + t
    return tl.where(
        src >= 0,
        tl.load(x_row + src * stride_x_token, mask=src >= 0, other=0.0).to(tl.float32),
        tl.load(
            state_base + (t + seqlen) * stride_state_token,
            mask=(src < 0) & load_init,
            other=0.0,
        ),
    )


def packed_causal_conv1d_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    part_dim: int,
) -> bool:
    """Whether `packed_causal_conv1d_fn` reproduces the stock conv bit for bit.

    The stock kernel computes in the conv state dtype; with an FP32 state that is
    FP32 for every input dtype, which this kernel does."""
    return (
        x.is_cuda
        and x.dim() == 2
        and x.stride(1) == 1
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and conv_states.dtype == torch.float32
        and weight.dtype == torch.float32
        and weight.dim() == 2
        and weight.size(0) == x.size(1)
        and weight.size(1) == 4
        and conv_states.dim() == 3
        and conv_states.size(1) == x.size(1)
        and conv_states.size(2) >= 3
        and part_dim % 64 == 0
        and x.size(1) % part_dim == 0
    )


def packed_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    part_dim: int,
    activation: str | None = "silu",
    null_block_id: int | None = NULL_BLOCK_ID,
) -> torch.Tensor:
    """Varlen causal depthwise conv of `x` (tokens, dim), dim = parts x part_dim.

    Same semantics as `causal_conv1d_fn(x.T, weight, bias, conv_states, ...).T`
    with `conv_states` (cache lines, dim, 3) updated in place; width-4 taps. Returns
    `(dim // part_dim, tokens, part_dim)` in `x.dtype`: plane i is the dense
    output of channels [i * part_dim, (i + 1) * part_dim).
    """
    num_tokens, dim = x.shape
    num_seqs = query_start_loc.size(0) - 1
    out = torch.empty(
        (dim // part_dim, num_tokens, part_dim), dtype=x.dtype, device=x.device
    )
    if num_tokens == 0 or num_seqs == 0:
        return out
    block_m, num_blocks, block_n, num_warps = _config(part_dim)
    # Radix-32 levels of the kernel's program search: 32 ** steps >= num_seqs.
    search_steps = 0
    while 32**search_steps < num_seqs:
        search_steps += 1
    # Every sequence adds at most one partial chunk (see the kernel's program map).
    num_col_blocks = triton.cdiv(dim, block_n)
    grid = (
        (triton.cdiv(num_tokens, block_m * num_blocks) + num_seqs) * num_col_blocks,
    )
    _packed_causal_conv1d_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        num_seqs,
        search_steps,
        32 ** max(search_steps - 1, 0),
        num_col_blocks,
        part_dim,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        out.stride(0),
        out.stride(1),
        null_block_id if null_block_id is not None else -1,
        HAS_BIAS=bias is not None,
        HAS_NULL_BLOCK=null_block_id is not None,
        SILU_ACTIVATION=activation in ("silu", "swish", True),
        BLOCK_M=block_m,
        NUM_BLOCKS=num_blocks,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out
