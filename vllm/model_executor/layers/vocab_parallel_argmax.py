# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the exact argmax over vocab-parallel logits shards.

Used by ``LogitsProcessor.get_top_tokens`` (``use_local_argmax_reduction``):

1. ``local_argmax_candidates``: one pass over the rank's ``[B, V_local]``
   logits shard with ``SPLIT`` programs per row. Each writes its best
   ``(value, global id)`` as fp32, which holds every bf16/fp16/fp32 value and
   every id below 2**24 exactly.
2. The caller all-gathers the ``[B, SPLIT * 2]`` candidates along dim 0
   (64 bytes per row and rank).
3. ``pick_from_candidates``: per row, the best of the ``tp_size * SPLIT``
   candidates, as int64 token ids.

Every comparison uses the order of ``torch.argmax``: NaN beats every number
(the first NaN wins), otherwise the larger value wins, ties go to the lower
id, and -0.0 == +0.0. An argmax under a total order with an index tie-break
composes over any partition of the vocabulary, so the result equals
``argmax`` over the concatenated logits bit for bit.
"""

import torch

from vllm.triton_utils import tl, triton

# Programs per row. SPLIT * 2 fp32 = 64 bytes per row keeps the gathered rows
# 16-byte aligned.
SPLIT = 8
BLOCK = 4096


@triton.jit
def _better(v, i, bv, bi):
    nv = v != v
    nb = bv != bv
    return (
        (nv & (~nb))
        | (nv & nb & (i < bi))
        | ((~nv) & (~nb) & ((v > bv) | ((v == bv) & (i < bi))))
    )


@triton.jit
def _combine(v1, i1, v2, i2):
    t = _better(v2, i2, v1, i1)
    return tl.where(t, v2, v1), tl.where(t, i2, i1)


@triton.jit
def _local_argmax_kernel(
    logits_ptr,
    stride_row,
    stride_col,
    n_valid,
    chunk,
    vocab_start,
    out_ptr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Program (row, s) scans [s * chunk, min((s + 1) * chunk, n_valid)) once
    # and writes its best (value, global id) to out[row, s, :]. Empty chunks
    # write (-inf, 2**31), which loses to every real entry.
    row = tl.program_id(0)
    s = tl.program_id(1)
    base = logits_ptr + row.to(tl.int64) * stride_row
    lo = s * chunk
    hi = tl.minimum(lo + chunk, n_valid)
    offs0 = tl.arange(0, BLOCK)
    BIG: tl.constexpr = 2147483647
    bv = tl.full([BLOCK], float("-inf"), tl.float32)
    bi = tl.full([BLOCK], BIG, tl.int32)
    for start in range(lo, hi, BLOCK):
        offs = start + offs0
        msk = offs < hi
        x = tl.load(
            base + offs.to(tl.int64) * stride_col, mask=msk, other=float("-inf")
        )
        x = x.to(tl.float32)
        # Per lane, indices only grow, so "better" means NaN-first or strictly
        # greater; unset lanes take any entry.
        nx = x != x
        nb = bv != bv
        take = msk & ((bi == BIG) | (nx & (~nb)) | ((~nx) & (~nb) & (x > bv)))
        bv = tl.where(take, x, bv)
        bi = tl.where(take, offs, bi)
    v, i = tl.reduce((bv, bi), 0, _combine)
    o = out_ptr + (row * SPLIT + s) * 2
    tl.store(o, v)
    tl.store(o + 1, tl.where(i == BIG, 2147483648.0, (i + vocab_start).to(tl.float32)))


@triton.jit
def _pick_kernel(
    g_ptr,
    num_rows,
    num_candidates,
    out_ptr,
    NCAND: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # g: [tp_size * num_rows, SPLIT * 2] fp32, the dim-0 all-gather of the
    # per-rank [num_rows, SPLIT * 2] candidates.
    row = tl.program_id(0)
    c = tl.arange(0, NCAND)  # candidate = rank * SPLIT + s
    msk = c < num_candidates
    r = c // SPLIT
    s = c % SPLIT
    p = g_ptr + ((r * num_rows + row) * SPLIT + s) * 2
    v = tl.load(p, mask=msk, other=float("-inf"))
    i = tl.load(p + 1, mask=msk, other=2147483648.0)
    _, bi = tl.reduce((v, i), 0, _combine)
    tl.store(out_ptr + row, bi.to(tl.int64))


def local_argmax_candidates(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    """[B, V_local] logits shard (any strides) -> [B, SPLIT * 2] fp32
    (value, global id)."""
    assert logits.dim() == 2
    num_rows, vocab_local = logits.shape
    out = torch.empty(num_rows, SPLIT * 2, dtype=torch.float32, device=logits.device)
    if num_rows:
        _local_argmax_kernel[(num_rows, SPLIT)](
            logits,
            logits.stride(0),
            logits.stride(1),
            vocab_local,
            triton.cdiv(vocab_local, SPLIT),
            vocab_start,
            out,
            SPLIT=SPLIT,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return out


def pick_from_candidates(
    gathered: torch.Tensor, num_rows: int, tp_size: int
) -> torch.Tensor:
    """[tp_size * B, SPLIT * 2] gathered candidates -> [B] int64 token ids."""
    out = torch.empty(num_rows, dtype=torch.int64, device=gathered.device)
    if num_rows:
        num_candidates = tp_size * SPLIT
        _pick_kernel[(num_rows,)](
            gathered,
            num_rows,
            num_candidates,
            out,
            NCAND=triton.next_power_of_2(num_candidates),
            SPLIT=SPLIT,
            num_warps=1,
        )
    return out
