# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the vocab-parallel argmax used by draft-token selection.

Replaces the elementwise chain around the all-gather in
`LogitsProcessor.get_top_tokens` with one kernel on each side of it: `pack_top`
builds the gathered payload, `select_top` consumes it.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_top_kernel(
    logits_ptr,
    out_ptr,
    n_cols,
    stride_m,
    vocab_start,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    best_v = float("-inf")
    best_i = 0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        vals = tl.load(
            logits_ptr + row * stride_m + cols,
            mask=cols < n_cols,
            other=float("-inf"),
        ).to(tl.float32)
        blk_v = tl.max(vals, axis=0)
        blk_i = tl.argmax(vals, axis=0) + off
        # strict > keeps the lowest index on ties, matching Tensor.max
        upd = blk_v > best_v
        best_v = tl.where(upd, blk_v, best_v)
        best_i = tl.where(upd, blk_i, best_i)
    tl.store(out_ptr + row * 2, best_v)
    tl.store(out_ptr + row * 2 + 1, (best_i + vocab_start).to(tl.float32))


@triton.jit
def _select_top_kernel(
    pairs_ptr,
    out_ptr,
    tp_size,
    BLOCK_TP: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_TP)
    base = pairs_ptr + row * tp_size * 2
    live = ranks < tp_size
    vals = tl.load(base + ranks * 2, mask=live, other=float("-inf"))
    idxs = tl.load(base + ranks * 2 + 1, mask=live, other=0.0)
    winner = tl.argmax(vals, axis=0)
    tl.store(out_ptr + row, tl.sum(tl.where(ranks == winner, idxs, 0.0)).to(tl.int64))


def pack_top(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    """Row max of a vocab shard as `[batch, 2]` float32 (value, global index).

    float32 holds vocab indices exactly well past any real vocabulary.
    """
    assert logits.ndim == 2 and logits.stride(1) == 1
    out = torch.empty(logits.shape[0], 2, dtype=torch.float32, device=logits.device)
    if logits.shape[0]:
        _pack_top_kernel[(logits.shape[0],)](
            logits,
            out,
            logits.shape[1],
            logits.stride(0),
            vocab_start,
            BLOCK=4096,
        )
    return out


def select_top(pairs: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Winning token id per row from gathered `[batch, tp_size, 2]` pairs."""
    assert pairs.is_contiguous() and pairs.shape[1:] == (tp_size, 2)
    out = torch.empty(pairs.shape[0], dtype=torch.int64, device=pairs.device)
    if pairs.shape[0]:
        _select_top_kernel[(pairs.shape[0],)](
            pairs,
            out,
            tp_size,
            BLOCK_TP=triton.next_power_of_2(tp_size),
        )
    return out
