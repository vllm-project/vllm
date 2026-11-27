# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-top-k sampling: select the top-k columns of each row once, then run
temperature, top-p and Gumbel-max sampling over those k survivors only.

The stock path masks the whole vocabulary (an iterative pivot search over the
fp32 logits) and then runs a Gumbel argmax over every column. When every
request has a small top_k, the work after the selection touches k values per
row. The Gumbel noise is keyed by (seed, position, token id) exactly as in
`gumbel_sample`, so the sampled token is the one the stock path would return
for the same support set.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# Largest per-request top_k the bounded path serves. Selection cost grows with
# k, and beyond this the stock full-vocab kernel is competitive.
BOUNDED_TOPK_MAX = 128


@triton.jit
def _bounded_topk_sample_kernel(
    sampled_ptr,
    logits_ptr,
    logits_stride,
    idx_ptr,
    idx_stride,
    expanded_idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    KMAX: tl.constexpr,
    KBLOCK: tl.constexpr,
    USE_TOP_P: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    cols = tl.arange(0, KBLOCK)
    in_range = cols < KMAX
    ids = tl.load(idx_ptr + row * idx_stride + cols, mask=in_range, other=0)
    ids = ids.to(tl.int64)
    top_k = tl.load(top_k_ptr + req)
    vals = tl.load(
        logits_ptr + row * logits_stride + ids, mask=in_range, other=float("-inf")
    )
    vals = vals.to(tl.float32)

    # The selector returns the KMAX best columns in no particular order; a
    # request with top_k < KMAX keeps only the top_k best of them (ties broken
    # by column position).
    cols_row = tl.broadcast_to(cols[None, :], (KBLOCK, KBLOCK))
    vals_row = tl.broadcast_to(vals[None, :], (KBLOCK, KBLOCK))
    better = (vals_row > vals[:, None]) | (
        (vals_row == vals[:, None]) & (cols_row < cols[:, None])
    )
    rank = tl.sum(better.to(tl.int32), axis=1)
    live = in_range & (rank < top_k)
    vals = tl.where(live, vals, float("-inf"))

    temp = tl.load(temp_ptr + req).to(tl.float32)
    if temp != 0.0 and temp != 1.0:
        vals = vals / temp

    if USE_TOP_P:
        top_p = tl.load(top_p_ptr + req)
        row_max = tl.max(vals, axis=0)
        e = tl.exp(vals - row_max)
        e = tl.where(live, e, 0.0)
        probs = e / tl.sum(e, axis=0)
        # Mass of the tokens ranked strictly above each token; a token is kept
        # while that mass is below top_p.
        probs_row = tl.broadcast_to(probs[None, :], (KBLOCK, KBLOCK))
        vals_row = tl.broadcast_to(vals[None, :], (KBLOCK, KBLOCK))
        higher = vals_row > vals[:, None]
        above = tl.sum(tl.where(higher, probs_row, 0.0), axis=1)
        keep = (above < top_p) | (top_p >= 1.0)
        vals = tl.where(keep, vals, float("-inf"))
        live = live & keep

    seed = tl.load(seeds_ptr + req)
    pos = tl.load(pos_ptr + row)
    _, arg = gumbel_noised_argmax(
        vals,
        ids,
        live,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    sampled = tl.sum(tl.where(cols == arg, ids, 0), axis=0)
    # Greedy requests take the lowest token id among tied maxima, as the stock
    # path does; the selector's column order is not stable between calls.
    best = tl.max(vals, axis=0)
    greedy = tl.min(tl.where(vals == best, ids, 0x7FFFFFFFFFFFFFFF), axis=0)
    tl.store(sampled_ptr + row, tl.where(temp == 0.0, greedy, sampled))


def bounded_topk_available() -> bool:
    try:
        from aiter.ops.topk_select import topk_select  # noqa: F401
    except Exception:
        return False
    return True


def bounded_topk_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size] bf16/fp16/fp32, unmodified
    max_top_k: int,
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    top_k: torch.Tensor,  # [max_num_reqs] int32
    top_p: torch.Tensor | None,  # [max_num_reqs] float32, None if all 1.0
    temperature: torch.Tensor,  # [max_num_reqs]
    seeds: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    from aiter.ops.topk_select import topk_select

    num_tokens = logits.shape[0]
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    select_in = logits if logits.dtype == torch.float32 else logits.float()
    _, idx = topk_select(select_in, max_top_k)
    sampled = torch.empty(num_tokens, dtype=torch.int64, device=logits.device)
    _bounded_topk_sample_kernel[(num_tokens,)](
        sampled,
        logits,
        logits.stride(0),
        idx,
        idx.stride(0),
        expanded_idx_mapping,
        top_k,
        top_p if top_p is not None else top_k,
        temperature,
        seeds,
        pos,
        KMAX=max_top_k,
        KBLOCK=triton.next_power_of_2(max_top_k),
        USE_TOP_P=top_p is not None,
        USE_FP64=use_fp64,
    )
    return sampled
