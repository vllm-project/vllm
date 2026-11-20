# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kitty write path: seal completed pages and write the fp16 residual.

The quantize-and-pack kernels are ports of Kitty's `quantize_pack_k_kernel`
and `quantize_pack_v_kernel` (https://github.com/Summer-Summer/Kitty), with
one program per (seal job, KV head) and the scale/min stored inside the page.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kitty.tiles import (
    K_TAIL_ROW,
    KITTY_PAGE_SIZE,
    V_SINK_ROW,
    V_TAIL_ROW,
    kitty_first_unsealed_k,
    kitty_first_unsealed_v,
)


@dataclass
class KittyStepPlan:
    """Per-step write-path plan, computed on the CPU from exact lengths."""

    old_lens: np.ndarray  # [num_reqs]
    # [J, 2] (request, block) per K / V block to seal this step.
    k_jobs: np.ndarray
    v_jobs: np.ndarray
    # [3, n] (token row, request, residual row) per residual K / V write.
    k_writes: np.ndarray
    v_writes: np.ndarray


def _block_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """All (request, block) pairs with block in [starts[r], ends[r])."""
    counts = np.maximum(ends - starts, 0)
    reqs = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
    offsets = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return np.stack([reqs, starts[reqs] + offsets], axis=1)


def plan_kitty_step(query_start_loc: np.ndarray, seq_lens: np.ndarray) -> KittyStepPlan:
    """Seal jobs and residual writes for one step, from the query start
    offsets and the cache lengths after the step."""
    qsl = query_start_loc.astype(np.int64)  # query start offsets, per request
    new = seq_lens.astype(np.int64)  # cached tokens per request AFTER this step
    query_lens = np.diff(qsl)  # number of query tokens per request
    old = new - query_lens  # cached tokens per request BEFORE this step

    # K and V blocks to seal per job, after this step
    k_jobs = _block_ranges(kitty_first_unsealed_k(old), kitty_first_unsealed_k(new))
    v_jobs = _block_ranges(kitty_first_unsealed_v(old), kitty_first_unsealed_v(new))

    num_tokens = int(qsl[-1])  # total number of tokens to process in this step
    tok = np.arange(num_tokens, dtype=np.int64)  # token ID for this step
    # request IDs that each token belongs to
    req = np.repeat(np.arange(len(new), dtype=np.int64), query_lens)
    # the actual position of the tokens, per each request
    pos = old[req] + tok - qsl[req]
    sink = pos < KITTY_PAGE_SIZE  # mask to get sink tokens

    # identify keys and values to keep in the residual, and their corresponding
    # rows in the residual cache
    k_keep = sink | (pos >= KITTY_PAGE_SIZE * kitty_first_unsealed_k(new[req]))
    k_rows = np.where(sink, pos, K_TAIL_ROW + pos % KITTY_PAGE_SIZE)
    v_keep = sink | (pos >= KITTY_PAGE_SIZE * kitty_first_unsealed_v(new[req]))
    v_rows = np.where(sink, V_SINK_ROW + pos, V_TAIL_ROW + pos % (2 * KITTY_PAGE_SIZE))

    return KittyStepPlan(
        old_lens=old,
        k_jobs=k_jobs,
        v_jobs=v_jobs,
        k_writes=np.stack([tok[k_keep], req[k_keep], k_rows[k_keep]]),
        v_writes=np.stack([tok[v_keep], req[v_keep], v_rows[v_keep]]),
    )


def kitty_gather_blocks(
    src: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    jobs: torch.Tensor,
    old_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    tail_row: int,
    ring: int,
) -> torch.Tensor:
    """The 128 fp16 tokens of each job's block as [J, H, 128, D].

    Positions below the request's old length come from its residual tail ring
    (row `tail_row + p % ring`), the rest from this step's `src` [N, H, D].
    """
    reqs = jobs[:, 0].long()
    pos = jobs[:, 1].long()[:, None] * KITTY_PAGE_SIZE + torch.arange(
        KITTY_PAGE_SIZE, device=jobs.device
    )
    old = old_lens[reqs].long()[:, None]
    step_rows = (query_start_loc[reqs].long()[:, None] + pos - old).clamp_(
        0, src.shape[0] - 1
    )
    res_rows = tail_row + pos % ring
    from_step = src[step_rows]
    from_res = residual[residual_block_ids[reqs].long()[:, None], :, res_rows]
    x = torch.where((pos < old)[:, :, None, None], from_res, from_step)
    return x.transpose(1, 2).contiguous()


def kitty_boost_ranks(k_blocks: torch.Tensor, d_boost: int) -> torch.Tensor:
    """Rank table [J, H, D] uint8: top-d_boost channels by mean |K| get their
    rank in topk order, the others get d_boost."""
    score = k_blocks.float().abs().mean(dim=2)
    _, top = score.topk(d_boost, dim=-1)
    ranks = torch.full_like(score, d_boost, dtype=torch.uint8)
    order = torch.arange(d_boost, device=score.device, dtype=torch.uint8)
    return ranks.scatter_(-1, top, order.expand_as(top))


@triton.jit
def _kitty_seal_k_kernel(
    x_ptr,
    stride_xj,
    stride_xh,
    rank_ptr,
    stride_rj,
    stride_rh,
    jobs_ptr,
    block_table_ptr,
    stride_bt,
    cache_ptr,
    stride_cb,
    stride_ch,
    D: tl.constexpr,
    DB: tl.constexpr,
):
    off_khi: tl.constexpr = 64 * D
    off_ksm: tl.constexpr = 64 * D + 32 * DB
    off_rank: tl.constexpr = off_ksm + 4 * D + 512
    j = tl.program_id(0)
    h = tl.program_id(1)
    req = tl.load(jobs_ptr + 2 * j)
    blk = tl.load(jobs_ptr + 2 * j + 1)
    page_id = tl.load(block_table_ptr + req.to(tl.int64) * stride_bt + blk)
    page_ptr = cache_ptr + page_id.to(tl.int64) * stride_cb + h * stride_ch

    offs_t = tl.arange(0, 128)
    offs_d = tl.arange(0, D)
    x = tl.load(
        x_ptr + j * stride_xj + h * stride_xh + offs_t[:, None] * D + offs_d[None, :]
    ).to(tl.float32)
    rank = tl.load(rank_ptr + j * stride_rj + h * stride_rh + offs_d)
    boosted = rank < DB

    x_max = tl.max(x, axis=0)
    x_min = tl.min(x, axis=0)
    qmax = tl.where(boosted, 15.0, 3.0)
    scale = tl.maximum(tl.math.div_rn(x_max - x_min, qmax), 1e-6).to(tl.float16)
    x_min = x_min.to(tl.float16)
    sm_ptr = (page_ptr + off_ksm).to(tl.pointer_type(tl.float16))
    tl.store(sm_ptr + 2 * offs_d, scale)
    tl.store(sm_ptr + 2 * offs_d + 1, x_min)

    x_f = tl.math.div_rn(
        x - x_min.to(tl.float32)[None, :], scale.to(tl.float32)[None, :]
    )
    q = tl.minimum(tl.maximum(tl.floor(x_f + 0.5), 0.0), qmax[None, :])
    q = q.to(tl.int32)

    # Four 2-bit codes per byte along tokens, first token in the lowest bits.
    shifts = (offs_t % 4) * 2
    low = tl.sum(tl.reshape((q & 3) << shifts[:, None], (32, 4, D)), axis=1)
    high = tl.sum(tl.reshape(((q >> 2) & 3) << shifts[:, None], (32, 4, D)), axis=1)
    offs_byte = tl.arange(0, 32)
    tl.store(page_ptr + offs_d[None, :] * 32 + offs_byte[:, None], low.to(tl.uint8))
    tl.store(
        page_ptr + off_khi + rank[None, :].to(tl.int32) * 32 + offs_byte[:, None],
        high.to(tl.uint8),
        mask=boosted[None, :],
    )
    tl.store(page_ptr + off_rank + offs_d, rank)


@triton.jit
def _kitty_seal_v_kernel(
    x_ptr,
    stride_xj,
    stride_xh,
    jobs_ptr,
    block_table_ptr,
    stride_bt,
    cache_ptr,
    stride_cb,
    stride_ch,
    D: tl.constexpr,
    DB: tl.constexpr,
):
    off_v: tl.constexpr = 32 * D
    off_vsm: tl.constexpr = 68 * D + 32 * DB
    j = tl.program_id(0)
    h = tl.program_id(1)
    req = tl.load(jobs_ptr + 2 * j)
    blk = tl.load(jobs_ptr + 2 * j + 1)
    page_id = tl.load(block_table_ptr + req.to(tl.int64) * stride_bt + blk)
    page_ptr = cache_ptr + page_id.to(tl.int64) * stride_cb + h * stride_ch

    offs_t = tl.arange(0, 128)
    offs_d = tl.arange(0, D)
    x = tl.load(
        x_ptr + j * stride_xj + h * stride_xh + offs_t[:, None] * D + offs_d[None, :]
    ).to(tl.float32)

    x_max = tl.max(x, axis=1)
    x_min = tl.min(x, axis=1)
    scale = tl.maximum(tl.math.div_rn(x_max - x_min, 3.0), 1e-6).to(tl.float16)
    x_min = x_min.to(tl.float16)
    sm_ptr = (page_ptr + off_vsm).to(tl.pointer_type(tl.float16))
    tl.store(sm_ptr + 2 * offs_t, scale)
    tl.store(sm_ptr + 2 * offs_t + 1, x_min)

    x_f = tl.math.div_rn(
        x - x_min.to(tl.float32)[:, None], scale.to(tl.float32)[:, None]
    )
    q = tl.minimum(tl.maximum(tl.floor(x_f + 0.5), 0.0), 3.0).to(tl.int32)

    # Four 2-bit codes per byte along channels, first channel in the lowest bits.
    shifts = (offs_d % 4) * 2
    packed = tl.sum(tl.reshape(q << shifts[None, :], (128, D // 4, 4)), axis=2)
    offs_byte = tl.arange(0, D // 4)
    tl.store(
        page_ptr + off_v + offs_t[:, None] * (D // 4) + offs_byte[None, :],
        packed.to(tl.uint8),
    )


def kitty_seal(
    key: torch.Tensor,
    value: torch.Tensor,
    packed_cache: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    block_table: torch.Tensor,
    k_jobs: torch.Tensor,
    v_jobs: torch.Tensor,
    old_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    d_boost: int,
) -> None:
    """Quantize this step's completed K and V blocks into the packed cache.

    key/value are this step's [N, H, D] K/V, packed_cache is [num_blocks, H,
    128, C] uint8, residual [num_res_blocks, H, 640, D], and k_jobs/v_jobs are
    [J, 2] (request, block) pairs.
    """
    num_heads, head_size = key.shape[1], key.shape[2]
    assert (
        packed_cache.stride(-1) == 1
        and packed_cache.stride(-2) == (packed_cache.shape[-1])
    )
    common = dict(
        residual=residual,
        residual_block_ids=residual_block_ids,
        old_lens=old_lens,
        query_start_loc=query_start_loc,
    )
    cache_args = (
        block_table,
        block_table.stride(0),
        packed_cache,
        packed_cache.stride(0),
        packed_cache.stride(1),
    )
    if k_jobs.shape[0] > 0:
        k_blocks = kitty_gather_blocks(
            key, jobs=k_jobs, tail_row=K_TAIL_ROW, ring=KITTY_PAGE_SIZE, **common
        )
        ranks = kitty_boost_ranks(k_blocks, d_boost)
        _kitty_seal_k_kernel[(k_jobs.shape[0], num_heads)](
            k_blocks,
            k_blocks.stride(0),
            k_blocks.stride(1),
            ranks,
            ranks.stride(0),
            ranks.stride(1),
            k_jobs,
            *cache_args,
            D=head_size,
            DB=d_boost,
        )
    if v_jobs.shape[0] > 0:
        v_blocks = kitty_gather_blocks(
            value, jobs=v_jobs, tail_row=V_TAIL_ROW, ring=2 * KITTY_PAGE_SIZE, **common
        )
        _kitty_seal_v_kernel[(v_jobs.shape[0], num_heads)](
            v_blocks,
            v_blocks.stride(0),
            v_blocks.stride(1),
            v_jobs,
            *cache_args,
            D=head_size,
            DB=d_boost,
        )


def kitty_write_residual(
    key: torch.Tensor,
    value: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    k_writes: torch.Tensor,
    v_writes: torch.Tensor,
) -> None:
    """Copy the new tokens that stay fp16 (sink, K tail, V tail) into the
    residual. `k_writes`/`v_writes` are [3, n] (token, request, row)."""
    for src, writes in ((key, k_writes), (value, v_writes)):
        if writes.shape[1] == 0:
            continue
        tok, req, row = writes
        residual[residual_block_ids[req].long(), :, row] = src[tok]
