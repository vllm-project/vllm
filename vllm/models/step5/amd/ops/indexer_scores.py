# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA prefill indexer scores from packed FP8 summary pages on ROCm."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["TABLE_ROW", "REGIONS"])
def _paged_scores_kernel(
    Q,
    KV,
    W,
    Lengths,
    Table,
    Output,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    KV_PAGE: tl.constexpr,
    TABLE_ROW,
    TABLE_COL: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    REGIONS,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    FNUZ: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    region = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    valid = (region < REGIONS) & (region < length)
    physical = tl.load(
        Table + batch * TABLE_ROW + (region // PAGE_SIZE) * TABLE_COL,
        valid,
        other=0,
    ).to(tl.int64)
    d = tl.arange(0, 256)
    h = tl.arange(0, 16)
    key = tl.load(
        KV
        + physical[:, None] * KV_PAGE
        + (region % PAGE_SIZE)[:, None] * 256
        + d[None, :],
        valid[:, None],
        other=0,
    )
    query = tl.load(
        Q.to(tl.pointer_type(tl.uint8))
        + batch * Q_BATCH
        + (row % NEXT_N) * Q_NEXT
        + h[None, :] * Q_HEAD
        + d[:, None],
        h[None, :] < HEADS,
        other=0,
    )
    if FNUZ:
        # e4m3fn bits read as e4m3fnuz halve each operand (x4); 0x80 is fnuz NaN.
        key = tl.where(key == 0x80, 0, key).to(tl.float8e4b8, bitcast=True)
        query = tl.where(query == 0x80, 0, query).to(tl.float8e4b8, bitcast=True)
        dot = tl.dot(key, query, out_dtype=tl.float32) * 4.0
    else:
        key = key.to(tl.float8e4nv, bitcast=True)
        query = query.to(tl.float8e4nv, bitcast=True)
        dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
    dot = tl.maximum(dot, 0.0)
    weights = tl.load(W + row * W_ROW + h * W_HEAD, h < HEADS, other=0)
    # Match the four independent FP32 FMA chains used by the decode indexer.
    sums = tl.full((TILE, 4), 0, tl.float32)
    lane = tl.arange(0, 4)
    for group in tl.static_range(HEADS // 4):
        index = group * 4 + lane
        a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
        w = tl.gather(weights, index, 0)
        sums = tl.fma(a, w[None, :], sums)
    a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
    b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
    c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
    e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
    scales = tl.load(
        (KV + physical * KV_PAGE + PAGE_SIZE * 256 + (region % PAGE_SIZE) * 4).to(
            tl.pointer_type(tl.float32)
        ),
        valid,
        other=0,
    )
    result = (((a + b) + c) + e) * scales
    tl.store(
        Output + row * REGIONS + region,
        tl.where(valid, result, float("-inf")),
        region < REGIONS,
    )


def paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Score packed summary pages: all FP8 keys, then one FP32 scale per key."""
    batch, next_n, heads, dim = q.shape
    assert q.dtype == torch.float8_e4m3fn and dim == 256
    assert heads in (4, 8, 16) and q.stride(-1) == 1
    assert kv_fused.dtype == torch.uint8 and kv_fused.is_contiguous()
    assert kv_fused.shape[2:] == (1, 260)
    assert weights.shape == (batch * next_n, heads)
    output = torch.empty(
        (batch * next_n, max_seq_len), dtype=torch.float32, device=q.device
    )
    if batch and max_seq_len:
        _paged_scores_kernel[(batch * next_n, triton.cdiv(max_seq_len, 32))](
            q,
            kv_fused,
            weights,
            seq_lens,
            block_tables,
            output,
            *q.stride()[:3],
            *weights.stride(),
            kv_fused.stride(0),
            *block_tables.stride(),
            seq_lens.stride(0),
            max_seq_len,
            next_n,
            heads,
            kv_fused.shape[1],
            32,
            FNUZ=current_platform.is_fp8_fnuz(),
            num_warps=4,
        )
    return output
