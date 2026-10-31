# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact small-k top-k over a wide last axis, such as a vocabulary.

torch.topk is slow for a small k over hundreds of thousands of columns. This
splits each row into chunks of GROUPS * GROUP_SIZE columns. A chunk keeps only
the k groups with the largest maxima, since no other group can hold one of the
chunk's top k, and selects its top k from those k * GROUP_SIZE columns. A second
kernel merges the per-chunk candidates.

Values and indices are packed into one int64 key that orders by value, then by
lowest index, so the result is deterministic and ties resolve like a stable
sort. NaN orders above +inf, as in torch.topk.
"""

import torch

from vllm.triton_utils import tl, triton

_GROUPS = 256
_GROUP_SIZE = 16
_CHUNK = _GROUPS * _GROUP_SIZE


@triton.jit
def _pack(x, idx):
    # Order-preserving int32 view of the fp32 value in the high word and the
    # inverted index in the low word, so equal values prefer the lower index.
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return (ordered.to(tl.int64) << 32) | (0xFFFFFFFF - idx.to(tl.int64))


@triton.jit
def _unpack(key):
    ordered = (key >> 32).to(tl.int32)
    bits = ordered ^ ((ordered >> 31) & 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True), 0xFFFFFFFF - (key & 0xFFFFFFFF)


@triton.jit
def _chunk_topk_kernel(
    x_ptr,
    keys_ptr,
    n,
    x_stride,
    num_chunks,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    row_ptr = x_ptr + row.to(tl.int64) * x_stride
    start = chunk * (GROUPS * GROUP_SIZE)
    group_ids = tl.arange(0, GROUPS)
    cols = start + group_ids[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x = tl.load(row_ptr + cols, mask=cols < n, other=float("-inf"))
    group_max = tl.max(x.to(tl.float32), axis=1)
    # Only the K groups with the largest maxima can hold a top-K element. Ties
    # pick the lowest group, which holds the lowest-index candidates.
    _, top_groups = _unpack(tl.topk(_pack(group_max, group_ids), K))
    cand = (start + top_groups * GROUP_SIZE)[:, None] + tl.arange(0, GROUP_SIZE)[
        None, :
    ]
    xc = tl.load(row_ptr + cand, mask=cand < n, other=float("-inf"))
    top = tl.topk(tl.reshape(_pack(xc, cand), (K * GROUP_SIZE,)), K)
    tl.store(keys_ptr + (row * num_chunks + chunk) * K + tl.arange(0, K), top)


@triton.jit
def _merge_topk_kernel(
    keys_ptr,
    values_ptr,
    indices_ptr,
    num_keys,
    k,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    keys = tl.load(
        keys_ptr + row * num_keys + offsets, mask=offsets < num_keys, other=-(2**63)
    )
    values, indices = _unpack(tl.topk(keys, K))
    out = tl.arange(0, K)
    tl.store(
        values_ptr + row * k + out,
        values.to(values_ptr.dtype.element_ty),
        mask=out < k,
    )
    tl.store(indices_ptr + row * k + out, indices, mask=out < k)


def supports_triton_topk(x: torch.Tensor, k: int) -> bool:
    """Whether triton_topk applies, and is faster than torch.topk."""
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.stride(-1) == 1
        and 0 < k <= 64
        and x.shape[-1] >= _CHUNK
    )


def triton_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted top-k along the last axis of a 2D tensor, like torch.topk.

    Returns values in x.dtype and int64 indices. Equal values resolve to the
    lowest index.
    """
    assert supports_triton_topk(x, k)
    num_rows, n = x.shape
    values = torch.empty(num_rows, k, dtype=x.dtype, device=x.device)
    indices = torch.empty(num_rows, k, dtype=torch.int64, device=x.device)
    if num_rows == 0:
        return values, indices

    # tl.topk needs at least two outputs.
    block_k = max(triton.next_power_of_2(k), 2)
    num_chunks = triton.cdiv(n, _CHUNK)
    keys = torch.empty(
        num_rows, num_chunks * block_k, dtype=torch.int64, device=x.device
    )
    # With few rows the grid underfills the device, so give each chunk more
    # warps; with many rows, one warp per chunk keeps occupancy high.
    _chunk_topk_kernel[(num_rows, num_chunks)](
        x,
        keys,
        n,
        x.stride(0),
        num_chunks,
        K=block_k,
        GROUPS=_GROUPS,
        GROUP_SIZE=_GROUP_SIZE,
        num_warps=4 if num_rows <= 32 else 1,
    )
    num_keys = num_chunks * block_k
    _merge_topk_kernel[(num_rows,)](
        keys,
        values,
        indices,
        num_keys,
        k,
        K=block_k,
        BLOCK=triton.next_power_of_2(num_keys),
        num_warps=4,
    )
    return values, indices
