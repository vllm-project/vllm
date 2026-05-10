# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for ``swap_blocks_batch`` on small batches."""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

_NUM_SMS = 12
_THRESHOLD_BYTES = 28 * 1024
_MIN_N = 16


@triton.jit
def _kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    gpu_to_cpu: bool = True,
) -> None:
    """Drop-in replacement for ``ops.swap_blocks_batch`` with Triton fast path.

    ``gpu_to_cpu`` gates the Triton path: only ``False`` (i.e., CPU->GPU
    reads) takes it. The dedicated copy engine matches or beats SM-issued
    stores on the GPU->CPU direction, so writes always defer to the C++
    DMA path. Default ``True`` is the safe choice for callers that don't
    know the direction.

    ``is_src_access_order_any`` is forwarded to the C++ DMA path on
    fallback (controls ``CU_MEMCPY_SRC_ACCESS_ORDER_ANY`` for the
    cuMemcpyBatchAsync attributes); it does not affect the Triton path.
    """
    n = src_addrs.numel()
    if n == 0:
        return
    if gpu_to_cpu or n < _MIN_N:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    max_bpj = int(sizes.max().item())
    if max_bpj >= _THRESHOLD_BYTES or bool((sizes % 8 != 0).any().item()):
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    chunk = min(triton.next_power_of_2(max_bpj), 8192)
    _kernel[(min(_NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        sizes.to("cuda", non_blocking=True),
        n,
        BYTES_PER_CHUNK=chunk,
    )
