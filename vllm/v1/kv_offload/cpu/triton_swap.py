# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for ``swap_blocks_batch`` on small uniform batches."""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

_NUM_SMS = 12
_THRESHOLD_BYTES = 28 * 1024


@triton.jit
def _kernel(
    src_addrs,
    dst_addrs,
    n_jobs,  # type: ignore[name-defined]
    bytes_per_job,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    words = bytes_per_job // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
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
) -> None:
    """Drop-in replacement for ``ops.swap_blocks_batch`` with Triton fast path.

    The ``is_src_access_order_any`` flag is forwarded to the C++ DMA path
    on fallback (controls ``CU_MEMCPY_SRC_ACCESS_ORDER_ANY`` for the
    cuMemcpyBatchAsync attributes). It does not apply to the Triton path —
    SM-issued copies don't go through the cuMemcpy descriptor pipeline.
    """
    n = src_addrs.numel()
    if n == 0:
        return
    bpj = int(sizes[0].item())
    if bpj >= _THRESHOLD_BYTES or bpj % 8 != 0 or not bool((sizes == bpj).all()):
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    chunk = min(triton.next_power_of_2(bpj), 8192)
    _kernel[(min(_NUM_SMS, n),)](
        src_addrs.to("cuda", non_blocking=True),
        dst_addrs.to("cuda", non_blocking=True),
        n,
        bpj,
        BYTES_PER_CHUNK=chunk,
    )
