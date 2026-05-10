# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel + tuned constants for the ``swap_blocks_batch`` fast path."""

from __future__ import annotations

from vllm.triton_utils import tl, triton

# Constants tuned empirically on H100 (PCIe Gen5):
#   _NUM_SMS - smallest SM slice that's within 5% of peak bandwidth at
#              the 8-32 KB block sizes that matter in practice
#   _THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA;
#                      above this the C++ cuMemcpyBatchAsync path takes the lead
#   _MIN_N - minimum batch size where Triton's per-launch cost is amortized;
#            below this DMA wins
_NUM_SMS = 12
_THRESHOLD_BYTES = 28 * 1024
_MIN_N = 16


@triton.jit
def _swap_blocks_kernel(
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
