# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage and gather token-sharded index-K pages for prefill."""

from vllm.triton_utils import tl, triton


# Runtime integers are not value-specialized: a new divisibility class would
# otherwise JIT-compile mid-step on one rank while its peers wait in the pull.
@triton.jit(do_not_specialize=["kv_page_stride", "num_programs", "rank", "epoch"])
def _index_k_stage_pages_kernel(
    kv_ptr,
    kv_page_stride,
    pages_ptr,
    stage_ptr,
    counter_ptr,
    num_programs,
    flag_bases_ptr,
    rank,
    epoch,
    W: tl.constexpr,
    PAGE_WORDS: tl.constexpr,
):
    p = tl.program_id(0)
    page = tl.load(pages_ptr + p).to(tl.int64)
    offs = tl.arange(0, PAGE_WORDS)
    vals = tl.load(kv_ptr + page * kv_page_stride + offs)
    tl.store(stage_ptr + p.to(tl.int64) * PAGE_WORDS + offs, vals)
    # Every thread's rows are stored before this program counts itself done;
    # the last one resets the counter and publishes the epoch to every rank.
    tl.debug_barrier()
    done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="sys")
    if done == num_programs - 1:
        tl.atomic_xchg(counter_ptr, 0, sem="relaxed", scope="gpu")
        for r in tl.static_range(W):
            peer = tl.load(flag_bases_ptr + r).to(tl.pointer_type(tl.int64))
            tl.atomic_xchg(peer + rank, epoch, sem="release", scope="sys")


@triton.jit(do_not_specialize=["epoch", "stage_off"])
def _index_k_pull_pages_kernel(
    flags_ptr,
    epoch,
    stage_bases_ptr,
    stage_off,
    out_ptr,
    W: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    # No timeout: reading a peer's staging buffer before it publishes would
    # return stale rows, so a stalled peer stalls this rank instead.
    for r in tl.static_range(W):
        seen = tl.atomic_add(flags_ptr + r, 0, sem="acquire", scope="sys")
        while seen < epoch:
            seen = tl.atomic_add(flags_ptr + r, 0, sem="acquire", scope="sys")
    tl.debug_barrier()
    p = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, ROWS * W)
    i = j // W
    src_rank = j % W
    c = tl.arange(0, ROW_WORDS)
    base = tl.load(stage_bases_ptr + src_rank).to(tl.pointer_type(tl.int32))
    row = stage_off + p * (ROWS * ROW_WORDS) + i * ROW_WORDS
    vals = tl.load(base[:, None] + row[:, None] + c[None, :])
    dst = out_ptr + p * (ROWS * W * ROW_WORDS) + j[:, None] * ROW_WORDS + c[None, :]
    tl.store(dst, vals)
