# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton launches of the DCP KV prefetch kernels, for their tests.

vLLM launches these kernels only directly, by their loaded handles
(`cuda_direct`). The Triton interpreter on CPU, and a device run that compares a
direct launch with Triton's, launch them through Triton instead, with the grids
and arguments below. The tests that run in a subprocess prepend this file's
source to their script.
"""

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops import kv_ce_rows


def stage_slots(
    restore_idx: torch.Tensor | None,
    slots: torch.Tensor,
    context_start: int,
    interleave: int,
    dcp_size: int,
    per_rank: int,
    new_rows: int | None = None,
) -> None:
    """Enqueue `_stage_slots_kernel` for the tokens `restore_idx` covers (or
    `new_rows` tokens in identity order when `restore_idx` is None), on the
    current stream. `slots` (int64) holds -1 where no token may map."""
    if restore_idx is not None:
        n = restore_idx.shape[0]
        assert restore_idx.dtype == slots.dtype == torch.int64
        assert restore_idx.is_contiguous() and slots.is_contiguous()
        ptr_arg = restore_idx
        has_restore = True
    else:
        assert new_rows is not None
        n = new_rows
        assert slots.dtype == torch.int64 and slots.is_contiguous()
        ptr_arg = slots
        has_restore = False
    if n == 0:
        return
    kv_ce_rows._stage_slots_kernel[(triton.cdiv(n, kv_ce_rows.SLOT_BLOCK),)](
        ptr_arg,
        slots,
        context_start,
        n,
        interleave,
        dcp_size,
        per_rank,
        BLOCK=kv_ce_rows.SLOT_BLOCK,
        HAS_RESTORE=has_restore,
    )


def stage_rows(
    cache: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    region: torch.Tensor,
    max_count: int,
) -> None:
    """Enqueue `_stage_rows_kernel` on the current stream: one program column per
    entry of `counts`, enough blocks for `max_count` rows (host-known; at least
    every count)."""
    assert cache.dtype == torch.uint8 and cache.is_contiguous()
    assert cache.shape[-1] == kv_ce_rows.FP8_ENTRY
    assert sources.dtype == targets.dtype == counts.dtype == torch.int32
    assert sources.numel() == targets.numel() == counts.numel()
    assert region.dtype == torch.bfloat16 and region.shape[1] == kv_ce_rows.ROW_DIM
    if max_count <= 0:
        return
    kv_ce_rows._stage_rows_kernel[
        (triton.cdiv(max_count, kv_ce_rows.STAGE_ROWS), counts.numel())
    ](cache, sources, targets, counts, region, ROWS=kv_ce_rows.STAGE_ROWS)


def plan_check(words: torch.Tensor, deltas: torch.Tensor, slot: int) -> None:
    """Enqueue `_plan_check_kernel` on the current stream. `words` (int64) are
    this rank's plan words and `deltas` (int64) the byte offsets of every rank's
    words from them."""
    assert words.dtype == deltas.dtype == torch.int64
    world = deltas.numel()
    kv_ce_rows._plan_check_kernel[(1,)](
        words.view(torch.uint8),
        deltas,
        slot,
        WORLD=world,
        BLOCK=triton.next_power_of_2(world),
    )
