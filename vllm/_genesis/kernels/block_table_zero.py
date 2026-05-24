# SPDX-License-Identifier: Apache-2.0
"""block_table tail zero-fill helper (Patch 14 / vLLM PR #39591, issue #39589).

Problem
-------
When a `block_table` row slot is reused by a shorter request after a longer
one, stale block IDs linger in the tail (positions ≥ num_blocks_per_row).
FlashInfer's `_copy_page_indices_kernel` can read past `num_blocks_per_row`,
hitting KV memory belonging to a concurrent live request. Result: silent
divergent output at `temperature=0` across retries, impossible to debug
without knowing the race exists.

Fix
---
After every `append_row` / `move_row` on the block table, zero the tail so
any out-of-bounds kernel read returns 0 (safe sentinel).

Integration
-----------
The apply_all orchestrator attaches a monkey-patch that wraps
`vllm.v1.worker.block_table.BlockTable.append_row` + `move_row` to call
`zero_block_table_tail` after the base implementation runs.

This module provides the pure-numpy helper so the logic is:
  - easily unit-testable without a running vLLM engine
  - identical on any numpy dtype / backend (CPU/GPU torch view)

Platform compatibility
----------------------
  All platforms ✅ — pure numpy/torch indexing, no CUDA-specific code.

Credits
-------
  - vLLM PR #39591 — upstream fix author
  - vLLM issue #39589 — original repro
  - Genesis Patch 14 — mirror into runtime patcher

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

from typing import Any


def zero_block_table_tail(
    block_table_np: Any,
    row_idx: int,
    end: int,
    max_per_req: int,
) -> None:
    """Zero `block_table_np[row_idx, end:max_per_req]` in-place.

    Safe no-op when `end >= max_per_req` (nothing to zero).

    Args:
        block_table_np: numpy ndarray shape (num_rows, max_per_req), int dtype.
        row_idx: row to clear tail of.
        end: first column index to zero from (inclusive).
        max_per_req: one past the last column index (exclusive upper bound).

    Raises:
        IndexError: row_idx out of range (caller bug).
        ValueError: end < 0 or max_per_req < end (caller bug).
    """
    if end < 0:
        raise ValueError(
            f"[Genesis P14] zero_block_table_tail got end={end} (must be >= 0)"
        )
    if max_per_req < end:
        raise ValueError(
            f"[Genesis P14] zero_block_table_tail got max_per_req={max_per_req} "
            f"< end={end}"
        )
    if end >= max_per_req:
        return  # nothing to zero
    block_table_np[row_idx, end:max_per_req] = 0


def append_row_with_tail_zero(
    block_table_np: Any,
    num_blocks_per_row: Any,
    row_idx: int,
    block_ids: list[int],
    max_per_req: int,
) -> None:
    """Combined append_row + tail-zero for direct tests and monkey-patch use.

    This matches the post-PR-#39591 upstream behavior exactly.

    Args:
        block_table_np: numpy ndarray shape (num_rows, max_per_req).
        num_blocks_per_row: numpy ndarray shape (num_rows,), int.
        row_idx: target row.
        block_ids: list of block IDs to append.
        max_per_req: upper bound of columns per request.
    """
    if len(block_ids) == 0:
        return
    num_blocks = len(block_ids)
    start = int(num_blocks_per_row[row_idx])
    end = start + num_blocks
    num_blocks_per_row[row_idx] = end
    block_table_np[row_idx, start:end] = block_ids
    # PR #39591 / issue #39589: zero the tail so kernels scanning past
    # num_blocks_per_row cannot read stale IDs from a prior request.
    zero_block_table_tail(block_table_np, row_idx, end, max_per_req)


def move_row_with_tail_zero(
    block_table_np: Any,
    num_blocks_per_row: Any,
    src: int,
    tgt: int,
    max_per_req: int,
) -> None:
    """Move src → tgt and zero tgt's tail.

    If tgt previously held more blocks than src, the tail of tgt's old entry
    would leak past num_blocks_per_row[tgt] without this zero.
    """
    num_blocks = int(num_blocks_per_row[src])
    block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
    zero_block_table_tail(block_table_np, tgt, num_blocks, max_per_req)
    num_blocks_per_row[tgt] = num_blocks
