# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DSpark draft context masking under prefix caching.

Cache-restored tokens never flow through the target forward, so the draft's
context KV is never written for them. shift_draft_block_tables hides those
slots from the draft's attention by left-shifting each request's block-table
row by the restored whole blocks (seq_lens is shortened to match by
_prepare_dflash_inputs_kernel).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    shift_draft_block_tables,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA"
)

DEVICE = "cuda"
BLOCK_SIZE = 16
MAX_BLOCKS = 64
MAX_NUM_REQS = 8


def _make_block_table(num_reqs: int) -> torch.Tensor:
    # Distinct block ids per (request, slot) so shifts are detectable.
    table = torch.arange(
        MAX_NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE
    ).view(MAX_NUM_REQS, MAX_BLOCKS)
    return table[:num_reqs].contiguous()


@pytest.mark.parametrize(
    "num_cached,expected_shift",
    [
        (0, 0),  # no cache hit: no-op
        (BLOCK_SIZE * 3, 3),  # block-aligned hit (the common APC case)
        (BLOCK_SIZE * 3 + 5, 3),  # unaligned: floor to whole blocks
        (BLOCK_SIZE - 1, 0),  # less than one block: no-op
    ],
)
def test_shift_single_request(num_cached: int, expected_shift: int):
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,), num_cached, dtype=torch.int32, device=DEVICE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        MAX_BLOCKS * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    kept = MAX_BLOCKS - expected_shift
    torch.testing.assert_close(
        block_table[0, :kept], original[0, expected_shift:]
    )


def test_shift_per_request_and_idx_mapping():
    # Requests in batch order 0..3 map to request-state slots 3..0, with a
    # different cached count per slot. Each row must shift by its own count.
    num_reqs = 4
    block_table = _make_block_table(num_reqs)
    original = block_table.clone()
    idx_mapping = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=DEVICE)
    # Slot i has i whole cached blocks.
    num_cached_tokens = torch.zeros(
        MAX_NUM_REQS, dtype=torch.int32, device=DEVICE
    )
    num_cached_tokens[:4] = (
        torch.arange(4, dtype=torch.int32, device=DEVICE) * BLOCK_SIZE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        MAX_BLOCKS * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    for batch_idx in range(num_reqs):
        shift = int(idx_mapping[batch_idx])  # slot id == cached blocks
        kept = MAX_BLOCKS - shift
        torch.testing.assert_close(
            block_table[batch_idx, :kept],
            original[batch_idx, shift:],
            msg=f"batch row {batch_idx} (slot {shift})",
        )


def test_shift_large_row_in_place_overlap():
    # Shift smaller than the copy chunk (1024) exercises the overlapping
    # in-place load-before-store path on a long row.
    max_blocks = 4096
    block_table = (
        torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)
        .unsqueeze(0)
        .contiguous()
    )
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (1,), 7 * BLOCK_SIZE, dtype=torch.int32, device=DEVICE
    )

    seq_lens = torch.full(
        (idx_mapping.shape[0],),
        max_blocks * BLOCK_SIZE,
        dtype=torch.int32,
        device=DEVICE,
    )
    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(
        block_table[0, : max_blocks - 7], original[0, 7:]
    )


def test_shift_copy_bounded_by_seq_len():
    # Only the blocks referenced by the shifted sequence move; the tail of the
    # row must stay untouched (perf guard for long-context block tables).
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,), 4 * BLOCK_SIZE, dtype=torch.int32, device=DEVICE
    )
    # Shifted draft length of 3.5 blocks -> exactly 4 blocks copied.
    seq_lens = torch.full(
        (1,), 3 * BLOCK_SIZE + BLOCK_SIZE // 2, dtype=torch.int32, device=DEVICE
    )

    shift_draft_block_tables(
        block_table, idx_mapping, num_cached_tokens, seq_lens, BLOCK_SIZE
    )

    torch.testing.assert_close(block_table[0, :4], original[0, 4:8])
    torch.testing.assert_close(block_table[0, 4:], original[0, 4:])
