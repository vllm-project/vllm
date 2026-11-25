# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand shared request pages to causal token rows inside the decode graph."""

import copy

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

from .config import MAX_VERIFY_ROWS


@triton.jit
def _expand_single_request_lengths(positions, slots, query_start, out_lengths):
    rows = tl.arange(0, 8)
    slot = tl.load(slots + rows)
    start = tl.load(query_start)
    end = tl.load(query_start + 1)
    active = (slot >= 0) & (start <= rows) & (rows < end)
    position = tl.load(positions + rows)
    tl.store(out_lengths + rows, tl.where(active, position + 1, 0))


@triton.jit
def _expand_verify_metadata(
    positions,
    query_start,
    slots,
    blocks,
    out_blocks,
    out_lengths,
    num_reqs: tl.constexpr,
    source_stride: tl.constexpr,
    source_columns: tl.constexpr,
    output_columns: tl.constexpr,
    block_size: tl.constexpr,
    REQS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.arange(0, REQS)
    start = tl.load(query_start + req, req < num_reqs, other=0)
    end = tl.load(query_start + req + 1, req < num_reqs, other=0)
    owner = tl.max(tl.where((req < num_reqs) & (start <= row) & (row < end), req, -1))
    position = tl.load(positions + row)
    slot = tl.load(slots + row)
    active = (slot >= 0) & (owner >= 0)
    length = tl.where(active, position + 1, 0)
    tl.store(out_lengths + row, length)
    columns = tl.arange(0, COLS)
    page = tl.load(
        blocks + tl.maximum(owner, 0) * source_stride + columns,
        active & (columns < source_columns) & (columns < tl.cdiv(length, block_size)),
        other=0,
    )
    tl.store(
        out_blocks + row * output_columns + columns, page, columns < output_columns
    )


class VerifyMetadata:
    def __init__(self, max_context_len, block_size, device, *, shared_pages=False):
        self.block_size = block_size
        self.shared_pages = shared_pages
        self.blocks = torch.empty(
            (MAX_VERIFY_ROWS, cdiv(max_context_len, block_size)),
            dtype=torch.int32,
            device=device,
        )
        self.lengths = torch.empty(MAX_VERIFY_ROWS, dtype=torch.int32, device=device)

    def supports(self, metadata):
        starts = getattr(metadata, "query_start_loc", None)
        blocks = metadata.block_table
        return (
            starts is not None
            and starts.ndim == 1
            and starts.dtype == torch.int32
            and 2 <= starts.numel() <= 9
            and blocks.ndim == 2
            and blocks.dtype == torch.int32
            and blocks.shape[0] >= starts.numel() - 1
            and 0 < blocks.shape[1] <= self.blocks.shape[1]
            and blocks.stride(1) == 1
        )

    def expand(self, metadata, positions):
        rows = positions.numel()
        blocks = metadata.block_table
        starts = metadata.query_start_loc
        num_reqs = starts.numel() - 1
        result = copy.copy(metadata)
        result.seq_lens = self.lengths[:rows]
        if self.shared_pages and rows == 8 and num_reqs == 1:
            # All verify rows read prefixes of the same request's table.
            # Keep the original graph-stable table and expand only lengths.
            _expand_single_request_lengths[(1,)](
                positions, metadata.slot_mapping, starts, self.lengths
            )
            result.block_table = blocks[:1]
            return result
        _expand_verify_metadata[(rows,)](
            positions,
            starts,
            metadata.slot_mapping,
            blocks,
            self.blocks,
            self.lengths,
            num_reqs,
            blocks.stride(0),
            blocks.shape[1],
            self.blocks.shape[1],
            self.block_size,
            triton.next_power_of_2(num_reqs),
            triton.next_power_of_2(self.blocks.shape[1]),
        )
        result.block_table = self.blocks[:rows]
        return result
