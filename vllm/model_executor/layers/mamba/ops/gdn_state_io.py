# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_gdn_initial_state_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    elements_per_state,
    total_elements,
    stride_state,
    stride_head,
    stride_row,
    stride_column,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < total_elements
    batch = offset // elements_per_state
    inner = offset % elements_per_state
    head = inner // (ROWS * COLUMNS)
    row = (inner // COLUMNS) % ROWS
    column = inner % COLUMNS

    state_index = tl.load(indices_ptr + batch, mask=mask, other=0).to(tl.int64)
    has_initial_state = tl.load(has_initial_state_ptr + batch, mask=mask, other=0).to(
        tl.int1
    )
    source_offset = (
        state_index * stride_state
        + head * stride_head
        + row * stride_row
        + column * stride_column
    )
    value = tl.load(
        state_ptr + source_offset,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + offset, value.to(tl.float32), mask=mask)


@triton.jit
def _scatter_gdn_final_state_kernel(
    cache_ptr,
    indices_ptr,
    final_state_ptr,
    elements_per_state,
    total_elements,
    stride_cache_state,
    stride_cache_head,
    stride_cache_row,
    stride_cache_column,
    stride_source_batch,
    stride_source_head,
    stride_source_row,
    stride_source_column,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < total_elements
    batch = offset // elements_per_state
    inner = offset % elements_per_state
    head = inner // (ROWS * COLUMNS)
    row = (inner // COLUMNS) % ROWS
    column = inner % COLUMNS

    state_index = tl.load(indices_ptr + batch, mask=mask, other=0).to(tl.int64)
    source_offset = (
        batch * stride_source_batch
        + head * stride_source_head
        + row * stride_source_row
        + column * stride_source_column
    )
    destination_offset = (
        state_index * stride_cache_state
        + head * stride_cache_head
        + row * stride_cache_row
        + column * stride_cache_column
    )
    value = tl.load(final_state_ptr + source_offset, mask=mask)
    tl.store(cache_ptr + destination_offset, value, mask=mask)


def gather_gdn_initial_state(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather, mask, and cast GDN recurrent states to contiguous FP32.

    The state pool must have shape ``[slots, heads, rows, columns]``. ``indices``
    and ``has_initial_state`` contain one entry for each output batch element.
    """
    batch = indices.numel()
    output = torch.empty(
        (batch, *state.shape[1:]), dtype=torch.float32, device=state.device
    )
    elements_per_state = state[0].numel()
    total_elements = output.numel()
    block = 256
    _gather_gdn_initial_state_kernel[(triton.cdiv(total_elements, block),)](
        state,
        indices,
        has_initial_state,
        output,
        elements_per_state,
        total_elements,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        ROWS=state.shape[2],
        COLUMNS=state.shape[3],
        BLOCK=block,
        num_warps=4,
    )
    return output


def scatter_gdn_final_state(
    cache: torch.Tensor,
    indices: torch.Tensor,
    final_state: torch.Tensor,
) -> None:
    """Cast and scatter GDN final states into the recurrent-state cache.

    ``cache`` and ``final_state`` must be four-dimensional with matching head,
    row, and column dimensions. ``indices`` selects one destination cache slot
    for each final-state batch element and may differ from gather source indices.
    """
    elements_per_state = final_state[0].numel()
    total_elements = final_state.numel()
    block = 256
    _scatter_gdn_final_state_kernel[(triton.cdiv(total_elements, block),)](
        cache,
        indices,
        final_state,
        elements_per_state,
        total_elements,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        final_state.stride(3),
        ROWS=cache.shape[2],
        COLUMNS=cache.shape[3],
        BLOCK=block,
        num_warps=4,
    )
