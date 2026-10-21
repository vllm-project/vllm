# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _gather_initial_states_launch(state, indices, valid, output, row_size):
    block_size = min(triton.next_power_of_2(row_size), 1024)
    return (triton.cdiv(row_size, block_size), indices.shape[0]), dict(
        state_ptr=state,
        indices_ptr=indices,
        has_initial_state_ptr=valid,
        output_ptr=output,
        stride_state_batch=state.stride(0),
        stride_indices=indices.stride(0),
        stride_has_initial_state=valid.stride(0),
        row_size=row_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )


def _gather_initial_states_warmup_inputs(*, row_size, state_dtype):
    # Mamba2 masks are fresh contiguous comparison outputs. Cache pages can
    # have padding; block-table column views have scalar stride classes 1,
    # generic, or divisible-by-16, and either pointer-alignment class.
    for state_stride, index_stride, aligned in itertools.product(
        (row_size, row_size + 1), (1, 2, 16), (True, False)
    ):
        grid, args = _gather_initial_states_launch(
            TritonWarmupTensor(
                state_dtype, shape=(2, row_size), strides=(state_stride, 1)
            ),
            TritonWarmupTensor(
                torch.int32, aligned=aligned, shape=(2,), strides=(index_stride,)
            ),
            TritonWarmupTensor(torch.bool, shape=(2,)),
            TritonWarmupTensor(state_dtype, shape=(2, row_size)),
            row_size,
        )
        yield dict(grid=grid, **args)


@triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_gather_initial_states_warmup_inputs
)
@triton.jit
def _gather_initial_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    stride_state_batch,
    stride_indices,
    stride_has_initial_state,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    has_initial_state = tl.load(
        has_initial_state_ptr + batch_idx * stride_has_initial_state
    ).to(tl.int1)
    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    state_idx = tl.where(has_initial_state, state_idx, 0)
    values = tl.load(
        state_ptr + state_idx * stride_state_batch + offsets,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + batch_idx * row_size + offsets, values, mask=mask)


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Gather dense state rows, replacing uninitialized rows with zeros."""
    assert state.ndim >= 2
    assert state.is_cuda or state.is_xpu
    assert indices.ndim == 1 and has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == state.device
    assert has_initial_state.device == state.device
    assert indices.dtype in (torch.int32, torch.int64)
    assert has_initial_state.dtype == torch.bool

    row_size = state[0].numel()
    # Mamba pages may pad stride(0), but each state row remains dense.
    assert state[0].is_contiguous()
    output = torch.empty(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype,
        device=state.device,
    )
    grid, args = _gather_initial_states_launch(
        state, indices, has_initial_state, output, row_size
    )
    _gather_initial_states_kernel[grid](**args)
    return output
