# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_states_kernel(
    state_ptr,
    src_ptr,
    indices_ptr,
    stride_state_batch,
    stride_src_batch,
    stride_indices,
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

    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    values = tl.load(src_ptr + batch_idx * stride_src_batch + offsets, mask=mask)
    tl.store(state_ptr + state_idx * stride_state_batch + offsets, values, mask=mask)


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    Equivalent to ``state[indices] = src`` but non-atomic and bandwidth-bound,
    since mamba cache slots are unique per sequence. ``gather_initial_states``
    is the read-side counterpart.
    """
    assert state.ndim >= 2
    assert state.is_cuda
    assert src.ndim == state.ndim
    assert indices.ndim == 1
    assert indices.device == state.device
    assert src.shape[1:] == state.shape[1:]
    assert src.shape[0] == indices.shape[0]
    assert indices.dtype in (torch.int32, torch.int64)

    row_size = state[0].numel()
    assert state[0].is_contiguous()
    assert src[0].is_contiguous()
    block_size = min(triton.next_power_of_2(row_size), 1024)
    grid = (triton.cdiv(row_size, block_size), indices.numel())
    _scatter_states_kernel[grid](
        state,
        src,
        indices,
        state.stride(0),
        src.stride(0),
        indices.stride(0),
        row_size=row_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
