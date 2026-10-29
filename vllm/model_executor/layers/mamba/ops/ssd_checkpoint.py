# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _store_prefill_checkpoint_kernel(
    x_ptr,
    conv_state_ptr,
    chunk_states_ptr,
    ssm_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    state_indices_ptr,
    chunk_idx_ptr,
    stride_x_dim,
    stride_x_token,
    stride_conv_block,
    stride_conv_dim,
    stride_conv_token,
    stride_chunk_states,
    stride_ssm_block,
    DIM: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SSM_ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    checkpoint_offset = tl.load(checkpoint_offsets_ptr + row)
    valid = checkpoint_offset > 0
    dst = tl.load(state_indices_ptr + row).to(tl.int64)

    conv_mask = valid & (cols < DIM * STATE_LEN)
    dim_idx = cols // STATE_LEN
    history_idx = cols % STATE_LEN
    token_idx = (
        tl.load(query_start_loc_ptr + row) + checkpoint_offset - STATE_LEN + history_idx
    ).to(tl.int64)
    conv = tl.load(
        x_ptr + dim_idx * stride_x_dim + token_idx * stride_x_token, mask=conv_mask
    )
    tl.store(
        conv_state_ptr
        + dst * stride_conv_block
        + dim_idx * stride_conv_dim
        + history_idx * stride_conv_token,
        conv,
        mask=conv_mask,
    )

    ssm_mask = valid & (cols < SSM_ROW_SIZE)
    src = tl.load(chunk_idx_ptr + row).to(tl.int64)
    ssm = tl.load(chunk_states_ptr + src * stride_chunk_states + cols, mask=ssm_mask)
    tl.store(ssm_state_ptr + dst * stride_ssm_block + cols, ssm, mask=ssm_mask)


def store_prefill_checkpoint(
    checkpoint: MambaPrefillCheckpointMetadata,
    chunk_idx: torch.Tensor,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    chunk_states: torch.Tensor,
    ssm_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    state_len: int,
) -> None:
    """Write each prefill row's mid-prefill checkpoint into its paged block.

    Rows without a checkpoint (offset 0) write nothing.

    Args:
        checkpoint: Per-row offsets and destination blocks.
        chunk_idx: Per-row index into ``chunk_states`` of the checkpoint.
        x: Pre-conv activations, ``(dim, num_prefill_tokens)``.
        conv_state: Paged conv state, ``(num_blocks, dim, state_len + num_spec)``.
        chunk_states: SSD states at every chunk end.
        ssm_state: Paged SSM state.
        query_start_loc: Prefill query start locations.
        state_len: ``conv_kernel_size - 1``, not the spec-widened conv width.
            Readers take the first ``state_len`` slots as the newest inputs.

    """
    dim = x.size(0)
    ssm_row_size = chunk_states[0].numel()
    block_size = 1024
    grid = (
        chunk_idx.numel(),
        triton.cdiv(max(dim * state_len, ssm_row_size), block_size),
    )
    _store_prefill_checkpoint_kernel[grid](
        x,
        conv_state,
        chunk_states,
        ssm_state,
        query_start_loc,
        checkpoint.checkpoint_offsets,
        checkpoint.state_indices,
        chunk_idx,
        x.stride(0),
        x.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        chunk_states.stride(0),
        ssm_state.stride(0),
        DIM=dim,
        STATE_LEN=state_len,
        SSM_ROW_SIZE=ssm_row_size,
        BLOCK_SIZE=block_size,
    )
