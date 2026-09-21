# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return 16 if backend == "flashkda" else None


@dataclass(frozen=True)
class FlashKDAPrefillCheckpointExporter(MambaPrefillCheckpointExporter):
    """Store FlashKDA recurrent and convolution checkpoint states."""

    state_len: int | None = None

    def export(
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *,
        raw_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_checkpoint: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        state_len = (
            self.state_len if self.state_len is not None else conv_state.shape[-1]
        )
        width = raw_qkv.shape[-1]
        recurrent_row_size = recurrent_checkpoint[0].numel()
        block_size = 256
        store_cache_checkpoints_kernel[
            (
                checkpoint.checkpoint_offsets.numel(),
                triton.cdiv(max(width * state_len, recurrent_row_size), block_size),
            )
        ](
            raw_qkv,
            conv_state,
            recurrent_checkpoint,
            recurrent_state,
            cu_seqlens,
            checkpoint.checkpoint_offsets,
            checkpoint.state_indices,
            raw_qkv.stride(0),
            raw_qkv.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            recurrent_checkpoint.stride(0),
            recurrent_state.stride(0),
            checkpoint.checkpoint_offsets.stride(0),
            state_len,
            width,
            recurrent_row_size,
            NULL_BLOCK_ID,
            block_size,
        )


@triton.jit
def store_cache_checkpoints_kernel(
    x_ptr,
    conv_state_ptr,
    recurrent_checkpoint_ptr,
    recurrent_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    checkpoint_state_indices_ptr,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    checkpoint_stride_0: tl.constexpr,
    recurrent_state_stride_0: tl.constexpr,
    checkpoint_offset_stride: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    RECURRENT_ROW_SIZE: tl.constexpr,
    NULL_STATE_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)
    checkpoint_offset = tl.load(
        checkpoint_offsets_ptr + seq_idx * checkpoint_offset_stride
    )
    valid_checkpoint = (state_idx != NULL_STATE_IDX) & (checkpoint_offset > 0)
    valid_conv = (
        (cols < WIDTH * STATE_LEN) & valid_checkpoint & (checkpoint_offset >= STATE_LEN)
    )
    width_idx = cols // STATE_LEN
    history_idx = cols % STATE_LEN
    checkpoint_end = tl.load(query_start_loc_ptr + seq_idx) + checkpoint_offset
    token_idx = checkpoint_end - STATE_LEN + history_idx
    values = tl.load(
        x_ptr + token_idx * x_stride_0 + width_idx * x_stride_1,
        mask=valid_conv,
    )
    tl.store(
        conv_state_ptr
        + state_idx * state_stride_0
        + width_idx * state_stride_1
        + history_idx * state_stride_2,
        values,
        mask=valid_conv,
    )

    valid_recurrent = (cols < RECURRENT_ROW_SIZE) & valid_checkpoint
    recurrent = tl.load(
        recurrent_checkpoint_ptr + seq_idx * checkpoint_stride_0 + cols,
        mask=valid_recurrent,
    )
    tl.store(
        recurrent_state_ptr + state_idx * recurrent_state_stride_0 + cols,
        recurrent,
        mask=valid_recurrent,
    )
