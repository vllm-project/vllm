# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv half of the ROCm Kimi-K3 prefill checkpoint.

The recurrent half `fused_kda_chunk` snapshots the recurrent state into the
cache row itself as the walk passes the boundary.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 256


@triton.jit
def _store_conv_checkpoints_kernel(
    x_ptr,
    conv_state_ptr,
    query_start_loc_ptr,
    checkpoint_offsets_ptr,
    checkpoint_state_indices_ptr,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    checkpoint_offset_stride: tl.constexpr,
    STATE_LEN: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # int64 because `state_idx * state_stride_0` spans the whole mamba page,
    # which the hybrid allocator pads up to the attention page: on Kimi-K3 that
    # product passes 2**31 a few thousand blocks into a large cache, and an
    # int32 wrap stores outside the tensor.
    state_idx = tl.load(checkpoint_state_indices_ptr + seq_idx).to(tl.int64)
    checkpoint_offset = tl.load(
        checkpoint_offsets_ptr + seq_idx * checkpoint_offset_stride
    )
    # `fused_kda_chunk` disables the export on a negative row rather than on
    # NULL_STATE_IDX as CUDA does, and both halves read the same tensor.
    valid_checkpoint = (state_idx >= 0) & (checkpoint_offset > 0)
    # Below STATE_LEN the window starts in an earlier forward and is no longer
    # in `x`.
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


def store_conv_checkpoints(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    state_len: int,
) -> None:
    """Save each sequence's conv window at its checkpoint offset.

    Args:
        x: pre-convolution projection, ``[tokens, width]``, covering exactly the
            sequences ``query_start_loc`` describes.
        conv_state: paged conv cache, ``[blocks, width, state_len + num_spec]``.
        query_start_loc: per-sequence token starts, ``[N + 1]``.
        checkpoint_offsets: int32 ``[N]``, relative to each sequence's first
            token; ``0`` skips the sequence.
        checkpoint_state_indices: int32 ``[N]`` destination block; negative
            skips the sequence.
        state_len: the convolution's own window, ``conv_kernel_size - 1``. Not
            ``conv_state.shape[-1]``, which speculative decoding widens by
            ``num_spec``: a checkpoint sized off the row would sit ``num_spec``
            tokens in the past, which every reader would silently accept.
    """
    num_seqs = checkpoint_offsets.numel()
    width = conv_state.shape[-2]
    assert x.shape[-1] == width
    assert 0 < state_len <= conv_state.shape[-1]
    _store_conv_checkpoints_kernel[
        (num_seqs, triton.cdiv(width * state_len, _BLOCK_SIZE))
    ](
        x,
        conv_state,
        query_start_loc,
        checkpoint_offsets,
        checkpoint_state_indices,
        x.stride(0),
        x.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        checkpoint_offsets.stride(0),
        state_len,
        width,
        _BLOCK_SIZE,
    )
