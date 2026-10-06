# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)


def compute_mamba_prefill_checkpoints(
    seq_lens: list[int],
    query_lens: list[int],
    hash_block_size: int,
    mamba_block_size: int,
    checkpoint_alignment: int | None,
    drop_eagle_block: bool,
) -> tuple[list[int], list[int]]:
    """Per-row internal prefill checkpoint offsets and cache block columns.

    Backends call this instead of re-deriving the rules, so they decline in
    lockstep with the scheduler and ``MambaManager``: allocating a checkpoint
    block without writing it leaves the prefix cache serving uninitialized
    state.

    Returns:
        ``(offsets, cols)``: the checkpoint's offset into each row's query and
        its block-table column. ``0`` and ``-1`` mean the row has none.

    """
    offsets: list[int] = []
    cols: list[int] = []
    for seq_len, query_len in zip(seq_lens, query_lens):
        query_start = seq_len - query_len
        position = get_mamba_prefill_checkpoint_position(
            seq_len, hash_block_size, drop_eagle_block=drop_eagle_block
        )
        valid = is_mamba_prefill_checkpoint_valid(
            query_start=query_start,
            query_end=seq_len,
            checkpoint_position=position,
            hash_block_size=hash_block_size,
            mamba_block_size=mamba_block_size,
            checkpoint_alignment=checkpoint_alignment,
        )
        offsets.append(position - query_start if valid else 0)
        cols.append(cdiv(seq_len, mamba_block_size) - 2 if valid else -1)
    return offsets, cols


def _gather_checkpoint_state_indices(
    block_table: torch.Tensor, request_rows: torch.Tensor, block_cols: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        block_cols >= 0, block_table[request_rows, block_cols], NULL_BLOCK_ID
    )


@dataclass
class MambaPrefillCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor
    # Host copy of ``checkpoint_offsets``, for planning kernel launches
    # without a device-to-host sync.
    offsets: list[int] | None = None
    # Gather indices into the block table, kept so that another KV cache group
    # with the same spec can re-derive its own ``state_indices``.
    request_rows: torch.Tensor | None = None
    block_cols: torch.Tensor | None = None

    def regather_state_indices(
        self, block_table: torch.Tensor
    ) -> "MambaPrefillCheckpointMetadata":
        assert self.request_rows is not None and self.block_cols is not None
        return replace(
            self,
            state_indices=_gather_checkpoint_state_indices(
                block_table, self.request_rows, self.block_cols
            ),
        )


class MambaPrefillCheckpointBuilder:
    """Build backend-neutral prefill checkpoint locations."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_spec: MambaSpec) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec

    def build(
        self,
        m: CommonAttentionMetadata,
        request_rows: list[int],
    ) -> MambaPrefillCheckpointMetadata | None:
        if self.vllm_config.cache_config.mamba_cache_mode != "align":
            return None
        if self.kv_cache_spec.num_prefill_checkpoint_blocks == 0:
            return None
        assert m.seq_lens_cpu_upper_bound is not None
        all_query_lens = m.query_start_loc_cpu.diff().tolist()
        query_lens = [all_query_lens[row] for row in request_rows]
        seq_lens = m.seq_lens_cpu_upper_bound.tolist()
        block_size = self.kv_cache_spec.block_size
        hash_block_size = self.vllm_config.cache_config.prefix_match_unit or block_size
        speculative_config = self.vllm_config.speculative_config
        drop_eagle_block = (
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        checkpoint_offsets, checkpoint_cols = compute_mamba_prefill_checkpoints(
            [seq_lens[row] for row in request_rows],
            query_lens,
            hash_block_size=hash_block_size,
            mamba_block_size=block_size,
            checkpoint_alignment=self.kv_cache_spec.prefill_checkpoint_alignment,
            drop_eagle_block=drop_eagle_block,
        )
        if not any(checkpoint_offsets):
            return None
        checkpoint_offsets_tensor = async_tensor_h2d(
            checkpoint_offsets,
            dtype=torch.int32,
            device=m.query_start_loc.device,
        )
        request_rows_tensor = async_tensor_h2d(
            request_rows, dtype=torch.int64, device=m.query_start_loc.device
        )
        checkpoint_cols_tensor = async_tensor_h2d(
            checkpoint_cols, dtype=torch.int64, device=m.query_start_loc.device
        )
        return MambaPrefillCheckpointMetadata(
            checkpoint_offsets_tensor,
            _gather_checkpoint_state_indices(
                m.block_table_tensor, request_rows_tensor, checkpoint_cols_tensor
            ),
            offsets=checkpoint_offsets,
            request_rows=request_rows_tensor,
            block_cols=checkpoint_cols_tensor,
        )


class MambaPrefillCheckpointExporter(ABC):
    """Export a mid-prefill checkpoint into backend-specific paged states."""

    @abstractmethod
    def export(
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *args,
        **kwargs,
    ) -> None:
        """Write checkpoint state into the paged cache."""
        raise NotImplementedError


@dataclass(frozen=True)
class ConvRecurrentCheckpointExporter(MambaPrefillCheckpointExporter):
    """Store a pre-convolution window and one recurrent state row per request."""

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
