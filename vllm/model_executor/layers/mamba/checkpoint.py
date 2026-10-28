# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    cache_hit_alignment_tokens: int,
    num_hash_lookahead_tokens: int = 0,
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
            seq_len,
            cache_hit_alignment_tokens,
            drop_eagle_block=drop_eagle_block,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
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
    # Host copies of ``checkpoint_offsets`` and of each row's query length, for
    # backends that lay out the checkpoint before any tensor exists (Mamba2's
    # SSD chunk split, FlashInfer GDN's checkpoint plan).
    offsets: list[int] | None = None
    query_lens: list[int] | None = None
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
        speculative_config = vllm_config.speculative_config
        self.drop_eagle_block = (
            speculative_config is not None
            and speculative_config.use_eagle_block_drop()
            and not vllm_config.num_hash_lookahead_tokens
        )

    def build(
        self,
        m: CommonAttentionMetadata,
        request_rows: list[int],
    ) -> MambaPrefillCheckpointMetadata | None:
        if self.vllm_config.cache_config.mamba_cache_mode != "align":
            return None
        if self.kv_cache_spec.num_prefill_checkpoint_blocks == 0:
            return None
        cache_config = self.vllm_config.cache_config
        hash_block_size = cache_config.hash_block_size
        cache_hit_alignment_tokens = cache_config.cache_hit_alignment_tokens
        if hash_block_size is None or cache_hit_alignment_tokens is None:
            # Unresolved only for the minimal KV cache built for profiling.
            return None
        assert m.seq_lens_cpu_upper_bound is not None
        all_query_lens = m.query_start_loc_cpu.diff().tolist()
        query_lens = [all_query_lens[row] for row in request_rows]
        seq_lens = m.seq_lens_cpu_upper_bound.tolist()
        if self.kv_cache_spec.fixed_state_slots:
            checkpoint_offsets, checkpoint_cols = self.get_fixed_slot_checkpoints(
                m, request_rows, query_lens, seq_lens
            )
        else:
            checkpoint_offsets, checkpoint_cols = compute_mamba_prefill_checkpoints(
                [seq_lens[row] for row in request_rows],
                query_lens,
                hash_block_size=hash_block_size,
                mamba_block_size=self.kv_cache_spec.block_size,
                checkpoint_alignment=self.kv_cache_spec.prefill_checkpoint_alignment,
                drop_eagle_block=self.drop_eagle_block,
                cache_hit_alignment_tokens=cache_hit_alignment_tokens,
                num_hash_lookahead_tokens=self.vllm_config.num_hash_lookahead_tokens,
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
            query_lens=query_lens,
            request_rows=request_rows_tensor,
            block_cols=checkpoint_cols_tensor,
        )

    def get_fixed_slot_checkpoints(
        self,
        m: CommonAttentionMetadata,
        request_rows: list[int],
        query_lens: list[int],
        seq_lens: list[int],
    ) -> tuple[list[int], list[int]]:
        """Checkpoints the scheduler planned into the fixed checkpoint slot."""
        offsets = [0] * len(request_rows)
        cols = [-1] * len(request_rows)
        positions = m.mamba_checkpoint_positions
        if positions is None:
            return offsets, cols
        for i, (row, query_len) in enumerate(zip(request_rows, query_lens)):
            position = int(positions[row])
            if position == 0:
                continue
            offset = position - (seq_lens[row] - query_len)
            assert 0 < offset < query_len, (position, seq_lens[row], query_len)
            offsets[i] = offset
            cols[i] = self.kv_cache_spec.checkpoint_slot
        return offsets, cols


@dataclass(frozen=True)
class MambaPrefillCheckpointExporter:
    """Store each checkpoint's pre-convolution window and recurrent state.

    The recurrent states are rows of a buffer the prefill kernel emitted;
    ``recurrent_rows`` picks each checkpoint row's state out of it, and by
    default row ``i`` holds checkpoint ``i``'s. Layers with only a conv state
    pass no recurrent buffers. ``token_indices`` maps the rows' tokens, laid out
    by ``cu_seqlens``, to their rows of ``raw_qkv``.
    """

    state_len: int | None = None

    def export(
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *,
        raw_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_checkpoint: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        recurrent_rows: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
    ) -> None:
        state_len = (
            self.state_len if self.state_len is not None else conv_state.shape[-1]
        )
        width = raw_qkv.shape[-1]
        has_recurrent = recurrent_checkpoint is not None
        if recurrent_checkpoint is None:
            # Unread placeholders for the recurrent pointers.
            recurrent_checkpoint = recurrent_state = conv_state
            recurrent_row_size = 0
        else:
            assert recurrent_state is not None
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
            recurrent_rows if recurrent_rows is not None else checkpoint.state_indices,
            token_indices if token_indices is not None else cu_seqlens,
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
            has_recurrent,
            recurrent_rows is not None,
            token_indices is not None,
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
    recurrent_rows_ptr,
    token_indices_ptr,
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
    HAS_RECURRENT: tl.constexpr,
    HAS_RECURRENT_ROWS: tl.constexpr,
    HAS_TOKEN_MAP: tl.constexpr,
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
    if HAS_TOKEN_MAP:
        token_idx = tl.load(token_indices_ptr + token_idx, mask=valid_conv, other=0)
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

    if HAS_RECURRENT:
        if HAS_RECURRENT_ROWS:
            recurrent_row = tl.load(recurrent_rows_ptr + seq_idx).to(tl.int64)
        else:
            recurrent_row = seq_idx.to(tl.int64)
        valid_recurrent = (cols < RECURRENT_ROW_SIZE) & valid_checkpoint
        recurrent = tl.load(
            recurrent_checkpoint_ptr + recurrent_row * checkpoint_stride_0 + cols,
            mask=valid_recurrent,
        )
        tl.store(
            recurrent_state_ptr + state_idx * recurrent_state_stride_0 + cols,
            recurrent,
            mask=valid_recurrent,
        )
