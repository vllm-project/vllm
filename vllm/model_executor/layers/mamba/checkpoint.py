# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
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


@dataclass
class MambaPrefillCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor


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
        checkpoint_state_indices = m.block_table_tensor[
            request_rows_tensor, checkpoint_cols_tensor
        ]
        checkpoint_state_indices = torch.where(
            checkpoint_cols_tensor >= 0,
            checkpoint_state_indices,
            NULL_BLOCK_ID,
        )
        return MambaPrefillCheckpointMetadata(
            checkpoint_offsets_tensor,
            checkpoint_state_indices,
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
