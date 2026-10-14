# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numba
import numpy as np
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


class CPUBlockTables(BlockTables):
    """Block tables with numba loops in place of the Triton kernels.

    Each KV cache group is handled with its own tensors, so no pointer arrays
    or fused multi-group writes are needed.
    """

    def apply_staged_writes(self) -> None:
        for block_table in self.block_tables:
            block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if out is None:
            out = tuple(self.input_block_tables)
        num_blocks = self.num_blocks.gpu.numpy()
        for group_id, (src, dst) in enumerate(zip(self.block_tables, out)):
            _gather_block_table(
                src.gpu.numpy(),
                dst[:num_reqs_padded].numpy(),
                idx_mapping.numpy(),
                num_blocks[group_id],
            )
        return tuple(dst[:num_reqs_padded] for dst in out)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slot_mappings = self.slot_mappings if out is None else out
        slot_mappings.fill_(PAD_SLOT_ID)
        for group_id, block_table in enumerate(self.block_tables):
            if not self._slot_mapping_enabled[group_id]:
                continue
            sharded = self.cp_size != 1 and bool(self.dcp_sharded[group_id])
            _compute_slot_mapping(
                slot_mappings[group_id].numpy(),
                block_table.gpu.numpy(),
                idx_mapping.numpy(),
                query_start_loc.numpy(),
                positions.numpy(),
                self.kernel_block_sizes[group_id],
                self.block_sizes[group_id],
                self.cp_size if sharded else 1,
                self.cp_rank,
                self.cp_interleave,
            )
        return slot_mappings[:, :num_tokens_padded]


@numba.njit(cache=True, nogil=True)
def _gather_block_table(
    src: np.ndarray, dst: np.ndarray, idx_mapping: np.ndarray, num_blocks: np.ndarray
) -> None:
    num_reqs = len(idx_mapping)
    dst[num_reqs:] = 0
    for i in range(num_reqs):
        n = num_blocks[idx_mapping[i]]
        dst[i, :n] = src[idx_mapping[i], :n]


@numba.njit(cache=True, nogil=True)
def _compute_slot_mapping(
    slot_mapping: np.ndarray,
    block_table: np.ndarray,
    idx_mapping: np.ndarray,
    query_start_loc: np.ndarray,
    positions: np.ndarray,
    block_size: int,
    kv_block_size: int,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
) -> None:
    virtual_block_size = kv_block_size * cp_size
    for i in range(len(idx_mapping)):
        req = idx_mapping[i]
        # A negative idx_mapping entry is a dummy request without blocks.
        if req < 0:
            continue
        for t in range(query_start_loc[i], query_start_loc[i + 1]):
            pos = positions[t]
            if cp_size != 1:
                offset = pos % virtual_block_size
                if offset // cp_interleave % cp_size != cp_rank:
                    continue
                pos = (
                    pos // virtual_block_size * kv_block_size
                    + offset // (cp_interleave * cp_size) * cp_interleave
                    + offset % cp_interleave
                )
            block_number = block_table[req, pos // block_size]
            slot_mapping[t] = block_number * block_size + pos % block_size
