# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


class CPUBlockTables(BlockTables):
    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups

        for group_id, output in enumerate(out):
            ops.gather_block_table_cpu(
                self.block_tables[group_id].gpu,
                output,
                idx_mapping,
                self.num_blocks.gpu[group_id],
                num_reqs_padded,
            )
        return tuple(block_table[:num_reqs_padded] for block_table in out)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slot_mappings = self.slot_mappings if out is None else out
        if self.num_kv_cache_groups == 0:
            return slot_mappings[:, :num_tokens_padded]

        for group_id in range(self.num_kv_cache_groups):
            dcp_sharded = self.cp_size != 1 and bool(self.dcp_sharded[group_id])
            ops.compute_slot_mapping_cpu(
                self.block_tables[group_id].gpu,
                slot_mappings[group_id],
                idx_mapping,
                query_start_loc,
                positions,
                self.block_sizes[group_id],
                self._slot_mapping_enabled[group_id],
                dcp_sharded,
                self.cp_size,
                self.cp_rank,
                self.cp_interleave,
                PAD_SLOT_ID,
            )
        return slot_mappings[:, :num_tokens_padded]
