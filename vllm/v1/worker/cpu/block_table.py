# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


class CPUBlockTables:
    # Set by the elastic EP warmup so KV writes land in the null block.
    redirect_writes_to_null_block = False

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        slot_mapping_enabled: list[bool] | None = None,
        sliding_window_reach: list[int] | None = None,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        assert len(slot_mapping_enabled) == self.num_kv_cache_groups
        self._slot_mapping_enabled = slot_mapping_enabled
        if sliding_window_reach is None:
            sliding_window_reach = [0] * self.num_kv_cache_groups
        assert len(sliding_window_reach) == self.num_kv_cache_groups
        self._sliding_window_reach = sliding_window_reach
        self.has_sliding_window_groups = any(r > 0 for r in sliding_window_reach)

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[torch.Tensor] = []
        for max_num_blocks in max_num_blocks_per_group:
            block_table = torch.zeros(
                (self.max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_tables.append(block_table)
        self.block_tables_np = [table.numpy() for table in self.block_tables]
        self.num_blocks = torch.zeros(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
            device=device,
        )
        self.num_blocks_np = self.num_blocks.numpy()

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(block_table) for block_table in self.block_tables
        ]
        self._num_dirty_rows = 0
        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks_np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            if self.redirect_writes_to_null_block:
                block_ids = [0] * len(block_ids)
            end = start + len(block_ids)
            row_capacity = self.block_tables[i].shape[1]
            if end > row_capacity:
                raise RuntimeError(
                    f"Block table write for request {req_index}, group {i} exceeds "
                    f"row capacity ({end} > {row_capacity})"
                )
            self.block_tables_np[i][req_index, start:end] = block_ids
            self.num_blocks_np[i, req_index] = end

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Gather the batch's block-table rows into the forward-pass tables.

        When num_computed_tokens ([max_num_reqs], indexed like the block tables)
        is given, sliding-window groups get every entry wholly below the window
        of the request's first scheduled token replaced by the null block. Those
        blocks may already have been freed by the sliding-window manager and
        reused, while attention kernels can still load them as part of a KV
        tile that straddles the window start: masked-out NaN in such a tile
        propagates into the output.
        """
        if self.num_kv_cache_groups == 0:
            return ()
        mask_sliding_window = (
            self.has_sliding_window_groups and num_computed_tokens is not None
        )
        num_reqs = idx_mapping.shape[0]
        num_rows = num_reqs_padded
        if out is None:
            out = tuple(self.input_block_tables)
            num_rows = max(num_rows, self._num_dirty_rows)
            self._num_dirty_rows = num_reqs
        else:
            assert len(out) == self.num_kv_cache_groups

        for group_id, output in enumerate(out):
            output[:num_rows].zero_()
            ops.gather_block_table_cpu(
                self.block_tables[group_id],
                output,
                idx_mapping,
                self.num_blocks[group_id],
                num_rows,
            )
            reach = self._sliding_window_reach[group_id]
            if mask_sliding_window and reach > 0 and num_reqs > 0:
                assert num_computed_tokens is not None
                first_positions = num_computed_tokens[idx_mapping].sub(reach)
                num_masked_blocks = first_positions.clamp_min_(0).div(
                    self.block_sizes[group_id], rounding_mode="floor"
                )
                block_indices = torch.arange(
                    output.shape[1], dtype=torch.int32, device=output.device
                )
                output[:num_reqs].masked_fill_(
                    block_indices.unsqueeze(0) < num_masked_blocks.unsqueeze(1), 0
                )
        return tuple(block_table[:num_reqs_padded] for block_table in out)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # Zero the rows so dummy runs write mamba state to the reserved null
        # block rather than through the previous real step's (stale) block
        # ids, which may point at blocks since freed and reallocated.
        num_rows = max(num_reqs, self._num_dirty_rows)
        for block_table in self.input_block_tables:
            block_table[:num_rows].zero_()
        # Profiling may populate the returned rows with valid block IDs.
        self._num_dirty_rows = num_reqs
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        slot_mappings = self.slot_mappings if out is None else out
        for group_id in range(self.num_kv_cache_groups):
            ops.compute_slot_mapping_cpu(
                self.block_tables[group_id],
                slot_mappings[group_id],
                idx_mapping,
                query_start_loc,
                positions,
                self.block_sizes[group_id],
                self._slot_mapping_enabled[group_id],
                PAD_SLOT_ID,
            )
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]
